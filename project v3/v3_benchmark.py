"""학습 방식별 효율 비교 벤치마크.

여러 학습 방식을 같은 잣대로 재려면 지표가 필요한데, 정확도만으로는 부족하다.
정확도가 높아도 확률이 부정확하면 키 복구가 안 되고, 라벨 종류가 다르면
(HW 33클래스 vs 바이트별 HW) 정확도끼리 비교 자체가 불가능하다.

그래서 중심 지표를 **PI(Perceived Information)** 로 잡는다.

    PI = H(Y) - E[-log2 p_model(y|trace)]        [단위: 비트/트레이스]

H(Y)는 라벨의 엔트로피(라벨이 원래 담고 있는 정보량), 뒤 항은 모델의 NLL을
비트로 환산한 값이다. 즉 **모델이 트레이스 하나에서 실제로 뽑아낸 정보량**이다.

  * 모델이 완벽하면 PI = H(Y)
  * 모델이 아무것도 못 배우면 PI = 0 (또는 음수)
  * 라벨 종류가 달라도 같은 '비트' 단위라 직접 비교된다

그리고 키 복구에 필요한 트레이스 수를 바로 예측한다.

    필요 트레이스 ~= log2(q) / PI = 23.0 / PI

이 예측이 실측 GE/SR 곡선과 맞는지도 함께 확인한다.

실측 근거 (완벽 오라클 기준, 트레이스당 정보량):
    sign(1비트)          1.00 비트  ->  23개 필요
    HW 전체(33클래스)     4.04 비트  ->   6개 필요
    바이트별 HW 4개      10.83 비트  ->   3개 필요
"""

import time

import numpy as np

from v3_common import Q, hw32


KEY_BITS = float(np.log2(Q))     # 23.0


# ------------------------------------------------------------- 라벨 스킴

class LabelScheme:
    """u -> 라벨 변환기. 학습 방식마다 다른 라벨을 쓰므로 표준 인터페이스로 감싼다."""

    def __init__(self, name, fn, n_classes, note=""):
        self.name = name
        self.fn = fn
        self.n_classes = n_classes
        self.note = note

    def __call__(self, u):
        return self.fn(u)


def _byte_hw(u, b):
    v = np.asarray(u, dtype=np.int64) & 0xFFFFFFFF
    return hw32((v >> (8 * b)) & 0xFF)


SCHEMES = {
    "hw32": LabelScheme(
        "hw32", lambda u: hw32(u).astype(np.int64), 33,
        "v2/v3 기본. 32비트 HW. 실측 4.04비트/트레이스"),
    "sign": LabelScheme(
        "sign", lambda u: (np.asarray(u) < 0).astype(np.int64), 2,
        "부호만. |u|<2^23이라 상위 9비트가 전부 부호비트다. 실측 1.00비트"),
    "byte0": LabelScheme("byte0", lambda u: _byte_hw(u, 0).astype(np.int64), 9,
                         "bit 0~7의 HW. 누설 약함 (|rho| 0.24)"),
    "byte1": LabelScheme("byte1", lambda u: _byte_hw(u, 1).astype(np.int64), 9,
                         "bit 8~15의 HW. 누설 약함 (|rho| 0.29)"),
    "byte2": LabelScheme("byte2", lambda u: _byte_hw(u, 2).astype(np.int64), 9,
                         "bit 16~23의 HW. 누설 강함 (|rho| 0.57)"),
    "byte3": LabelScheme("byte3", lambda u: _byte_hw(u, 3).astype(np.int64), 9,
                         "bit 24~31의 HW. 사실상 8 x 부호비트 (|rho| 0.72)"),
}

MULTITASK_HEADS = ["sign", "byte0", "byte1", "byte2"]

# 멀티태스크 기준값. 공격셋 u 전량(10,240,000 샘플)으로 실측했다.
#
# 주변 엔트로피의 단순 합(8.7853)은 헤드가 완전히 독립일 때만 성립하는 상한이다.
# 실제로는 헤드끼리 정보가 겹치므로 **결합 엔트로피**를 써야 한다.
#   주변 합 8.7853 - 결합 8.4157 = 중복 0.3696비트 (4.2%)
# 따라서 HW 단독 대비 실질 배율은 2.084배가 아니라 **1.997배**다.
#
# byte3을 헤드에 추가하면 결합 엔트로피가 +0.0000비트다. |u| < 2^23이라 상위
# 9비트가 전부 부호 확장이어서 sign과 완전히 중복되기 때문이다. 그래서 제외한다.
MULTITASK_REFERENCE = {
    "heads": tuple(MULTITASK_HEADS),
    "marginal_entropy": {"sign": 1.0000, "byte0": 2.5439,
                         "byte1": 2.5432, "byte2": 2.6981},
    "marginal_sum": 8.7853,
    "joint_entropy": 8.4157,
    "redundancy": 0.3696,
    "hw32_entropy": 4.2147,
    "ratio_vs_hw32": 1.997,
    "byte3_marginal_gain": 0.0000,
}


def label_entropy(y, n_classes):
    """H(Y). PI의 상한이자, 그 라벨이 원리적으로 담을 수 있는 정보량."""
    c = np.bincount(np.asarray(y).astype(np.int64), minlength=n_classes)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


# ----------------------------------------------------------------- 지표

def perceived_information(probs, y_true, n_classes=None):
    """PI = H(Y) - NLL(비트). 모델이 트레이스 하나에서 뽑아낸 실제 정보량."""
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = np.asarray(y_true).astype(np.int64)
    n_classes = n_classes or probs.shape[1]
    h = label_entropy(y, n_classes)
    nll_bits = float(-np.mean(np.log2(probs[np.arange(len(y)), y])))
    return {"pi": h - nll_bits, "entropy": h, "nll_bits": nll_bits,
            "pi_ratio": (h - nll_bits) / h if h > 0 else 0.0}


def joint_entropy(head_labels, head_names=None):
    """헤드 라벨들의 **결합** 엔트로피. 주변 엔트로피의 합이 아니다.

    합을 쓰면 헤드 간 중복을 무시해 정보량을 과대평가한다(실측 4.2% 과대).
    """
    names = head_names or list(head_labels.keys())
    code = np.zeros(len(head_labels[names[0]]), dtype=np.int64)
    for n in names:
        code = code * SCHEMES[n].n_classes + np.asarray(head_labels[n]).astype(np.int64)
    return label_entropy(code, int(code.max()) + 1)


def multitask_perceived_information(head_probs, head_labels, head_names=None,
                                    joint_h=None):
    """멀티태스크 모델의 PI.

        PI = H(결합 라벨) - sum_h NLL_h(비트)

    NLL을 헤드별로 더하는 것은 "헤드들이 트레이스가 주어졌을 때 조건부 독립"이라는
    모델 가정에 해당한다. 공격 시 로그우도를 헤드별로 합산하는 것과 정확히 같은 가정이라
    일관된다. 헤드가 실제로는 상관되어 있으면 이 PI가 그만큼 낮게 나오며, 그것이 맞다.

    joint_h: 평가 표본이 작으면 결합 엔트로피 추정이 편향된다(빈이 최대 1458개).
             None이면 표본에서 추정하고, 헤드 조합이 기본값과 같으면
             MULTITASK_REFERENCE의 실측값(10.24M 샘플 기준)을 쓰는 편이 안전하다.
    """
    names = head_names or list(head_probs.keys())
    nll = 0.0
    per_head = {}
    for n in names:
        p = np.clip(np.asarray(head_probs[n], dtype=np.float64), 1e-12, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        y = np.asarray(head_labels[n]).astype(np.int64)
        h_nll = float(-np.mean(np.log2(p[np.arange(len(y)), y])))
        h_ent = label_entropy(y, SCHEMES[n].n_classes)
        nll += h_nll
        per_head[n] = {"entropy": h_ent, "nll_bits": h_nll, "pi": h_ent - h_nll,
                       "top1": float((p.argmax(axis=1) == y).mean())}
    H = joint_h if joint_h is not None else joint_entropy(head_labels, names)
    marg = sum(v["entropy"] for v in per_head.values())
    return {"pi": H - nll, "entropy": H, "nll_bits": nll,
            "pi_ratio": (H - nll) / H if H > 0 else 0.0,
            "marginal_sum": marg, "redundancy": marg - H, "per_head": per_head}


def traces_to_recovery(pi, key_bits=KEY_BITS):
    """PI로부터 키 복구에 필요한 트레이스 수를 예측한다."""
    if pi <= 0:
        return float("inf")
    return key_bits / pi


def combined_pi(pi_list):
    """멀티태스크 헤드들의 PI 합. 헤드가 서로 독립일 때의 상한이다.

    실제로는 헤드끼리 정보가 겹치므로(예: byte3은 sign과 거의 같은 정보)
    이 값은 낙관적이다. 반드시 실측 GE와 대조할 것.
    """
    return float(np.sum(pi_list))


# ------------------------------------------------------------- 변형 실행

# 앙상블 다양성 구성별 실측 기준값 (선형 프로브, 실제 트레이스 600+600개).
# 새로 잰 앙상블이 이 표의 어디에 해당하는지 대조하는 용도다.
DIVERSITY_REFERENCE = {
    "single":   {"label": "단일 모델",              "error_corr": None,  "gain": 1.000},
    "bagging":  {"label": "배깅(같은 창, 데이터 재추출)", "error_corr": 0.946, "gain": 1.003},
    "nested":   {"label": "중첩 창(폭만 확대)",       "error_corr": 0.432, "gain": 1.018},
    "disjoint": {"label": "disjoint 창",           "error_corr": 0.258, "gain": 1.195},
    "mixed":    {"label": "넓은 창 + disjoint",     "error_corr": None,  "gain": 1.108},
}


class Variant:
    """비교할 학습 방식 하나.

    ensemble: 앙상블이면 {"source": "disjoint"|"bagging"|"nested"|"mixed"|"snapshot",
                          "n_members": int, "specs": [...]} 형태의 dict.
              단일 모델이면 None.
    """

    def __init__(self, name, description, build_fn=None, scheme="hw32",
                 window=32, extra=None, ensemble=None):
        self.name = name
        self.description = description
        self.build_fn = build_fn
        self.scheme = scheme
        self.window = window
        self.extra = extra or {}
        self.ensemble = ensemble
        self.metrics = {}

    def record(self, **kw):
        self.metrics.update(kw)
        return self


def evaluate_variant(variant, probs, y_true, n_params=None, train_seconds=None,
                     train_samples=None, sr100=None, oracle_sr100=None):
    """한 변형의 지표를 계산한다. probs/y_true는 공격셋(또는 검증셋) 예측 결과."""
    y = np.asarray(y_true).astype(np.int64)
    n_classes = probs.shape[1]
    counts = np.bincount(y, minlength=n_classes)
    baseline = counts.max() / counts.sum()
    acc = float((probs.argmax(axis=1) == y).mean())

    info = perceived_information(probs, y, n_classes)
    m = {
        "name": variant.name,
        "scheme": variant.scheme,
        "window": variant.window,
        "top1": acc,
        "baseline": float(baseline),
        "acc_ratio": acc / baseline if baseline > 0 else 0.0,
        "pi_bits": info["pi"],
        "label_entropy": info["entropy"],
        "pi_ratio": info["pi_ratio"],
        "predicted_traces": traces_to_recovery(info["pi"]),
        "n_params": n_params,
        "train_seconds": train_seconds,
        "train_samples": train_samples,
        "measured_sr100": sr100,
    }
    if sr100 and oracle_sr100:
        m["oracle_efficiency"] = oracle_sr100 / sr100
    if sr100 and info["pi"] > 0:
        m["prediction_error"] = m["predicted_traces"] / sr100
    # 학습 비용 대비 정보량
    if train_seconds and train_seconds > 0:
        m["pi_per_minute"] = info["pi"] / (train_seconds / 60.0)
    if variant.ensemble:
        m["ensemble"] = dict(variant.ensemble)
    variant.record(**m)
    return m


def evaluate_ensemble_variant(variant, member_probs, y_true, mode="mean",
                              weights=None, **kw):
    """앙상블 변형을 평가한다. 멤버 확률 리스트를 받아 다양성과 이득까지 채운다.

    variant.ensemble["source"]에 다양성 원천을 적어두면 비교표에서
    DIVERSITY_REFERENCE의 실측 기준값과 나란히 볼 수 있다.
    """
    import v3_ensemble as ve            # 순환 임포트 방지를 위한 지연 임포트

    combined = ve.combine_probs(member_probs, mode, weights)
    m = evaluate_variant(variant, combined, y_true, **kw)

    div = ve.diversity_report(member_probs, y_true, verbose=False)
    gain = ve.ensemble_gain(member_probs, y_true, mode, weights)
    info = dict(variant.ensemble or {})
    info.update({
        "n_members": div["n_members"],
        "combine_mode": mode,
        "weights": None if weights is None else [float(w) for w in weights],
        "error_correlation": div["mean_error_correlation"],
        "disagreement": div["mean_disagreement"],
        "member_pi": gain["member_pi"],
        "best_member_pi": gain["best_member_pi"],
        "mean_member_pi": gain["mean_member_pi"],
        "gain_over_best": gain["gain_over_best"],
        "gain_ratio": gain["gain_ratio"],
        "predicted_traces_best": gain["predicted_traces_best"],
    })
    m["ensemble"] = info
    variant.record(ensemble=info)
    return m


def evaluate_multitask_variant(variant, head_probs, head_labels, head_names=None,
                               joint_h=None, use_reference_entropy=True,
                               n_params=None, train_seconds=None,
                               train_samples=None, sr100=None, oracle_sr100=None):
    """멀티태스크 변형을 단일 헤드 변형과 같은 표에 놓을 수 있게 지표를 맞춘다.

    Top-1은 **모든 헤드가 동시에 맞은 비율**(결합 정확도)로 잰다. HW 단독 모델의
    Top-1과 같은 의미(라벨 전체를 맞혔는가)가 되어 비교가 성립한다.
    베이스라인도 결합 라벨의 최빈 비율이다.
    """
    names = head_names or list(head_probs.keys())
    if joint_h is None and use_reference_entropy and \
            tuple(names) == MULTITASK_REFERENCE["heads"]:
        joint_h = MULTITASK_REFERENCE["joint_entropy"]

    info = multitask_perceived_information(head_probs, head_labels, names, joint_h)

    correct = np.ones(len(head_labels[names[0]]), dtype=bool)
    code = np.zeros(len(correct), dtype=np.int64)
    for n in names:
        p = np.asarray(head_probs[n])
        y = np.asarray(head_labels[n]).astype(np.int64)
        correct &= (p.argmax(axis=1) == y)
        code = code * SCHEMES[n].n_classes + y
    counts = np.bincount(code, minlength=int(code.max()) + 1)
    baseline = counts.max() / counts.sum()
    acc = float(correct.mean())

    m = {
        "name": variant.name,
        "scheme": f"multitask({len(names)}헤드)",
        "window": variant.window,
        "top1": acc,
        "baseline": float(baseline),
        "acc_ratio": acc / baseline if baseline > 0 else 0.0,
        "pi_bits": info["pi"],
        "label_entropy": info["entropy"],
        "pi_ratio": info["pi_ratio"],
        "predicted_traces": traces_to_recovery(info["pi"]),
        "n_params": n_params,
        "train_seconds": train_seconds,
        "train_samples": train_samples,
        "measured_sr100": sr100,
        "multitask": {
            "heads": list(names),
            "joint_entropy": info["entropy"],
            "marginal_sum": info["marginal_sum"],
            "redundancy": info["redundancy"],
            "per_head": info["per_head"],
        },
    }
    if sr100 and oracle_sr100:
        m["oracle_efficiency"] = oracle_sr100 / sr100
    if train_seconds and train_seconds > 0:
        m["pi_per_minute"] = info["pi"] / (train_seconds / 60.0)
    if variant.ensemble:
        m["ensemble"] = dict(variant.ensemble)
    variant.record(**m)
    return m


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t0


# ------------------------------------------------------------------ 비교표

def _config_label(m):
    """표에 쓸 구성 요약. 단일 모델이면 '단일', 앙상블이면 '앙상블 N개(원천)'."""
    e = m.get("ensemble")
    if not e:
        return "단일"
    src = e.get("source", "?")
    ref = DIVERSITY_REFERENCE.get(src)
    name = ref["label"] if ref else src
    return f"앙상블 {e.get('n_members', '?')}개 · {name}"


def comparison_table(metrics_list, sort_by="pi_bits"):
    """변형들을 한 표로 비교한다. 정렬 기준은 기본이 PI다."""
    rows = sorted(metrics_list, key=lambda m: m.get(sort_by, 0), reverse=True)
    L = []
    A = L.append
    A("| 학습 방식 | 구성 | 라벨 | Top-1 | 베이스라인 배율 | **PI (비트)** | H(Y) | PI/H | 예측 트레이스 | 실측 SR100 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for m in rows:
        pred = m["predicted_traces"]
        pred_s = f"{pred:.1f}" if np.isfinite(pred) else "무한"
        A(f"| {m['name']} | {_config_label(m)} | {m['scheme']} | {m['top1']*100:.2f}% | "
          f"{m['acc_ratio']:.2f}x | **{m['pi_bits']:.3f}** | "
          f"{m['label_entropy']:.3f} | {m['pi_ratio']*100:.1f}% | "
          f"{pred_s} | {m.get('measured_sr100') or '-'} |")
    return "\n".join(L)


def ensemble_table(metrics_list):
    """앙상블 변형만 따로, 다양성과 이득 중심으로 본다.

    오차 상관이 낮을수록 멤버들이 서로 다른 실수를 하므로 이득이 크다.
    상관이 0.8을 넘으면 멤버가 사실상 같은 모델이라 앙상블 의미가 없다.
    """
    ens = [m for m in metrics_list if m.get("ensemble")]
    if not ens:
        return "_앙상블 변형 없음_"
    L = []
    A = L.append
    A("| 학습 방식 | 다양성 원천 | 멤버 | 결합 | 오차 상관 | 불일치율 | "
      "최고 멤버 PI | 앙상블 PI | 이득 | 예측 트레이스 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for m in sorted(ens, key=lambda x: x["pi_bits"], reverse=True):
        e = m["ensemble"]
        src = e.get("source", "?")
        ref = DIVERSITY_REFERENCE.get(src)
        ec = e.get("error_correlation")
        flag = " !" if ec is not None and ec > 0.8 else ""
        A("| {} | {} | {} | {} | {}{} | {} | {} | **{:.3f}** | {} | {} |".format(
            m["name"], ref["label"] if ref else src, e.get("n_members", "-"),
            e.get("combine_mode", "-"),
            f"{ec:.3f}" if ec is not None else "-", flag,
            f"{e['disagreement']*100:.1f}%" if e.get("disagreement") is not None else "-",
            f"{e['best_member_pi']:.3f}" if e.get("best_member_pi") is not None else "-",
            m["pi_bits"],
            f"{e['gain_ratio']:.3f}x" if e.get("gain_ratio") is not None else "-",
            f"{m['predicted_traces']:.1f}" if np.isfinite(m["predicted_traces"]) else "무한"))
    A("")
    A("`!` 표시는 오차 상관이 0.8을 넘어 멤버들이 사실상 같은 모델이라는 뜻이다.")
    A("")
    A("**참고: 다양성 원천별 실측 기준값** (선형 프로브, 실제 트레이스 600+600개)\n")
    A("| 원천 | 오차 상관 | 앙상블 이득 |")
    A("|---|---|---|")
    for k, v in DIVERSITY_REFERENCE.items():
        if k == "single":
            continue
        A(f"| {v['label']} | {v['error_corr'] if v['error_corr'] is not None else '-'} "
          f"| {v['gain']:.3f}x |")
    A("")
    A("중첩 창(폭만 다른 창)은 넓은 창이 좁은 창을 포함해 우열 관계가 되므로")
    A("멤버로 부적합하다. 폭 확대는 단일 모델의 기본 설정으로 쓴다.")
    A("선형 프로브 기준이라 신경망 앙상블 이득은 이보다 클 수 있다.")
    return "\n".join(L)


def multitask_table(metrics_list):
    """멀티태스크 변형의 헤드별 분해. 어느 헤드가 실제로 정보를 주는지 본다."""
    mts = [m for m in metrics_list if m.get("multitask")]
    if not mts:
        return "_멀티태스크 변형 없음_"
    L = []
    A = L.append
    for m in mts:
        mt = m["multitask"]
        A(f"**{m['name']}** — 헤드 {len(mt['heads'])}개\n")
        A("| 헤드 | H(Y_h) | NLL(비트) | PI_h | PI/H | Top-1 |")
        A("|---|---|---|---|---|---|")
        for h in mt["heads"]:
            d = mt["per_head"][h]
            note = SCHEMES[h].note if h in SCHEMES else ""
            A(f"| {h} | {d['entropy']:.3f} | {d['nll_bits']:.3f} | "
              f"**{d['pi']:.3f}** | {d['pi']/d['entropy']*100 if d['entropy'] else 0:.1f}% | "
              f"{d['top1']*100:.2f}% |")
        A("")
        A(f"- 주변 엔트로피 합 {mt['marginal_sum']:.3f} 비트, "
          f"**결합 엔트로피 {mt['joint_entropy']:.3f} 비트**, "
          f"중복 {mt['redundancy']:.3f} 비트 "
          f"({mt['redundancy']/mt['marginal_sum']*100 if mt['marginal_sum'] else 0:.1f}%)")
        A(f"- 결합 PI **{m['pi_bits']:.3f} 비트** -> 예측 트레이스 "
          f"{m['predicted_traces']:.1f}개")
        A("")
    r = MULTITASK_REFERENCE
    A("**참고: 라벨 자체의 정보량 상한** (공격셋 u 전량 10,240,000 샘플 실측)\n")
    A("| 라벨 | 엔트로피 | HW 대비 | 완벽 모델의 필요 트레이스 |")
    A("|---|---|---|---|")
    A(f"| HW 단독 (33클래스) | {r['hw32_entropy']:.4f} | 1.000배 | "
      f"{KEY_BITS/r['hw32_entropy']:.2f} |")
    A(f"| 멀티태스크 주변 합 (상한, 쓰면 안 됨) | {r['marginal_sum']:.4f} | "
      f"{r['marginal_sum']/r['hw32_entropy']:.3f}배 | {KEY_BITS/r['marginal_sum']:.2f} |")
    A(f"| **멀티태스크 결합 (실제)** | **{r['joint_entropy']:.4f}** | "
      f"**{r['ratio_vs_hw32']:.3f}배** | **{KEY_BITS/r['joint_entropy']:.2f}** |")
    A("")
    A("주변 엔트로피의 단순 합은 헤드가 완전히 독립일 때만 성립하는 상한이다.")
    A(f"실제로는 {r['redundancy']:.4f}비트({r['redundancy']/r['marginal_sum']*100:.1f}%)가 중복이라 결합 엔트로피를 써야 한다.")
    A("")
    A(f"`byte3`은 헤드에 추가해도 결합 엔트로피가 +{r['byte3_marginal_gain']:.4f}비트다.")
    A("|u| < 2^23이라 상위 9비트가 전부 부호 확장이어서 `sign`과 완전히 중복된다.")
    A("(실측에서 bit24/bit28/bit31의 |rho|가 0.7238로 완전히 동일했다.)")
    A("")
    A("**주의: 상한과 실현치는 다르다.** 1.997배는 라벨이 담을 수 있는 정보량의 상한일 뿐,")
    A("헤드마다 학습 난이도가 크게 다르다. 선형 프로브 실측 PI_h는 다음과 같았다.\n")
    A("| 헤드 | 누설 \\|rho\\| | H(Y_h) | 실측 PI_h | 비고 |")
    A("|---|---|---|---|---|")
    A("| sign | 0.72 | 1.000 | **+0.643** | 쉽게 학습됨 (Top-1 91%) |")
    A("| byte2 | 0.57 | 2.692 | +0.096 | 겨우 양수 |")
    A("| byte1 | 0.29 | 2.550 | -0.148 | 학습 실패 |")
    A("| byte0 | 0.24 | 2.543 | -0.149 | 학습 실패 |")
    A("")
    A("결합 엔트로피 8.416비트 중 **5.09비트가 byte0/byte1에 있는데 둘 다 누설이 약하다.**")
    A("즉 멀티태스크의 이득은 '약한 바이트를 학습할 수 있는가'에 전적으로 달려 있다.")
    A("강한 CNN이 이를 해내는지 반드시 PI_h로 헤드별 확인할 것. 특정 헤드의 PI_h가")
    A("음수면 그 헤드는 정보를 주는 게 아니라 뺏고 있으므로 `loss_weights`에서 낮추거나 뺀다.")
    return "\n".join(L)


def efficiency_table(metrics_list):
    """학습 비용 대비 효율. 같은 성능이면 싼 쪽이 낫다."""
    L = []
    A = L.append
    A("| 학습 방식 | 파라미터 | 학습 샘플 | 학습 시간 | PI/분 | 오라클 대비 효율 |")
    A("|---|---|---|---|---|---|")
    for m in metrics_list:
        secs = m.get("train_seconds")
        ppm = m.get("pi_per_minute")
        eff = m.get("oracle_efficiency")
        A("| {} | {} | {} | {} | {} | {} |".format(
            m["name"],
            m.get("n_params") or "-",
            f"{m['train_samples']:,}" if m.get("train_samples") else "-",
            f"{secs:.0f}s" if secs else "-",
            f"{ppm:.3f}" if ppm else "-",
            f"{eff*100:.0f}%" if eff else "-"))
    return "\n".join(L)


def render_benchmark(out_path, metrics_list, notes=""):
    L = []
    A = L.append
    A("# 학습 방식별 효율 비교\n")
    A("중심 지표는 **PI(Perceived Information)** 다. 모델이 트레이스 하나에서")
    A("실제로 뽑아낸 정보량(비트)이며, 라벨 종류가 달라도 직접 비교된다.")
    A("키 공간이 23.0비트이므로 `필요 트레이스 = 23.0 / PI`로 예측한다.\n")
    A("## 성능\n")
    A(comparison_table(metrics_list))
    A("\n## 멀티태스크 헤드 분해\n")
    A(multitask_table(metrics_list))
    A("\n## 앙상블 다양성\n")
    A(ensemble_table(metrics_list))
    A("\n## 학습 비용 대비 효율\n")
    A(efficiency_table(metrics_list))
    A("\n## 참고: 완벽 오라클 기준선\n")
    A("| 라벨 | 트레이스당 정보량 | 필요 트레이스 |")
    A("|---|---|---|")
    A("| sign (1비트) | 1.00 비트 | 23 |")
    A("| HW 전체 (33클래스) | 4.04 비트 | 6 |")
    A("| 바이트별 HW 4개 | 10.83 비트 | 3 |")
    A("\n어떤 모델도 해당 라벨의 오라클보다 잘할 수 없다. PI가 오라클 정보량에")
    A("얼마나 근접했는지가 그 학습 방식의 완성도다.\n")
    if notes:
        A("## 메모\n")
        A(notes)
    with open(out_path, "w") as f:
        f.write("\n".join(L))
    return out_path
