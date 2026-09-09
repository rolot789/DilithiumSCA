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

# 멀티태스크 학습에 쓸 헤드 조합. 바이트별 HW는 합쳐서 10.83비트로 HW 단독의 2.7배다.
MULTITASK_HEADS = ["sign", "byte0", "byte1", "byte2"]


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
