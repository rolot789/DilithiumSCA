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

라벨 자체의 정보량 상한 (공격셋 u 전량 10,240,000 샘플로 실측한 엔트로피):
    sign(1비트)               1.0000 비트  ->  23.0개 필요
    HW 전체(33클래스)          4.2147 비트  ->   5.46개 필요
    멀티태스크 결합(4헤드)      8.4157 비트  ->   2.73개 필요

주변 엔트로피의 단순 합(8.7853)은 헤드가 독립일 때만 성립하는 상한이므로 쓰지
않는다. 실제 중복이 0.3696비트(4.2%)다. 자세한 값은 MULTITASK_REFERENCE 참조.
"""

import json
import os
import time
from datetime import datetime

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
        "v2/v3 기본. 32비트 HW. H(Y)=4.2147비트"),
    "sign": LabelScheme(
        "sign", lambda u: (np.asarray(u) < 0).astype(np.int64), 2,
        "부호만. |u|<2^23이라 상위 9비트가 전부 부호 확장이다. H(Y)=1.0000비트"),
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

# ---------------------------------------------------------------------------
# 두 종류의 기준값을 구분한다. 섞으면 갱신할 때 사고가 난다.
#
#   MULTITASK_REFERENCE (아래, 상수)
#       라벨 자체의 엔트로피. u 분포의 성질이라 **모델과 무관**하다.
#       CNN을 학습해도 바뀌지 않으므로 상수로 못박는다.
#
#   head_baseline.json (아래 load/record 함수)
#       헤드별 PI, 한계 기여, 선택된 헤드. **모델에 따라 달라진다.**
#       현재는 선형 프로브로 잰 값이 들어 있고, CNN 결과가 나오면 교체된다.
# ---------------------------------------------------------------------------

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


HEAD_BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "head_baseline.json")


def load_head_baseline(path=None):
    """모델 의존적인 헤드 기준값을 읽는다. 없으면 None."""
    p = path or HEAD_BASELINE_PATH
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def record_head_baseline(source, per_head, chosen_heads, combos=None,
                         notes="", path=None, model_info=None, sync=True):
    """헤드 기준값을 갱신한다. **CNN 학습 결과가 나오면 이 함수를 호출한다.**

    source     : "linear_probe" | "cnn" 등 무엇으로 잰 값인지
    per_head   : {헤드: {"top1":…, "pi_standalone":…, "marginal":…}}
    chosen_heads: select_heads()가 고른 헤드 목록
    combos     : [{"heads":[…], "pi":…, "predicted_traces":…}, …] (선택)

    이전 기준값은 `history`에 쌓이므로 선형 프로브 -> CNN 변화를 추적할 수 있다.
    라벨 엔트로피(MULTITASK_REFERENCE)는 여기서 건드리지 않는다. 모델과 무관하기 때문이다.
    """
    p = path or HEAD_BASELINE_PATH
    prev = load_head_baseline(p)
    payload = {
        "source": source,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "per_head": per_head,
        "chosen_heads": list(chosen_heads),
        "combos": combos or [],
        "notes": notes,
        "model_info": model_info or {},
        "history": (prev.get("history", []) + [{
            k: prev[k] for k in ("source", "recorded_at", "per_head",
                                 "chosen_heads", "combos", "notes")
            if k in prev}]) if prev else [],
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if sync and path is None:      # 임시 경로로 쓴 경우엔 문서를 건드리지 않는다
        sync_docs(baseline=payload, verbose=False)
    return p


AUTO_BLOCK_START = "<!-- AUTO:head-baseline:start -->"
AUTO_BLOCK_END = "<!-- AUTO:head-baseline:end -->"

_HERE = os.path.dirname(os.path.abspath(__file__))
DOC_TARGETS = [os.path.join(_HERE, "instruction_v3.md"),
               os.path.join(_HERE, "README.md")]


def render_head_baseline_md(baseline=None):
    """head_baseline.json을 사람이 읽는 마크다운 블록으로 만든다."""
    b = baseline or load_head_baseline()
    if not b:
        return "_헤드 기준값 미기록. `baseline_from_ablation()`으로 기록할 것._"
    src = b.get("source", "?")
    tag = {"linear_probe": "선형 프로브", "cnn": "CNN"}.get(src, src)
    L = [f"**측정 출처: {tag}** (기록 {b.get('recorded_at', '?')})"]
    mi = b.get("model_info") or {}
    if mi:
        L.append("")
        L.append("- " + ", ".join(f"{k}={v}" for k, v in mi.items()))
    L += ["", "| 헤드 | Top-1 | PI_h(단독) | 한계 기여 | 판정 |",
          "|---|---|---|---|---|"]
    for h, d in b.get("per_head", {}).items():
        marg = d.get("marginal")
        L.append("| {} | {} | {} | {} | {} |".format(
            h,
            f"{d['top1']*100:.2f}%" if d.get("top1") is not None else "-",
            f"{d['pi_standalone']:+.3f}" if d.get("pi_standalone") is not None else "-",
            f"**{marg:+.3f}**" if marg is not None else "-",
            "**빼는 게 이득**" if d.get("drop_is_better") else "유지"))
    if b.get("combos"):
        L += ["", "| 헤드 조합 | 결합 PI | 예측 트레이스 |", "|---|---|---|"]
        for c in b["combos"]:
            pt = c.get("predicted_traces")
            pt_s = f"{pt:.1f}" if pt is not None and np.isfinite(pt) else "무한"
            L.append("| {} | {:.3f} | {} |".format("+".join(c["heads"]),
                                                   c["pi"], pt_s))
    L += ["", f"**선택된 헤드: {'+'.join(b.get('chosen_heads', [])) or '-'}**"]
    if b.get("notes"):
        L += ["", b["notes"]]
    if src == "linear_probe":
        L += ["", "**이 값은 선형 프로브 기준이다.** 약한 바이트를 학습해내는 강한 "
                  "CNN이라면 결론이 달라진다. 스킬 Step 6-5에서 "
                  "`baseline_from_ablation('cnn', ...)`을 호출하면 이 블록이 "
                  "CNN 실측치로 자동 교체된다."]
    return "\n".join(L)


def sync_docs(paths=None, baseline=None, verbose=True):
    """문서의 AUTO 블록을 현재 기준값으로 다시 쓴다.

    마커(`AUTO:head-baseline:start`/`end`) 사이만 교체하므로 손으로 쓴 설명은
    보존된다. 마커가 없는 파일은 건너뛴다.
    """
    block = render_head_baseline_md(baseline)
    updated = []
    for p in (paths or DOC_TARGETS):
        if not os.path.exists(p):
            continue
        with open(p) as f:
            txt = f.read()
        i, j = txt.find(AUTO_BLOCK_START), txt.find(AUTO_BLOCK_END)
        if i < 0 or j < 0 or j < i:
            if verbose:
                print(f"  건너뜀(마커 없음): {p}")
            continue
        new = (txt[:i + len(AUTO_BLOCK_START)] + "\n\n" + block + "\n\n"
               + txt[j:])
        if new != txt:
            with open(p, "w") as f:
                f.write(new)
            updated.append(p)
            if verbose:
                print(f"  갱신: {p}")
        elif verbose:
            print(f"  변경 없음: {p}")
    return updated


def baseline_from_ablation(source, head_probs, head_labels, head_names=None,
                           model_info=None, notes="", path=None):
    """한 번의 호출로 한계 기여를 계산하고 기준값까지 기록한다.

    Step 6-5 끝에서 이것만 호출하면 된다. **캘리브레이션셋 확률을 넘길 것.**
    """
    names = list(head_names or head_probs.keys())
    info = multitask_perceived_information(head_probs, head_labels, names)
    abl = head_ablation(head_probs, head_labels, names)
    per_head = {}
    for h in names:
        d = info["per_head"][h]
        a = abl["heads"].get(h) or {}
        per_head[h] = {"top1": d["top1"], "pi_standalone": d["pi"],
                       "marginal": a.get("marginal"),
                       "drop_is_better": a.get("drop_is_better")}
    chosen = select_heads(head_probs, head_labels, names, verbose=False)
    combos = [{"heads": chosen["heads"], "pi": chosen["pi"],
               "predicted_traces": traces_to_recovery(chosen["pi"])}]
    if chosen["heads"] != names:          # 전체 조합이 곧 선택이면 중복이다
        combos.append({"heads": names, "pi": info["pi"],
                       "predicted_traces": traces_to_recovery(info["pi"])})
    return record_head_baseline(source, per_head, chosen["heads"], combos,
                                notes=notes, path=path, model_info=model_info)


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


def head_ablation(head_probs, head_labels, head_names=None):
    """헤드를 하나씩 빼보며 결합 PI가 얼마나 떨어지는지 잰다.

    **헤드 선택은 PI_h(단독)가 아니라 이 한계 기여도로 해야 한다.**

    PI_h(단독) = H(Y_h) - NLL_h 는 그 헤드만 봤을 때의 정보량이다. 그런데 결합
    PI에 대한 실제 기여는 `ΔH(결합) - NLL_h` 이고, 헤드끼리 겹치면 ΔH가 H(Y_h)보다
    작아져 부호가 뒤집힐 수 있다.

    실측 예 (선형 프로브): byte2는 PI_h = +0.241로 양수지만, 결합에 넣으면
    H(결합)을 +2.326만 올리는데 NLL은 2.452를 더해 한계 기여가 **-0.126**이다.
    sign과 0.366비트 겹치기 때문이다. 실제로 byte2를 빼면 PI가 0.535에서
    0.660으로 올랐다.

    반환: {헤드: 그 헤드를 뺐을 때의 PI 변화}. **양수면 빼는 게 이득**이다.
    """
    names = list(head_names or head_probs.keys())
    full = multitask_perceived_information(head_probs, head_labels, names)["pi"]
    out = {}
    for h in names:
        rest = [n for n in names if n != h]
        if not rest:
            out[h] = None
            continue
        sub = multitask_perceived_information(
            {n: head_probs[n] for n in rest},
            {n: head_labels[n] for n in rest}, rest)["pi"]
        out[h] = {"pi_without": sub, "marginal": full - sub,
                  "drop_is_better": sub > full}
    return {"full_pi": full, "heads": out}


def select_heads(head_probs, head_labels, head_names=None, verbose=True):
    """결합 PI가 가장 높은 헤드 부분집합을 탐욕적으로 고른다.

    한계 기여가 음수인 헤드를 하나씩 제거하며, 더 이상 개선되지 않으면 멈춘다.
    반드시 **캘리브레이션셋**에서 호출할 것. 공격셋으로 고르면 테스트셋 튜닝이 된다.
    """
    names = list(head_names or head_probs.keys())
    best_pi = multitask_perceived_information(head_probs, head_labels, names)["pi"]
    while len(names) > 1:
        cand = None
        for h in names:
            rest = [n for n in names if n != h]
            pi = multitask_perceived_information(
                {n: head_probs[n] for n in rest},
                {n: head_labels[n] for n in rest}, rest)["pi"]
            if pi > best_pi and (cand is None or pi > cand[1]):
                cand = (h, pi)
        if cand is None:
            break
        if verbose:
            print(f"    헤드 '{cand[0]}' 제거 -> PI {best_pi:.3f} -> {cand[1]:.3f}")
        names.remove(cand[0])
        best_pi = cand[1]
    return {"heads": names, "pi": best_pi}


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
                 window=96, extra=None, ensemble=None):
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
            "ablation": head_ablation(head_probs, head_labels, names)
            if len(names) > 1 else None,
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


def evaluate_multitask_ensemble_variant(variant, member_head_probs, head_labels,
                                        mode="mean", per_head_weights=None,
                                        head_names=None, joint_h=None, **kw):
    """멀티태스크 모델들로 구성한 앙상블. 헤드별로 결합한 뒤 결합 PI로 평가한다."""
    import v3_ensemble as ve

    names = head_names or list(member_head_probs[0].keys())
    combined = ve.combine_multitask(member_head_probs, mode, per_head_weights)
    m = evaluate_multitask_variant(variant, combined, head_labels,
                                   head_names=names, joint_h=joint_h, **kw)

    div = ve.diversity_report_multitask(member_head_probs, head_labels, verbose=False)
    if joint_h is None and tuple(names) == MULTITASK_REFERENCE["heads"]:
        joint_h = MULTITASK_REFERENCE["joint_entropy"]
    member_pi = [multitask_perceived_information(mp, head_labels, names, joint_h)["pi"]
                 for mp in member_head_probs]
    best = max(member_pi)
    info = dict(variant.ensemble or {})
    info.update({
        "n_members": len(member_head_probs),
        "combine_mode": mode,
        "error_correlation": div["mean_error_correlation"],
        "disagreement": div["mean_disagreement"],
        "member_pi": member_pi,
        "best_member_pi": float(best),
        "mean_member_pi": float(np.mean(member_pi)),
        "gain_over_best": float(m["pi_bits"] - best),
        "gain_ratio": float(m["pi_bits"] / best) if best > 0 else float("inf"),
        "predicted_traces_best": traces_to_recovery(best),
        "per_head_diversity": {h: v["mean_error_correlation"]
                               for h, v in div["per_head"].items()},
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
        abl = (mt.get("ablation") or {}).get("heads", {})
        A("| 헤드 | H(Y_h) | NLL(비트) | PI_h (단독) | **한계 기여** | 빼면? | Top-1 |")
        A("|---|---|---|---|---|---|---|")
        for h in mt["heads"]:
            d = mt["per_head"][h]
            a = abl.get(h) or {}
            marg = a.get("marginal")
            A("| {} | {:.3f} | {:.3f} | {:+.3f} | {} | {} | {:.2f}% |".format(
                h, d["entropy"], d["nll_bits"], d["pi"],
                f"**{marg:+.3f}**" if marg is not None else "-",
                "**빼는 게 이득**" if a.get("drop_is_better") else "유지",
                d["top1"] * 100))
        A("")
        A("**헤드 선택은 `PI_h(단독)`이 아니라 `한계 기여`로 한다.** 헤드끼리 정보가")
        A("겹치면 단독 PI가 양수여도 결합에 넣었을 때 손해일 수 있다.")
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
    A("헤드마다 학습 난이도가 크게 다르다.\n")

    base = load_head_baseline()
    if base:
        src = base.get("source", "?")
        tag = {"linear_probe": "선형 프로브", "cnn": "CNN"}.get(src, src)
        A(f"**기준값 출처: {tag}** (기록 {base.get('recorded_at', '?')})\n")
        A("| 헤드 | Top-1 | PI_h(단독) | 한계 기여 | 판정 |")
        A("|---|---|---|---|---|")
        for h, d in base.get("per_head", {}).items():
            marg = d.get("marginal")
            A("| {} | {} | {} | {} | {} |".format(
                h,
                f"{d['top1']*100:.2f}%" if d.get("top1") is not None else "-",
                f"{d['pi_standalone']:+.3f}" if d.get("pi_standalone") is not None else "-",
                f"**{marg:+.3f}**" if marg is not None else "-",
                "빼는 게 이득" if d.get("drop_is_better") else "유지"))
        A("")
        if base.get("combos"):
            A("| 헤드 조합 | 결합 PI | 예측 트레이스 |")
            A("|---|---|---|")
            for c in base["combos"]:
                pt = c.get("predicted_traces")
                pt_s = f"{pt:.1f}" if pt is not None and np.isfinite(pt) else "무한"
                A("| {} | {:.3f} | {} |".format("+".join(c["heads"]), c["pi"], pt_s))
            A("")
        A(f"선택된 헤드: **{'+'.join(base.get('chosen_heads', [])) or '-'}**")
        if base.get("notes"):
            A(f"\n{base['notes']}")
        if src == "linear_probe":
            A("")
            A("**이 값은 선형 프로브 기준이다.** 약한 바이트를 학습해내는 강한 CNN이라면")
            A("결론이 달라진다. Step 6-5에서 `baseline_from_ablation('cnn', ...)`을 호출해")
            A("CNN 실측치로 교체할 것.")
        A("")
    else:
        A("_헤드 기준값 미기록. `baseline_from_ablation()`으로 기록할 것._\n")

    A("헤드 선택은 **단독 PI_h가 아니라 한계 기여**로 한다. 헤드끼리 정보가 겹치면")
    A("단독 PI가 양수여도 결합에 넣었을 때 손해일 수 있다(선형 프로브에서 byte2가")
    A("단독 +0.241, 한계 -0.140이었다). `select_heads()`가 캘리브레이션셋에서 수행한다.")
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
    r = MULTITASK_REFERENCE
    A("\n## 참고: 완벽 오라클 기준선\n")
    A("라벨 자체의 엔트로피 상한이다 (공격셋 u 전량 10,240,000 샘플 실측).\n")
    A("| 라벨 | 트레이스당 정보량 | 필요 트레이스 |")
    A("|---|---|---|")
    A(f"| sign (1비트) | {r['marginal_entropy']['sign']:.4f} 비트 | "
      f"{KEY_BITS / r['marginal_entropy']['sign']:.2f} |")
    A(f"| HW 전체 (33클래스) | {r['hw32_entropy']:.4f} 비트 | "
      f"{KEY_BITS / r['hw32_entropy']:.2f} |")
    A(f"| 멀티태스크 결합 (4헤드) | {r['joint_entropy']:.4f} 비트 | "
      f"{KEY_BITS / r['joint_entropy']:.2f} |")
    A("\n어떤 모델도 해당 라벨의 오라클보다 잘할 수 없다. PI가 오라클 정보량에")
    A("얼마나 근접했는지가 그 학습 방식의 완성도다.")
    A("\n참고로 GE/SR 시뮬레이션에서 HW 오라클은 6개 트레이스에서 SR 100%에")
    A("도달했다. 위 5.46개는 유한 후보 효과를 뺀 이론값이다.\n")
    if notes:
        A("## 메모\n")
        A(notes)
    with open(out_path, "w") as f:
        f.write("\n".join(L))
    return out_path
