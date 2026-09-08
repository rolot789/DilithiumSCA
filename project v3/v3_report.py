"""학습 결과 평가 및 리포트 생성.

평가를 3단계로 나눈다. 각 단계는 앞 단계가 통과해야 의미가 있다.

  1단계 분류 성능   모델이 HW를 맞히는가
  2단계 캘리브레이션 모델이 내놓는 '확률'이 진짜 확률인가
  3단계 키 복구     실제로 비밀키를 얻는가

2단계를 따로 두는 이유가 중요하다. 부채널 공격은 여러 트레이스의 확률을 곱해
나가는 방식이라 **확률값 자체가 정확해야** 한다. 정확도가 높아도 확률이
과신/과소 상태면 공격이 무너진다. v2가 T=5.0이라는 임의 보정을 넣어야 했던
근본 원인이 여기 있었고, 그래서 v3는 이것을 독립 지표로 측정한다.
"""

import json
import os
from datetime import datetime

import numpy as np

from v3_common import L_POLY, N_COEFF, N_HW_CLASSES, Q
import v3_evaluate as ev


# ---------------------------------------------------------------- 1단계: 분류

def classification_report(probs, y_true, label=""):
    """Top-k 정확도를 항상 다수 클래스 베이스라인과 함께 보고한다.

    무작위 1/33 = 3.03%와 비교하면 성능이 과장된다. 라벨이 이봉분포라
    '무조건 최빈 HW'만 답해도 8.68%가 나오기 때문이다.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true).astype(np.int64)
    order = np.argsort(probs, axis=1)[:, ::-1]

    counts = np.bincount(y_true, minlength=N_HW_CLASSES)
    baseline = counts.max() / counts.sum()
    rep = {
        "label": label,
        "n": int(len(y_true)),
        "majority_baseline": float(baseline),
        "random_baseline": 1.0 / N_HW_CLASSES,
        "majority_class": int(counts.argmax()),
    }
    for k in (1, 3, 5):
        hit = (order[:, :k] == y_true[:, None]).any(axis=1).mean()
        rep[f"top{k}"] = float(hit)
    rep["top1_over_baseline"] = float(rep["top1"] / baseline)

    err = order[:, 0] - y_true
    rep["mean_abs_error"] = float(np.abs(err).mean())
    rep["error_median"] = float(np.median(err))
    return rep


def per_coefficient_report(predict_fn, poly_coeff_pairs, verbose=True):
    """계수별 정확도 분포. 계수 불변 모델이 정말 모든 계수에서 동작하는지 본다.

    predict_fn(j, k) -> (probs, y_true)
    편차가 크면 시간 매핑(time_map.json)이 일부 계수에서 어긋났다는 신호다.
    """
    rows = []
    for j, k in poly_coeff_pairs:
        probs, y = predict_fn(j, k)
        acc = float((np.argmax(probs, axis=1) == y).mean())
        rows.append({"poly": int(j), "coeff": int(k), "top1": acc})
    accs = np.array([r["top1"] for r in rows])
    summary = {
        "n_coefficients": len(rows),
        "mean": float(accs.mean()),
        "std": float(accs.std()),
        "min": float(accs.min()),
        "max": float(accs.max()),
        "p05": float(np.percentile(accs, 5)),
        "worst": sorted(rows, key=lambda r: r["top1"])[:5],
    }
    if verbose:
        print(f"  계수별 Top-1: 평균 {summary['mean']*100:.2f}% "
              f"(표준편차 {summary['std']*100:.2f}%, 최소 {summary['min']*100:.2f}%)")
        if summary["std"] > 0.05:
            print("  >> 편차가 크다. 시간 매핑이 일부 계수에서 어긋났을 수 있다.")
    return summary, rows


# ------------------------------------------------------- 2단계: 캘리브레이션

def calibration_report(probs, y_true, n_bins=15):
    """확률이 실제 빈도와 얼마나 일치하는가.

    ECE(Expected Calibration Error): 모델이 '70% 확신'이라고 말한 예측들이
    실제로 70% 맞는지를 잰다. 0에 가까울수록 좋다.
    ECE가 크면 GE/SR이 나빠지므로 온도 보정이 필수다.
    """
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    y_true = np.asarray(y_true).astype(np.int64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if not m.any():
            continue
        acc_b, conf_b, w = correct[m].mean(), conf[m].mean(), m.mean()
        ece += w * abs(acc_b - conf_b)
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                     "n": int(m.sum()), "accuracy": float(acc_b),
                     "confidence": float(conf_b)})
    nll = float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true])))
    return {"ece": float(ece), "nll": nll, "mean_confidence": float(conf.mean()),
            "accuracy": float(correct.mean()), "bins": bins,
            "overconfident": bool(conf.mean() > correct.mean())}


# ------------------------------------------------------------ 3단계: 키 복구

def attack_report(probs_cal, y_cal, probs_atk, c_vals_atk, coeff_k, poly_j,
                  true_s, u_true=None, n_candidates=65536, n_experiments=100,
                  max_traces=40, seed=0, verbose=True):
    """정직한 프로토콜로 GE/SR을 재고 완벽 오라클과 비교한다.

    온도와 사전확률은 캘리브레이션셋에서만 학습한다(공격셋 미사용).
    후보군은 정답 키 파일과 무관하게 [0,q)에서 균등추출한다.
    """
    T, nll = ev.fit_temperature(probs_cal, y_cal)
    log_prior = ev.implicit_log_prior(probs_cal)
    L = ev.trace_log_likelihood(probs_atk, temperature=T, log_prior=log_prior)

    cands, tidx = ev.build_candidate_residues(n_candidates, seed=seed, true_s=true_s)
    ge, sr, ge_full = ev.guessing_entropy(L, c_vals_atk, coeff_k, cands, tidx,
                                          n_experiments=n_experiments,
                                          max_traces=max_traces, seed=seed,
                                          verbose=False)
    rep = {
        "temperature": float(T), "calibration_nll": float(nll),
        "n_candidates": int(len(cands)),
        "full_space": int(Q),
        "scale_factor": float(Q / len(cands)),
        "ge": ge.tolist(), "sr": sr.tolist(), "ge_full": ge_full.tolist(),
        "traces_to_sr100": None, "traces_to_ge0": None,
    }
    if np.any(sr >= 1.0):
        rep["traces_to_sr100"] = int(np.argmax(sr >= 1.0)) + 1
    if np.any(ge <= 0.5):
        rep["traces_to_ge0"] = int(np.argmax(ge <= 0.5)) + 1

    if u_true is not None:
        o_ge, o_sr = ev.perfect_oracle_curve(u_true, c_vals_atk, coeff_k, poly_j,
                                             cands, tidx,
                                             n_experiments=min(n_experiments, 40),
                                             max_traces=min(max_traces, 12), seed=seed)
        rep["oracle_ge"] = o_ge.tolist()
        rep["oracle_sr"] = o_sr.tolist()
        rep["oracle_traces_to_sr100"] = (int(np.argmax(o_sr >= 1.0)) + 1
                                         if np.any(o_sr >= 1.0) else None)
        if rep["traces_to_sr100"] and rep["oracle_traces_to_sr100"]:
            rep["efficiency"] = rep["oracle_traces_to_sr100"] / rep["traces_to_sr100"]

    if verbose:
        print(f"  학습된 온도 T={T:.3f} (캘리브레이션셋 전용)")
        print(f"  후보 {len(cands):,}개 -> 전체공간 환산 x{rep['scale_factor']:.1f}")
        print(f"  SR 100% 도달: {rep['traces_to_sr100'] or '미도달'} 트레이스")
        if "oracle_traces_to_sr100" in rep:
            print(f"  완벽 오라클: {rep['oracle_traces_to_sr100']} 트레이스 "
                  f"(효율 {rep.get('efficiency', 0)*100:.0f}%)")
    return rep


# ------------------------------------------------------------------ 리포트

def render_markdown(out_path, meta, cls_val, cls_atk, calib, attack,
                    per_coeff=None, portability=None):
    """모든 결과를 하나의 마크다운 리포트로 묶는다."""
    L = []
    A = L.append
    A(f"# Dilithium v3 학습·공격 리포트\n")
    A(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    A("## 1. 실행 환경 및 설정\n")
    A("| 항목 | 값 |")
    A("|---|---|")
    for k, v in meta.items():
        A(f"| {k} | {v} |")
    A("")

    A("## 2. 분류 성능\n")
    A("정확도는 반드시 다수 클래스 베이스라인과 비교해야 한다. 라벨이 이봉분포라")
    A("무작위(3.03%)가 아니라 최빈 클래스 비율이 진짜 기준선이다.\n")
    A("| 구분 | Top-1 | Top-3 | Top-5 | 베이스라인 | 배율 |")
    A("|---|---|---|---|---|---|")
    for r in (cls_val, cls_atk):
        if r is None:
            continue
        A(f"| {r['label']} | {r['top1']*100:.2f}% | {r['top3']*100:.2f}% | "
          f"{r['top5']*100:.2f}% | {r['majority_baseline']*100:.2f}% | "
          f"{r['top1_over_baseline']:.2f}x |")
    A("")
    if cls_val and cls_atk:
        gap = (cls_val["top1"] - cls_atk["top1"]) * 100
        A(f"검증셋 대비 공격셋 성능 차이: **{gap:+.2f}%p**. "
          f"이 값이 크면 캠페인 간 이식성 문제다 (v2는 4.5%p였다).\n")

    if per_coeff:
        A("### 2.1 계수별 편차\n")
        A(f"- 계수 {per_coeff['n_coefficients']}개 표본, "
          f"평균 {per_coeff['mean']*100:.2f}%, 표준편차 {per_coeff['std']*100:.2f}%")
        A(f"- 최소 {per_coeff['min']*100:.2f}% / 최대 {per_coeff['max']*100:.2f}%")
        if per_coeff["std"] > 0.05:
            A("- **편차가 크다.** 시간 매핑이 일부 계수에서 어긋났을 수 있다.")
        else:
            A("- 편차가 작다. 계수 불변 모델이 1024계수 전반에서 균일하게 동작한다.")
        A("")

    A("## 3. 캘리브레이션 품질\n")
    A("부채널 공격은 확률을 곱해 나가므로 확률값 자체가 정확해야 한다.")
    A("정확도가 높아도 여기가 나쁘면 키 복구가 무너진다.\n")
    A(f"- **ECE** (0에 가까울수록 좋음): {calib['ece']:.4f}")
    A(f"- **NLL**: {calib['nll']:.4f}")
    A(f"- 평균 확신도 {calib['mean_confidence']*100:.2f}% vs 실제 정확도 "
      f"{calib['accuracy']*100:.2f}% -> "
      f"{'과신(overconfident)' if calib['overconfident'] else '과소확신(underconfident)'}")
    A("")

    A("## 4. 키 복구 성능\n")
    A(f"- 후보 공간: {attack['n_candidates']:,}개 (전체 q={attack['full_space']:,}, "
      f"환산 배율 x{attack['scale_factor']:.1f})")
    A(f"- **후보군은 정답 키 파일과 무관하게 생성**했다 (v2는 정답에서 뽑아 "
      f"32,736배 축소된 상태로 측정했다)")
    A(f"- 학습된 온도 T={attack['temperature']:.3f} (캘리브레이션셋 전용, 공격셋 미사용)")
    A("")
    A("| 트레이스 수 | GE (부분집합) | GE (전체공간 환산) | SR |")
    A("|---|---|---|---|")
    ge, sr, gef = attack["ge"], attack["sr"], attack["ge_full"]
    for n in [1, 2, 4, 8, 16, 24, 32, len(ge)]:
        if n <= len(ge):
            A(f"| {n} | {ge[n-1]:.2f} | {gef[n-1]:,.0f} | {sr[n-1]*100:.1f}% |")
    A("")
    A(f"- **SR 100% 도달: {attack['traces_to_sr100'] or '미도달'} 트레이스**")
    if "oracle_traces_to_sr100" in attack:
        A(f"- 완벽 오라클(HW를 100% 맞히는 이론적 상한): "
          f"{attack['oracle_traces_to_sr100']} 트레이스")
        if "efficiency" in attack:
            A(f"- **효율 {attack['efficiency']*100:.0f}%** "
              f"(1.0에 가까울수록 이론적 한계에 근접)")
    A("")

    if portability:
        A("## 5. 캠페인 간 이식성\n")
        A(f"- 평균 이동(프로파일링 sigma 단위): 중앙값 "
          f"{portability['mean_shift_median']:.4f}, p95 {portability['mean_shift_p95']:.4f}")
        A(f"- 분산 비율: 중앙값 {portability['scale_ratio_median']:.4f}, "
          f"p95 {portability['scale_ratio_p95']:.4f}")
        A("")

    A("## 6. 판정\n")
    checks = []
    if cls_atk:
        checks.append((cls_atk["top1_over_baseline"] > 3.0,
                       f"공격셋 정확도가 베이스라인의 {cls_atk['top1_over_baseline']:.1f}배"))
    checks.append((calib["ece"] < 0.10, f"ECE {calib['ece']:.4f} < 0.10"))
    checks.append((attack["traces_to_sr100"] is not None,
                   "정직한 후보 공간에서 SR 100% 도달"))
    if per_coeff:
        checks.append((per_coeff["std"] < 0.05,
                       f"계수별 표준편차 {per_coeff['std']*100:.2f}% < 5%"))
    for ok, txt in checks:
        A(f"- {'[통과]' if ok else '[미달]'} {txt}")
    A("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))
    return out_path


def save_json(out_path, payload):
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path
