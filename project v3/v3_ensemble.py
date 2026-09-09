"""앙상블 학습.

SCA에서 앙상블이 효과적인 것은 문헌의 정설이지만(Perin et al., "Strength in
Numbers"), 이 데이터셋에는 **특별히 잘 맞는 다양성 원천**이 하나 더 있다.

실측 결과 계수 하나가 6~7개 지점에서 누설한다.
    poly0/coeff50 상대위치: -1(0.807) +15(0.650) +24(0.662)
                            +36(0.311) +55(0.251) +64(0.202)

서로 다른 윈도우를 보는 모델은 물리적으로 다른 누설 지점을 학습한다. 다만
**겹치지 않게(disjoint) 배치해야 실제 이득이 난다.** 폭만 키운 중첩 구성은
넓은 멤버가 좁은 멤버를 포함해버려 우열 관계가 되고, 앙상블이 사실상
"최고 멤버 고르기"로 퇴화한다.

선형 프로브로 실측한 다양성과 이득:

    구성                       오차 상관   앙상블 이득   앙상블 PI
    배깅(같은 창, 데이터 재추출)   0.946      1.003배      0.954
    중첩 창(폭만 확대)            0.432      1.018배      1.081
    disjoint 창                  0.258      1.195배      0.784
    넓은 창 + disjoint (권장)      -         1.108배      **1.177**

주의: 선형 모델은 분산이 매우 작아 이 수치가 **신경망 앙상블을 과소평가한다.**
신경망은 초기화와 SGD 잡음 때문에 분산이 크고, SCA 문헌에서도 시드 앙상블
이득이 크게 보고된다. 실기에서 PI로 직접 확인할 것.

결합 방식은 세 가지를 제공하고 PI로 고르게 한다. 어느 쪽이 나은지는 모델의
캘리브레이션 상태에 달렸으므로 미리 정하지 않는다.

  mean     확률의 산술평균. 무난하고 캘리브레이션이 잘 유지된다.
  logmean  로그확률의 평균(기하평균). 날카로워지지만 과신 위험이 있다.
  weighted 캘리브레이션셋에서 EM으로 가중치를 학습. 멤버 품질이 고르지 않을 때 유리.

주의: 결합 가중치는 **반드시 캘리브레이션셋에서만** 학습한다. 공격셋으로 고르면
v2가 온도를 공격셋에서 고른 것과 같은 오류가 된다.
"""

import numpy as np

from v3_benchmark import perceived_information, traces_to_recovery


# ------------------------------------------------------------------ 결합

def _norm(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def combine_probs(prob_list, mode="mean", weights=None):
    """멤버 확률들을 하나로 합친다. prob_list: [(n, n_classes), ...]"""
    P = np.stack([_norm(p) for p in prob_list])          # (M, n, C)
    M = len(P)
    if weights is None:
        w = np.full(M, 1.0 / M)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()

    if mode == "mean" or mode == "weighted":
        out = np.tensordot(w, P, axes=(0, 0))
    elif mode == "logmean":
        out = np.exp(np.tensordot(w, np.log(P), axes=(0, 0)))
    else:
        raise ValueError(f"알 수 없는 결합 방식: {mode}")
    return _norm(out)


def fit_ensemble_weights(prob_list, y_cal, n_iter=200, tol=1e-8):
    """캘리브레이션셋 NLL을 최소화하는 혼합 가중치를 EM으로 학습한다.

    p_ens(y|x) = sum_m w_m p_m(y|x) 는 정확히 혼합모형이므로 EM이 그대로 적용되고
    우도가 단조 증가한다. scipy 없이 동작한다.
    """
    P = np.stack([_norm(p) for p in prob_list])
    y = np.asarray(y_cal).astype(np.int64)
    L = P[:, np.arange(len(y)), y]                        # (M, n) 정답 클래스 확률
    M = len(P)
    w = np.full(M, 1.0 / M)
    prev = -np.inf
    for _ in range(n_iter):
        num = w[:, None] * L
        den = num.sum(axis=0, keepdims=True)
        den = np.where(den <= 0, 1e-300, den)
        r = num / den
        w = r.mean(axis=1)
        w = np.clip(w, 1e-12, None)
        w /= w.sum()
        ll = float(np.mean(np.log(np.clip((w[:, None] * L).sum(axis=0), 1e-300, None))))
        if abs(ll - prev) < tol:
            break
        prev = ll
    return w


def combine_multitask(member_head_probs, mode="mean", per_head_weights=None):
    """멀티태스크 멤버들을 **헤드별로** 결합한다.

    member_head_probs: [{헤드: (n, C_h) 확률}, ...]

    헤드마다 따로 결합하는 이유: 멤버마다 잘하는 헤드가 다르다. sign은 잘 맞히지만
    byte0은 못 맞히는 멤버가 있을 수 있고, 그럴 때 하나의 공통 가중치를 쓰면
    한쪽을 망친다.
    """
    heads = list(member_head_probs[0].keys())
    out = {}
    for h in heads:
        w = (per_head_weights or {}).get(h)
        out[h] = combine_probs([m[h] for m in member_head_probs], mode, w)
    return out


def fit_multitask_weights(member_head_probs, head_labels_cal):
    """헤드별 혼합 가중치를 캘리브레이션셋에서 각각 학습한다."""
    heads = list(member_head_probs[0].keys())
    return {h: fit_ensemble_weights([m[h] for m in member_head_probs],
                                    head_labels_cal[h]) for h in heads}


def select_combine_mode_multitask(member_head_probs, head_labels_cal, joint_h=None):
    """결합 방식을 헤드 전체의 합산 PI 기준으로 고른다. 공격셋은 쓰지 않는다."""
    from v3_benchmark import multitask_perceived_information

    best = None
    for mode in ("mean", "logmean", "weighted"):
        w = fit_multitask_weights(member_head_probs, head_labels_cal) \
            if mode == "weighted" else None
        comb = combine_multitask(member_head_probs, mode, w)
        pi = multitask_perceived_information(comb, head_labels_cal,
                                             joint_h=joint_h)["pi"]
        if best is None or pi > best[1]:
            best = (mode, pi, w)
    return {"mode": best[0], "cal_pi": best[1], "per_head_weights": best[2]}


def diversity_report_multitask(member_head_probs, head_labels, verbose=True):
    """헤드별 다양성. 어떤 헤드에서 멤버들이 실제로 다른지 본다."""
    heads = list(member_head_probs[0].keys())
    per_head = {}
    for h in heads:
        per_head[h] = diversity_report([m[h] for m in member_head_probs],
                                       head_labels[h], verbose=False)
    ec = [v["mean_error_correlation"] for v in per_head.values()]
    dis = [v["mean_disagreement"] for v in per_head.values()]
    rep = {"n_members": len(member_head_probs), "per_head": per_head,
           "mean_error_correlation": float(np.mean(ec)),
           "mean_disagreement": float(np.mean(dis))}
    if verbose:
        for h, v in per_head.items():
            print(f"    {h:6s} 오차 상관 {v['mean_error_correlation']:.3f}, "
                  f"불일치 {v['mean_disagreement']*100:.1f}%")
    return rep


def select_combine_mode(prob_list, y_cal, weights=None):
    """캘리브레이션셋 PI가 가장 높은 결합 방식을 고른다. 공격셋은 쓰지 않는다."""
    best = None
    for mode in ("mean", "logmean", "weighted"):
        w = weights if mode == "weighted" else None
        if mode == "weighted" and w is None:
            w = fit_ensemble_weights(prob_list, y_cal)
        pi = perceived_information(combine_probs(prob_list, mode, w), y_cal)["pi"]
        if best is None or pi > best[1]:
            best = (mode, pi, w)
    return {"mode": best[0], "cal_pi": best[1], "weights": best[2]}


# ------------------------------------------------------------------ 진단

def diversity_report(prob_list, y_true, verbose=True):
    """멤버들이 실제로 서로 다른가. 다양성이 없으면 앙상블 이득도 없다."""
    y = np.asarray(y_true).astype(np.int64)
    preds = np.stack([_norm(p).argmax(axis=1) for p in prob_list])
    M = len(preds)
    errs = (preds != y[None, :]).astype(np.float64)

    dis, ecorr = [], []
    for a in range(M):
        for b in range(a + 1, M):
            dis.append(float((preds[a] != preds[b]).mean()))
            ea, eb = errs[a] - errs[a].mean(), errs[b] - errs[b].mean()
            d = np.sqrt((ea ** 2).sum() * (eb ** 2).sum())
            ecorr.append(float((ea * eb).sum() / d) if d > 0 else 1.0)
    rep = {
        "n_members": M,
        "mean_disagreement": float(np.mean(dis)) if dis else 0.0,
        "mean_error_correlation": float(np.mean(ecorr)) if ecorr else 1.0,
        "member_accuracy": [float((preds[i] == y).mean()) for i in range(M)],
    }
    if verbose:
        print(f"  멤버 {M}개, 예측 불일치율 {rep['mean_disagreement']*100:.2f}%, "
              f"오차 상관 {rep['mean_error_correlation']:.3f}")
        if rep["mean_error_correlation"] > 0.8:
            print("  >> 멤버들이 너무 비슷하다. 앙상블 이득이 작을 것이다.")
    return rep


def ensemble_gain(prob_list, y_true, mode="mean", weights=None):
    """앙상블이 최고 멤버보다 얼마나 나은가. PI 기준으로 잰다."""
    member_pi = [perceived_information(p, y_true)["pi"] for p in prob_list]
    ens = combine_probs(prob_list, mode, weights)
    ens_pi = perceived_information(ens, y_true)["pi"]
    best = max(member_pi)
    return {
        "member_pi": member_pi,
        "best_member_pi": float(best),
        "mean_member_pi": float(np.mean(member_pi)),
        "ensemble_pi": float(ens_pi),
        "gain_over_best": float(ens_pi - best),
        "gain_ratio": float(ens_pi / best) if best > 0 else float("inf"),
        "predicted_traces_best": traces_to_recovery(best),
        "predicted_traces_ensemble": traces_to_recovery(ens_pi),
    }


# ------------------------------------------------------------- 멤버 구성

def nested_window_specs():
    """폭만 키운 중첩 윈도우. **앙상블 멤버로는 부적합하다.**

    실측: 오차 상관 0.432로 다양성이 있어 보이지만 앙상블 이득은 1.018배뿐이다.
    넓은 윈도우가 좁은 윈도우를 완전히 포함하므로 상호보완이 아니라 우열 관계이고,
    학습된 가중치가 가장 넓은 멤버에 0.93으로 몰려 사실상 "최고 멤버 고르기"가 된다.

    폭 자체는 단일 모델 성능에 크게 기여하므로(PI 0.658 -> 1.060, +61%)
    **앙상블이 아니라 단일 모델의 기본 설정으로 쓸 것.**
    """
    return [
        {"window": 32, "window_offset": 0},
        {"window": 64, "window_offset": 16},
        {"window": 96, "window_offset": 24},
        {"window": 128, "window_offset": 32},
    ]


def disjoint_window_specs(width=32, offsets=(-32, 0, 32, 64)):
    """겹치지 않는 구간을 보는 멤버들. 진짜 상호보완이 나오는 구성이다.

    실측 누설점 -1/+15/+24/+36/+55/+64가 구간별로 나뉘어 배정된다.
        [-16,+16] -> -1, +15      [+16,+48] -> +24, +36
        [+48,+80] -> +55, +64     [-48,-16] -> (거의 없음)

    실측: 오차 상관 0.258(중첩 0.432, 배깅 0.946 대비 훨씬 낮음),
    앙상블 이득 **1.195배**. 다만 멤버 개개가 약해서 절대 PI는 단일 넓은 창보다 낮다.
    """
    return [{"window": width, "window_offset": o} for o in offsets]


def recommended_specs():
    """실측상 가장 좋았던 구성: 넓은 창 1개(주력) + 겹치지 않는 보조 멤버들.

    실측 PI 비교 (선형 프로브 기준)
        단일 넓은 창(w=128,off=32)      1.062
        disjoint 4개만                  0.784  (1.195배 이득이지만 절대값이 낮음)
        넓은 창 + disjoint 4개 = 5멤버  **1.177**  (1.108배)

    학습된 가중치가 [0, 0.007, 0.021, 0.001, 0.97]로 넓은 창에 몰리지만,
    나머지가 더하는 0.115비트가 실제 이득이다.
    """
    return [{"window": 128, "window_offset": 32}] + disjoint_window_specs()


def seed_diverse_specs(base_spec, seeds=(0, 1, 2, 3, 4)):
    """같은 구조를 시드만 바꿔 학습.

    주의: 선형 프로브로 잰 배깅 다양성은 오차 상관 0.946, 앙상블 이득 1.003배로
    거의 무의미했다. 다만 **선형 모델은 분산이 매우 작아 이 결과가 신경망을
    과소평가한다.** 신경망은 초기화/SGD 잡음/드롭아웃 때문에 분산이 훨씬 크고,
    SCA 문헌에서도 시드 앙상블 이득이 크게 보고된다.
    실기에서 반드시 PI로 직접 확인할 것.
    """
    return [dict(base_spec, seed=s) for s in seeds]


class Ensemble:
    """학습된 멤버들을 묶어 예측을 결합한다.

    멤버마다 윈도우가 다를 수 있으므로 (model, spec) 쌍으로 보관하고
    예측 시 각자에 맞는 윈도우를 만들어 넣는다.
    """

    def __init__(self, members, mode="mean", weights=None):
        self.members = members          # [(model, spec_dict), ...]
        self.mode = mode
        self.weights = weights

    def member_probs(self, sampler_factory, trace_indices, poly, coeff,
                     batch_size=256):
        """멤버별 확률 행렬 리스트. sampler_factory(spec) -> 해당 윈도우 샘플러."""
        import v3_model as vm
        out = []
        for model, spec in self.members:
            sam = sampler_factory(spec)
            out.append(vm.predict_coefficient_probs(model, sam, trace_indices,
                                                    poly, coeff, batch_size))
        return out

    def predict(self, sampler_factory, trace_indices, poly, coeff, batch_size=256):
        probs = self.member_probs(sampler_factory, trace_indices, poly, coeff,
                                  batch_size)
        return combine_probs(probs, self.mode, self.weights), probs

    def calibrate(self, prob_list, y_cal):
        """결합 방식과 가중치를 캘리브레이션셋에서 결정한다."""
        sel = select_combine_mode(prob_list, y_cal)
        self.mode, self.weights = sel["mode"], sel["weights"]
        return sel


# -------------------------------------------------- 스냅샷 앙상블 (저비용)

def make_snapshot_callback(out_prefix, n_snapshots=5, epochs=50, max_lr=1e-3):
    """한 번의 학습으로 앙상블 멤버를 얻는다(cyclic LR 스냅샷).

    맥북에서 모델을 5개 따로 학습하는 것은 비싸다. 학습률을 주기적으로 올렸다
    내리면서 각 주기 끝에서 가중치를 저장하면, 서로 다른 국소최적해에 있는
    멤버들을 한 번의 학습 비용으로 얻는다.

    다양성은 윈도우 앙상블보다 작지만 비용이 1/N이다.
    """
    import tensorflow as tf

    cycle = max(1, epochs // n_snapshots)

    class Snapshot(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.saved = []

        def on_epoch_begin(self, epoch, logs=None):
            t = (epoch % cycle) / cycle
            lr = max_lr / 2 * (np.cos(np.pi * t) + 1)
            self.model.optimizer.learning_rate.assign(lr)

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % cycle == 0:
                path = f"{out_prefix}_snap{len(self.saved)}.keras"
                self.model.save(path)
                self.saved.append(path)

    return Snapshot()
