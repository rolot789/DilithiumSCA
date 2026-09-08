"""Ranking Loss (Zaid et al., CHES 2021) — 키 순위를 직접 최적화하는 SCA 전용 손실.

교차 엔트로피는 "HW를 잘 맞히기"를 최적화하지만, 우리가 정말 원하는 것은
"정답 키가 1등이 되기"다. 둘은 같지 않다. 흔한 HW를 잘 맞히는 것보다 드물지만
키를 갈라주는 HW를 맞히는 것이 공격에는 더 유용할 수 있다.

Ranking Loss는 후보 키들의 누적 점수를 직접 보고, 정답 키 점수가 나머지보다
높아지도록 민다.

    s(m)  = sum_i  log p_model( HW(MR(c_i * cand_m)) | trace_i )
    RkL   = sum_{m != m*}  log2( 1 + exp( -alpha * (s(m*) - s(m)) ) )

즉 배치 하나가 "트레이스 여러 개로 계수 하나를 공격한 결과"가 되고,
그 공격이 성공하도록 학습한다.

## 이 데이터셋의 걸림돌과 해결

원 논문은 (평문, 키) 쌍이 있어야 한다. 우리 쪽 대응물은 (c, s)인데
**DSCAD 프로파일링 데이터에는 c도 s도 없다.** u만 주어진다
(attack 쪽에만 attack_10000_c.npy와 attack_s.npy가 있다).

해결: u로부터 일관된 (가짜키, challenge) 쌍을 역산한다.
MR(c*s) = c*s*2^-32 (mod q) 이므로, 가짜키 s*를 아무거나 고정하면

    c_i = u_i * 2^32 * (s*)^-1   (mod q)

로 각 트레이스의 challenge를 정할 수 있다. 이러면 MR(c_i * s*) = u_i 가 정확히
성립한다(실측 잔여류 일치 100%, 정수까지 일치 99.5~100%).

모델은 c를 입력으로 받지 않고 트레이스만 본다. 라벨 u는 실제 측정값 그대로다.
c와 s*는 **손실 계산의 장부**로만 쓰이므로 이 구성은 타당하다.

## 경고: 이 데이터셋 실측에서는 RkL이 해로웠다

선형 모델로 CE와 비교한 결과다(동일 시드/스텝, poly0의 계수 32개, 실제 트레이스).

| 손실 | Top-1 | PI(보정 전) | 학습된 T | PI(보정 후) |
|---|---|---|---|---|
| CE 단독 | 20.39% | **0.691** | 1.20 | **0.710** |
| CE + RkL | 14.89% | -10.143 | 8.80 | 0.215 |
| RkL 위주(ce=0.2) | 15.10% | -20.119 | 10.00(상한) | -0.219 |

**PI가 -10 ~ -20으로 붕괴한다.** 확신을 갖고 틀리는 상태이며, 온도 보정을 해도
CE의 1/3 수준까지밖에 회복되지 않는다.

원인은 손실의 구조 자체다. RkL은 누적 점수의 **상대적 순서**만 제약하므로
로그확률 전체를 상수배해도 값이 거의 변하지 않는다. 즉 **절대 확률 크기를
붙잡아주는 항이 없다.** 이는 선형 모델의 한계가 아니라 손실의 성질이므로
신경망에서도 같은 방향으로 나타날 가능성이 높다.

우리 공격 파이프라인은 로그확률을 누적하므로 캘리브레이션이 곧 성능이다.
그래서 이 데이터셋에서는 **CE를 기본으로 쓰고 RkL은 실험용으로만** 둔다.

쓴다면 반드시:
  1. `ce_weight`를 충분히 크게(>= 1.0) 두어 확률 크기를 CE가 붙잡게 한다
  2. 학습 후 **반드시 온도 보정**하고 PI를 다시 잰다
  3. PI가 CE 단독보다 낮으면 채택하지 않는다

## 그 밖의 주의

- 후보 전체(q=8,380,417)를 배치마다 쓸 수는 없다. 배치당 M개(기본 64)를
  무작위 추출하며, 이는 문서화된 근사다.
- 배치는 **한 계수의 여러 트레이스**여야 한다. 계수를 섞은 배치에서는 키 개념이
  성립하지 않는다. `RankingBatchSampler`가 계수별 그룹으로 배치를 만든다.
- 선형 프로브 기준이라는 한계는 있다. 신경망에서 다를 수 있으니 PI로 직접 확인할 것.
  다만 위에서 설명한 캘리브레이션 붕괴 메커니즘은 모델 종류와 무관하다.
"""

import numpy as np

from v3_common import L_POLY, N_COEFF, Q, hw32, mr

_R32 = pow(2, 32, Q)


def synthesize_key_assignment(u_labels, poly, coeff, s_star=None, seed=0):
    """실제 u로부터 일관된 (가짜키 s*, 트레이스별 challenge c) 쌍을 만든다.

    프로파일링 데이터에 c/s가 없어서 필요한 단계다. 반환된 c로
    MR(c_i * s_star) == u_i 가 성립한다.
    """
    u = np.asarray(u_labels)[:, poly, coeff].astype(np.int64)
    if s_star is None:
        rng = np.random.default_rng(seed)
        s_star = int(rng.integers(1, Q))
    inv = pow(int(s_star) % Q, Q - 2, Q)
    c = (u.astype(object) % Q * _R32 % Q * inv % Q)
    return int(s_star), np.array(c.tolist(), dtype=np.int64)


def verify_assignment(c_vals, s_star, u_labels, poly, coeff):
    """역산이 실제로 맞는지 확인한다. 학습 전에 반드시 호출할 것."""
    u = np.asarray(u_labels)[:, poly, coeff].astype(np.int64)
    got = mr(np.asarray(c_vals, dtype=np.int64) * int(s_star))
    return {
        "residue_match": float(np.mean(((got - u) % Q) == 0)),
        "exact_match": float(np.mean(got == u)),
    }


def sample_candidates(true_s, n_candidates, rng):
    """후보 키 M개. 정답을 0번 자리에 넣고 나머지는 [0,q)에서 균등추출한다.

    학습용이므로 정답 포함이 정당하다(지도학습 라벨에 해당). 평가에서
    후보를 정답에서 만드는 것과는 완전히 다른 이야기다 — 평가는
    v3_evaluate.build_candidate_residues를 쓴다.
    """
    cands = rng.integers(0, Q, size=n_candidates, dtype=np.int64)
    cands[0] = int(true_s) % Q
    return cands


def hypothesis_table(c_vals, candidates):
    """(n_traces, n_cand) HW 가설표. hyp[i, m] = HW(MR(c_i * cand_m))."""
    d = np.asarray(c_vals, dtype=np.int64)[:, None] * np.asarray(candidates,
                                                                dtype=np.int64)[None, :]
    return hw32(mr(d)).astype(np.int64)


# ------------------------------------------------------- numpy 참조 구현

def _softplus(x):
    return np.logaddexp(0.0, x)


def ranking_loss_numpy(log_probs, hyp, true_idx=0, alpha=1.0):
    """RkL 참조 구현. log_probs: (n_traces, n_classes) 로그 확률.

    TF 구현과 수식을 맞추기 위한 기준이자, 단위 테스트용이다.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    scores = np.take_along_axis(lp, hyp, axis=1).sum(axis=0)      # (n_cand,)
    delta = scores[true_idx] - scores                             # (n_cand,)
    mask = np.ones(len(scores), dtype=bool)
    mask[true_idx] = False
    return float(np.sum(_softplus(-alpha * delta[mask]) / np.log(2.0)))


def ranking_loss_grad_numpy(log_probs, hyp, true_idx=0, alpha=1.0):
    """RkL의 log_probs에 대한 해석적 기울기. 유한차분 검증용."""
    lp = np.asarray(log_probs, dtype=np.float64)
    n, C = lp.shape
    scores = np.take_along_axis(lp, hyp, axis=1).sum(axis=0)
    delta = scores[true_idx] - scores
    w = np.zeros_like(scores)
    mask = np.ones(len(scores), dtype=bool)
    mask[true_idx] = False
    # d/dz softplus(-alpha*z)/ln2 = -alpha*sigmoid(-alpha*z)/ln2
    sig = 1.0 / (1.0 + np.exp(alpha * delta[mask]))
    w[mask] = -alpha * sig / np.log(2.0)
    g = np.zeros_like(lp)
    # delta_m = s_{m*} - s_m 이므로 s_{m*}에는 +sum(w), s_m에는 -w_m
    np.add.at(g, (np.arange(n), hyp[:, true_idx]), w[mask].sum())
    for m in np.where(mask)[0]:
        np.add.at(g, (np.arange(n), hyp[:, m]), -w[m])
    return g


def key_rank_numpy(log_probs, hyp, true_idx=0):
    """현재 배치에서 정답 키의 순위(동점은 평균). 학습 모니터링용."""
    scores = np.take_along_axis(np.asarray(log_probs, dtype=np.float64),
                                hyp, axis=1).sum(axis=0)
    t = scores[true_idx]
    return float((scores > t).sum() + 0.5 * ((scores == t).sum() - 1))


# --------------------------------------------------------- 배치 샘플러

class RankingBatchSampler:
    """계수별 그룹으로 배치를 만든다.

    일반 샘플러는 한 배치에 여러 계수를 섞는데, 그러면 "키" 개념이 성립하지 않는다.
    여기서는 배치를 (n_groups x group_size)로 만들고 **그룹 하나 = 계수 하나**,
    그룹 안의 group_size개가 그 계수를 공격하는 트레이스들이 된다.

    산출: (X (B, window, 1), hw (B,), hyp (n_groups, group_size, n_cand))
    """

    def __init__(self, traces, hw_labels, u_labels, centers, trace_indices,
                 window=128, window_offset=32, n_groups=8, group_size=16,
                 n_candidates=64, shift_aug=1, poly_subset=None,
                 coeff_subset=None, seed=0):
        self.traces = traces
        self.hw = hw_labels
        self.u = u_labels
        self.centers = np.asarray(centers) + int(window_offset)
        self.idx = np.asarray(trace_indices)
        self.window = window
        self.n_groups = n_groups
        self.group_size = group_size
        self.n_candidates = n_candidates
        self.shift_aug = shift_aug
        self.polys = np.asarray(poly_subset if poly_subset is not None else range(L_POLY))
        self.coeffs = np.asarray(coeff_subset if coeff_subset is not None else range(N_COEFF))
        self.rng = np.random.default_rng(seed)

        half = window // 2
        sub = self.centers[np.ix_(self.polys, self.coeffs)]
        if (sub - half - shift_aug).min() < 0 or (sub + half + shift_aug).max() > traces.shape[1]:
            raise ValueError("윈도우가 트레이스 경계를 벗어남. window/offset/shift_aug 조정 필요")

        # 계수별 (가짜키, challenge)를 미리 만들어 둔다. 계수마다 키가 달라야
        # 계수 불변 학습이 유지된다.
        self._assign = {}

    def _assignment(self, j, k):
        key = (int(j), int(k))
        if key not in self._assign:
            s_star, c = synthesize_key_assignment(self.u, j, k,
                                                  seed=hash(key) & 0xFFFF)
            self._assign[key] = (s_star, c)
        return self._assign[key]

    def batches(self):
        half = self.window // 2
        B = self.n_groups * self.group_size
        while True:
            X = np.empty((B, self.window), dtype=np.float32)
            y = np.empty(B, dtype=np.int32)
            hyp = np.empty((self.n_groups, self.group_size, self.n_candidates),
                           dtype=np.int32)
            pos = 0
            for g in range(self.n_groups):
                j = int(self.rng.choice(self.polys))
                k = int(self.rng.choice(self.coeffs))
                s_star, c_all = self._assignment(j, k)
                cands = sample_candidates(s_star, self.n_candidates, self.rng)
                sel = self.rng.choice(self.idx, self.group_size, replace=False)
                hyp[g] = hypothesis_table(c_all[sel], cands)
                cen = self.centers[j, k]
                for t in sel:
                    c0 = cen + (self.rng.integers(-self.shift_aug, self.shift_aug + 1)
                                if self.shift_aug else 0)
                    X[pos] = np.asarray(self.traces[t], dtype=np.float32)[c0 - half:c0 + half]
                    y[pos] = self.hw[t, j, k]
                    pos += 1
            yield X[:, :, None], y, hyp


# ----------------------------------------------------------- TF 구현

def make_ranking_model(base_model, alpha=1.0, ce_weight=1.0, n_groups=8,
                       group_size=16):
    """base_model을 감싸 RkL + CE로 학습하는 Keras 모델을 만든다.

    ce_weight 기본값이 1.0인 이유: RkL 단독은 로그확률의 절대 크기를 제약하지
    않아 캘리브레이션이 무너진다. 실측에서 ce_weight=0.2면 온도 보정 후에도
    PI가 음수였다(-0.219). CE가 확률 크기를 붙잡아 주어야 한다.

    학습 후 반드시 온도 보정하고 PI를 CE 단독과 비교할 것.
    """
    import tensorflow as tf

    class RankingModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.base = base_model
            self.alpha = float(alpha)
            self.ce_weight = float(ce_weight)
            self.n_groups = int(n_groups)
            self.group_size = int(group_size)
            self.ce = tf.keras.losses.SparseCategoricalCrossentropy()
            self.rk_metric = tf.keras.metrics.Mean(name="rkl")
            self.ce_metric = tf.keras.metrics.Mean(name="ce")

        def call(self, x, training=False):
            return self.base(x, training=training)

        def _rkl(self, log_probs, hyp):
            # log_probs (B, C) -> (n_groups, group_size, C)
            lp = tf.reshape(log_probs, (self.n_groups, self.group_size, -1))
            # hyp (n_groups, group_size, n_cand) -> 각 후보의 클래스 인덱스
            picked = tf.gather(lp, hyp, batch_dims=2)          # (G, S, M)
            scores = tf.reduce_sum(picked, axis=1)             # (G, M)
            true = scores[:, :1]                               # 정답은 0번 자리
            delta = true - scores[:, 1:]                       # (G, M-1)
            return tf.reduce_mean(
                tf.reduce_sum(tf.math.softplus(-self.alpha * delta), axis=1)
                / tf.math.log(2.0))

        def train_step(self, data):
            x, y, hyp = data
            with tf.GradientTape() as tape:
                p = self.base(x, training=True)
                logp = tf.math.log(tf.clip_by_value(p, 1e-12, 1.0))
                rkl = self._rkl(logp, hyp)
                ce = self.ce(y, p)
                loss = rkl + self.ce_weight * ce
            g = tape.gradient(loss, self.base.trainable_variables)
            self.optimizer.apply_gradients(zip(g, self.base.trainable_variables))
            self.rk_metric.update_state(rkl)
            self.ce_metric.update_state(ce)
            return {"loss": loss, "rkl": self.rk_metric.result(),
                    "ce": self.ce_metric.result()}

        @property
        def metrics(self):
            return [self.rk_metric, self.ce_metric]

    return RankingModel()


def to_tf_dataset(sampler, window):
    """RankingBatchSampler를 tf.data로 감싼다."""
    import tensorflow as tf
    B = sampler.n_groups * sampler.group_size
    sig = (
        tf.TensorSpec(shape=(B, window, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(B,), dtype=tf.int32),
        tf.TensorSpec(shape=(sampler.n_groups, sampler.group_size,
                             sampler.n_candidates), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(sampler.batches, output_signature=sig)
    return ds.prefetch(tf.data.AUTOTUNE)
