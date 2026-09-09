"""정직한 키 복구 평가.

v2 평가가 무효였던 이유
  attack_evaluation.ipynb:  s_candidates = np.unique(s_atk_all[0])
  -> 후보군을 정답 비밀키 파일에서 만들었다. 실제 탐색공간 q=8,380,417을
     정답이 반드시 포함된 256개로 32,736배 줄여놓고 GE를 잰 셈이다.
     또 temperature=5.0을 공격셋에서 직접 골랐다(테스트셋 튜닝).

여기서는
  * 후보를 [0,q)에서 시드 고정 균등추출한다. 정답은 순위 계산 목적으로만 삽입하며,
    부분집합 순위는 q/n_candidates 배로 전체공간 순위로 환산한다.
  * temperature와 사전확률 보정은 공격셋과 분리된 캘리브레이션셋에서만 학습한다.
  * GE와 함께 SR을 보고하고, 완벽 오라클 기준선을 같이 그린다.

MR 출력은 s의 대표원 선택에 대해 사실상 불변이다(실측 불일치 0.058%,
차이는 정확히 +-q인 int32 오버플로 경계 사례). 따라서 잔여류 열거가 타당하다.
"""

import numpy as np

from v3_common import N_HW_CLASSES, Q, hw32, mr


def centered_residue(x):
    """s를 (-q/2, q/2] 대표원으로. 실측상 [0,q) 대표원보다 불일치가 적다."""
    r = np.asarray(x, dtype=np.int64) % Q
    return np.where(r > Q // 2, r - Q, r)


def build_candidate_residues(n_candidates, seed, true_s):
    """후보군 생성. 정답 키 파일은 *후보 생성에 절대 쓰이지 않는다*.

    true_s는 오직 (a) 순위를 매길 대상 지정, (b) 부분집합에 정답이 존재하도록
    보장하는 용도로만 삽입된다. GE는 부분집합 기준이므로 scale_to_full_space()로
    전체공간 순위로 환산해야 한다.
    """
    rng = np.random.default_rng(seed)
    cands = rng.integers(0, Q, size=n_candidates, dtype=np.int64)
    true_r = int(centered_residue(true_s)) % Q
    cands[0] = true_r                      # 순위 계산을 위해 정답 보장
    cands = np.unique(cands)
    true_idx = int(np.searchsorted(cands, true_r))
    return centered_residue(cands), true_idx


def scale_to_full_space(rank, n_candidates):
    """부분집합 순위 -> 전체 q 공간 순위 추정치."""
    return np.asarray(rank, dtype=np.float64) * (Q / float(n_candidates))


def fit_temperature(probs_cal, y_cal, grid=None):
    """캘리브레이션셋 NLL을 최소화하는 온도. 공격셋을 절대 쓰지 않는다."""
    if grid is None:
        # 양쪽 경계를 넉넉히 둔다. 실제로 두 번 경계에 닿았다.
        #   ranking loss 모델(과신)  -> 상한 10에 닿음
        #   one-vs-rest 프로브(과소확신) -> 하한 0.2에 닿음
        # 경계에 닿으면 보정이 덜 된 채로 통과해 PI가 왜곡된다.
        grid = np.concatenate([np.linspace(0.05, 0.95, 19), np.linspace(1.0, 3.0, 21),
                               np.linspace(3.2, 10.0, 35), np.linspace(10.5, 30.0, 20)])
    P = np.clip(np.asarray(probs_cal, dtype=np.float64), 1e-12, 1.0)
    logp = np.log(P)
    best_t, best_nll = 1.0, np.inf
    idx = np.arange(len(y_cal))
    for t in grid:
        scaled = logp / t
        scaled -= scaled.max(axis=1, keepdims=True)
        lse = np.log(np.exp(scaled).sum(axis=1))
        nll = -np.mean(scaled[idx, y_cal] - lse)
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t, best_nll


def apply_temperature(probs, t):
    P = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    logp = np.log(P) / t
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def implicit_log_prior(probs_cal):
    """모델이 실제로 갖고 있는 사전확률. class_weight/focal loss를 쓰면 이것이
    데이터 사전확률과 달라지므로, 우도로 바꾸려면 이 값을 빼야 한다.
    v2 report.md는 이 보정을 성과로 적었지만 코드에는 없었다."""
    prior = np.clip(np.asarray(probs_cal, dtype=np.float64).mean(axis=0), 1e-12, 1.0)
    return np.log(prior / prior.sum())


def trace_log_likelihood(probs, temperature=1.0, log_prior=None):
    """공격에 쓸 로그 우도 행렬 (n_traces, 33)."""
    P = apply_temperature(probs, temperature) if temperature != 1.0 else \
        np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    L = np.log(P)
    if log_prior is not None:
        L = L - log_prior[None, :]
    return L


def _hw_table(c_k_sel, cands):
    """(n_traces, n_cand) HW 가설 테이블."""
    d = np.asarray(c_k_sel, dtype=np.int64)[:, None] * cands[None, :]
    return hw32(mr(d))


def _mid_rank(cum, true_idx):
    """동점을 포함한 평균 순위.

    HW는 33개 값뿐이라 서로 다른 s 후보가 동일한 HW 시퀀스를 내는 일이 흔하고,
    트레이스가 적을수록 심하다. 단순히 (score > true_score) 개수만 세면 동점
    후보가 전부 무시되어 GE가 0으로 붕괴한다 -- 실제로는 아직 구분되지 않았는데
    복구에 성공한 것처럼 보이는 심각한 과대평가다.
    """
    true_score = cum[:, true_idx][:, None]
    greater = (cum > true_score).sum(axis=1)
    ties = (cum == true_score).sum(axis=1) - 1
    return greater + 0.5 * ties


def guessing_entropy(log_lik, c_vals, coeff_k, cands, true_idx,
                     n_experiments=100, max_traces=50, seed=0, verbose=True):
    """GE와 SR 곡선을 함께 계산한다.

    log_lik : (n_pool, 33) 공격셋 트레이스별 로그 우도
    c_vals  : (n_pool, 256) 해당 트레이스들의 challenge
    반환    : ge(부분집합 순위), sr, ge_full(전체공간 환산)
    """
    rng = np.random.default_rng(seed)
    n_pool = len(log_lik)
    if n_pool < max_traces:
        raise ValueError(f"공격 풀 {n_pool} < max_traces {max_traces}")
    ranks = np.zeros((n_experiments, max_traces))
    for e in range(n_experiments):
        sel = rng.choice(n_pool, max_traces, replace=False)
        hw = _hw_table(c_vals[sel, coeff_k], cands)          # (T, n_cand)
        contrib = np.take_along_axis(log_lik[sel], hw.astype(np.int64), axis=1)
        cum = np.cumsum(contrib, axis=0)                     # (T, n_cand)
        ranks[e] = _mid_rank(cum, true_idx)
        if verbose and (e + 1) % 20 == 0:
            print(f"    실험 {e+1}/{n_experiments}")
    ge = ranks.mean(axis=0)
    sr = (ranks == 0).mean(axis=0)
    return ge, sr, scale_to_full_space(ge, len(cands))


def perfect_oracle_curve(u_true, c_vals, coeff_k, poly_j, cands, true_idx,
                         n_experiments=100, max_traces=16, seed=0):
    """HW를 100% 맞히는 오라클의 GE. 모델 성능의 상한(정보이론적 하한 트레이스 수).

    실측상 이 값은 4개 트레이스 근처에서 0으로 떨어진다. 어떤 모델도 이보다
    적은 트레이스로 복구할 수 없으므로, v3 성능은 항상 이 곡선과 함께 보고한다.
    """
    rng = np.random.default_rng(seed)
    n_pool = len(c_vals)
    hw_true = hw32(u_true[:, poly_j, coeff_k])
    ranks = np.zeros((n_experiments, max_traces))
    for e in range(n_experiments):
        sel = rng.choice(n_pool, max_traces, replace=False)
        hw = _hw_table(c_vals[sel, coeff_k], cands)
        match = (hw == hw_true[sel][:, None]).astype(np.float64)
        cum = np.cumsum(np.log(np.clip(match, 1e-12, 1.0)), axis=0)
        ranks[e] = _mid_rank(cum, true_idx)
    return ranks.mean(axis=0), (ranks == 0).mean(axis=0)


def summarize(ge, sr, ge_full, label=""):
    first_sr1 = np.argmax(sr >= 1.0) + 1 if np.any(sr >= 1.0) else None
    print(f"[{label}]")
    print(f"  후보 부분집합 기준 최종 GE = {ge[-1]:.3f}")
    print(f"  전체공간(q={Q:,}) 환산 GE  = {ge_full[-1]:,.1f}")
    print(f"  최종 SR = {sr[-1]*100:.1f}%")
    print(f"  SR 100% 도달 트레이스 수 = {first_sr1 if first_sr1 else '미도달'}")
    return {"ge_last": float(ge[-1]), "ge_full_last": float(ge_full[-1]),
            "sr_last": float(sr[-1]), "traces_to_sr1": first_sr1}
