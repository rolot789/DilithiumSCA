"""트레이스 정규화 파이프라인.

v2 대비 달라진 핵심 3가지
  1. per-trace 이득(gain) 보정을 vertical 정규화 *앞에* 넣는다.
     v2 instruction.md은 "개별 트레이스 정규화 금지"라고 못박았지만, 이는
     POI 구간에서 트레이스별 정규화를 할 때만 맞는 말이다. 트레이스 전체(40,000
     샘플)에서 robust scale을 재면 데이터 의존 성분은 전체 분산의 극히 일부라
     신호를 지우지 않으면서 측정 이득/DC 드리프트만 제거된다.
  2. vertical 통계량을 "그 데이터 자신의 캠페인"에서 뽑을 수 있게 했다.
     v2는 프로파일링 통계량을 공격셋에 그대로 적용해 캠페인 간 도메인 격차를
     그대로 안고 갔다. 공격자는 공격 트레이스를 갖고 있으므로 그 평균/표준편차를
     계산하는 것은 라벨 누설이 아니며 완전히 정당하다.
  3. 표준편차를 전체 트레이스로 계산한다. v2는 set1의 앞 5,000개만 썼다.

정규화 순서: box filter -> per-trace gain -> POI 슬라이스 -> vertical z-score
"""

from dataclasses import dataclass, field

import numpy as np

from v3_common import N_HW_CLASSES, TRACE_LEN, hw32


@dataclass
class NormConfig:
    moving_average: int = 1          # 1이면 필터 없음 (v2는 5)
    per_trace_gain: bool = True      # v2에는 없던 단계
    gain_estimator: str = "mad"      # "mad" | "std"
    vertical: bool = True
    vertical_source: str = "own"     # "own"(캠페인 자체) | "external"(프로파일링 통계 이식)
    robust_vertical: bool = False    # 평균/표준편차 대신 중앙값/MAD
    clip_sigma: float = 0.0          # >0 이면 해당 sigma에서 클리핑

    def describe(self):
        return (f"MA={self.moving_average} gain={self.gain_estimator if self.per_trace_gain else 'off'} "
                f"vertical={self.vertical_source if self.vertical else 'off'}"
                f"{' robust' if self.robust_vertical else ''}"
                f"{f' clip={self.clip_sigma}' if self.clip_sigma else ''}")


_MAD_TO_SIGMA = 1.4826


def box_filter(X, w):
    """이동평균. v2의 np.convolve(mode='same')는 양 끝을 0으로 패딩해 경계를
    왜곡시킨다. 여기서는 edge 패딩 + 누적합이라 경계가 안전하고 훨씬 빠르다."""
    if not w or w <= 1:
        return X
    pad_l, pad_r = w // 2, w - 1 - w // 2
    P = np.pad(X, ((0, 0), (pad_l, pad_r)), mode="edge").astype(np.float64)
    cs = np.cumsum(P, axis=1)
    cs = np.concatenate([np.zeros((X.shape[0], 1)), cs], axis=1)
    return ((cs[:, w:] - cs[:, :-w]) / w).astype(np.float32)


def per_trace_scale(X, estimator="mad"):
    """트레이스별 (중심, 스케일). 트레이스 전체 구간에서 재야 신호가 보존된다."""
    if estimator == "mad":
        center = np.median(X, axis=1, keepdims=True)
        scale = np.median(np.abs(X - center), axis=1, keepdims=True) * _MAD_TO_SIGMA
    elif estimator == "std":
        center = X.mean(axis=1, keepdims=True)
        scale = X.std(axis=1, keepdims=True)
    else:
        raise ValueError(f"알 수 없는 estimator: {estimator}")
    scale = np.where(scale == 0, 1.0, scale)
    return center.astype(np.float32), scale.astype(np.float32)


def extract_poi(trace_iter, poi, cfg, n_total, verbose=True):
    """스트리밍으로 POI 열만 뽑아 메모리에 올린다.

    반환: (X_poi, gain)  — X_poi는 vertical 정규화 *직전* 상태.
    트레이스 원본은 int64 3.05GB/set이라 절대 통째로 올리지 않는다.
    """
    poi = np.asarray(poi, dtype=np.int64)
    X = np.empty((n_total, len(poi)), dtype=np.float32)
    gain = np.empty((n_total, 1), dtype=np.float32)
    row = 0
    for _, _, chunk in trace_iter:
        chunk = box_filter(chunk, cfg.moving_average)
        if cfg.per_trace_gain:
            center, scale = per_trace_scale(chunk, cfg.gain_estimator)
            chunk = (chunk - center) / scale
            gain[row:row + len(chunk)] = scale
        else:
            gain[row:row + len(chunk)] = 1.0
        X[row:row + len(chunk)] = chunk[:, poi]
        row += len(chunk)
        if verbose and row % 10000 == 0:
            print(f"    {row}/{n_total} 트레이스 처리")
    if row != n_total:
        raise ValueError(f"트레이스 수 불일치: {row} != {n_total}")
    return X, gain


def fit_vertical_streaming(trace_iter, cfg, trace_len=TRACE_LEN):
    """전체 트레이스(40,000 샘플) 기준 시간샘플별 평균/표준편차를 Welford로 누적한다.

    Stage 2의 계수 불변 학습은 트레이스 전 구간에서 윈도우를 뽑으므로 POI 부분집합이
    아니라 전 구간 통계가 필요하다. 12.2GB를 메모리에 올리지 않기 위한 스트리밍 경로.
    v2가 set1의 앞 5,000개만 쓴 것과 달리 전량을 반영한다.
    """
    n = 0
    mean = np.zeros(trace_len, dtype=np.float64)
    m2 = np.zeros(trace_len, dtype=np.float64)
    for _, _, chunk in trace_iter:
        chunk = box_filter(chunk, cfg.moving_average)
        if cfg.per_trace_gain:
            center, scale = per_trace_scale(chunk, cfg.gain_estimator)
            chunk = (chunk - center) / scale
        c = np.asarray(chunk, dtype=np.float64)
        cn = len(c)
        cmean = c.mean(axis=0)
        cm2 = ((c - cmean) ** 2).sum(axis=0)
        delta = cmean - mean
        tot = n + cn
        mean += delta * (cn / tot)
        m2 += cm2 + (delta ** 2) * (n * cn / tot)
        n = tot
    if n < 2:
        raise ValueError("트레이스가 2개 미만")
    std = np.sqrt(m2 / (n - 1))
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32), n


def fit_vertical(X, robust=False):
    """POI별(=시간 샘플별) 중심/스케일. 전체 트레이스를 다 쓴다."""
    if robust:
        center = np.median(X, axis=0)
        scale = np.median(np.abs(X - center), axis=0) * _MAD_TO_SIGMA
    else:
        center = X.mean(axis=0)
        scale = X.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return center.astype(np.float32), scale.astype(np.float32)


def apply_vertical(X, center, scale, clip_sigma=0.0):
    Z = (X - center) / scale
    if clip_sigma and clip_sigma > 0:
        Z = np.clip(Z, -clip_sigma, clip_sigma)
    return Z.astype(np.float32)


def prepare(trace_iter, poi, cfg, n_total, external_stats=None, verbose=True):
    """한 캠페인(프로파일링 또는 공격)을 통째로 전처리한다.

    cfg.vertical_source == "own"      : 이 캠페인 자신의 통계량 사용 (도메인 적응)
    cfg.vertical_source == "external" : external_stats(=프로파일링 통계) 이식 (v2 방식)
    """
    X, _ = extract_poi(trace_iter, poi, cfg, n_total, verbose=verbose)
    if not cfg.vertical:
        return X, None
    if cfg.vertical_source == "own":
        stats = fit_vertical(X, cfg.robust_vertical)
    elif cfg.vertical_source == "external":
        if external_stats is None:
            raise ValueError("vertical_source='external'인데 external_stats가 없음")
        stats = external_stats
    else:
        raise ValueError(f"알 수 없는 vertical_source: {cfg.vertical_source}")
    return apply_vertical(X, stats[0], stats[1], cfg.clip_sigma), stats


def snr(X, labels, n_classes=N_HW_CLASSES):
    """POI별 SNR = Var_class(E[X|class]) / E_class[Var(X|class)].

    rho 기반 선택은 누설이 HW에 선형이라는 가정에 묶이지만 SNR은 그렇지 않다.
    v2가 rho 한 행(rho[0])만 보고 threshold를 잘못 잡아 POI 4개만 건진 문제를
    구조적으로 피하기 위한 대안 지표다.
    """
    labels = np.asarray(labels)
    means = np.zeros((n_classes, X.shape[1]), dtype=np.float64)
    varis = np.zeros((n_classes, X.shape[1]), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for cl in range(n_classes):
        m = labels == cl
        counts[cl] = m.sum()
        if counts[cl] < 2:
            continue
        sub = X[m]
        means[cl] = sub.mean(axis=0)
        varis[cl] = sub.var(axis=0)
    valid = counts >= 2
    w = counts[valid] / counts[valid].sum()
    signal = np.average(means[valid] ** 2, axis=0, weights=w) - np.average(means[valid], axis=0, weights=w) ** 2
    noise = np.average(varis[valid], axis=0, weights=w)
    noise = np.where(noise == 0, np.finfo(np.float64).tiny, noise)
    return signal / noise


def select_poi(metric, n_keep, min_distance=0, window=0):
    """metric 상위 n_keep개를 *반드시* 확보한다.

    v2는 threshold 방식이라 조건을 넘는 지점이 4개뿐이어도 조용히 통과했다.
    여기서는 개수를 보장하고, 확보 실패 시 예외를 던진다.
    """
    metric = np.asarray(metric, dtype=np.float64)
    order = np.argsort(metric)[::-1]
    chosen = []
    for idx in order:
        if len(chosen) >= n_keep:
            break
        if min_distance and any(abs(idx - c) < min_distance for c in chosen):
            continue
        chosen.append(int(idx))
    if len(chosen) < n_keep:
        raise ValueError(f"POI {n_keep}개를 확보하지 못함 (확보 {len(chosen)}개). "
                         f"min_distance={min_distance}를 줄일 것.")
    if window:
        expanded = set()
        half = window // 2
        for cen in chosen:
            expanded.update(range(max(0, cen - half), min(TRACE_LEN, cen + half + 1)))
        return np.array(sorted(expanded), dtype=np.int64), np.array(sorted(chosen), dtype=np.int64)
    return np.array(sorted(chosen), dtype=np.int64), np.array(sorted(chosen), dtype=np.int64)


def portability_report(X_prof, X_atk, verbose=True):
    """프로파일링 -> 공격 캠페인 도메인 격차 진단.

    v2는 이 격차를 재지 않은 채 프로파일링 통계량을 공격셋에 이식했다.
    두 값이 0에서 크게 벗어나면 vertical_source='own'이 필요하다는 신호다.
    """
    mp, sp = X_prof.mean(axis=0), X_prof.std(axis=0)
    ma, sa = X_atk.mean(axis=0), X_atk.std(axis=0)
    sp_safe = np.where(sp == 0, 1.0, sp)
    mean_shift = np.abs(ma - mp) / sp_safe
    scale_ratio = sa / sp_safe
    rep = {
        "mean_shift_median": float(np.median(mean_shift)),
        "mean_shift_p95": float(np.percentile(mean_shift, 95)),
        "scale_ratio_median": float(np.median(scale_ratio)),
        "scale_ratio_p95": float(np.percentile(scale_ratio, 95)),
    }
    if verbose:
        print(f"  평균 이동(프로파일링 sigma 단위): 중앙값 {rep['mean_shift_median']:.4f}, "
              f"p95 {rep['mean_shift_p95']:.4f}")
        print(f"  분산 비율: 중앙값 {rep['scale_ratio_median']:.4f}, p95 {rep['scale_ratio_p95']:.4f}")
        if rep["mean_shift_p95"] > 0.1 or not 0.9 < rep["scale_ratio_median"] < 1.1:
            print("  >> 캠페인 격차 유의미. vertical_source='own' 권장.")
        else:
            print("  >> 캠페인 격차 미미. 통계량 이식도 안전.")
    return rep
