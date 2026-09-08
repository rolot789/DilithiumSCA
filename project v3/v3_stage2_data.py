"""Stage 2 데이터 파이프라인: 계수 불변 학습용 윈도우 샘플러.

핵심 아이디어
    트레이스 1개(40,000 샘플) 안에 1024개 계수 연산이 순차적으로 들어있고
    라벨 u[i, j, k]도 1024개가 전부 있다. v2는 [:, 0, 50] 하나만 썼다.
    윈도우를 옮겨가며 전부 쓰면 학습 샘플이 40,000 -> 약 4천만 개가 되고,
    계수 불변 모델 하나로 1024계수 전체 복구가 자동 해결된다.

Apple Silicon 최적화
  * 정규화된 트레이스를 float16 memmap(3.2GB)으로 한 번만 만들어 재사용한다.
    통합 메모리라 host->device 복사가 없고, OS 페이지 캐시가 사실상 GPU 캐시다.
  * 배치를 만들 때 무작위 (trace, 계수) 쌍을 흩뿌리면 memmap 랜덤 접근이 되어
    페이지 폴트가 폭증한다. 대신 **적은 수의 트레이스를 읽고 그 안에서 계수를
    많이 뽑는다**. 트레이스 행 하나가 80KB 연속 읽기라 지역성이 크게 좋아진다.
  * 제너레이터가 샘플 단위가 아니라 **배치 단위**로 yield 한다. Python 오버헤드가
    샘플당이 아니라 배치당으로 줄어든다.
"""

import numpy as np

from v3_common import L_POLY, N_COEFF, N_PER_SET, TRACE_LEN, hw32
from v3_normalize import box_filter, per_trace_scale


def build_time_map(poly_params):
    """계수 -> 시간샘플 중심 매핑 (L_POLY, N_COEFF).

    실측 테이블 (L_POLY, N_COEFF)을 그대로 넘기는 것을 권장한다
    (v3_common.calibrate_time_map_dense / load_time_map).

    [(stride, offset), ...] 형태의 선형 파라미터도 받지만 어디까지나 근사다.
    실측 결과 계수 간격이 27~37 사이에서 흔들리고 poly1은 앞부분 35에서 뒷부분
    30으로 바뀌어, 직선 적합은 poly1에서 145샘플까지 어긋난다.
    """
    arr = np.asarray(poly_params)
    if arr.ndim == 2 and arr.shape == (L_POLY, N_COEFF):
        centers = arr.astype(np.int64)
    else:
        if len(poly_params) != L_POLY:
            raise ValueError(f"다항식 {L_POLY}개 파라미터 필요, {len(poly_params)}개 받음")
        k = np.arange(N_COEFF, dtype=np.float64)
        centers = np.rint(np.stack([offset + stride * k
                                    for stride, offset in poly_params])).astype(np.int64)
    if centers.min() < 0 or centers.max() >= TRACE_LEN:
        raise ValueError(f"시간 매핑이 트레이스 범위를 벗어남: "
                         f"{centers.min()} ~ {centers.max()} (허용 0~{TRACE_LEN-1})")
    return centers


def precompute_hw_labels(u_labels):
    """(n, 4, 256) u -> HW int8. 40,000개 기준 40MB라 전량 RAM에 올린다."""
    return hw32(np.asarray(u_labels)).astype(np.int8)


def build_normalized_memmap(trace_iter_factory, out_path, cfg, n_total,
                            trace_len=TRACE_LEN, verbose=True):
    """정규화된 트레이스를 float16 memmap으로 1회 생성한다.

    2패스 구조다. 1패스에서 시간샘플별 통계를 Welford로 누적하고,
    2패스에서 정규화 결과를 디스크에 쓴다. 원본 int64(12.2GB)를 메모리에
    올리지 않으며, 결과물은 float16으로 3.2GB다.

    trace_iter_factory: 호출할 때마다 새 이터레이터를 반환하는 콜러블
                        (2회 순회해야 하므로 이터레이터가 아니라 팩토리를 받는다)
    """
    from v3_normalize import fit_vertical_streaming

    if verbose:
        print("  [1/2] 시간샘플별 통계 계산 중 (전체 트레이스)...")
    mean, std, n_seen = fit_vertical_streaming(trace_iter_factory(), cfg, trace_len)
    if n_seen != n_total:
        raise ValueError(f"통계 패스 트레이스 수 {n_seen} != {n_total}")

    if verbose:
        print(f"  [2/2] float16 memmap 기록 중 -> {out_path} "
              f"({n_total * trace_len * 2 / 1e9:.2f} GB)")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                    shape=(n_total, trace_len))
    row = 0
    for _, _, chunk in trace_iter_factory():
        chunk = box_filter(chunk, cfg.moving_average)
        if cfg.per_trace_gain:
            center, scale = per_trace_scale(chunk, cfg.gain_estimator)
            chunk = (chunk - center) / scale
        z = (chunk - mean) / std
        if cfg.clip_sigma and cfg.clip_sigma > 0:
            z = np.clip(z, -cfg.clip_sigma, cfg.clip_sigma)
        out[row:row + len(z)] = z.astype(np.float16)
        row += len(z)
        if verbose and row % 10000 == 0:
            print(f"      {row}/{n_total}")
    out.flush()
    if row != n_total:
        raise ValueError(f"기록된 트레이스 {row} != {n_total}")
    return mean, std


def set_split_indices(val_set=4, n_sets=4, n_per_set=N_PER_SET):
    """leave-one-set-out 분할. 랜덤 분할은 같은 캠페인 내부라 낙관적이다."""
    if not 1 <= val_set <= n_sets:
        raise ValueError(f"val_set은 1~{n_sets}")
    all_idx = np.arange(n_sets * n_per_set)
    lo, hi = (val_set - 1) * n_per_set, val_set * n_per_set
    val = all_idx[lo:hi]
    train = np.concatenate([all_idx[:lo], all_idx[hi:]])
    return train, val


class CoefficientWindowSampler:
    """트레이스 전 구간에서 (윈도우, HW) 배치를 생성한다.

    한 배치는 traces_per_batch개의 트레이스만 읽고, 각 트레이스에서
    batch_size // traces_per_batch개의 계수를 뽑는다. memmap 지역성을 위한 구조다.
    """

    def __init__(self, traces, hw_labels, centers, trace_indices,
                 window=64, batch_size=512, traces_per_batch=32,
                 shift_aug=0, poly_subset=None, coeff_subset=None, seed=0):
        if batch_size % traces_per_batch != 0:
            raise ValueError("batch_size는 traces_per_batch의 배수여야 함")
        if window % 2 != 0:
            raise ValueError("window는 짝수를 권장 (중심 정렬)")
        self.traces = traces
        self.hw = hw_labels
        self.centers = centers
        self.idx = np.asarray(trace_indices)
        self.window = window
        self.batch_size = batch_size
        self.traces_per_batch = traces_per_batch
        self.per_trace = batch_size // traces_per_batch
        self.shift_aug = shift_aug
        self.polys = np.asarray(poly_subset if poly_subset is not None else range(L_POLY))
        self.coeffs = np.asarray(coeff_subset if coeff_subset is not None else range(N_COEFF))
        self.rng = np.random.default_rng(seed)

        half = window // 2
        lo = centers[np.ix_(self.polys, self.coeffs)] - half - shift_aug
        hi = centers[np.ix_(self.polys, self.coeffs)] + half + shift_aug
        if lo.min() < 0 or hi.max() > TRACE_LEN:
            raise ValueError(f"윈도우가 트레이스 경계를 벗어남 ({lo.min()} ~ {hi.max()}). "
                             f"window/shift_aug를 줄이거나 계수 부분집합을 조정할 것.")

    def samples_per_epoch(self):
        return len(self.idx) * len(self.polys) * len(self.coeffs)

    def steps_per_epoch(self):
        return self.samples_per_epoch() // self.batch_size

    def batches(self):
        """무한 배치 제너레이터. (X (B, window, 1) float32, y (B,) int32)"""
        half = self.window // 2
        while True:
            trace_sel = self.rng.choice(self.idx, self.traces_per_batch, replace=False)
            X = np.empty((self.batch_size, self.window), dtype=np.float32)
            y = np.empty(self.batch_size, dtype=np.int32)
            pos = 0
            for ti in trace_sel:
                row = np.asarray(self.traces[ti], dtype=np.float32)   # 연속 읽기
                pj = self.rng.choice(self.polys, self.per_trace)
                pk = self.rng.choice(self.coeffs, self.per_trace)
                cen = self.centers[pj, pk]
                if self.shift_aug:
                    cen = cen + self.rng.integers(-self.shift_aug,
                                                  self.shift_aug + 1, self.per_trace)
                for m in range(self.per_trace):
                    s = cen[m] - half
                    X[pos] = row[s:s + self.window]
                    y[pos] = self.hw[ti, pj[m], pk[m]]
                    pos += 1
            yield X[:, :, None], y

    def full_coefficient_batch(self, trace_index, poly, coeff_list):
        """공격/평가용: 한 트레이스에서 지정 계수들의 윈도우를 한 번에 뽑는다."""
        half = self.window // 2
        row = np.asarray(self.traces[trace_index], dtype=np.float32)
        cen = self.centers[poly, np.asarray(coeff_list)]
        X = np.stack([row[c - half:c + half] for c in cen])
        return X[:, :, None]


def to_tf_dataset(sampler, n_classes=33):
    """tf.data 래핑. 배치 단위 제너레이터라 Python 오버헤드가 배치당으로 줄어든다."""
    import tensorflow as tf

    sig = (
        tf.TensorSpec(shape=(sampler.batch_size, sampler.window, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(sampler.batch_size,), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(sampler.batches, output_signature=sig)
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, n_classes)),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
