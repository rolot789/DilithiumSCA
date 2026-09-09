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


def precompute_multitask_labels(u_labels, head_names=None):
    """멀티태스크 헤드별 라벨을 미리 만든다.

    HW 하나(H=4.2147비트)보다 나눠 예측하는 쪽이 트레이스당 정보량이 크다
    (**결합 엔트로피 8.4157비트, 1.997배**). 정보량이 2배면 필요 트레이스가 절반이다.
    주변 엔트로피의 합(8.7853)은 헤드가 독립일 때만 성립하는 상한이라 쓰지 않는다.

    주의: byte3은 사실상 8 x 부호비트다. |u| < 2^23이라 상위 9비트가 전부 부호
    확장이기 때문이며, 실측에서 bit24/bit28/bit31의 |rho|가 0.7238로 완전히 같다.
    그래서 기본 헤드 조합에서 byte3을 빼고 sign을 쓴다.
    """
    from v3_benchmark import MULTITASK_HEADS, SCHEMES
    names = head_names or MULTITASK_HEADS
    u = np.asarray(u_labels)
    return {n: SCHEMES[n](u).astype(np.int8) for n in names}


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


def inverse_strength_weights(quality, power=1.0, floor=0.2):
    """시간매핑 품질(|rho|)에서 계수 추출 가중치를 만든다.

    누설이 약한 계수일수록 더 자주 뽑는다. 약한 계수가 전체 키 복구의 병목이기
    때문이다. quality는 `calibrate_time_map_dense()`가 준 (L_POLY, N_COEFF) 배열.

    power=0이면 균등(기존 동작), 1이면 |rho|에 반비례, 클수록 약한 계수에 집중.
    floor는 강한 계수가 완전히 굶지 않도록 하는 하한이다.

    주의: 가중치를 주면 강한 계수의 학습량이 줄어든다. **반드시 PI와
    full_key_metrics로 전후를 비교할 것.** 전체 키 관점에서 이득이 없으면 쓰지 않는다.
    """
    qmean = np.asarray(quality, dtype=np.float64).mean(axis=0)   # 계수별 평균 |rho|
    qmean = np.maximum(qmean, 1e-6)
    w = (qmean.max() / qmean) ** power
    return np.maximum(w / w.max(), floor)


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
                 shift_aug=0, poly_subset=None, coeff_subset=None, seed=0,
                 window_offset=0, extra_labels=None, coeff_weights=None):
        """window_offset: 윈도우 중심을 피크에서 얼마나 뒤로 밀 것인가.

        실측 결과 계수 하나가 **여러 지점에서 누설한다**. poly0/coeff50 기준
        피크 상대위치와 |rho|:
            -1(0.807)  +15(0.650)  +24(0.662)  +36(0.311)  +55(0.251)  +64(0.202)
        window=32(-16~+16)는 앞의 두 개만 담고 +24 이후를 통째로 버린다.
        window=96, window_offset=24로 두면 -24~+72를 담아 6개를 모두 포함한다.

        extra_labels: {헤드이름: (n,4,256) 배열} 멀티태스크 학습용.
        주면 배치의 y가 dict로 나온다.
        """
        if batch_size % traces_per_batch != 0:
            raise ValueError("batch_size는 traces_per_batch의 배수여야 함")
        if window % 2 != 0:
            raise ValueError("window는 짝수를 권장 (중심 정렬)")
        self.traces = traces
        self.hw = hw_labels
        self.extra_labels = extra_labels or {}
        self.window_offset = int(window_offset)
        centers = np.asarray(centers) + self.window_offset
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

        # 계수별 추출 확률. 누설이 계수 인덱스 k에 따라 물리적으로 약해지므로
        # (|rho| k=0~31에서 0.721 -> k=224~255에서 0.611) 약한 계수가 전체 키
        # 복구의 병목이 된다. 가중치를 주면 그쪽을 더 자주 학습하게 할 수 있다.
        if coeff_weights is None:
            self.coeff_p = None
        else:
            w = np.asarray(coeff_weights, dtype=np.float64)[self.coeffs]
            if w.min() < 0 or w.sum() <= 0:
                raise ValueError("coeff_weights는 음수가 없고 합이 양수여야 한다")
            self.coeff_p = w / w.sum()

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
            extra = {k: np.empty(self.batch_size, dtype=np.int32)
                     for k in self.extra_labels}
            pos = 0
            for ti in trace_sel:
                row = np.asarray(self.traces[ti], dtype=np.float32)   # 연속 읽기
                pj = self.rng.choice(self.polys, self.per_trace)
                pk = self.rng.choice(self.coeffs, self.per_trace, p=self.coeff_p)
                cen = self.centers[pj, pk]
                if self.shift_aug:
                    cen = cen + self.rng.integers(-self.shift_aug,
                                                  self.shift_aug + 1, self.per_trace)
                for m in range(self.per_trace):
                    s = cen[m] - half
                    X[pos] = row[s:s + self.window]
                    y[pos] = self.hw[ti, pj[m], pk[m]]
                    for k, arr in self.extra_labels.items():
                        extra[k][pos] = arr[ti, pj[m], pk[m]]
                    pos += 1
            yield (X[:, :, None], dict(extra, hw=y) if extra else y)

    def full_coefficient_batch(self, trace_index, poly, coeff_list):
        """공격/평가용: 한 트레이스에서 지정 계수들의 윈도우를 한 번에 뽑는다."""
        half = self.window // 2
        row = np.asarray(self.traces[trace_index], dtype=np.float32)
        cen = self.centers[poly, np.asarray(coeff_list)]
        X = np.stack([row[c - half:c + half] for c in cen])
        return X[:, :, None]


def to_tf_dataset(sampler, n_classes=33):
    """tf.data 래핑. 배치 단위 제너레이터라 Python 오버헤드가 배치당으로 줄어든다.

    단일 헤드와 멀티태스크를 모두 처리한다. 라벨 형식이 손실 함수와 짝이 맞아야 한다.

      단일 헤드   : y를 one-hot으로. compile_model()이 CategoricalCrossentropy를 쓴다.
      멀티태스크  : y를 정수 dict 그대로. compile_multitask()가
                   SparseCategoricalCrossentropy를 쓰므로 one-hot을 씌우면 안 되고,
                   헤드마다 클래스 수가 달라(2 또는 9) 공통 one_hot도 불가능하다.
    """
    import tensorflow as tf

    x_spec = tf.TensorSpec(shape=(sampler.batch_size, sampler.window, 1),
                           dtype=tf.float32)
    if getattr(sampler, "extra_labels", None):
        # 샘플러는 extra_labels + "hw"를 내놓는다. 모델 출력 이름과 정확히 맞아야
        # 하므로 build_multitask_model(include_hw=True)여야 한다(기본값).
        heads = list(sampler.extra_labels.keys()) + ["hw"]
        y_spec = {h: tf.TensorSpec(shape=(sampler.batch_size,), dtype=tf.int32)
                  for h in heads}
        ds = tf.data.Dataset.from_generator(sampler.batches,
                                            output_signature=(x_spec, y_spec))
        return ds.prefetch(tf.data.AUTOTUNE)

    y_spec = tf.TensorSpec(shape=(sampler.batch_size,), dtype=tf.int32)
    ds = tf.data.Dataset.from_generator(sampler.batches,
                                        output_signature=(x_spec, y_spec))
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, n_classes)),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
