"""DSCAD 데이터셋 공통 상수 / 연산 / 검증 유틸리티.

이 파일의 모든 상수는 원본 데이터셋 헤더와 공식 Verification.ipynb에서
실측 확인된 값이다. v1/v2 instruction.md의 "총 160,000 트레이스" 서술은 오류이며,
실제 프로파일링 트레이스는 10,000 x 4 = 40,000개다.
"""

import os

import numpy as np

Q = 8380417
QINV = 58728449
N_COEFF = 256           # 다항식당 계수 수
L_POLY = 4              # 비밀키 다항식 수
TRACE_LEN = 40000       # 트레이스당 시간 샘플 수
N_PROFILING_SETS = 4
N_PER_SET = 10000
N_PROFILING = N_PROFILING_SETS * N_PER_SET
N_ATTACK = 10000
N_HW_CLASSES = 33

# 원본 파일별 (shape, dtype). 로딩 시 전량 assert 한다.
EXPECTED = {
    "profiling_40000traces_set1.npy": ((N_PER_SET, TRACE_LEN), np.int64),
    "profiling_40000traces_set2.npy": ((N_PER_SET, TRACE_LEN), np.int64),
    "profiling_40000traces_set3.npy": ((N_PER_SET, TRACE_LEN), np.int64),
    "profiling_40000traces_set4.npy": ((N_PER_SET, TRACE_LEN), np.int64),
    "profiling_40000_u=cs.npy": ((N_PROFILING, L_POLY, N_COEFF), np.int32),
    "attack_10000traces.npy": ((N_ATTACK, TRACE_LEN), np.int64),
    "attack_10000_u=cs.npy": ((N_ATTACK, L_POLY, N_COEFF), np.int32),
    "attack_10000_c.npy": ((N_ATTACK, N_COEFF), np.int32),
    "attack_s.npy": ((L_POLY, N_COEFF), np.int64),
}

_POPCNT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def mr(d):
    """Montgomery reduction. 공식 Verification.ipynb의 스칼라 MR과 완전히 일치한다.

    d * QINV 은 int64를 넘치지만 mod 2^32 결과만 필요하므로 uint64 랩어라운드를
    그대로 이용한다 (원본 노트북이 낸 overflow 경고의 정체이기도 하다).
    """
    d = np.asarray(d, dtype=np.int64)
    g = (d.astype(np.uint64) * np.uint64(QINV)) & np.uint64(0xFFFFFFFF)
    g = g.astype(np.int64)
    g = np.where(g >= 2 ** 31, g - 2 ** 32, g)
    return (d - g * Q) >> np.int64(32)


def hw32(x):
    """32비트 비트패턴 기준 Hamming Weight. 음수는 2의 보수로 해석된다."""
    x = np.asarray(x, dtype=np.int64) & 0xFFFFFFFF
    lo = _POPCNT16[(x & 0xFFFF).astype(np.uint16)].astype(np.int16)
    hi = _POPCNT16[((x >> 16) & 0xFFFF).astype(np.uint16)].astype(np.int16)
    return lo + hi


def u_from_hypothesis(c_k, s_hyp):
    """가설 s에 대한 중간값 u = MR(c[k] * s). c_k와 s_hyp는 브로드캐스트 가능해야 한다."""
    return mr(np.asarray(c_k, dtype=np.int64) * np.asarray(s_hyp, dtype=np.int64))


def load_array(dataset_dir, name, mmap=True):
    """shape/dtype을 assert하며 로드한다. v2의 조용한 실패를 막는 유일한 방어선이다."""
    path = os.path.join(dataset_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 없음. DSCAD V2/1-Dataset 전체를 받았는지 확인할 것.")
    arr = np.load(path, mmap_mode="r" if mmap else None)
    exp_shape, exp_dtype = EXPECTED[name]
    if arr.shape != exp_shape:
        raise ValueError(f"{name}: shape {arr.shape}, 기대값 {exp_shape}")
    if arr.dtype != exp_dtype:
        raise ValueError(f"{name}: dtype {arr.dtype}, 기대값 {exp_dtype}")
    return arr


def iter_profiling_traces(dataset_dir, chunk=1000, dtype=np.float32):
    """프로파일링 4개 set을 순서대로 청크 단위 스트리밍한다.

    set 하나가 int64로 3.05GB이므로 절대 통째로 올리지 않는다. 산출 순서는
    profiling_40000_u=cs.npy의 행 순서와 일치한다 (set1 -> set4).
    """
    for si in range(1, N_PROFILING_SETS + 1):
        arr = load_array(dataset_dir, f"profiling_40000traces_set{si}.npy")
        for start in range(0, N_PER_SET, chunk):
            yield si, start, np.asarray(arr[start:start + chunk], dtype=dtype)


def coefficient_time_center(j, k, poly_stride=10000.0, coeff_stride=35.0, offset=78.0):
    """계수 (poly j, coeff k) 연산이 누설되는 대략적 시간 샘플 위치.

    rho.npy 20개 행의 피크 위치에서 역산한 가설이다. 반드시
    calibrate_time_map()으로 실측 보정한 뒤 사용할 것 (기본값은 poly0 기준이라
    poly1~3에서는 어긋난다).
    """
    return offset + poly_stride * j + coeff_stride * k


def calibrate_time_map(traces, u_labels, j, coeff_list, search_lo=0, search_hi=TRACE_LEN):
    """실측 트레이스로 (poly j)의 계수->시간 선형 매핑을 보정한다.

    각 계수의 HW와 트레이스 간 상관계수 피크 위치를 찾은 뒤 k에 대해 직선을 적합한다.
    반환값은 (coeff_stride, offset) 이며 coefficient_time_center에 그대로 넣으면 된다.
    """
    peaks = []
    seg = np.asarray(traces[:, search_lo:search_hi], dtype=np.float64)
    seg -= seg.mean(axis=0, keepdims=True)
    seg_ss = np.sqrt((seg ** 2).sum(axis=0))
    seg_ss[seg_ss == 0] = 1.0
    for k in coeff_list:
        h = hw32(u_labels[:len(seg), j, k]).astype(np.float64)
        h -= h.mean()
        denom = np.sqrt((h ** 2).sum())
        if denom == 0:
            raise ValueError(f"coeff {k}: HW 분산이 0")
        rho = (seg * h[:, None]).sum(axis=0) / (seg_ss * denom)
        peaks.append(search_lo + int(np.argmax(np.abs(rho))))
    coeff_stride, offset = np.polyfit(np.asarray(coeff_list, dtype=np.float64), peaks, 1)
    return float(coeff_stride), float(offset), peaks


def verify_dataset(dataset_dir, n_check=200, verbose=True):
    """Stage 0 게이트. 여기를 통과하지 못하면 이후 단계를 진행하지 않는다."""
    report = {}
    for name in EXPECTED:
        arr = load_array(dataset_dir, name)
        report[name] = (arr.shape, str(arr.dtype))

    u = load_array(dataset_dir, "attack_10000_u=cs.npy")
    c = load_array(dataset_dir, "attack_10000_c.npy")
    s = load_array(dataset_dir, "attack_s.npy")

    # u[i][j][k] == MR(c[i][k] * s[j][k]) 관계 검증
    cc = np.asarray(c[:n_check], dtype=np.int64)
    ss = np.asarray(s, dtype=np.int64)
    got = mr(cc[:, None, :] * ss[None, :, :])
    if not np.array_equal(got, np.asarray(u[:n_check], dtype=np.int64)):
        raise ValueError("u = MR(c*s) 관계 검증 실패. 파일 조합이 잘못되었을 수 있음.")
    report["mr_relation_ok"] = True

    # s -> s+q 불변성: 후보 공간이 q개임을 보증하는 근거
    inv = np.array_equal(mr(cc[:, 50] * (int(ss[0, 50]) + Q)),
                         np.asarray(u[:n_check, 0, 50], dtype=np.int64))
    report["s_plus_q_invariant"] = bool(inv)

    hw = hw32(u[:, 0, 50])
    counts = np.bincount(hw, minlength=N_HW_CLASSES)
    report["hw_distribution"] = counts
    report["majority_class_accuracy"] = float(counts.max() / counts.sum())
    report["n_negative_u"] = int(np.sum(np.asarray(u[:, 0, 50]) < 0))

    if verbose:
        for k, v in report.items():
            if k == "hw_distribution":
                nz = np.nonzero(v)[0]
                print(f"  HW 분포: {nz.min()}~{nz.max()} 사용, 최빈 HW={int(v.argmax())}")
            else:
                print(f"  {k}: {v}")
        print(f"  >> 다수 클래스 베이스라인 = {report['majority_class_accuracy']*100:.2f}% "
              f"(모델 정확도는 반드시 이 값과 비교해야 함)")
    return report
