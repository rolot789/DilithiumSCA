"""Apple Silicon(M시리즈) 실행 환경 설정.

설치:
    pip install tensorflow-macos tensorflow-metal

M시리즈에서 주의할 점
  * 통합 메모리(Unified Memory): CPU/GPU가 RAM을 공유하므로 host->device 복사
    비용이 없다. 대신 총량을 공유하므로 데이터 사본을 늘리면 그대로 손해다.
    numpy memmap + tf.data 조합이 이 구조에 가장 잘 맞는다.
  * P코어/E코어가 섞여 있다. intra-op 스레드를 논리코어 전체로 잡으면 E코어에
    작업이 밀려 오히려 느려진다. P코어 수에 맞춘다.
  * XLA(jit_compile=True)는 Metal 백엔드에서 지원되지 않는다.
  * tensorflow-metal에서 mixed_float16은 버전에 따라 불안정했다. 기본은 float32로
    두고, 직접 수치 일치를 확인한 경우에만 켠다.
  * TF 2.11+ 신규 Adam은 M시리즈에서 legacy Adam보다 느린 사례가 보고되어 있다.
    make_optimizer()가 legacy 경로를 우선 시도한다.
"""

import os
import platform
import subprocess


def performance_core_count(default=8):
    """M시리즈 P코어(성능 코어) 수. Apple Silicon이 아니면 default."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return default


def is_apple_silicon():
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def configure_numpy_threads(n_threads=None):
    """Accelerate(vecLib) 스레드 수 제한. numpy import 전에 호출해야 효과가 있다."""
    n = n_threads or performance_core_count()
    for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    return n


def configure_tensorflow(mixed_precision=False, verbose=True):
    """TF 스레드/디바이스/정밀도 설정. 반환값은 진단 정보 dict."""
    import tensorflow as tf

    n_p = performance_core_count()
    tf.config.threading.set_intra_op_parallelism_threads(n_p)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    gpus = tf.config.list_physical_devices("GPU")
    info = {
        "apple_silicon": is_apple_silicon(),
        "tf_version": tf.__version__,
        "performance_cores": n_p,
        "gpu_devices": [g.name for g in gpus],
        "metal": bool(gpus) and is_apple_silicon(),
        "mixed_precision": False,
    }

    if mixed_precision:
        # Metal에서 불안정했던 이력이 있어 명시적으로 켤 때만 적용한다.
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")
        info["mixed_precision"] = True

    if verbose:
        print(f"  Apple Silicon: {info['apple_silicon']} / TF {info['tf_version']}")
        print(f"  P코어 {n_p}개, GPU 디바이스 {info['gpu_devices'] or '없음(CPU 실행)'}")
        if info["apple_silicon"] and not gpus:
            print("  >> Metal GPU 미인식. 'pip install tensorflow-metal' 확인 필요.")
        if mixed_precision:
            print("  >> mixed_float16 활성. 손실이 NaN이 되면 즉시 끌 것.")
    return info


def make_optimizer(learning_rate=1e-3):
    """M시리즈에서 더 빠른 legacy Adam을 우선 사용한다."""
    import tensorflow as tf
    if is_apple_silicon():
        try:
            return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        except AttributeError:
            pass
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def compile_kwargs():
    """Metal에서는 XLA를 끄고 컴파일해야 한다."""
    return {"jit_compile": False}
