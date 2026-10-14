"""v3 학습 실행기.

기본값은 v3 README의 계수 불변 단일 HW 헤드 설정이다. 실제 실행은
프로젝트 루트에서 다음처럼 한다::

    .venv-v3/bin/python "project v3/v3_train.py"

이 파일은 데이터 전처리를 다시 하지 않고, 이미 생성된 float16 memmap을
사용한다. 장시간 실행 중에도 배치/epoch 로그를 stdout과 CSV에 남긴다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf

import v3_backend as bk


class LiveProgress(tf.keras.callbacks.Callback):
    """Keras callback that emits compact live progress without a noisy progress bar."""

    def __init__(self, report_every: int = 1000):
        super().__init__()
        self.report_every = max(1, int(report_every))
        self.started = None
        self.epoch_started = None
        self.steps = None

    def on_train_begin(self, logs=None):
        self.started = time.time()
        print("[v3] 학습 시작", flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_started = time.time()
        self.steps = None
        print(f"[v3] epoch {epoch + 1} 시작", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        step = batch + 1
        if step == 1 or step % self.report_every == 0:
            elapsed = time.time() - self.epoch_started
            loss = logs.get("loss", float("nan"))
            acc = logs.get("accuracy")
            suffix = f" accuracy={acc:.4f}" if acc is not None else ""
            print(f"[v3] batch={step} elapsed={elapsed/60:.1f}m "
                  f"loss={loss:.5f}{suffix}", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.epoch_started
        pairs = []
        for key in ("loss", "accuracy", "top3", "val_loss", "val_accuracy", "val_top3"):
            if key in logs:
                pairs.append(f"{key}={logs[key]:.5f}")
        print(f"[v3] epoch {epoch + 1} 완료 ({elapsed/60:.1f}m) "
              + " ".join(pairs), flush=True)

    def on_train_end(self, logs=None):
        total = 0.0 if self.started is None else time.time() - self.started
        print(f"[v3] 학습 종료: {total/3600:.2f}h", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Train the v3 coefficient-invariant model")
    p.add_argument("--dataset-dir", type=Path,
                   default=ROOT / "Dataset" / "1-Dataset")
    p.add_argument("--prof-norm", type=Path,
                   default=ROOT / "artifacts" / "v3" / "prof_norm.npy")
    p.add_argument("--time-map", type=Path,
                   default=ROOT / "project v3" / "time_map.json")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "artifacts" / "v3" / "training_single")
    p.add_argument("--mode", choices=("single", "multitask"), default="single")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--traces-per-batch", type=int, default=32)
    p.add_argument("--validation-steps", type=int, default=200)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--window-offset", type=int, default=24)
    p.add_argument("--shift-aug", type=int, default=1)
    p.add_argument("--val-set", type=int, default=4)
    p.add_argument("--report-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="기존 best 체크포인트가 있으면 가중치를 불러와 이어서 실행")
    return p.parse_args()


def check_paths(args):
    required = [args.dataset_dir, args.prof_norm, args.time_map]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("필수 입력 없음: " + ", ".join(missing))
    if args.epochs < 1 or args.validation_steps < 1:
        raise ValueError("epochs와 validation_steps는 양수여야 합니다.")
    if args.batch_size % args.traces_per_batch:
        raise ValueError("batch-size는 traces-per-batch의 배수여야 합니다.")


def main():
    args = parse_args()
    check_paths(args)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bk.configure_numpy_threads()
    info = bk.configure_tensorflow(verbose=True)

    import v3_common as vc
    import v3_model as vm
    import v3_stage2_data as s2
    print("[v3] 데이터 검증 게이트", flush=True)
    vc.verify_dataset(str(args.dataset_dir), verbose=True)
    X = np.load(args.prof_norm, mmap_mode="r")
    if X.shape != (vc.N_PROFILING, vc.TRACE_LEN) or X.dtype != np.float16:
        raise ValueError(f"prof_norm shape/dtype 오류: {X.shape} {X.dtype}")
    u = vc.load_array(str(args.dataset_dir), "profiling_40000_u=cs.npy")
    hw = s2.precompute_hw_labels(u)
    centers = s2.build_time_map(vc.load_time_map(str(args.time_map)))
    train_idx, val_idx = s2.set_split_indices(val_set=args.val_set)

    extra = None
    if args.mode == "multitask":
        extra = s2.precompute_multitask_labels(u)

    train = s2.CoefficientWindowSampler(
        X, hw, centers, train_idx,
        window=args.window, window_offset=args.window_offset,
        batch_size=args.batch_size, traces_per_batch=args.traces_per_batch,
        shift_aug=args.shift_aug, seed=args.seed, extra_labels=extra,
    )
    val = s2.CoefficientWindowSampler(
        X, hw, centers, val_idx,
        window=args.window, window_offset=args.window_offset,
        batch_size=args.batch_size, traces_per_batch=args.traces_per_batch,
        shift_aug=0, seed=args.seed + 1, extra_labels=extra,
    )
    train_ds = s2.to_tf_dataset(train)
    val_ds = s2.to_tf_dataset(val)

    if args.mode == "multitask":
        model = vm.compile_multitask(vm.build_multitask_model(window=args.window))
    else:
        model = vm.compile_model(vm.build_model(window=args.window))

    best_path = args.out_dir / "best.keras"
    final_path = args.out_dir / "final.keras"
    csv_path = args.out_dir / "history.csv"
    config_path = args.out_dir / "config.json"
    config = {
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
        "tensorflow": info,
        "model_params": model.count_params(),
        "train_samples_per_epoch": train.samples_per_epoch(),
        "train_steps_per_epoch": train.steps_per_epoch(),
        "val_samples_per_epoch": val.samples_per_epoch(),
        "val_steps_per_epoch": val.steps_per_epoch(),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    callbacks = vm.make_callbacks(checkpoint_path=str(best_path))
    callbacks.extend([
        tf.keras.callbacks.CSVLogger(str(csv_path), append=args.resume),
        tf.keras.callbacks.TerminateOnNaN(),
        LiveProgress(report_every=args.report_every),
    ])

    if args.resume and best_path.exists():
        print(f"[v3] best 체크포인트 로드: {best_path}", flush=True)
        model.load_weights(best_path)
    elif args.resume:
        print("[v3] 재개할 체크포인트가 없어 새로 시작합니다.", flush=True)

    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=train.steps_per_epoch(),
        validation_steps=args.validation_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
    )
    model.save(final_path)
    print(f"[v3] 최종 모델 저장: {final_path}", flush=True)
    print(f"[v3] 최적 모델 저장: {best_path}", flush=True)


if __name__ == "__main__":
    main()
