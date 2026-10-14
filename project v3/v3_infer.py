"""v3 단일 HW 모델의 재현 가능한 추론·공격 평가 실행기.

학습에 사용하지 않은 공격 캠페인에서 분류 성능을 재고, profiling set4를
캘리브레이션 데이터로 사용해 대표 계수의 GE/SR을 계산한다. GE/SR은 기본적으로
65,536개 균등 후보 표본에 대한 값이며, 전체 q 공간 값은 선형 환산 추정치다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import v3_backend as bk

bk.configure_numpy_threads()

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained v3 single-head model")
    p.add_argument("--dataset-dir", type=Path,
                   default=ROOT / "Dataset" / "1-Dataset")
    p.add_argument("--prof-norm", type=Path,
                   default=ROOT / "artifacts" / "v3" / "prof_norm.npy")
    p.add_argument("--attack-norm", type=Path,
                   default=ROOT / "artifacts" / "v3" / "attack_norm.npy")
    p.add_argument("--time-map", type=Path,
                   default=ROOT / "project v3" / "time_map.json")
    p.add_argument("--model", type=Path,
                   default=ROOT / "artifacts" / "v3" / "training_single" / "best.keras")
    p.add_argument("--final-model", type=Path,
                   default=ROOT / "artifacts" / "v3" / "training_single" / "final.keras")
    p.add_argument("--history", type=Path,
                   default=ROOT / "artifacts" / "v3" / "training_single" / "history.csv")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "artifacts" / "v3" / "evaluation_single")
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--window-offset", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--classification-batches", type=int, default=200)
    p.add_argument("--per-coeff-traces", type=int, default=2000)
    p.add_argument("--coefficients-per-poly", type=int, default=9)
    p.add_argument("--target-poly", type=int, default=0)
    p.add_argument("--target-coeff", type=int, default=50)
    p.add_argument("--n-candidates", type=int, default=65536)
    p.add_argument("--n-experiments", type=int, default=100)
    p.add_argument("--max-traces", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def require_paths(args):
    required = [args.dataset_dir, args.prof_norm, args.attack_norm, args.time_map,
                args.model, args.history]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("필수 입력 없음: " + ", ".join(missing))


def load_history(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"학습 이력이 비어 있음: {path}")
    parsed = []
    for row in rows:
        parsed.append({k: int(v) if k == "epoch" else float(v)
                       for k, v in row.items()})
    best = min(parsed, key=lambda r: r["val_loss"])
    return {
        "epochs_completed": len(parsed),
        "best_epoch": best["epoch"] + 1,
        "best": best,
        "last": parsed[-1],
        "early_stopped": len(parsed) < 50,
    }


def collect_random_predictions(model, sampler, n_batches, predict_batch_size):
    gen = sampler.batches()
    xs, ys = [], []
    for _ in range(n_batches):
        xb, yb = next(gen)
        xs.append(xb)
        ys.append(yb)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    probs = model.predict(x, batch_size=predict_batch_size, verbose=0)
    return np.asarray(probs), y


def predict_one_coefficient(model, sampler, trace_indices, poly, coeff,
                            predict_batch_size):
    trace_indices = np.asarray(trace_indices, dtype=np.int64)
    x = np.empty((len(trace_indices), sampler.window, 1), dtype=np.float32)
    for pos, ti in enumerate(trace_indices):
        x[pos] = sampler.full_coefficient_batch(int(ti), poly, [coeff])[0]
    return np.asarray(model.predict(x, batch_size=predict_batch_size, verbose=0))


def representative_pairs(n_per_poly, target_poly, target_coeff):
    coeffs = np.unique(np.rint(np.linspace(0, 255, n_per_poly)).astype(np.int64))
    pairs = [(j, int(k)) for j in range(4) for k in coeffs]
    target = (int(target_poly), int(target_coeff))
    if target not in pairs:
        pairs.append(target)
    return sorted(pairs)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    args = parse_args()
    require_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    info = bk.configure_tensorflow(verbose=True)
    import tensorflow as tf
    import v3_common as vc
    import v3_evaluate as ev
    import v3_model as vm
    import v3_report as vr
    import v3_stage2_data as s2

    print("[infer] 데이터 검증", flush=True)
    dataset_check = vc.verify_dataset(str(args.dataset_dir), verbose=True)

    prof = np.load(args.prof_norm, mmap_mode="r")
    attack = np.load(args.attack_norm, mmap_mode="r")
    if prof.shape != (vc.N_PROFILING, vc.TRACE_LEN) or prof.dtype != np.float16:
        raise ValueError(f"prof_norm shape/dtype 오류: {prof.shape} {prof.dtype}")
    if attack.shape != (vc.N_ATTACK, vc.TRACE_LEN) or attack.dtype != np.float16:
        raise ValueError(f"attack_norm shape/dtype 오류: {attack.shape} {attack.dtype}")

    centers = s2.build_time_map(vc.load_time_map(str(args.time_map)))
    u_prof = vc.load_array(str(args.dataset_dir), "profiling_40000_u=cs.npy")
    u_attack = vc.load_array(str(args.dataset_dir), "attack_10000_u=cs.npy")
    c_attack = vc.load_array(str(args.dataset_dir), "attack_10000_c.npy")
    s_attack = vc.load_array(str(args.dataset_dir), "attack_s.npy")
    hw_prof = s2.precompute_hw_labels(u_prof)
    hw_attack = s2.precompute_hw_labels(u_attack)
    _, val_idx = s2.set_split_indices(val_set=4)
    attack_idx = np.arange(vc.N_ATTACK, dtype=np.int64)

    val_sampler = s2.CoefficientWindowSampler(
        prof, hw_prof, centers, val_idx,
        window=args.window, window_offset=args.window_offset,
        batch_size=args.batch_size, traces_per_batch=32, shift_aug=0,
        seed=args.seed,
    )
    attack_sampler = s2.CoefficientWindowSampler(
        attack, hw_attack, centers, attack_idx,
        window=args.window, window_offset=args.window_offset,
        batch_size=args.batch_size, traces_per_batch=32, shift_aug=0,
        seed=args.seed + 1,
    )

    print(f"[infer] 모델 로드: {args.model}", flush=True)
    model = tf.keras.models.load_model(args.model, compile=False)
    history = load_history(args.history)

    print("[infer] 전체 계수 무작위 표본 분류 평가", flush=True)
    val_probs, val_y = collect_random_predictions(
        model, val_sampler, args.classification_batches, args.batch_size)
    attack_probs, attack_y = collect_random_predictions(
        model, attack_sampler, args.classification_batches, args.batch_size)
    cls_val = vr.classification_report(val_probs, val_y, label="profiling set4 표본")
    cls_attack = vr.classification_report(attack_probs, attack_y, label="attack 표본")

    temperature, calibration_nll = ev.fit_temperature(val_probs, val_y)
    val_probs_cal = ev.apply_temperature(val_probs, temperature)
    attack_probs_cal = ev.apply_temperature(attack_probs, temperature)
    calibration = {
        "temperature": temperature,
        "fit_nll": calibration_nll,
        "validation_raw": vr.calibration_report(val_probs, val_y),
        "validation_calibrated": vr.calibration_report(val_probs_cal, val_y),
        "attack_raw": vr.calibration_report(attack_probs, attack_y),
        "attack_calibrated": vr.calibration_report(attack_probs_cal, attack_y),
    }

    print(f"[infer] 대표 계수 poly={args.target_poly} coeff={args.target_coeff}", flush=True)
    target_val_probs = predict_one_coefficient(
        model, val_sampler, val_idx, args.target_poly, args.target_coeff, args.batch_size)
    target_attack_probs = predict_one_coefficient(
        model, attack_sampler, attack_idx, args.target_poly, args.target_coeff,
        args.batch_size)
    target_val_y = np.asarray(hw_prof[val_idx, args.target_poly, args.target_coeff])
    target_attack_y = np.asarray(hw_attack[:, args.target_poly, args.target_coeff])
    target_cls_val = vr.classification_report(
        target_val_probs, target_val_y, label="profiling set4 / poly0 coeff50")
    target_cls_attack = vr.classification_report(
        target_attack_probs, target_attack_y, label="attack / poly0 coeff50")

    key_attack = vr.attack_report(
        target_val_probs, target_val_y, target_attack_probs, c_attack,
        args.target_coeff, args.target_poly,
        true_s=int(s_attack[args.target_poly, args.target_coeff]),
        u_true=u_attack, n_candidates=args.n_candidates,
        n_experiments=args.n_experiments, max_traces=args.max_traces,
        seed=args.seed, verbose=True,
    )

    print("[infer] 대표 계수별 편차 평가", flush=True)
    rng = np.random.default_rng(args.seed)
    coeff_trace_idx = np.sort(rng.choice(
        attack_idx, size=min(args.per_coeff_traces, len(attack_idx)), replace=False))
    pairs = representative_pairs(args.coefficients_per_poly,
                                 args.target_poly, args.target_coeff)
    cache = {}
    for pos, (j, k) in enumerate(pairs, start=1):
        probs = predict_one_coefficient(
            model, attack_sampler, coeff_trace_idx, j, k, args.batch_size)
        cache[(j, k)] = (
            ev.apply_temperature(probs, temperature),
            np.asarray(hw_attack[coeff_trace_idx, j, k]),
        )
        if pos % 8 == 0 or pos == len(pairs):
            print(f"  {pos}/{len(pairs)} 계수 완료", flush=True)

    per_coeff, per_coeff_rows = vr.per_coefficient_report(
        lambda j, k: cache[(j, k)], pairs, verbose=True)
    full_key_estimate = ev.full_key_metrics(
        per_coeff["pi_values"], n_coefficients=vc.L_POLY * vc.N_COEFF,
        verbose=True)

    final_equivalence = None
    if args.final_model.exists():
        print("[infer] best/final 예측 일치 검사", flush=True)
        final_model = tf.keras.models.load_model(args.final_model, compile=False)
        probe = np.concatenate([
            val_sampler.full_coefficient_batch(int(ti), args.target_poly,
                                               [args.target_coeff])
            for ti in val_idx[:512]
        ])
        best_probe = np.asarray(model.predict(probe, batch_size=args.batch_size, verbose=0))
        final_probe = np.asarray(final_model.predict(probe, batch_size=args.batch_size, verbose=0))
        diff = np.abs(best_probe - final_probe)
        final_equivalence = {
            "n": len(probe),
            "max_abs_probability_difference": float(diff.max()),
            "mean_abs_probability_difference": float(diff.mean()),
            "argmax_agreement": float((best_probe.argmax(axis=1)
                                         == final_probe.argmax(axis=1)).mean()),
            "equivalent": bool(np.allclose(best_probe, final_probe, rtol=1e-6, atol=1e-7)),
        }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": {
            "model": args.model,
            "target_poly": args.target_poly,
            "target_coeff": args.target_coeff,
            "classification_samples_per_split": int(len(val_y)),
            "representative_coefficients": len(pairs),
            "traces_per_representative_coefficient": int(len(coeff_trace_idx)),
            "candidate_protocol": "uniform sampled candidates with mid-rank ties",
            "full_q_enumerated": False,
        },
        "environment": info,
        "dataset_check": dataset_check,
        "training_history": history,
        "classification": {
            "validation": cls_val,
            "attack": cls_attack,
            "target_validation": target_cls_val,
            "target_attack": target_cls_attack,
        },
        "calibration": calibration,
        "key_attack": key_attack,
        "per_coefficient": {
            "summary": per_coeff,
            "rows": per_coeff_rows,
        },
        "full_key_estimate_from_sample": full_key_estimate,
        "best_final_equivalence": final_equivalence,
    }

    result_path = args.out_dir / "results.json"
    result_path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))
    np.savez_compressed(
        args.out_dir / "target_predictions.npz",
        validation_probs=target_val_probs,
        validation_labels=target_val_y,
        attack_probs=target_attack_probs,
        attack_labels=target_attack_y,
    )
    print(f"[infer] 결과 저장: {result_path}", flush=True)


if __name__ == "__main__":
    main()
