"""저장된 v3 결과의 독립 재계산. 모델 학습/추론은 실행하지 않는다."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

import v3_common as vc
import v3_evaluate as ev
import v3_report as vr

ROOT = Path(__file__).resolve().parents[1]


def layer_weights(path):
    with zipfile.ZipFile(path) as archive:
        data = archive.read("model.weights.h5")
    weights = {}
    with h5py.File(io.BytesIO(data), "r") as h5:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = np.asarray(obj)
        h5["layers"].visititems(collect)
    return weights


def main():
    out = ROOT / "artifacts/v3/evaluation_single"
    results_path = out / "results.json"
    d = json.loads(results_path.read_text())
    checks = []

    def check(name, ok, detail=None):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})
        if not ok:
            raise AssertionError(name)

    with open(ROOT / "artifacts/v3/training_single/history.csv", newline="") as f:
        history = list(csv.DictReader(f))
    check("history_epochs", len(history) == 44 and
          [int(r["epoch"]) for r in history] == list(range(44)))
    best = min(history, key=lambda r: float(r["val_loss"]))
    check("best_epoch", int(best["epoch"]) + 1 == d["training_history"]["best_epoch"])
    check("best_metrics", all(np.isclose(float(best[k]), v, atol=1e-12)
          for k, v in d["training_history"]["best"].items()))

    with np.load(out / "target_predictions.npz") as saved:
        for split, report_name in (("validation", "target_validation"),
                                   ("attack", "target_attack")):
            p, y = saved[f"{split}_probs"], saved[f"{split}_labels"]
            check(f"{split}_probabilities", p.shape == (10000, 33) and
                  np.isfinite(p).all() and (p >= 0).all() and
                  np.allclose(p.sum(axis=1), 1, atol=2e-6))
            rep = vr.classification_report(p, y)
            expected = d["classification"][report_name]
            check(f"{split}_classification", all(
                np.isclose(rep[k], expected[k], atol=1e-12)
                for k in ("top1", "top3", "top5", "majority_baseline", "mean_abs_error")))
        t, nll = ev.fit_temperature(saved["validation_probs"], saved["validation_labels"])
        check("target_temperature", np.isclose(t, d["key_attack"]["temperature"]) and
              np.isclose(nll, d["key_attack"]["calibration_nll"]))
        c = vc.load_array(str(ROOT / "Dataset/1-Dataset"), "attack_10000_c.npy")
        s = vc.load_array(str(ROOT / "Dataset/1-Dataset"), "attack_s.npy")
        j, k = d["scope"]["target_poly"], d["scope"]["target_coeff"]
        cands, tidx = ev.build_candidate_residues(65536, 42, int(s[j, k]))
        check("candidate_count", len(cands) == d["key_attack"]["n_candidates"])
        prior = ev.implicit_log_prior(saved["validation_probs"])
        likelihood = ev.trace_log_likelihood(saved["attack_probs"], t, prior)
        ge, sr, ge_full = ev.guessing_entropy(
            likelihood, c, k, cands, tidx, n_experiments=100,
            max_traces=40, seed=42, verbose=False)
        check("attack_curves_recomputed", all(np.allclose(actual, d["key_attack"][key])
              for key, actual in (("ge", ge), ("sr", sr), ("ge_full", ge_full))))

    check("mid_rank_ties", np.array_equal(
        ev._mid_rank(np.array([[1., 1., 0.], [2., 1., 0.], [1., 1., 1.]]), 0),
        np.array([0.5, 0., 1.])))
    for name, metrics in d["calibration"].items():
        if not isinstance(metrics, dict):
            continue
        bins = metrics["bins"]
        n = sum(b["n"] for b in bins)
        ece = sum(b["n"] * abs(b["accuracy"] - b["confidence"]) for b in bins) / n
        check(f"{name}_ece", n == 102400 and np.isclose(ece, metrics["ece"]))

    rows = d["per_coefficient"]["rows"]
    top1 = np.array([r["top1"] for r in rows])
    pi = np.array([r["pi"] for r in rows])
    summary = d["per_coefficient"]["summary"]
    check("coefficient_summary", np.isclose(top1.mean(), summary["mean"]) and
          np.isclose(top1.std(), summary["std"]) and
          np.isclose(np.median(pi), summary["pi_median"]))

    best_w = layer_weights(ROOT / "artifacts/v3/training_single/best.keras")
    final_w = layer_weights(ROOT / "artifacts/v3/training_single/final.keras")
    check("best_final_all_layer_weights", best_w.keys() == final_w.keys() and all(
        np.array_equal(value, final_w[name]) for name, value in best_w.items()),
        {"arrays_compared": len(best_w)})

    u = vc.load_array(str(ROOT / "Dataset/1-Dataset"), "attack_10000_u=cs.npy")
    raw_mismatch = centered_mismatch = 0
    for start in range(0, len(c), 500):
        cc = np.asarray(c[start:start + 500], dtype=np.int64)[:, None, :]
        reference = np.asarray(u[start:start + 500], dtype=np.int64)
        raw_mismatch += int(np.count_nonzero(vc.mr(cc * np.asarray(s)[None]) != reference))
        centered_mismatch += int(np.count_nonzero(
            vc.mr(cc * ev.centered_residue(s)[None]) != reference))
    check("mr_original_key_full_attack", raw_mismatch == 0,
          {"values_checked": int(u.size)})
    boundary = [r for r in rows if r["coeff"] in (0, 255)]
    interior = [r for r in rows if r["coeff"] not in (0, 255)]
    payload = {
        "audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "representative_groups": {
            "boundary": {"n_coefficients": len(boundary), "mean_top1": float(np.mean(
                [r["top1"] for r in boundary]))},
            "interior": {"n_coefficients": len(interior), "mean_top1": float(np.mean(
                [r["top1"] for r in interior]))},
        },
        "mr_centered_representative": {
            "mismatches": centered_mismatch, "total": int(u.size),
            "fraction": centered_mismatch / u.size,
            "note": "Residue-equivalent MR outputs can have different signed HW at overflow boundaries.",
        },
        "limits": [
            "Random 102400-window probabilities were not saved; top-k metrics for that sample cannot be independently recomputed from NPZ.",
            "Calibration reuses profiling set4, also used for model selection.",
            "Candidate ranks cover a fixed sampled candidate set, not exhaustive q-space recovery.",
            "PI/log2(q) trace counts are heuristics, not empirical full-key recovery.",
        ],
    }
    path = out / "audit.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
