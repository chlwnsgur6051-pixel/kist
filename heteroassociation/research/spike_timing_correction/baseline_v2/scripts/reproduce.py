"""Reproduce baseline_v2 (Step and Exponential) and check against the archived histories.

Usage (from the repository root):
  python research/spike_timing_correction/baseline_v2/scripts/reproduce.py --model both --dataset both
  python research/spike_timing_correction/baseline_v2/scripts/reproduce.py --legacy-v1   # baseline_v1 parity

Checks per epoch: exact count, bit accuracy, FP/FN, weight min/max/norm (bitwise),
and the epoch-1000 weight matrix. --epochs 5000 compares the whole archived history.
--legacy-v1 runs the same code in legacy mode (15.1 ms, old gap, no bound) and checks
every training spike and the final weight against research/.../baseline_v1.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))
import stc_model as M  # noqa: E402

KEYS = ("exact_count", "bit_accuracy_percent", "false_positive_bits", "false_negative_bits",
        "weight_min", "weight_max", "weight_norm")


def check_v2(model, dataset, epochs):
    spec = M.STEP_V2 if model == "step" else M.EXP_V2
    x, y, orders = M.load_reference_dataset(dataset)
    rows, w, state, extra = M.train(spec, x, y, orders, epochs, keep_weights={1000})
    saved = list(csv.DictReader(open(V2 / f"results/csv/{dataset.lower()}_{model}_v2_epoch_history.csv")))
    bad = [r["epoch"] for r, s in zip(rows, saved) if any(float(r[k]) != float(s[k]) for k in KEYS)]
    with np.load(V2 / f"results/checkpoints/{dataset.lower()}_{model}_v2_weights.npz") as z:
        w_ok = np.array_equal(extra["weights"][1000], z["weight_epoch1000"]) if epochs >= 1000 else None
        if epochs >= 5000:
            w_ok = w_ok and np.array_equal(w, z["weight_epoch5000"])
    first, sus = M.first_and_sustained(rows[:1000])
    ok = not bad and w_ok is not False
    print(f"{model:11s} {dataset:8s} epochs={epochs} first10={first} sustained10={sus} "
          f"exact@{min(epochs,1000)}={rows[min(epochs,1000)-1]['exact_count']}/10 "
          f"gap_spikes={sum(r['training_gap_spikes'] for r in rows)} clips={sum(r['clip_events'] for r in rows)} "
          f"mismatched_epochs={len(bad)} weights_identical={w_ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_v1(dataset):
    x, y, orders = M.load_reference_dataset(dataset)
    rows, w, _, extra = M.train(M.STEP_V1, x, y, orders, 1000, keep_events=True)
    with np.load(M.BASELINE_V1 / f"results/checkpoints/{dataset.lower()}_training.npz") as z:
        ok = (np.array_equal(extra["events"].astype(np.int64), z["output_spike_ticks"].astype(np.int64))
              and np.array_equal(w, z["final_weight"]))
    print(f"baseline_v1 legacy {dataset}: spikes and final weight identical -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("step", "exponential", "both"), default="both")
    ap.add_argument("--dataset", choices=("Original", "Modified", "both"), default="both")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--legacy-v1", action="store_true")
    a = ap.parse_args()
    datasets = ("Original", "Modified") if a.dataset == "both" else (a.dataset,)
    if a.legacy_v1:
        ok = all([check_v1(d) for d in datasets])
    else:
        models = ("step", "exponential") if a.model == "both" else (a.model,)
        ok = all([check_v2(m, d, a.epochs) for m in models for d in datasets])
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
