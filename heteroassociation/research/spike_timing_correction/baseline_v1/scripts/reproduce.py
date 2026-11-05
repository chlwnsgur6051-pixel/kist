"""Reproduce the two archived baseline conditions without changing their model.

Run from any working directory. The only mutable globals set in the copied
sweep runner are the archived condition selectors: CS end, lambda, and cap.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "src/run_sweep.py"
    spec = importlib.util.spec_from_file_location("archived_run_sweep", path)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    runner.MODEL_PATH = ROOT / "src/sweep_model.py"
    return runner


def check_dataset(root, name, x, y, orders):
    with np.load(root / "datasets" / f"{name.lower()}_K10.npz") as saved:
        assert np.array_equal(x, saved["input_binary"])
        assert np.array_equal(y, saved["target_binary"])
        assert np.array_equal(orders, saved["presentation_orders"])


def run_one(runner, name, destination):
    model, x, y, weight, history, summary, timing, logs = runner.simulate(
        name, 4.5, 0.5, epochs=1000, current_cap=None, save_events=True
    )
    orders = model.load_data(name)[2]
    check_dataset(ROOT, name, x, y, orders)
    with (ROOT / "results/csv" / f"{name.lower()}_epoch_history.csv").open(newline="") as f:
        archived_history = list(csv.DictReader(f))
    with np.load(ROOT / "results/checkpoints" / f"{name.lower()}_training.npz") as archived:
        assert np.array_equal(x, archived["input_binary"])
        assert np.array_equal(y, archived["target_binary"])
        assert np.array_equal(orders, archived["presentation_orders"])
        assert np.array_equal(logs[0], archived["output_spike_ticks"])
        assert np.array_equal(logs[1], archived["output_spike_count"])
        weight_error = float(np.max(np.abs(weight - archived["final_weight"])))
    assert len(history) == len(archived_history) == 1000
    assert all(int(a["exact_count"]) == int(b["exact_count"]) and
               abs(float(a["bit_accuracy_percent"]) - float(b["bit_accuracy_percent"])) < 1e-10
               for a, b in zip(history, archived_history))
    assert weight_error < 1e-10, weight_error
    expected = {"Original": (113, 226), "Modified": (21, 37)}[name]
    assert (summary["first_10_epoch"], summary["sustained_10_start_epoch"]) == expected
    assert summary["final_exact_count"] == 10
    assert summary["teacher_spikes_expected"] == summary["teacher_spikes_observed"]
    assert summary["extra_natural_spikes"] == 0
    assert float(weight.min()) < 0  # This signed-weight condition has no floor.
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / f"{name.lower()}_final_weights.npz", weight=weight,
                        input_binary=x, target_binary=y)
    with (destination / f"{name.lower()}_epoch_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return {"dataset": name, "first_10_epoch": summary["first_10_epoch"],
            "sustained_10_start_epoch": summary["sustained_10_start_epoch"],
            "final_exact_count": summary["final_exact_count"],
            "final_bit_accuracy_percent": summary["final_bit_accuracy_percent"],
            "max_abs_final_weight_error": weight_error,
            "archived_epoch_accuracy_rows_matched": 1000,
            "archived_spike_events_matched": 1,
            "teacher_spikes_expected": summary["teacher_spikes_expected"],
            "teacher_spikes_observed": summary["teacher_spikes_observed"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Original", "Modified", "both"), default="both")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    runner = load_runner()
    datasets = ("Original", "Modified") if args.dataset == "both" else (args.dataset,)
    rows = [run_one(runner, name, args.output) for name in datasets]
    with (args.output / "verification.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
