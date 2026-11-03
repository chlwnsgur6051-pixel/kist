"""Check bundled datasets, epoch trajectories, weights, and source provenance."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = "6142a720dbd205fe9b75a34f76fd2ba264cfe226821c4a2c285ef4cfe7a63177"
EXPECTED = {("Original", "Baseline"): (295, 10),
            ("Original", "W >= 0"): (None, 5),
            ("Modified", "Baseline"): (48, 10),
            ("Modified", "W >= 0"): (52, 10)}


def main() -> None:
    source = ROOT / "src/continuous_stdp_experiment.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_HASH
    with ZipFile(ROOT / "src/reference_20261102_105031.zip") as z:
        for dataset, filename in (("original", "original_K10.npz"),
                                  ("modified", "modified_K10.npz")):
            with np.load(io.BytesIO(z.read(
                    f"20261102_105031/npz/{dataset}_K10_seed0_Basic_Correction.npz"))) as a, \
                 np.load(ROOT / "datasets" / filename) as b:
                assert np.array_equal(a["input"], b["X_binary"])
                assert np.array_equal(a["target"], b["Y_binary"])
                assert a["input"].shape == a["target"].shape == (10, 15)
    with (ROOT / "results/csv/summary.csv").open() as f:
        summary = list(csv.DictReader(f))
    with (ROOT / "results/csv/epoch_comparison.csv").open() as f:
        comparison = list(csv.DictReader(f))
    assert len(summary) == 4 and len(comparison) == 4000
    assert all(int(r["delta_continuous"]) == int(r["delta_independent"]) == 0
               for r in comparison)
    with np.load(ROOT / "results/checkpoints/final_weights.npz") as new, \
         np.load(ROOT / "references/previous_final_weights.npz") as prior:
        weight_diff = {}
        for dataset in ("Original", "Modified"):
            for condition, new_suffix, old_suffix in (("Baseline", "baseline", "baseline"),
                                                       ("W >= 0", "floor", "nonnegative")):
                key = (dataset, condition)
                row = next(s for s in summary if (s["dataset"], s["model"]) == key)
                first, final = EXPECTED[key]
                assert (row["first_10_continuous"] == "" if first is None
                        else int(row["first_10_continuous"]) == first)
                assert int(row["final_continuous"]) == final
                assert int(row["final_independent"]) == final
                assert int(row["non_4ms_natural_spike_count"]) == 0
                a = new[f"{dataset}_{new_suffix}"]
                b = prior[f"{dataset.lower()}_{old_suffix}"]
                assert a.shape == b.shape == (15, 15)
                difference = float(np.max(np.abs(a - b)))
                assert difference < 1e-12
                weight_diff[f"{dataset}_{new_suffix}"] = difference
    print(json.dumps({"status": "pass", "epoch_comparisons": len(comparison),
                      "accuracy_mismatches": 0, "max_weight_differences": weight_diff},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
