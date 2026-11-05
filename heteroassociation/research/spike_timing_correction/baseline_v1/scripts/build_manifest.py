"""Write SHA-256 and provenance for every archived file."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
SOURCE = {
    "src/sweep_model.py": "results/timing_correction_cs_duration_lr_sweep/scripts/sweep_model.py",
    "src/run_sweep.py": "results/timing_correction_cs_duration_lr_sweep/scripts/run_sweep.py",
    "src/reference_20261102_105031.zip": "research/continuous_stdp_heteroassociation/src/reference_20261102_105031.zip",
    "src/reference_nonnegative_weight_results.zip": "research/continuous_stdp_heteroassociation/src/reference_nonnegative_weight_results.zip",
    "results/csv/original_epoch_history.csv": "results/timing_correction_cs_duration_lr_sweep/csv/epoch_original_end4.5_lambda0.5.csv",
    "results/csv/modified_epoch_history.csv": "results/timing_correction_cs_duration_lr_sweep/csv/epoch_modified_end4.5_lambda0.5.csv",
    "results/checkpoints/original_training.npz": "results/timing_correction_cs_duration_lr_sweep/npz/original_end4.5_lambda0.5.npz",
    "results/checkpoints/modified_training.npz": "results/timing_correction_cs_duration_lr_sweep/npz/modified_end4.5_lambda0.5.npz",
    "results/checkpoints/selected_traces.npz": "results/timing_correction_final_visualization/visualization_traces.npz",
    "configs/original_archived_condition.json": "results/timing_correction_cs_duration_lr_sweep/csv/condition_original_end4.5_lambda0.5.json",
    "configs/modified_archived_condition.json": "results/timing_correction_cs_duration_lr_sweep/csv/condition_modified_end4.5_lambda0.5.json",
    "results/csv/original_association_correctness.csv": "results/timing_correction_final_visualization/original_timing_association_correctness.csv",
    "results/csv/modified_association_correctness.csv": "results/timing_correction_final_visualization/modified_timing_association_correctness.csv",
    "figures/stdp_early_late.png": "results/timing_correction_final_visualization/timing_correction_stdp_window_events.png",
    "figures/original_learning.png": "results/timing_correction_final_visualization/original_timing_correctness_zoom.png",
    "figures/modified_learning.png": "results/timing_correction_final_visualization/modified_timing_correctness_zoom.png",
    "figures/original_training_vs_recall.png": "results/timing_correction_final_visualization/final_training_vs_recall.png",
}
DERIVED = {
    "datasets/original_K10.npz": "results/checkpoints/original_training.npz: input_binary,target_binary,presentation_orders",
    "datasets/modified_K10.npz": "results/checkpoints/modified_training.npz: input_binary,target_binary,presentation_orders",
    "datasets/original_K10.csv": "datasets/original_K10.npz: human-readable input/target",
    "datasets/modified_K10.csv": "datasets/modified_K10.npz: human-readable input/target",
    "results/checkpoints/final_weights.npz": "results/checkpoints/{original,modified}_training.npz: final_weight",
    "results/csv/summary.csv": "results/timing_correction_cs_duration_lr_sweep/summary.csv: two selected rows",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or "runs" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == "artifact_manifest.csv":
            continue
        source = SOURCE.get(rel, "")
        if source:
            source_path = PROJECT / source
            assert source_path.exists(), source
            assert sha(path) == sha(source_path), rel
            kind = "exact_copy"
        elif rel in DERIVED:
            source = DERIVED[rel]
            kind = "derived_from_archived_numeric_data"
        else:
            kind = "archive_documentation_or_generated_figure"
        rows.append({"archived_path": rel, "sha256": sha(path), "size_bytes": path.stat().st_size,
                     "provenance": kind, "original_source_path": source})
    with (ROOT / "artifact_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified and inventoried {len(rows)} files")


if __name__ == "__main__":
    main()
