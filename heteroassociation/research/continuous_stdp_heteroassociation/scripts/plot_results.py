"""Render archived accuracy and final weight figures without new training."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "results/csv/epoch_history.csv").open() as f:
        history = list(csv.DictReader(f))
    with np.load(ROOT / "results/checkpoints/final_weights.npz") as z:
        weights = {name: z[name].copy() for name in z.files}
    colors = {"Baseline": "#2563a6", "W >= 0": "#d26a23"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, dataset in zip(axes, ("Original", "Modified")):
        for condition in ("Baseline", "W >= 0"):
            rows = [r for r in history if r["dataset"] == dataset and r["model"] == condition]
            ax.plot([int(r["epoch"]) for r in rows],
                    [10 * int(r["exact_count_continuous_recall"]) for r in rows],
                    color=colors[condition], lw=1.2, label=condition)
        ax.set(title=dataset, xlabel="Epoch", xlim=(0, 1000), ylim=(0, 102))
        ax.grid(alpha=.2)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Exact Recall Accuracy (%)")
    fig.suptitle("Continuous-State Natural LIF Recall (1000 Epochs)")
    fig.tight_layout()
    save(fig, "accuracy_by_epoch")

    scale = max(float(np.max(np.abs(w))) for w in weights.values())
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, dataset in enumerate(("Original", "Modified")):
        for col, suffix in enumerate(("baseline", "floor")):
            ax = axes[row, col]
            im = ax.imshow(weights[f"{dataset}_{suffix}"], origin="lower",
                           cmap="RdBu_r", norm=norm, interpolation="nearest")
            ax.set(title=f"{dataset} — {'Baseline' if col == 0 else 'W ≥ 0'}",
                   xlabel="Output neuron", ylabel="Input neuron",
                   xticks=range(0, 15, 2), yticks=range(0, 15, 2))
    fig.suptitle("Final 15×15 Synaptic Weights")
    fig.colorbar(im, ax=axes, label="Synaptic weight", shrink=.75)
    save(fig, "final_weights")


if __name__ == "__main__":
    main()
