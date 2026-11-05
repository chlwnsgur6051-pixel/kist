"""Regenerate baseline accuracy, final-weight, and selected trace figures.

Reads archived numeric artifacts only; does not train or alter the model.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def accuracy():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, color, first, stable in (
        ("Original", "#b34743", 113, 226),
        ("Modified", "#286fa1", 21, 37),
    ):
        with (ROOT / "results/csv" / f"{name.lower()}_epoch_history.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        epochs = [int(r["epoch"]) for r in rows]
        exact = [float(r["exact_accuracy_percent"]) for r in rows]
        assert len(rows) == 1000 and exact[-1] == 100
        ax.plot(epochs, exact, color=color, lw=1.3, label=name)
        ax.scatter([first, stable], [100, 100], color=color, s=[26, 50], zorder=4)
    ax.set(xlabel="Epoch", ylabel="Teacher-free exact recall (%)",
           xlim=(0, 1000), ylim=(0, 105),
           title="Spike-Timing Correction | Learning Accuracy")
    ax.axhline(100, color="0.65", lw=.8, ls="--")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, "accuracy_by_epoch")


def weights():
    with np.load(ROOT / "results/checkpoints/final_weights.npz") as d:
        matrices = [d["original"], d["modified"]]
    limit = max(float(np.max(np.abs(w))) for w in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, name, w in zip(axes, ("Original", "Modified"), matrices):
        image = ax.imshow(w, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set(title=name, xlabel="Output neuron", ylabel="Input neuron",
               xticks=range(0, 15, 2), yticks=range(0, 15, 2))
    fig.colorbar(image, ax=axes, label="Weight", shrink=.8)
    fig.suptitle("Final Signed Synaptic Weights (Full Precision in NPZ)")
    fig.subplots_adjust(top=.83, right=.91, wspace=.22)
    save(fig, "final_weights")


def selected_original_trace():
    with np.load(ROOT / "results/checkpoints/selected_traces.npz") as d:
        x = d["input_binary"][0]
        ticks = d["epoch_0006_event_ticks"]
        voltage = d["epoch_0006_voltage"]
        source = d["epoch_0006_cs_source_current"]
        teacher = d["epoch_0006_teacher_current"]
    assert ticks[0, 0] == 0 and ticks[0, 1] == 45
    time = np.arange(voltage.shape[0]) * .1
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    for i in np.flatnonzero(x):
        axes[0].scatter(1, i, color="0.2", s=14)
    axes[0].set(ylabel="Input neuron", ylim=(-.5, 14.5))
    for j in range(15):
        for t in ticks[j]:
            if t >= 0:
                axes[1].scatter(t * .1, j, color="#b34743" if t == 0 else "#286fa1",
                                marker="D" if t == 0 else "o", s=16)
    axes[1].set(ylabel="Output neuron", ylim=(-.5, 14.5))
    axes[2].step(time, source[:, 0], where="post", color="#b34743", label="CS source, output 0")
    ax2 = axes[2].twinx()
    ax2.step(time, teacher[:, 0], where="post", color="#9b3a87", label="Teacher current")
    axes[2].set_ylabel("CS current")
    ax2.set_ylabel("Teacher current")
    axes[3].plot(time, voltage[:, 0], color="#b34743", label="Output 0, after reset")
    axes[3].axhline(1, color="0.4", ls="--", lw=.8, label="Threshold")
    axes[3].set(xlabel="Time (ms)", ylabel="Membrane V", xlim=(-.1, 10))
    axes[3].legend(frameon=False, fontsize=8)
    fig.suptitle("Original 0→A | Epoch 6 Before Update | Recorded LIF Events")
    fig.tight_layout()
    save(fig, "original_epoch6_spike_membrane")


if __name__ == "__main__":
    accuracy()
    weights()
    selected_original_trace()
