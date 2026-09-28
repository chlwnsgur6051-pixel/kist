"""
Contrastive (closed-form) theoretical optimal weights for digit→letter association.

Teacher: target=1 → +I, target=0 → −I. Weights via least-squares (pinv).
Recall: simple LIF with fixed threshold/tau.

Humans: edit PARAMETER_SET. Agents: use argparse.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import binary_character as data

# ---------------------------------------------------------------------------
# PARAMETER SET — human-editable
# ---------------------------------------------------------------------------
PARAMETER_SET = {
    "threshold": 0.8,
    "tau": 20e-3,
    "dt": 1e-3,
    "basis": "quadratic",  # "linear" | "quadratic"
    "target_one_current": 1.0,
    "target_zero_current": -1.0,
    "weight_clip": 20.0,
    "bias_clip": 20.0,
    "recall_steps": 1,
    "count_threshold": 1,
    "device": "cpu",
    "output_dir": "results/contrastive_optimal_weight",
    "run_name": "contrastive_quadratic",
}


def parse_args():
    p = argparse.ArgumentParser(description="Contrastive optimal weight (pinv teacher).")
    p.add_argument("--basis", choices=["linear", "quadratic"], default=PARAMETER_SET["basis"])
    p.add_argument("--target-one-current", type=float, default=PARAMETER_SET["target_one_current"])
    p.add_argument("--target-zero-current", type=float, default=PARAMETER_SET["target_zero_current"])
    p.add_argument("--weight-clip", type=float, default=PARAMETER_SET["weight_clip"])
    p.add_argument("--bias-clip", type=float, default=PARAMETER_SET["bias_clip"])
    p.add_argument("--recall-steps", type=int, default=PARAMETER_SET["recall_steps"])
    p.add_argument("--count-threshold", type=int, default=PARAMETER_SET["count_threshold"])
    p.add_argument("--device", default=PARAMETER_SET["device"])
    p.add_argument("--output-dir", default=PARAMETER_SET["output_dir"])
    p.add_argument("--run-name", default=PARAMETER_SET["run_name"])
    return p.parse_args()


def load_patterns(device):
    digit_images, _ = data.make_digit_dataset()
    letter_images, _ = data.make_letter_dataset()
    cs = torch.tensor(digit_images.reshape(10, -1), dtype=torch.float32, device=device)
    target = torch.tensor(letter_images.reshape(10, -1), dtype=torch.float32, device=device)
    return cs, target


def expand_features(cs, basis):
    if basis == "linear":
        return cs
    features = [cs]
    pair_features = []
    n_features = cs.shape[1]
    for i in range(n_features):
        for j in range(i, n_features):
            pair_features.append((cs[:, i] * cs[:, j]).unsqueeze(1))
    features.append(torch.cat(pair_features, dim=1))
    return torch.cat(features, dim=1)


def train_contrastive_association(cs, target, basis, target_one_current, target_zero_current):
    phi = expand_features(cs, basis)
    bias_column = torch.ones(phi.shape[0], 1, dtype=phi.dtype, device=phi.device)
    phi_aug = torch.cat([phi, bias_column], dim=1)
    teacher_current = target * target_one_current + (1.0 - target) * target_zero_current
    weight_bias = torch.linalg.pinv(phi_aug) @ teacher_current
    return phi, weight_bias[:-1], weight_bias[-1]


def lif_recall(phi, weight, bias, threshold, tau, dt, recall_steps, count_threshold):
    beta = math.exp(-dt / tau)
    mem = torch.zeros(phi.shape[0], weight.shape[1], dtype=phi.dtype, device=phi.device)
    counts = torch.zeros_like(mem)
    for _ in range(recall_steps):
        mem = beta * mem + phi @ weight + bias
        spikes = (mem >= threshold).to(phi.dtype)
        counts += spikes
        mem = mem * (1.0 - spikes)
    return (counts >= count_threshold).to(phi.dtype)


def compute_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    target_np = target.detach().cpu().numpy().astype(np.float32)
    pair_rows = []
    bit_accs, mses, cosines = [], [], []
    for idx in range(pred_np.shape[0]):
        p, t = pred_np[idx], target_np[idx]
        bit_acc = float((p == t).mean())
        mse = float(np.mean((p - t) ** 2))
        denom = float(np.linalg.norm(p) * np.linalg.norm(t) + 1e-8)
        cosine = float(np.dot(p, t) / denom)
        exact = bool(np.array_equal(p, t))
        bit_accs.append(bit_acc)
        mses.append(mse)
        cosines.append(cosine)
        pair_rows.append(
            {
                "pair": f"{idx}->{chr(ord('A') + idx)}",
                "exact": exact,
                "bit_acc": bit_acc,
                "mse": mse,
                "cosine": cosine,
                "pred_ones": int(p.sum()),
                "target_ones": int(t.sum()),
                "pred": p.reshape(5, 3).astype(int).tolist(),
                "target": t.reshape(5, 3).astype(int).tolist(),
            }
        )
    return {
        "exact_count": int(sum(row["exact"] for row in pair_rows)),
        "mean_bit_acc": float(np.mean(bit_accs)),
        "mean_mse": float(np.mean(mses)),
        "mean_cosine": float(np.mean(cosines)),
        "pair_rows": pair_rows,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    cs, target = load_patterns(device)

    phi, weight, bias = train_contrastive_association(
        cs, target, args.basis, args.target_one_current, args.target_zero_current
    )
    weight = torch.clamp(weight, -args.weight_clip, args.weight_clip)
    bias = torch.clamp(bias, -args.bias_clip, args.bias_clip)

    pred = lif_recall(
        phi,
        weight,
        bias,
        PARAMETER_SET["threshold"],
        PARAMETER_SET["tau"],
        PARAMETER_SET["dt"],
        args.recall_steps,
        args.count_threshold,
    )
    metrics = compute_metrics(pred, target)
    rank = int(
        torch.linalg.matrix_rank(
            torch.cat([phi, torch.ones(phi.shape[0], 1, device=device)], dim=1)
        ).item()
    )

    result = {
        "run_name": args.run_name,
        "fixed_neuron": {
            "threshold": PARAMETER_SET["threshold"],
            "tau": PARAMETER_SET["tau"],
            "dt": PARAMETER_SET["dt"],
        },
        "config": {
            "basis": args.basis,
            "target_one_current": args.target_one_current,
            "target_zero_current": args.target_zero_current,
            "weight_clip": args.weight_clip,
            "bias_clip": args.bias_clip,
            "feature_dim": int(phi.shape[1]),
            "augmented_feature_rank": rank,
        },
        "weight_stats": {
            "min": float(weight.min().item()),
            "max": float(weight.max().item()),
            "mean": float(weight.mean().item()),
            "bias_min": float(bias.min().item()),
            "bias_max": float(bias.max().item()),
            "bias_mean": float(bias.mean().item()),
        },
        **metrics,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_name}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print("=== Contrastive optimal weight ===")
    print(
        f"basis={args.basis}, feature_dim={phi.shape[1]}, rank={rank}, "
        f"exact={metrics['exact_count']}/10, bit_acc={metrics['mean_bit_acc']:.3f}"
    )
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
