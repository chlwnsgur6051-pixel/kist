"""Train baseline_v2 from zero weights and report teacher-free recall after every epoch.

Usage (from anywhere):
  python scripts/train.py --model step --dataset Original --epochs 1000
  python scripts/train.py --model exponential --dataset Modified --epochs 1000 --save runs/exp_modified.npz

The presentation order is the archived one in datasets/*_K10.npz (a different
permutation in every epoch, rows reused cyclically after epoch 1000).  Pass
--order sequential to present 0,1,...,K-1 in every epoch instead; that is NOT the
validated baseline condition and its results are not comparable with results/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))
import stc_model as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("step", "exponential"), required=True)
    ap.add_argument("--dataset", choices=("Original", "Modified"), required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--order", choices=("archived", "sequential"), default="archived")
    ap.add_argument("--save", type=Path, help="write final weight and per-epoch exact counts (.npz)")
    a = ap.parse_args()
    spec = M.STEP_V2 if a.model == "step" else M.EXP_V2
    x, y, orders = M.load_reference_dataset(a.dataset)
    if a.order == "sequential":
        orders = np.arange(len(x), dtype=np.int64)[None, :]
    rows, w, state, extra = M.train(spec, x, y, orders, a.epochs)
    first, sustained = M.first_and_sustained(rows, K=len(x))
    last = rows[-1]
    print(f"{spec.name} | {a.dataset} | order={a.order} | epochs={a.epochs}")
    print(f"  first {len(x)}/{len(x)} epoch: {first}   sustained from epoch: {sustained}")
    print(f"  final exact recall: {last['exact_count']}/{len(x)}   bit accuracy: {last['bit_accuracy_percent']:.2f}%")
    print(f"  weight range: [{last['weight_min']:.4f}, {last['weight_max']:.4f}]   "
          f"clip events: {sum(r['clip_events'] for r in rows)}   "
          f"gap spikes: {sum(r['training_gap_spikes'] for r in rows)}")
    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.save, weight=w, exact_count=np.array([r["exact_count"] for r in rows]),
                            input_binary=x, target_binary=y)
        print(f"  saved {a.save}")


if __name__ == "__main__":
    main()
