"""Teacher-free recall of the archived v2 weights (epoch 1000 and 5000).

Each cue is presented without a teacher and the output bit is 1 if the neuron
spikes at 0.1..10.0 ms.  The readout starts from a resting LIF state, so this is
a checkpoint-loading check; the official per-epoch accuracy in results/csv uses
the same readout on a copy of the continuous training state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))
import stc_model as M  # noqa: E402


def main():
    ok = True
    for model, spec in (("step", M.STEP_V2), ("exponential", M.EXP_V2)):
        for dataset in ("Original", "Modified"):
            x, y, _ = M.load_reference_dataset(dataset)
            with np.load(V2 / f"results/checkpoints/{dataset.lower()}_{model}_v2_weights.npz") as z:
                for key in ("weight_epoch1000", "weight_epoch5000"):
                    w = np.ascontiguousarray(z[key], dtype=np.float64)
                    r = M.recall(spec, x, y, w, M.State())
                    ok &= r["exact"] == len(x)
                    print(f"{model:11s} {dataset:8s} {key}: exact={r['exact']}/{len(x)} "
                          f"bit={100 * r['bit']:.1f}% max|W|={np.abs(w).max():.3f}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
