"""Teacher-free readout of archived final weights from fresh resting LIF state.

This is a checkpoint-loading diagnostic. The official per-epoch accuracy uses
the sequential copy-state readout in the archived training runner.
"""
from __future__ import annotations

import numpy as np

from reproduce import ROOT, load_runner


def main():
    model = load_runner().model_module()
    model.CAP = None
    model.CS_END_TICK = 45
    model.LAMBDA = 0.5
    for name in ("Original", "Modified"):
        x, y, orders = model.load_data(name)
        with np.load(ROOT / "results/checkpoints" / f"{name.lower()}_training.npz") as d:
            weight = d["final_weight"].copy()
        prediction = np.zeros_like(y)
        for association in range(10):
            voltage = np.zeros(15)
            refractory = np.full(15, -1_000_000_000, dtype=np.int64)
            prediction[association], _, post_ticks = model.lif_trial(
                x[association], weight, np.zeros(15, dtype=np.int64),
                voltage, refractory, 0, False
            )
            assert np.all(post_ticks[:, 0] != 0)
        exact = int(np.all(prediction == y, axis=1).sum())
        bit = float(np.mean(prediction == y))
        print(f"{name}: teacher-free checkpoint readout exact={exact}/10, bit={bit:.1%}")
        assert exact == 10 and bit == 1.0


if __name__ == "__main__":
    main()
