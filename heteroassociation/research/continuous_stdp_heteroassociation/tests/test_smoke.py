"""Fast integrity check for the bundled runnable model; full replay is separate."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/continuous_stdp_experiment.py"


class SmokeTest(unittest.TestCase):
    def test_teacher_natural_spikes_and_pair_window(self):
        if importlib.util.find_spec("numba") is None:
            shim = types.ModuleType("numba")
            shim.njit = lambda function: function
            sys.modules["numba"] = shim
        spec = importlib.util.spec_from_file_location("continuous_model_smoke", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x, y, orders = module.load_data("Original")
        self.assertEqual(orders.shape, (1000, 10))
        weight = np.full((15, 15), module.CAP / x[0].sum())
        voltage = np.zeros(15)
        refractory = np.full(15, -10**9, dtype=np.int64)
        prediction, count, posts = module.lif_trial(x[0], weight, y[0],
                                                      voltage, refractory, 0, True)
        for j in range(15):
            self.assertEqual(posts[j, 0], 0 if y[0, j] else 40)
            self.assertEqual(count[j], 2 if y[0, j] else 1)
        self.assertTrue(prediction.all())
        self.assertAlmostEqual(module.LAMBDA * module.window(-1), .05, places=8)
        self.assertAlmostEqual(module.LAMBDA * module.window(3), -.05, places=8)


if __name__ == "__main__":
    unittest.main()
