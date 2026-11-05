"""Short mechanism and artifact checks; full 1000-epoch parity is reproduce.py."""
from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reproduce import load_runner  # noqa: E402


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_runner().model_module()
        cls.model.CAP = None
        cls.model.CS_END_TICK = 45
        cls.model.LAMBDA = 0.5

    def test_window_and_all_pairs(self):
        m = self.model
        self.assertAlmostEqual(m.window(-1), 0.05, places=7)
        self.assertAlmostEqual(m.window(3), -0.05, places=7)
        pre = np.zeros(15, dtype=np.int64)
        pre[0] = 1
        times = np.full((15, 12), -1, dtype=np.int64)
        times[0, :3] = (0, 40, 70)
        counts = np.zeros(15, dtype=np.int64)
        counts[0] = 3
        accumulator = m.ContinuousAllPairs()
        delta = accumulator.process(pre, counts, times, 0, 1)
        expected = .5 * sum(m.window(d) for d in (-1, 3, 6))
        self.assertAlmostEqual(delta[0, 0], expected, places=12)
        self.assertEqual(accumulator.total_count, 3)
        self.assertEqual(np.count_nonzero(delta[1:]), 0)
        # No spikes from an inactive presynaptic neuron, even with a teacher post.
        empty = m.ContinuousAllPairs().process(np.zeros(15, dtype=np.int64),
                                                counts, times, 0, 1)
        self.assertEqual(np.count_nonzero(empty), 0)

    def test_real_teacher_and_teacher_free_recall(self):
        m = self.model
        with np.load(ROOT / "datasets/original_K10.npz") as d:
            x, y = d["input_binary"][0], d["target_binary"][0]
        w = np.zeros((15, 15))
        v = np.zeros(15)
        r = np.full(15, -1_000_000_000, dtype=np.int64)
        prediction, counts, posts = m.lif_trial(x, w, y, v, r, 0, True)
        self.assertFalse(np.any(prediction))
        for j in range(15):
            if y[j]:
                self.assertEqual(posts[j, 0], 0)
                self.assertEqual(counts[j], 1)
            else:
                self.assertEqual(counts[j], 0)
        v = np.zeros(15)
        r = np.full(15, -1_000_000_000, dtype=np.int64)
        no_teacher, counts, posts = m.lif_trial(x, w, y, v, r, 0, False)
        self.assertFalse(np.any(no_teacher))
        self.assertFalse(np.any(counts))
        self.assertFalse(np.any(posts >= 0))

    def test_archived_datasets_and_checkpoints(self):
        m = self.model
        for name in ("Original", "Modified"):
            x, y, order = m.load_data(name)
            with np.load(ROOT / "datasets" / f"{name.lower()}_K10.npz") as d:
                self.assertTrue(np.array_equal(x, d["input_binary"]))
                self.assertTrue(np.array_equal(y, d["target_binary"]))
                self.assertTrue(np.array_equal(order, d["presentation_orders"]))
            with np.load(ROOT / "results/checkpoints" / f"{name.lower()}_training.npz") as d:
                self.assertEqual(d["final_weight"].shape, (15, 15))
                self.assertLess(float(d["final_weight"].min()), 0)
                self.assertEqual(m.recall_independent(x, y, d["final_weight"]), 10)
            with (ROOT / "results/csv" / f"{name.lower()}_epoch_history.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1000)
            self.assertEqual(int(rows[-1]["exact_count"]), 10)


if __name__ == "__main__":
    unittest.main()
