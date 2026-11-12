"""Fast self-checks of the archived baseline_v2 code (about 2-3 minutes).

Run:  python tests/test_baseline_v2.py        (or: pytest tests/test_baseline_v2.py)

1. Archived datasets: shapes, binary values, and the presentation-order type.
2. Model constants: Step 15.0 ms / Exponential 100 ms, lambda, bound, corrected gap.
3. The first 60 epochs of all four conditions equal the archived epoch histories.
4. Archived epoch-1000/5000 weights recall 10/10 and stay inside [-1, 1].
5. Gap spikes and clipping: in a stress condition where both really occur
   (Exponential, tau_s 16 ms, 15.1 ms interval), the C core equals the independent
   pure-Python reference association by association (spikes, dW, W, recall).
   This check reads the archived STDP class from ../baseline_v1 and is skipped
   when that folder is absent.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))
import stc_model as M  # noqa: E402

KEYS = ("exact_count", "bit_accuracy_percent", "false_positive_bits", "false_negative_bits",
        "weight_min", "weight_max", "weight_norm")
CONDITIONS = [(m, d) for m in ("step", "exponential") for d in ("Original", "Modified")]


def spec_of(model):
    return M.STEP_V2 if model == "step" else M.EXP_V2


def test_datasets_and_order():
    orders = {}
    for name in ("Original", "Modified"):
        x, y, o = M.load_reference_dataset(name)
        assert x.shape == y.shape == (10, 15) and set(np.unique(x)) <= {0, 1} and set(np.unique(y)) <= {0, 1}
        assert o.shape == (1000, 10) and all(sorted(r) == list(range(10)) for r in o.tolist())
        orders[name] = o
    assert np.array_equal(orders["Original"], orders["Modified"])
    # The archived order is a different permutation in every epoch (seed 0), not a fixed order.
    rng = np.random.default_rng(0)
    assert np.array_equal(orders["Original"], np.stack([rng.permutation(10) for _ in range(1000)]))
    assert len({tuple(r) for r in orders["Original"].tolist()}) == 1000


def test_constants():
    s, e = M.STEP_V2, M.EXP_V2
    assert (s.interval, s.rate, s.gap_mode, s.bound, s.exponential, s.cs_end) == (150, 0.5, 1, 1.0, False, 45)
    assert (e.interval, e.rate, e.gap_mode, e.bound, e.exponential, e.tau_s, e.truncate, e.factor) == \
           (1000, 0.25, 1, 1.0, True, 8.0, False, 1.0)
    assert M.TRIAL_TICKS == 101 and s.interval - M.TRIAL_TICKS == 49


def test_first_epochs_match_archive(epochs=60):
    for model, dataset in CONDITIONS:
        x, y, orders = M.load_reference_dataset(dataset)
        rows, *_ = M.train(spec_of(model), x, y, orders, epochs)
        saved = list(csv.DictReader(open(V2 / f"results/csv/{dataset.lower()}_{model}_v2_epoch_history.csv")))
        assert len(saved) == 5000
        for r, s in zip(rows, saved):
            assert all(float(r[k]) == float(s[k]) for k in KEYS), (model, dataset, r["epoch"])


def test_archived_weights_recall():
    for model, dataset in CONDITIONS:
        x, y, _ = M.load_reference_dataset(dataset)
        saved = list(csv.DictReader(open(V2 / f"results/csv/{dataset.lower()}_{model}_v2_epoch_history.csv")))
        assert saved[999]["exact_count"] == "10" and saved[4999]["exact_count"] == "10"
        assert sum(int(s["clip_events"]) for s in saved) == 0
        assert sum(int(s["training_gap_spikes"]) for s in saved) == 0
        with np.load(V2 / f"results/checkpoints/{dataset.lower()}_{model}_v2_weights.npz") as z:
            for key in ("weight_epoch1000", "weight_epoch5000"):
                w = np.ascontiguousarray(z[key], dtype=np.float64)
                assert np.abs(w).max() < 1.0
                assert M.recall(spec_of(model), x, y, w, M.State())["exact"] == 10


def test_gap_spikes_and_clipping_against_reference(epochs=30):
    if not (M.BASELINE_V1 / "src/sweep_model.py").exists():   # reference STDP class lives in baseline_v1
        if "pytest" in sys.modules:
            import pytest
            pytest.skip("../baseline_v1 not present")
        return None
    import reference_model as R
    spec = M.variant(M.EXP_V2, "stress: tau_s 16 ms, 15.1 ms interval", interval=151, tau_s=16.0)
    x, y, orders = M.load_reference_dataset("Original")
    rows, w, state, extra = M.train(spec, x, y, orders, epochs, detail_epochs=set(range(1, epochs + 1)))
    gap = sum(r["training_gap_spikes"] for r in rows)
    recall_gap = sum(r["recall_gap_spikes"] for r in rows)
    clips = sum(r["clip_events"] for r in rows)
    assert gap > 0 and recall_gap > 0 and clips > 0, (gap, recall_gap, clips)
    assert np.abs(w).max() <= 1.0 and rows[-1]["synapses_at_bound"] > 0
    log, w_ref, v_ref, refr_ref, tail_ref = R.train(spec, x, y, orders, epochs)
    n = 0
    for ep in range(1, epochs + 1):
        d = extra["details"][ep]
        for t in d["train"]:
            ref = log[n]; n += 1
            assert [list(map(int, e)) for e in t["events"]] == ref["events"], (ep, t["position"])
            assert np.array_equal(t["delta"], ref["delta"]), (ep, t["position"])
        assert np.array_equal(d["train"][-1]["w_after"], log[n - 1]["weight"]), ep
        assert np.array_equal(d["recall"]["pred"], log[n - 1]["recall"]), ep
    assert np.array_equal(w, w_ref) and np.array_equal(state.v, v_ref)
    assert np.array_equal(state.refr, refr_ref) and np.array_equal(state.tail, tail_ref)
    return gap, recall_gap, clips


if __name__ == "__main__":
    test_datasets_and_order(); print("datasets and presentation order: PASS")
    test_constants(); print("model constants: PASS")
    test_first_epochs_match_archive(); print("first 60 epochs equal the archived histories (4 conditions): PASS")
    test_archived_weights_recall(); print("archived epoch-1000/5000 weights recall 10/10, no clipping, no gap spikes: PASS")
    out = test_gap_spikes_and_clipping_against_reference()
    if out is None:
        print("stress condition vs Python reference: SKIPPED (../baseline_v1 not present)")
    else:
        print(f"stress condition equals the Python reference (training gap spikes {out[0]}, "
              f"recall gap spikes {out[1]}, clip events {out[2]}): PASS")
