"""Freeze and verify the three fixed datasets and the calibrated teacher-event baseline.

Run from the heteroassociation directory. This never edits the original sources.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_research_checkpoint")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
HANDOFF = ROOT / "local_stdp_audit_dataset_handoff_20261020"
DIAGNOSTICS = ROOT / "my_research/experiments/diagnostics"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DIAGNOSTICS))
import binary_character
import shared_spike_stdp_training as shared
import shared_spike_timing_calibration as timing

KS = (1, 2, 3, 5, 7, 10)
EXPECTED = {
    "Original": (100, 100, 64.4444444444, 37.7777777778, 25.8333333333, 10),
    "A": (100, 100, 90.5555555556, 54.6825396825, 36.5476190476, 40),
    "B": (100, 100, 96.9444444444, 70.3968253968, 46.4285714286, 20),
}
TIMING = dict(pre_plus_ms=3.0, pre_minus_ms=7.0, post_plus_ms=5.0,
              post_minus_early_ms=1.3, post_minus_late_ms=8.7)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source(tag: str):
    if tag == "Original":
        x = binary_character.make_digit_dataset()[0].reshape(10, 15).astype(np.int8)
        y = binary_character.make_letter_dataset()[0].reshape(10, 15).astype(np.int8)
        source = ROOT / "binary_character.py"
        return x, 2*x-1, y, 2*y-1, source
    source = HANDOFF / f"dataset_{tag}.npz"
    with np.load(source, allow_pickle=False) as d:
        assert set(d.files) == {"X_binary", "X_bipolar", "Y_bipolar"}
        x, xb, yb = (d[k].astype(np.int8) for k in ("X_binary", "X_bipolar", "Y_bipolar"))
    return x, xb, ((yb+1)//2).astype(np.int8), yb, source


def freeze():
    DATASETS.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for tag in EXPECTED:
        x, xb, y, yb, source = read_source(tag)
        assert all(a.shape == (10, 15) for a in (x, xb, y, yb))
        assert np.array_equal(2*x-1, xb) and np.array_equal(2*y-1, yb)
        dest = DATASETS / f"{tag}.npz"
        if dest.exists():
            with np.load(dest, allow_pickle=False) as prior:
                assert set(prior.files) == {"X_binary", "X_bipolar", "Y_binary", "Y_bipolar"}
                assert all(np.array_equal(prior[k], v) for k, v in
                           (("X_binary", x), ("X_bipolar", xb), ("Y_binary", y), ("Y_bipolar", yb)))
        else:
            np.savez_compressed(dest, X_binary=x, X_bipolar=xb, Y_binary=y, Y_bipolar=yb)
        manifest[tag] = dict(file=str(dest.relative_to(ROOT)), sha256=sha(dest),
                             source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                             mapping=[f"{i}->{chr(65+i)}" for i in range(10)],
                             description=("Current rank-10 binary_character.py patterns" if tag == "Original"
                                          else "Fixed supplied dataset; target bitmap is from Y_bipolar in handoff NPZ"))
    manifest_path = DATASETS / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(tag):
    p = DATASETS / f"{tag}.npz"
    with np.load(p, allow_pickle=False) as d:
        x, xb, y, yb = (d[k].copy() for k in ("X_binary", "X_bipolar", "Y_binary", "Y_bipolar"))
    source = ROOT / json.loads((DATASETS / "manifest.json").read_text())[tag]["source"]
    if source.exists():
        sx, sxb, sy, syb, _ = read_source(tag)
        assert all(np.array_equal(a, b) for a, b in zip((x, xb, y, yb), (sx, sxb, sy, syb)))
    return x, xb, y, yb


def recall(x, y, w):
    score = x @ w
    pred = np.where(score > 0, 1, -1)
    signed = y * score
    return score, pred, signed, np.all(pred == y, axis=1), np.all(signed > 0, axis=1)


def similarity(x):
    dots = x @ x.T
    vals = np.abs(dots[np.triu_indices(10, 1)])
    return float(vals.mean())


def run(tag, out, writer):
    x01, x, y01, y = load(tag)
    rank = int(np.linalg.matrix_rank(x))
    assert rank == 10
    corr_rows = []
    for k, expected in zip(KS, EXPECTED[tag]):
        vals = []
        for ids in itertools.combinations(range(10), k):
            xx, yy = x[list(ids)], y[list(ids)]
            score, pred, signed, exact, strict = recall(xx, yy, xx.T @ yy)
            vals.append((exact.mean(), np.mean(pred == yy), signed.min(), strict.mean()))
        arr = np.asarray(vals)
        exact_percent = 100 * arr[:, 0].mean()
        assert abs(exact_percent - expected) < 1e-7, (tag, k, exact_percent, expected)
        row = dict(dataset=tag, method="Original Correlation", K=k, subsets=len(vals),
                   exact_percent=exact_percent, bit_percent=100*arr[:, 1].mean(),
                   minimum_signed_margin=float(arr[:, 2].min()),
                   strictly_positive_margin_percent=100*arr[:, 3].mean(),
                   input_rank=rank, mean_absolute_input_inner_product=similarity(x))
        writer.writerow(row)
        corr_rows.append(row)
    wc = x.T @ y
    corr_score, corr_pred, corr_signed, corr_exact, corr_strict = recall(x, y, wc)
    wp = np.linalg.pinv(x) @ y
    pinv_score, pinv_pred, pinv_signed, pinv_exact, pinv_strict = recall(x, y, wp)
    assert pinv_exact.sum() == 10
    writer.writerow(dict(dataset=tag, method="Pseudoinverse", K=10, subsets=1,
                         exact_percent=100*pinv_exact.mean(), bit_percent=100*np.mean(pinv_pred == y),
                         minimum_signed_margin=float(pinv_signed.min()),
                         strictly_positive_margin_percent=100*pinv_strict.mean(),
                         input_rank=rank, mean_absolute_input_inner_product=similarity(x)))
    b2 = shared.source_module()
    assert (b2.W_INIT, b2.W_MIN, b2.W_MAX, b2.EPOCHS, b2.DT, b2.TAU_STDP) == (0.05, -1.5, 1.5, 60, 0.1, 2.0)
    b2.A_PLUS = b2.A_MINUS = 0.0075
    cases, case_summary = timing.four_cases(b2, TIMING)
    assert case_summary["all_signs_correct"]
    for i in range(10):
        pre = timing.event_vector(b2, x01[i], TIMING)
        assert pre.shape == (15,)
        assert np.array_equal(pre, timing.event_vector(b2, x01[i], TIMING))
        assert set(pre.tolist()) <= {30, 70}
    history, clips, suppressed, reversals = timing.train(
        b2, x, y, x01, tuple(range(10)), TIMING, True, "random_seed0")
    assert reversals.sum() == 0
    if tag in ("A", "B"):
        assert clips.sum() == 0
    w_eff = history[-1]-history[0]
    score, pred, signed, exact, strict = recall(x, y, w_eff)
    sim = shared.similarity(w_eff, wc)
    saturation = int(np.count_nonzero(np.isclose(history[-1], b2.W_MIN) |
                                       np.isclose(history[-1], b2.W_MAX)))
    if tag in ("A", "B"):
        assert saturation == 0
    writer.writerow(dict(dataset=tag, method="Calibrated Shared-Spike Additive STDP", K=10, subsets=1,
                         exact_percent=100*exact.mean(), bit_percent=100*np.mean(pred == y),
                         minimum_signed_margin=float(signed.min()),
                         strictly_positive_margin_percent=100*strict.mean(),
                         input_rank=rank, mean_absolute_input_inner_product=similarity(x)))
    if tag in ("A", "B"):
        assert int(exact.sum()) == (4 if tag == "A" else 2)
    np.savez_compressed(out / f"{tag}_K10_weights.npz", W_correlation=wc,
                        W_pseudoinverse=wp, W_initial=history[0], W_final=history[-1],
                        W_effective=w_eff)
    result = dict(dataset=tag, input_rank=rank, mean_absolute_input_inner_product=similarity(x),
                  correlation_exact=int(corr_exact.sum()),
                  correlation_recalled_ids=np.flatnonzero(corr_exact).tolist(),
                  correlation_strictly_positive_count=int(corr_strict.sum()),
                  correlation_zero_score_bits=int(np.count_nonzero(corr_score == 0)),
                  pseudoinverse_exact=int(pinv_exact.sum()),
                  stdp_exact=int(exact.sum()), stdp_recalled_ids=np.flatnonzero(exact).tolist(),
                  stdp_bit_percent=100*float(np.mean(pred == y)),
                  clipping_events=int(clips.sum()), suppressed_updates=int(suppressed.sum()),
                  saturation_synapses=saturation, cosine=sim["cosine"],
                  normalized_fro_error=sim["normalized_fro_error"],
                  best_fit_scale=sim["scale"], four_case_updates=cases,
                  four_case_relative_spread=case_summary["relative_spread"])
    with (out / f"{tag}_K10_bit_scores.csv").open("w", newline="") as f:
        cols = ("digit", "output_neuron", "target_sign", "correlation_score", "correlation_prediction",
                "correlation_signed_margin", "pseudoinverse_score", "stdp_score", "stdp_prediction",
                "stdp_signed_margin")
        bit_writer = csv.DictWriter(f, cols); bit_writer.writeheader()
        for i, j in itertools.product(range(10), range(15)):
            bit_writer.writerow(dict(digit=i, output_neuron=j, target_sign=int(y[i,j]),
                                     correlation_score=float(corr_score[i,j]),
                                     correlation_prediction=int(corr_pred[i,j]),
                                     correlation_signed_margin=float(corr_signed[i,j]),
                                     pseudoinverse_score=float(pinv_score[i,j]),
                                     stdp_score=float(score[i,j]), stdp_prediction=int(pred[i,j]),
                                     stdp_signed_margin=float(signed[i,j])))
    return result


def bitmap_figure(out):
    fig, axes = plt.subplots(3, 2, figsize=(8, 11))
    for row, tag in enumerate(EXPECTED):
        x, _, y, _ = load(tag)
        axes[row,0].imshow(x.reshape(10, 5, 3).reshape(50, 3), cmap="Greys", vmin=0, vmax=1)
        axes[row,1].imshow(y.reshape(10, 5, 3).reshape(50, 3), cmap="Greys", vmin=0, vmax=1)
        for col in range(2):
            axes[row,col].set(title=f"{tag}: {'digit inputs' if col==0 else 'letter targets'}",
                              xlabel="Pixel column", ylabel="10 patterns × 5 pixel rows")
    fig.tight_layout()
    fig.savefig(out / "reference_bitmaps.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true", help="Create the three immutable reference NPZs")
    p.add_argument("--dataset", choices=("all", *EXPECTED), default="all")
    args = p.parse_args()
    if args.freeze:
        freeze()
    assert (DATASETS / "manifest.json").exists(), "Run --freeze first"
    manifest = json.loads((DATASETS / "manifest.json").read_text())
    tags = list(EXPECTED) if args.dataset == "all" else [args.dataset]
    for tag in tags:
        assert sha(ROOT / manifest[tag]["file"]) == manifest[tag]["sha256"]
        source = ROOT / manifest[tag]["source"]
        if source.exists():
            assert sha(source) == manifest[tag]["source_sha256"]
    out = HERE / "validation" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "baseline_metrics.csv").open("w", newline="") as f:
        fields = ("dataset", "method", "K", "subsets", "exact_percent", "bit_percent",
                  "minimum_signed_margin", "strictly_positive_margin_percent", "input_rank",
                  "mean_absolute_input_inner_product")
        writer = csv.DictWriter(f, fields); writer.writeheader()
        results = [run(tag, out, writer) for tag in tags]
    bitmap_figure(out)
    (out / "verification.json").write_text(json.dumps(dict(timestamp=datetime.now().astimezone().isoformat(),
         datasets=tags, source_kernel_sha256=sha(shared.SOURCE), results=results,
         command="python3 my_research/reference/verify_checkpoint.py --dataset " + args.dataset,
         status="PASS"), indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": str(out), "results":
                      [{k:v for k,v in r.items() if k in ("dataset", "correlation_exact", "pseudoinverse_exact",
                           "stdp_exact", "clipping_events", "cosine")} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
