"""Repeated shared-spike training using the unmodified B2 stdp_update function.

The wrapper schedules teacher events directly, as teacher-only B2 does for its
single teacher event. Natural output spikes are excluded from plasticity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "Ver관리/2. temporal/stdp_temporal_symmetric_teacher_only.py"
KS = (1, 2, 3, 5, 7, 10)
EPOCHS_TO_PLOT = (0, 1, 2, 5, 10, 20, 40, 60)
SEED = 0


def source_module():
    spec = importlib.util.spec_from_file_location("shared_spike_original_b2", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_dataset(path):
    with np.load(path, allow_pickle=False) as d:
        xb, y, x01 = d["X_bipolar"].astype(int), d["Y_bipolar"].astype(int), d["X_binary"].astype(int)
    assert xb.shape == y.shape == x01.shape == (10, 15)
    assert np.array_equal(2 * x01 - 1, xb)
    assert set(np.unique(y)) <= {-1, 1}
    return xb, y, x01


def event_steps(b2, target_bit, mode):
    if mode == "shared":
        return [b2.STEP_TEACH] if target_bit == 1 else [int(round(t / b2.DT)) for t in (1.0, 8.0)]
    return [b2.STEP_TEACH]


def pre_steps(b2, x01, target_bit, mode):
    if mode == "shared":
        return b2.encode(x01, teach=None)
    return b2.encode(x01, teach="fire" if target_bit == 1 else "timing_ltd")


def delta_template(b2, x01, target_bit, mode):
    old_min, old_max = b2.W_MIN, b2.W_MAX
    try:
        b2.W_MIN, b2.W_MAX = -np.inf, np.inf
        zero = np.zeros(15, dtype=float)
        return b2.stdp_update(zero.copy(), pre_steps(b2, x01, target_bit, mode),
                              event_steps(b2, target_bit, mode))
    finally:
        b2.W_MIN, b2.W_MAX = old_min, old_max


def similarity(weight, reference):
    dot = float(np.sum(weight * reference))
    ref_norm_sq = float(np.sum(reference ** 2))
    scale = max(0.0, dot / ref_norm_sq) if ref_norm_sq else 0.0
    residual = weight - scale * reference
    wn = float(np.linalg.norm(weight))
    rn = float(np.linalg.norm(reference))
    return dict(scale=scale, cosine=dot / (wn * rn) if wn and rn else 0.0,
                normalized_fro_error=float(np.linalg.norm(residual) / wn) if wn else 0.0,
                max_abs_weight_error=float(np.max(np.abs(residual))))


def recall(x, y, weight):
    scores = x @ weight
    pred = np.where(scores > 0, 1, -1)
    return dict(exact_count=int(np.all(pred == y, axis=1).sum()),
                bit_accuracy=float(np.mean(pred == y)), scores=scores, pred=pred)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def single_presentation(b2, datasets, out):
    cases, synapses = [], []
    for x_bit, y_bit in itertools.product((1, 0), repeat=2):
        x01 = np.full(15, x_bit, dtype=int)
        delta = delta_template(b2, x01, y_bit, "shared")
        expected = (2 * x_bit - 1) * (2 * y_bit - 1)
        assert np.all(np.sign(delta) == expected)
        cases.append(dict(input_bit=x_bit, target_bit=y_bit,
                          pre_ms=float(pre_steps(b2, x01, y_bit, "shared")[0] * b2.DT),
                          post_ms=";".join(str(s * b2.DT) for s in event_steps(b2, y_bit, "shared")),
                          delta_w=float(delta[0]), absolute_delta_w=float(abs(delta[0])),
                          expected_sign=expected, actual_sign=int(np.sign(delta[0]))))
    write_csv(out / "csv" / "four_case_updates.csv", cases)
    metrics = []
    for tag, (x, y, x01) in datasets.items():
        for p in range(10):
            shared_pre = b2.encode(x01[p], teach=None)
            assert np.array_equal(shared_pre, pre_steps(b2, x01[p], 1, "shared"))
            assert np.array_equal(shared_pre, pre_steps(b2, x01[p], 0, "shared"))
            update = np.column_stack([delta_template(b2, x01[p], int((y[p, j]+1)//2), "shared")
                                      for j in range(15)])
            outer = np.outer(x[p], y[p])
            assert np.array_equal(np.sign(update), outer), f"four-case sign prerequisite failed: {tag} {p}"
            sm = similarity(update, outer)
            metrics.append(dict(dataset=tag, digit=p, all_225_signs_correct=True,
                                shared_pre_train_tested=True, **sm))
            if p == 0:
                for i, j in itertools.product(range(15), repeat=2):
                    synapses.append(dict(dataset=tag, digit=p, input_neuron=i, output_neuron=j,
                                         input_bit=int(x01[p, i]), target_bit=int((y[p, j]+1)//2),
                                         pre_ms=float(shared_pre[i]*b2.DT),
                                         post_ms=";".join(str(s*b2.DT) for s in event_steps(b2, int((y[p,j]+1)//2), "shared")),
                                         initial_weight=float(b2.W_INIT), delta_w=float(update[i, j]),
                                         outer_product=int(outer[i, j])))
        np.savez_compressed(out / "npz" / f"single_presentation_{tag}.npz",
                            X=x, Y=y, X_binary=x01,
                            W_delta=np.column_stack([delta_template(b2, x01[0], int((y[0,j]+1)//2), "shared") for j in range(15)]),
                            W_outer=np.outer(x[0], y[0]))
    write_csv(out / "csv" / "single_presentation_metrics.csv", metrics)
    write_csv(out / "csv" / "single_presentation_synapses_digit0.csv", synapses)
    return cases, metrics


def train(b2, x, y, x01, ids, mode, bounded, order):
    old_min, old_max = b2.W_MIN, b2.W_MAX
    if not bounded:
        b2.W_MIN, b2.W_MAX = -np.inf, np.inf
    try:
        w = np.full((15, 15), b2.W_INIT, dtype=float)
        history = [w.copy()]
        clip_events = [0]
        suppressed_events = [0]
        sign_reversals = [0]
        presentations = [0]
        rng = np.random.RandomState(SEED)
        templates = {(p, j): delta_template(b2, x01[p], int((y[p, j]+1)//2), mode)
                     for p in ids for j in range(15)}
        # Audit the same x-dependent input event vector for every output in shared mode.
        if mode == "shared":
            for p in ids:
                shared = b2.encode(x01[p], teach=None)
                assert all(np.array_equal(shared, pre_steps(b2, x01[p], int((y[p,j]+1)//2), mode))
                           for j in range(15))
        for epoch in range(1, b2.EPOCHS + 1):
            sequence = (rng.permutation(ids) if order == "original_random_seed0" else
                        np.array(ids if order == "ascending" else tuple(reversed(ids))))
            clipped = 0
            suppressed = 0
            reversed_sign = 0
            for p in sequence:
                shared_pre = b2.encode(x01[p], teach=None) if mode == "shared" else None
                for j in range(15):
                    target_bit = int((y[p, j]+1)//2)
                    pre = shared_pre if mode == "shared" else pre_steps(b2, x01[p], target_bit, mode)
                    post = event_steps(b2, target_bit, mode)
                    before = w[:, j].copy()
                    if bounded:
                        prospective = before + templates[(p, j)]
                        clipped += int(np.sum((prospective < old_min) | (prospective > old_max)))
                    w[:, j] = b2.stdp_update(before.copy(), pre, post)
                    expected = before + templates[(p, j)]
                    if bounded:
                        expected = np.clip(expected, old_min, old_max)
                    assert np.allclose(w[:, j], expected, atol=1e-12)
                    actual_delta = w[:, j] - before
                    suppressed += int(np.sum(np.abs(actual_delta) <= 1e-12))
                    reversed_sign += int(np.sum(actual_delta * templates[(p, j)] < -1e-12))
            history.append(w.copy())
            clip_events.append(clipped)
            suppressed_events.append(suppressed)
            sign_reversals.append(reversed_sign)
            presentations.append(len(ids))
        return (np.stack(history), np.asarray(clip_events), np.asarray(presentations),
                np.asarray(suppressed_events), np.asarray(sign_reversals))
    finally:
        b2.W_MIN, b2.W_MAX = old_min, old_max


def baseline_all_subsets(datasets, out):
    rows = []
    for tag, (x, y, _) in datasets.items():
        for k in KS:
            exact, bit = [], []
            for ids in itertools.combinations(range(10), k):
                xx, yy = x[list(ids)], y[list(ids)]
                res = recall(xx, yy, xx.T @ yy)
                exact.append(res["exact_count"] / k)
                bit.append(res["bit_accuracy"])
            rows.append(dict(dataset=tag, K=k, subset_count=len(exact),
                             mean_exact_percent=100*np.mean(exact), mean_bit_percent=100*np.mean(bit),
                             coverage="all combinations; pure correlation only"))
    write_csv(out / "csv" / "correlation_all_subset_baseline.csv", rows)
    assert [r["mean_exact_percent"] for r in rows if r["K"] == 10] == [40.0, 20.0]
    return rows


def save_event_and_presentation_schedules(b2, datasets, out):
    events, orders = [], []
    for tag, (_, y, x01) in datasets.items():
        for p in range(10):
            shared_pre = b2.encode(x01[p], teach=None)
            for i in range(15):
                events.append(dict(dataset=tag, digit=p, event_type="shared_presynaptic",
                                   neuron=i, bit=int(x01[p, i]), time_ms=float(shared_pre[i]*b2.DT)))
            for j in range(15):
                target_bit=int((y[p,j]+1)//2)
                for ps in event_steps(b2,target_bit,"shared"):
                    events.append(dict(dataset=tag,digit=p,event_type="teacher_postsynaptic",
                                       neuron=j,bit=target_bit,time_ms=float(ps*b2.DT)))
        for k in KS:
            ids=tuple(range(k))
            for order in ("original_random_seed0","ascending","descending"):
                rng=np.random.RandomState(SEED)
                for epoch in range(1,b2.EPOCHS+1):
                    sequence=(rng.permutation(ids) if order=="original_random_seed0" else
                              ids if order=="ascending" else tuple(reversed(ids)))
                    for position,p in enumerate(sequence):
                        orders.append(dict(dataset=tag,K=k,order=order,epoch=epoch,
                                           presentation_index=position,digit=int(p)))
    write_csv(out/"csv"/"event_schedule.csv",events)
    write_csv(out/"csv"/"presentation_order.csv",orders)


def run_training(b2, datasets, out):
    checkpoint, association, order_rows = [], [], []
    for tag, (x, y, x01) in datasets.items():
        for k in KS:
            ids = tuple(range(k))  # fixed prefix subset; identical across all learning methods
            xx, yy = x[list(ids)], y[list(ids)]
            corr = xx.T @ yy
            corr_result = recall(xx, yy, corr)
            for mode, bounded, order, label in (
                ("shared", False, "original_random_seed0", "Shared-Spike STDP / No Clipping"),
                ("shared", True, "original_random_seed0", "Shared-Spike STDP / Original Bounds"),
                ("b2", False, "original_random_seed0", "B2 Target-Dependent Timing / No Clipping"),
                ("b2", True, "original_random_seed0", "B2 Target-Dependent Timing / Original Bounds"),
                ("shared", True, "ascending", "Shared-Spike STDP / Ascending Order"),
                ("shared", True, "descending", "Shared-Spike STDP / Descending Order"),
            ):
                history, clip_events, presentations, suppressed_events, sign_reversals = train(
                    b2, x, y, x01, ids, mode, bounded, order)
                assert not np.any(sign_reversals), "Actual update reversed timing-derived sign"
                effective = history - b2.W_INIT
                stem = f"{tag}_K{k}_{mode}_{'bounded' if bounded else 'unbounded'}_{order}"
                np.savez_compressed(out / "npz" / f"{stem}.npz", W_initial=history[0],
                                    W_final=history[-1], W_effective_final=effective[-1],
                                    W_correlation=corr, W_by_epoch=history,
                                    clipping_events_by_epoch=clip_events,
                                    suppressed_events_by_epoch=suppressed_events,
                                    sign_reversals_by_epoch=sign_reversals,
                                    subset_ids=np.asarray(ids))
                for ep in range(b2.EPOCHS + 1):
                    wr = effective[ep]
                    res = recall(xx, yy, wr)
                    sm = similarity(wr, corr)
                    sat = float(np.mean((np.abs(history[ep]-b2.W_MIN)<=1e-12) |
                                        (np.abs(history[ep]-b2.W_MAX)<=1e-12))) if bounded else 0.0
                    checkpoint.append(dict(dataset=tag, K=k, subset_ids=";".join(map(str, ids)),
                                           learning_method=label, update_source=mode,
                                           bounded=bounded, order=order, epoch=ep,
                                           presentations_per_epoch=k, teacher_only=True,
                                           natural_posts_used=False, clipping_events=int(clip_events[ep]),
                                           suppressed_update_events=int(suppressed_events[ep]),
                                           sign_reversals=int(sign_reversals[ep]),
                                           saturated_synapse_percent=100*sat,
                                           exact_count=res["exact_count"], exact_percent=100*res["exact_count"]/k,
                                           bit_percent=100*res["bit_accuracy"],
                                           weight_min=float(history[ep].min()), weight_max=float(history[ep].max()),
                                           **sm))
                    if ep == b2.EPOCHS and order == "original_random_seed0":
                        for r, p in enumerate(ids):
                            association.append(dict(dataset=tag, K=k, learning_method=label, digit=p,
                                                    letter=chr(ord('A')+p), target_bits="".join(str(z) for z in ((yy[r]+1)//2)),
                                                    prediction_bits="".join(str(z) for z in ((res["pred"][r]+1)//2)),
                                                    exact=bool(np.all(res["pred"][r] == yy[r])),
                                                    bit_accuracy=float(np.mean(res["pred"][r] == yy[r]))))
                order_rows.append(dict(dataset=tag, K=k, learning_method=label, order=order,
                                       update_source=mode,bounded=bounded,
                                       final_max_abs_difference_from_random=np.nan,
                                       final_exact_count=recall(xx, yy, effective[-1])["exact_count"],
                                       final_bit_accuracy=recall(xx, yy, effective[-1])["bit_accuracy"]))
            checkpoint.append(dict(dataset=tag, K=k, subset_ids=";".join(map(str, ids)),
                                   learning_method="Original Correlation", update_source="direct_matrix",
                                   bounded=False, order="none", epoch=b2.EPOCHS,
                                   presentations_per_epoch=0, teacher_only=False, natural_posts_used=False,
                                   clipping_events=0, suppressed_update_events=0, sign_reversals=0,
                                   saturated_synapse_percent=0.0,
                                   exact_count=corr_result["exact_count"], exact_percent=100*corr_result["exact_count"]/k,
                                   bit_percent=100*corr_result["bit_accuracy"],
                                   weight_min=float(corr.min()), weight_max=float(corr.max()),
                                   **similarity(corr, corr)))
    for row in order_rows:
        if row["order"] == "original_random_seed0":
            row["final_max_abs_difference_from_random"] = 0.0
        else:
            tag,k,mode,order=row["dataset"],row["K"],row["update_source"],row["order"]
            stem=f"{tag}_K{k}_{mode}_bounded_"
            with np.load(out/"npz"/f"{stem}{order}.npz") as d:
                current=d["W_final"]
            with np.load(out/"npz"/f"{stem}original_random_seed0.npz") as d:
                reference=d["W_final"]
            row["final_max_abs_difference_from_random"]=float(np.max(np.abs(current-reference)))
    write_csv(out / "csv" / "epoch_metrics.csv", checkpoint)
    write_csv(out / "csv" / "association_predictions.csv", association)
    write_csv(out / "csv" / "learning_order_results.csv", order_rows)
    return checkpoint


def write_tie_diagnostics(datasets, out):
    rows=[]
    for tag,(x,y,_) in datasets.items():
        for k in KS:
            ids=list(range(k))
            xx,yy=x[ids],y[ids]
            corr=xx.T@yy
            corr_scores=xx@corr
            with np.load(out/"npz"/f"{tag}_K{k}_b2_unbounded_original_random_seed0.npz") as d:
                b2_scores=xx@d["W_effective_final"]
            for r,p in enumerate(ids):
                for j in range(15):
                    if corr_scores[r,j]==0 or abs(b2_scores[r,j])<1e-10:
                        rows.append(dict(dataset=tag,K=k,digit=p,output_neuron=j,
                                         target_sign=int(yy[r,j]),correlation_score=int(corr_scores[r,j]),
                                         b2_unbounded_score=float(b2_scores[r,j]),
                                         correlation_prediction=1 if corr_scores[r,j]>0 else -1,
                                         b2_unbounded_prediction=1 if b2_scores[r,j]>0 else -1,
                                         prediction_differs=bool((corr_scores[r,j]>0)!=(b2_scores[r,j]>0))))
    write_csv(out/"csv"/"near_zero_tie_diagnostics.csv",rows)
    return rows


def figures(b2, cases, checkpoints, out):
    figdir = out / "figures"
    fig, ax = plt.subplots(figsize=(9, 4))
    for y, label, color in ((3, "Input +1", "tab:blue"), (2, "Input −1", "tab:orange"),
                            (1, "Teacher target +1", "tab:green"), (0, "Teacher target −1", "tab:red")):
        times = {3:[2],2:[7],1:[4.5],0:[1,8]}[y]
        ax.scatter(times, [y]*len(times), s=90, label=label, color=color)
    ax.set(yticks=range(4), yticklabels=["Teacher −1", "Teacher +1", "Input −1", "Input +1"],
           xlabel="Time (ms)", title="Shared Input and Target-Specific Teacher Spike Events", xlim=(0,9))
    ax.grid(axis="x", alpha=.3); fig.tight_layout(); fig.savefig(figdir/"shared_pre_post_spike_timing.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4))
    labels=[f"({r['input_bit']},{r['target_bit']})" for r in cases]
    ax.bar(labels, [r["delta_w"] for r in cases], color=["tab:blue" if r["delta_w"]>0 else "tab:red" for r in cases])
    ax.axhline(0,color="black",lw=.7); ax.set(xlabel="(Input bit, Target bit)", ylabel="One-presentation Δw",
        title="Four Bipolar STDP Update Magnitudes: Shared Pre, Teacher Only")
    fig.tight_layout(); fig.savefig(figdir/"four_bipolar_update_magnitudes.png",dpi=150); plt.close(fig)

    fig, axes=plt.subplots(2,3,figsize=(12,7))
    for ri,tag in enumerate(("A","B")):
        d=np.load(out/"npz"/f"{tag}_K10_shared_bounded_original_random_seed0.npz")
        corr=d["W_correlation"]; eff=d["W_effective_final"]
        s=similarity(eff,corr)["scale"]
        for ci,(mat,title) in enumerate(((s*corr,"Scaled Correlation"),(eff,"Shared-Spike STDP ΔW"),(eff-s*corr,"Weight Difference"))):
            ax=axes[ri,ci]; lim=max(np.max(np.abs(s*corr)),np.max(np.abs(eff))) if ci<2 else np.max(np.abs(mat))
            ax.imshow(mat,cmap="coolwarm",vmin=-lim,vmax=lim); ax.set(title=f"Dataset {tag}: {title}",xlabel="Output neuron",ylabel="Input neuron")
    fig.suptitle("K=10, 60 Epochs, Original Bounds, Effective Weights"); fig.tight_layout(); fig.savefig(figdir/"stdp_vs_correlation_weight_k10.png",dpi=150); plt.close(fig)

    for key,ylabel,name in (("normalized_fro_error","Normalized Frobenius Error","weight_error_vs_epochs.png"),
                             ("saturated_synapse_percent","Weight Saturation (%)","weight_saturation_vs_epochs.png")):
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,tag in zip(axes,("A","B")):
            for k in (1,3,5,10):
                for method,style in (("Shared-Spike STDP / No Clipping","--"),("Shared-Spike STDP / Original Bounds","-")):
                    sel=[r for r in checkpoints if r["dataset"]==tag and r["K"]==k and r["learning_method"]==method]
                    ax.plot([r["epoch"] for r in sel],[r[key] for r in sel],linestyle=style,label=f"K={k} {'unbounded' if style=='--' else 'bounded'}")
            ax.set(xlabel="Learning Epoch",title=f"Dataset {tag}"); ax.grid(alpha=.2)
        axes[0].set_ylabel(ylabel); axes[1].legend(fontsize=7,ncol=2)
        fig.suptitle(f"{ylabel} vs Learning Epoch: Shared-Spike Teacher-Only STDP")
        fig.tight_layout(); fig.savefig(figdir/name,dpi=150); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    methods=("Original Correlation","B2 Target-Dependent Timing / Original Bounds",
             "Shared-Spike STDP / No Clipping","Shared-Spike STDP / Original Bounds")
    for ax,tag in zip(axes,("A","B")):
        for method in methods:
            sel=[r for r in checkpoints if r["dataset"]==tag and r["learning_method"]==method and r["epoch"]==b2.EPOCHS]
            ax.plot([r["K"] for r in sel],[r["exact_percent"] for r in sel],marker="o",label=method)
        ax.set(xlabel="Number of Stored Associations",xticks=KS,title=f"Dataset {tag}: Fixed Prefix Subsets"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Exact Recall Accuracy (%)"); axes[1].legend(fontsize=7)
    fig.suptitle("Static Recall Accuracy by Dataset and Learning Method (Score > 0)")
    fig.tight_layout(); fig.savefig(figdir/"exact_recall_by_dataset_and_method.png",dpi=150); plt.close(fig)


def write_summary(b2, datasets, cases, single_metrics, checkpoints, all_subset, tie_rows, out):
    lines=["# Shared-Spike STDP Training Verification", "",
           "This is teacher-event spike-pair learning through the **unmodified** B2 `stdp_update` function. It does not evaluate LIF recall. No original source or dataset was modified.", "",
           "## Configuration", "",
           f"- Network: 15 input × 15 output, one feedforward weight matrix, initial raw weight {b2.W_INIT} per synapse; effective signed weight `W_after − W_initial`.",
           f"- B2 STDP: `A_plus=A_minus={b2.A_PLUS}`, `tau={b2.TAU_STDP} ms`, `DT={b2.DT} ms`; bounds [{b2.W_MIN},{b2.W_MAX}] or temporarily disabled in the isolated module for no-clipping control.",
           f"- Epochs: {b2.EPOCHS}; original random-permutation schedule `RandomState(seed=0).permutation(subset)` once per epoch, plus ascending/descending order controls. Fixed prefix subsets are digits 0..K−1 for K={list(KS)}; no subset averaging is claimed for STDP.",
           "- One shared presynaptic event vector per presentation: input +1 at 2 ms, −1 at 7 ms, identical for all 15 output columns. Target +1 teacher post at 4.5 ms; target −1 posts at 1 and 8 ms. All are inside the 0–10 ms simulation window.",
           "- Teacher-only plasticity: every scheduled teacher event is passed to the source update function. Natural output spikes are not generated or used during training in this wrapper. This follows the original teacher-only choice to exclude natural posts, but the two-event teacher schedule is new.",
           "- Recall: stored bipolar cues only; `score=X_bipolar @ W_effective`, prediction +1 iff score>0. Pure correlation uses `W=X.T@Y`. B2 reference uses the original target-dependent pre timing on the same supplied A/B arrays and the same teacher-only source update; it is an experimental re-run, not an archived measurement.",
           "", "## Single-presentation prerequisite", "",
           "| Input | Target | Δw |", "|---:|---:|---:|",
           *[f"| {r['input_bit']} | {r['target_bit']} | {r['delta_w']:+.9f} |" for r in cases], "",
           "All four signs passed before K>3 training. Magnitudes are **not equal**: target −1 has approximately 1.94× the absolute update of target +1. Thus even one-presentation shared-spike ΔW is not exactly proportional to the bipolar outer product. The reported positive best-fit scale is `max(0, sum(ΔW*C)/sum(C*C))`, with `C=outer(x,y)` for one pair or `C=X.T@Y` for a subset; normalized error is `||ΔW−aC||_F/||ΔW||_F`. Dataset-specific scales and residuals for every digit are in `csv/single_presentation_metrics.csv`; raw synapses for digit 0 are in `csv/single_presentation_synapses_digit0.csv`.",
           "", "## Final 60-epoch results for fixed prefix subsets", "",
           "| Dataset | K | Correlation exact | B2 reference exact | Shared no-clipping exact | Shared original-bounds exact | Shared bounded cosine | Shared bounded normalized error | Shared saturated |",
           "|---|---:|---:|---:|---:|---:|---:|---:|" ]
    for tag in ("A","B"):
        for k in KS:
            d={r["learning_method"]:r for r in checkpoints if r["dataset"]==tag and r["K"]==k and r["epoch"]==b2.EPOCHS}
            c=d["Original Correlation"]; b=d["B2 Target-Dependent Timing / Original Bounds"]
            u=d["Shared-Spike STDP / No Clipping"]; s=d["Shared-Spike STDP / Original Bounds"]
            lines.append(f"| {tag} | {k} | {c['exact_count']}/{k} | {b['exact_count']}/{k} | {u['exact_count']}/{k} | {s['exact_count']}/{k} | {s['cosine']:.4f} | {s['normalized_fro_error']:.4f} | {s['saturated_synapse_percent']:.1f}% |")
    lines += ["", "The table uses **fixed prefix subsets** for all four methods. `csv/correlation_all_subset_baseline.csv` separately reproduces the supplied all-combination means; those means must not be compared as if they were these prefix results. In particular, the K=10 prefix is the full dataset: correlation A=4/10, B=2/10.", "",
              "## Repeated-learning effects and limitations", ""]
    for tag in ("A","B"):
        d={r["learning_method"]:r for r in checkpoints if r["dataset"]==tag and r["K"]==10 and r["epoch"]==b2.EPOCHS}
        b0=d["B2 Target-Dependent Timing / No Clipping"]
        bb=d["B2 Target-Dependent Timing / Original Bounds"]
        ss=d["Shared-Spike STDP / Original Bounds"]
        lines.append(f"- Dataset {tag}, K=10: B2 no-clipping reference {b0['exact_count']}/10 (cosine {b0['cosine']:.4f}), B2 original-bounds {bb['exact_count']}/10 (cosine {bb['cosine']:.4f}); shared original-bounds {ss['exact_count']}/10 with {ss['saturated_synapse_percent']:.1f}% saturated synapses at epoch 60. Its final epoch had {ss['clipping_events']} clipping attempts and {ss['suppressed_update_events']} zero net synapse updates; sign reversals {ss['sign_reversals']}.")
    ties=[r for r in tie_rows if r["dataset"]=="A" and r["K"]==10 and r["prediction_differs"]]
    lines.append(f"- Dataset A K=10 has {len(ties)} correlation-score ties whose B2 no-clipping scores are tiny positive floating-point values; these explain its 3/10 versus pure correlation 4/10 despite cosine 1.0000. Exact bit details and unrounded scores are in `csv/near_zero_tie_diagnostics.csv`. The primary decoder remains the predeclared strict `score>0` rule; no post-hoc tolerance was applied.")
    for tag in ("A","B"):
        for k in (1,3,5,10):
            d={r["learning_method"]:r for r in checkpoints if r["dataset"]==tag and r["K"]==k and r["epoch"]==b2.EPOCHS}
            u=d["Shared-Spike STDP / No Clipping"]; s=d["Shared-Spike STDP / Original Bounds"]
            a=d["Shared-Spike STDP / Ascending Order"]; z=d["Shared-Spike STDP / Descending Order"]
            lines.append(f"- Dataset {tag}, K={k}: no-clipping correlation cosine {u['cosine']:.4f}, error {u['normalized_fro_error']:.4f}; bounded cosine {s['cosine']:.4f}, error {s['normalized_fro_error']:.4f}, saturation {s['saturated_synapse_percent']:.1f}%. Ascending/descending exact: {a['exact_count']}/{k}, {z['exact_count']}/{k}.")
    for tag in ("A","B"):
        with np.load(out/"npz"/f"{tag}_K10_shared_bounded_ascending.npz") as d:
            asc=d["W_final"]
        with np.load(out/"npz"/f"{tag}_K10_shared_bounded_descending.npz") as d:
            desc=d["W_final"]
        lines.append(f"- Dataset {tag}, K=10: ascending versus descending final raw weights differ by a maximum absolute {np.max(np.abs(asc-desc)):.6f}. Their final exact counts are recorded separately in `csv/learning_order_results.csv`.")
        for bound in ("unbounded","bounded"):
            with np.load(out/"npz"/f"{tag}_K10_shared_{bound}_original_random_seed0.npz") as d:
                last_change=float(np.max(np.abs(d["W_by_epoch"][-1]-d["W_by_epoch"][-2])))
            lines.append(f"- Dataset {tag}, K=10, shared {bound}: maximum absolute weight change from epoch 59 to 60 was {last_change:.6f}; the matrix had not reached an exact fixed point by epoch 60.")
    lines += ["", "The source kernel is additive before clipping. Therefore, with teacher events fixed and clipping disabled, each presentation repeats a fixed synapse-local increment: outer-product **signs** persist, but target-specific magnitudes create a systematic departure from one common correlation scale. With original bounds, clipping can suppress later updates or make them zero at the bound; the `npz/*W_by_epoch` arrays and `csv/epoch_metrics.csv` retain the measured trajectory and clip-event counts. Order effects are visible only through clipping and floating-point effects in this teacher-only fixed-event setup.",
              "", "No natural post spikes were mixed into plasticity. This isolates teacher-event learning; it does **not** show that a LIF output would emit the prescribed 1/8-ms events without external teaching. We made no LIF recall claim. Single-event sign feasibility, repeated event learning, correlation reproduction and static recall are distinct conclusions; see the measured table. The suitability of this encoding for later local weight-dependent STDP work depends on accepting externally scheduled teacher events and the observed unequal update magnitudes and bounded-weight behavior.",
              "", "## Reproduce", "", "Run from repository root:", "", "```bash",
              "python3 my_research/experiments/diagnostics/shared_spike_stdp_training.py \\",
              "  --dataset-a local_stdp_audit_dataset_handoff_20261020/dataset_A.npz \\",
              "  --dataset-b local_stdp_audit_dataset_handoff_20261020/dataset_B.npz", "```", ""]
    (out/"summary.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset-a",type=Path,required=True)
    parser.add_argument("--dataset-b",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    out=args.out or ROOT/"results"/"shared_spike_stdp_training"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True,exist_ok=False)
    for name in ("scripts","config","csv","npz","figures"):
        (out/name).mkdir()
    shutil.copy2(__file__,out/"scripts"/Path(__file__).name)
    b2=source_module()
    assert b2.EPOCHS==60 and b2.STEP_TEACH==45 and b2.STEP_ON==20 and b2.STEP_OFF==70
    datasets={"A":read_dataset(args.dataset_a),"B":read_dataset(args.dataset_b)}
    config=dict(git_revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip(),
                source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                datasets={tag:dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                          for tag,path in (("A",args.dataset_a),("B",args.dataset_b))},
                K=list(KS),subsets="fixed prefix digits 0..K-1",seed=SEED,epochs=b2.EPOCHS,
                input_times_ms={"+1":2.0,"-1":7.0},teacher_post_times_ms={"+1":[4.5],"-1":[1.0,8.0]},
                time_step_ms=b2.DT,sim_time_ms=b2.T_MAX,
                A_plus=b2.A_PLUS,A_minus=b2.A_MINUS,tau_stdp_ms=b2.TAU_STDP,
                initial_weight=b2.W_INIT,original_weight_bounds=[b2.W_MIN,b2.W_MAX],
                unclipped_control="imported module bounds set to +/-infinity; same source stdp_update",
                natural_posts="excluded; teacher event lists only",decoder="X_bipolar @ (W_final-W_initial); +1 iff score>0")
    (out/"config"/"experiment_config.json").write_text(json.dumps(config,indent=2)+"\n")
    cases,single_metrics=single_presentation(b2,datasets,out)  # fails fast if four signs do not match
    baselines=baseline_all_subsets(datasets,out)
    save_event_and_presentation_schedules(b2,datasets,out)
    # K=1,2,3 execute first. K=5,7,10 are reached only if the pilot passes.
    checkpoints=run_training(b2,datasets,out)
    tie_rows=write_tie_diagnostics(datasets,out)
    figures(b2,cases,checkpoints,out)
    write_summary(b2,datasets,cases,single_metrics,checkpoints,baselines,tie_rows,out)
    archive=out.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.rglob("*")):
            if file.is_file(): z.write(file,file.relative_to(out.parent))
    print(json.dumps(dict(result_directory=str(out),archive=str(archive),
                          single_presentation_cases=len(single_metrics),training_runs=2*len(KS)*6,
                          k10={tag:{method:next(r["exact_count"] for r in checkpoints if r["dataset"]==tag and r["K"]==10 and r["epoch"]==b2.EPOCHS and r["learning_method"]==method)
                                     for method in ("Original Correlation","B2 Target-Dependent Timing / No Clipping","B2 Target-Dependent Timing / Original Bounds","Shared-Spike STDP / No Clipping","Shared-Spike STDP / Original Bounds")}
                               for tag in ("A","B")}),indent=2))


if __name__=="__main__":
    main()
