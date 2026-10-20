"""Calibrate shared-spike teacher timing with the original B2 STDP kernel.

Timing selection uses only the four event updates, never dataset recall. This
is teacher-event learning; natural output spikes and LIF recall are excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shared_spike_stdp_training as prior


ROOT = Path(__file__).resolve().parents[3]
KS = (1, 2, 3, 5, 7, 10)
SEED = 0
NEAR_ZERO_TOL = 1e-10  # diagnostic label only; never used in the decoder


def steps(b2, times):
    result = tuple(int(round(float(t) / b2.DT)) for t in times)
    assert all(0 <= s < b2.n_steps for s in result)
    return result


def four_cases(b2, timing):
    old_min, old_max = b2.W_MIN, b2.W_MAX
    b2.W_MIN, b2.W_MAX = -np.inf, np.inf
    try:
        rows = []
        for xbit, ybit in itertools.product((1, 0), repeat=2):
            pre_ms = timing["pre_plus_ms"] if xbit else timing["pre_minus_ms"]
            post_ms = (timing["post_plus_ms"],) if ybit else (
                timing["post_minus_early_ms"], timing["post_minus_late_ms"])
            w0 = np.zeros(15, dtype=float)
            pre = np.full(15, steps(b2, (pre_ms,))[0], dtype=int)
            post = steps(b2, post_ms)
            updated = b2.stdp_update(w0.copy(), pre, post)
            assert np.allclose(updated, updated[0])
            d = float(updated[0])
            expected = (2*xbit-1)*(2*ybit-1)
            rows.append(dict(input_bit=xbit, target_bit=ybit, pre_ms=pre_ms,
                             post_ms=";".join(str(t) for t in post_ms),
                             delta_w=d, absolute_delta_w=abs(d),
                             expected_sign=expected, sign_correct=bool(np.sign(d)==expected)))
        magnitudes = np.array([r["absolute_delta_w"] for r in rows])
        return rows, dict(all_signs_correct=all(r["sign_correct"] for r in rows),
                          relative_spread=float((magnitudes.max()-magnitudes.min())/magnitudes.mean()),
                          minimum_absolute_update=float(magnitudes.min()),
                          mean_absolute_update=float(magnitudes.mean()))
    finally:
        b2.W_MIN, b2.W_MAX = old_min, old_max


def calibration_search(b2):
    proposed = dict(pre_plus_ms=3.0, pre_minus_ms=7.0, post_plus_ms=5.0,
                    post_minus_early_ms=1.3, post_minus_late_ms=8.7)
    candidates = []
    # Deliberately small timing-only grid; no dataset recall is evaluated here.
    for post_plus_step, early_step, late_step in itertools.product(
        range(48, 53), range(10, 17), range(84, 91)):
        timing = dict(pre_plus_ms=3.0, pre_minus_ms=7.0,
                      post_plus_ms=post_plus_step/10,
                      post_minus_early_ms=early_step/10,
                      post_minus_late_ms=late_step/10)
        rows, metric = four_cases(b2, timing)
        candidates.append(dict(**timing, **metric))
    feasible = [r for r in candidates if r["all_signs_correct"]]
    assert feasible
    chosen = min(feasible, key=lambda r: (
        r["relative_spread"], -r["minimum_absolute_update"],
        r["post_plus_ms"], r["post_minus_early_ms"], r["post_minus_late_ms"]))
    timing = {k:chosen[k] for k in proposed}
    proposed_rows, proposed_metric = four_cases(b2, proposed)
    selected_rows, selected_metric = four_cases(b2, timing)
    assert proposed_metric["all_signs_correct"] and selected_metric["all_signs_correct"]
    return proposed, proposed_rows, proposed_metric, timing, selected_rows, selected_metric, candidates


def event_vector(b2, x01, timing):
    pre = np.where(x01 > 0, steps(b2, (timing["pre_plus_ms"],))[0],
                   steps(b2, (timing["pre_minus_ms"],))[0]).astype(int)
    assert pre.shape == (15,)
    return pre


def teacher_steps(b2, target_sign, timing):
    return steps(b2, (timing["post_plus_ms"],)) if target_sign == 1 else steps(
        b2, (timing["post_minus_early_ms"], timing["post_minus_late_ms"]))


def event_delta(b2, pre, post):
    old_min, old_max = b2.W_MIN, b2.W_MAX
    b2.W_MIN, b2.W_MAX = -np.inf, np.inf
    try:
        return b2.stdp_update(np.zeros(15, dtype=float), pre, post)
    finally:
        b2.W_MIN, b2.W_MAX = old_min, old_max


def single_presentation_matrices(b2,datasets,timing,out):
    rows=[]
    for tag,(xb,y,x01) in datasets.items():
        for p in range(10):
            pre=event_vector(b2,x01[p],timing)
            update=np.column_stack([event_delta(b2,pre,teacher_steps(b2,int(y[p,j]),timing))
                                    for j in range(15)])
            outer=np.outer(xb[p],y[p])
            assert np.array_equal(np.sign(update),outer)
            sim=prior.similarity(update,outer)
            rows.append(dict(dataset=tag,digit=p,all_225_signs_correct=True,**sim))
            np.savez_compressed(out/"npz"/f"single_presentation_{tag}_digit{p}.npz",
                                W_delta=update,W_outer=outer,shared_pre_steps=pre)
    prior.write_csv(out/"csv"/"single_presentation_matrix_metrics.csv",rows)
    return rows


def presentation_orders(b2,out):
    rows=[]
    for k in KS:
        ids=tuple(range(k))
        for order in ("random_seed0","ascending","descending"):
            rng=np.random.RandomState(SEED)
            for epoch in range(1,b2.EPOCHS+1):
                sequence=(rng.permutation(ids) if order=="random_seed0" else
                          ids if order=="ascending" else tuple(reversed(ids)))
                for position,p in enumerate(sequence):
                    rows.append(dict(K=k,order=order,epoch=epoch,presentation_index=position,digit=int(p)))
    prior.write_csv(out/"csv"/"presentation_order.csv",rows)


def train(b2, xb, y, x01, ids, timing, bounded, order):
    old_min, old_max = b2.W_MIN, b2.W_MAX
    if not bounded:
        b2.W_MIN, b2.W_MAX = -np.inf, np.inf
    try:
        w = np.full((15,15), b2.W_INIT, dtype=float)
        weights = [w.copy()]
        clipped_counts, suppressed_counts, reversal_counts = [0], [0], [0]
        rng = np.random.RandomState(SEED)
        pre_by_digit = {p:event_vector(b2, x01[p], timing) for p in ids}
        posts = {(p,j):teacher_steps(b2, int(y[p,j]), timing) for p in ids for j in range(15)}
        deltas = {(p,j):event_delta(b2, pre_by_digit[p], posts[(p,j)]) for p in ids for j in range(15)}
        for p in ids:
            assert all(pre_by_digit[p] is pre_by_digit[p] for _ in range(15))
            for j in range(15):
                assert np.array_equal(np.sign(deltas[(p,j)]), xb[p]*y[p,j])
        for ep in range(b2.EPOCHS):
            sequence = (rng.permutation(ids) if order=="random_seed0" else
                        ids if order=="ascending" else tuple(reversed(ids)))
            clipped = suppressed = reversed_sign = 0
            for p in sequence:
                pre = pre_by_digit[p]  # same object supplied to every output
                for j in range(15):
                    before = w[:,j].copy()
                    update = deltas[(p,j)]
                    if bounded:
                        prospective = before+update
                        clipped += int(np.sum((prospective<old_min)|(prospective>old_max)))
                    w[:,j] = b2.stdp_update(before.copy(), pre, posts[(p,j)])
                    expected = np.clip(before+update,old_min,old_max) if bounded else before+update
                    assert np.allclose(w[:,j],expected,atol=1e-12)
                    actual = w[:,j]-before
                    suppressed += int(np.sum(np.abs(actual)<=1e-12))
                    reversed_sign += int(np.sum(actual*update < -1e-12))
            weights.append(w.copy())
            clipped_counts.append(clipped)
            suppressed_counts.append(suppressed)
            reversal_counts.append(reversed_sign)
        assert sum(reversal_counts)==0
        return (np.stack(weights), np.asarray(clipped_counts),
                np.asarray(suppressed_counts), np.asarray(reversal_counts))
    finally:
        b2.W_MIN, b2.W_MAX = old_min, old_max


def margins_and_recall(xx, yy, weight):
    score=xx@weight
    pred=np.where(score>0,1,-1)
    signed=yy*score
    exact=np.all(pred==yy,axis=1)
    strict=np.all(signed>0,axis=1)
    return score,pred,signed,exact,strict


def run_learning(b2,datasets,previous,chosen,out):
    epoch_rows, association_rows, bit_rows, order_rows=[],[],[],[]
    for tag,(xb,y,x01) in datasets.items():
        for k in KS:
            ids=tuple(range(k))
            xx,yy=xb[list(ids)],y[list(ids)]
            corr=xx.T@yy
            for label,timing,bounded,order in (
                ("Previous Shared-Spike Timing",previous,False,"random_seed0"),
                ("Calibrated Shared-Spike Timing",chosen,False,"random_seed0"),
                ("Previous Shared-Spike Timing",previous,True,"random_seed0"),
                ("Calibrated Shared-Spike Timing",chosen,True,"random_seed0"),
                ("Calibrated Shared-Spike Timing",chosen,True,"ascending"),
                ("Calibrated Shared-Spike Timing",chosen,True,"descending"),
            ):
                history,clipped,suppressed,reversals=train(b2,xb,y,x01,ids,timing,bounded,order)
                stem=f"{tag}_K{k}_{'previous' if label.startswith('Previous') else 'calibrated'}_{'bounded' if bounded else 'unbounded'}_{order}"
                np.savez_compressed(out/"npz"/f"{stem}.npz",W_initial=history[0],
                                    W_final=history[-1],W_effective=history[-1]-history[0],
                                    W_correlation=corr,W_by_epoch=history,
                                    clipping_events_by_epoch=clipped,
                                    suppressed_updates_by_epoch=suppressed,
                                    sign_reversals_by_epoch=reversals,
                                    subset_ids=np.asarray(ids))
                for ep,w in enumerate(history):
                    effective=w-history[0]
                    score,pred,signed,exact,strict=margins_and_recall(xx,yy,effective)
                    sim=prior.similarity(effective,corr)
                    sat=100*np.mean((np.abs(w-b2.W_MIN)<=1e-12)|(np.abs(w-b2.W_MAX)<=1e-12)) if bounded else 0.0
                    epoch_rows.append(dict(dataset=tag,K=k,subset_ids=";".join(map(str,ids)),
                                           method=label,bounded=bounded,order=order,epoch=ep,
                                           exact_count=int(exact.sum()),exact_percent=100*exact.mean(),
                                           strictly_positive_margin_count=int(strict.sum()),
                                           bit_percent=100*np.mean(pred==yy),
                                           min_signed_margin=float(signed.min()),
                                           near_zero_score_count=int(np.sum(np.abs(score)<NEAR_ZERO_TOL)),
                                           saturated_synapse_percent=float(sat),
                                           clipping_events=int(clipped[ep]),
                                           suppressed_updates=int(suppressed[ep]),
                                           sign_reversals=int(reversals[ep]),
                                           **sim))
                    if ep==b2.EPOCHS and order=="random_seed0":
                        for row,p in enumerate(ids):
                            association_rows.append(dict(dataset=tag,K=k,method=label,bounded=bounded,
                                                         digit=p,exact=bool(exact[row]),
                                                         strictly_positive_margin=bool(strict[row]),
                                                         bit_accuracy=float(np.mean(pred[row]==yy[row])),
                                                         minimum_signed_margin=float(signed[row].min()),
                                                         target_bits="".join(map(str,((yy[row]+1)//2).tolist())),
                                                         predicted_bits="".join(map(str,((pred[row]+1)//2).tolist()))))
                            for j in range(15):
                                bit_rows.append(dict(dataset=tag,K=k,method=label,bounded=bounded,
                                                     digit=p,output_neuron=j,target_sign=int(yy[row,j]),
                                                     score=float(score[row,j]),prediction_sign=int(pred[row,j]),
                                                     signed_margin=float(signed[row,j]),
                                                     correct=bool(pred[row,j]==yy[row,j]),
                                                     strictly_positive_margin=bool(signed[row,j]>0),
                                                     near_zero=bool(abs(score[row,j])<NEAR_ZERO_TOL)))
                order_rows.append(dict(dataset=tag,K=k,method=label,bounded=bounded,
                                       order=order,exact_count=int(exact.sum()),bit_accuracy=float(np.mean(pred==yy)),
                                       max_abs_weight_difference_from_random=np.nan))
            # Direct correlation on precisely the same prefix subset.
            score,pred,signed,exact,strict=margins_and_recall(xx,yy,corr)
            epoch_rows.append(dict(dataset=tag,K=k,subset_ids=";".join(map(str,ids)),
                                   method="Original Correlation",bounded=False,order="none",epoch=b2.EPOCHS,
                                   exact_count=int(exact.sum()),exact_percent=100*exact.mean(),
                                   strictly_positive_margin_count=int(strict.sum()),
                                   bit_percent=100*np.mean(pred==yy),min_signed_margin=float(signed.min()),
                                   near_zero_score_count=int(np.sum(np.abs(score)<NEAR_ZERO_TOL)),
                                   saturated_synapse_percent=0.0,clipping_events=0,
                                   suppressed_updates=0,sign_reversals=0,**prior.similarity(corr,corr)))
            for row,p in enumerate(ids):
                association_rows.append(dict(dataset=tag,K=k,method="Original Correlation",bounded=False,
                                             digit=p,exact=bool(exact[row]),
                                             strictly_positive_margin=bool(strict[row]),
                                             bit_accuracy=float(np.mean(pred[row]==yy[row])),
                                             minimum_signed_margin=float(signed[row].min()),
                                             target_bits="".join(map(str,((yy[row]+1)//2).tolist())),
                                             predicted_bits="".join(map(str,((pred[row]+1)//2).tolist()))))
                for j in range(15):
                    bit_rows.append(dict(dataset=tag,K=k,method="Original Correlation",bounded=False,
                                         digit=p,output_neuron=j,target_sign=int(yy[row,j]),
                                         score=float(score[row,j]),prediction_sign=int(pred[row,j]),
                                         signed_margin=float(signed[row,j]),correct=bool(pred[row,j]==yy[row,j]),
                                         strictly_positive_margin=bool(signed[row,j]>0),
                                         near_zero=bool(abs(score[row,j])<NEAR_ZERO_TOL)))
    for row in order_rows:
        if row["order"]=="random_seed0":
            row["max_abs_weight_difference_from_random"]=0.0
        else:
            tag,k,order=row["dataset"],row["K"],row["order"]
            a=np.load(out/"npz"/f"{tag}_K{k}_calibrated_bounded_{order}.npz")["W_final"]
            b=np.load(out/"npz"/f"{tag}_K{k}_calibrated_bounded_random_seed0.npz")["W_final"]
            row["max_abs_weight_difference_from_random"]=float(np.max(np.abs(a-b)))
    prior.write_csv(out/"csv"/"epoch_metrics.csv",epoch_rows)
    prior.write_csv(out/"csv"/"association_results.csv",association_rows)
    prior.write_csv(out/"csv"/"bit_scores_and_margins.csv",bit_rows)
    prior.write_csv(out/"csv"/"learning_order_results.csv",order_rows)
    return epoch_rows,association_rows,bit_rows,order_rows


def make_figures(b2,proposed_rows,selected_rows,candidates,epochs,out):
    fdir=out/"figures"
    fig,ax=plt.subplots(figsize=(8,4))
    for y,label,times,color in ((3,"Pre +1",[3],"tab:blue"),(2,"Pre −1",[7],"tab:orange"),
                                 (1,"Target +1 post",[5],"tab:green"),(0,"Target −1 posts",[selected_rows[1]["post_ms"].split(';')[0],selected_rows[1]["post_ms"].split(';')[1]],"tab:red")):
        ax.scatter([float(t) for t in times],[y]*len(times),label=label,color=color,s=80)
    ax.set(xlim=(0,10),yticks=range(4),yticklabels=["Target −1 post","Target +1 post","Pre −1","Pre +1"],
           xlabel="Time (ms)",title="Calibrated Shared Pre-Spike and Teacher Post-Spike Timing")
    ax.grid(axis="x",alpha=.25);fig.tight_layout();fig.savefig(fdir/"calibrated_spike_timing.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    xs=np.arange(4);labels=[f"({r['input_bit']},{r['target_bit']})" for r in selected_rows]
    ax.bar(xs-.18,[r["delta_w"] for r in proposed_rows],width=.36,label="Proposed Timing")
    ax.bar(xs+.18,[r["delta_w"] for r in selected_rows],width=.36,label="Selected Timing")
    ax.set(xticks=xs,xticklabels=labels,xlabel="(Input bit, Target bit)",ylabel="One-Presentation Δw",
           title="Four Bipolar Updates from Unmodified STDP Kernel")
    ax.axhline(0,color="black",lw=.7);ax.legend();fig.tight_layout();fig.savefig(fdir/"four_case_update_magnitudes.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,tag in zip(axes,("A","B")):
        for label,style in (("Previous Shared-Spike Timing","--"),("Calibrated Shared-Spike Timing","-")):
            for bounded,color in ((False,"tab:blue"),(True,"tab:orange")):
                rows=[r for r in epochs if r["dataset"]==tag and r["K"]==10 and r["method"]==label and r["bounded"]==bounded and r["order"]=="random_seed0"]
                ax.plot([r["epoch"] for r in rows],[r["normalized_fro_error"] for r in rows],
                        linestyle=style,color=color,label=f"{'Previous' if style=='--' else 'Calibrated'}, {'Bounded' if bounded else 'Unbounded'}")
        ax.set(title=f"Dataset {tag}, K=10",xlabel="Learning Epoch");ax.grid(alpha=.2)
    axes[0].set_ylabel("Normalized Frobenius Error");axes[1].legend(fontsize=8)
    fig.suptitle("Weight Error vs Learning Epoch");fig.tight_layout();fig.savefig(fdir/"weight_error_by_epoch.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,tag in zip(axes,("A","B")):
        for label in ("Previous Shared-Spike Timing","Calibrated Shared-Spike Timing"):
            rows=[r for r in epochs if r["dataset"]==tag and r["K"]==10 and r["method"]==label and r["bounded"]==True and r["order"]=="random_seed0"]
            ax.plot([r["epoch"] for r in rows],[r["saturated_synapse_percent"] for r in rows],label=label)
        ax.set(title=f"Dataset {tag}, K=10",xlabel="Learning Epoch");ax.grid(alpha=.2)
    axes[0].set_ylabel("Weight Saturation (%)");axes[1].legend(fontsize=8)
    fig.suptitle("Weight Saturation vs Learning Epoch, Original Bounds");fig.tight_layout();fig.savefig(fdir/"weight_saturation_by_epoch.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ri,tag in enumerate(("A","B")):
        d0=np.load(out/"npz"/f"{tag}_K10_previous_unbounded_random_seed0.npz")
        d1=np.load(out/"npz"/f"{tag}_K10_calibrated_unbounded_random_seed0.npz")
        corr=d0["W_correlation"]; prev=d0["W_effective"]; cal=d1["W_effective"]
        a=prior.similarity(cal,corr)["scale"]
        mats=(a*corr,prev,cal,cal-a*corr)
        titles=("Scaled Correlation","Previous STDP ΔW","Calibrated STDP ΔW","Calibrated Difference")
        for ci,(mat,title) in enumerate(zip(mats,titles)):
            lim=max(np.max(np.abs(a*corr)),np.max(np.abs(prev)),np.max(np.abs(cal))) if ci<3 else max(np.max(np.abs(mat)),1e-12)
            axes[ri,ci].imshow(mat,cmap="coolwarm",vmin=-lim,vmax=lim)
            axes[ri,ci].set(title=f"Dataset {tag}: {title}",xlabel="Output neuron",ylabel="Input neuron")
    fig.suptitle("K=10, 60 Epochs, No Clipping");fig.tight_layout();fig.savefig(fdir/"correlation_vs_stdp_weights.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,tag in zip(axes,("A","B")):
        for method,bounded in (("Original Correlation",False),
                               ("Previous Shared-Spike Timing",False),
                               ("Calibrated Shared-Spike Timing",False),
                               ("Calibrated Shared-Spike Timing",True)):
            rows=[r for r in epochs if r["dataset"]==tag and r["method"]==method and r["bounded"]==bounded and r["epoch"]==b2.EPOCHS and r["order"] in ("none","random_seed0")]
            ax.plot([r["K"] for r in rows],[r["exact_percent"] for r in rows],marker="o",
                    label=method+(" / bounded" if bounded else " / unbounded" if method!="Original Correlation" else ""))
        ax.set(title=f"Dataset {tag}: Fixed Prefix Subsets",xlabel="Number of Stored Associations",xticks=KS)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Exact Recall Accuracy (%)");axes[1].legend(fontsize=7)
    fig.suptitle("Static Recall by Timing and Weight Bounds: Score > 0")
    fig.tight_layout();fig.savefig(fdir/"exact_recall_by_method.png",dpi=150);plt.close(fig)


def write_summary(b2,proposed,proposed_metric,chosen,chosen_metric,epochs,associations,bits,orders,out):
    lines=["# Shared-Spike STDP Timing Calibration", "",
           "This is an event-timing calibration of the **unmodified** B2 `stdp_update`. Timing selection used only four-case update magnitude and sign, never Dataset A/B recall. Teacher events were prescribed; natural output spikes and LIF recall were not tested.","",
           "## Timing and calibration", "",
           f"- Proposed: {proposed}; measured four-case relative spread {100*proposed_metric['relative_spread']:.6f}%, minimum |Δw| {proposed_metric['minimum_absolute_update']:.9f}.",
           f"- Selected from a predeclared 245-point grid with pre times fixed at 3/7 ms, target+ post 4.8–5.2 ms, target− early 1.0–1.6 ms and late 8.4–9.0 ms (0.1-ms steps): {chosen}.",
           f"- Measured selected spread {100*chosen_metric['relative_spread']:.6f}%, minimum |Δw| {chosen_metric['minimum_absolute_update']:.9f}; all four signs correct. Objective: minimize `(max|Δw|−min|Δw|)/mean|Δw|`, then maximize minimum |Δw|. No dataset score entered this choice.",
           "- All times are on the 0.1-ms grid within the 0–10-ms window; input timing is independent of output target and one 15-element pre-event vector is passed to every output column.",
           f"- 60 epochs, W_initial={b2.W_INIT}, A_plus=A_minus={b2.A_PLUS}, tau={b2.TAU_STDP} ms, original bounds [{b2.W_MIN},{b2.W_MAX}] or no-clipping control; random permutation seed {SEED}. K=[1,2,3,5,7,10] uses one fixed prefix subset 0..K−1 per K, identical across methods. Ascending/descending controls test bounded calibrated timing.",
           "- Primary static decoder: `score = X_bipolar @ (W_final−W_initial)`; `score>0` predicts +1, otherwise −1. Strict positive signed-margin recall and |score|<1e−10 are separate diagnostics; this tolerance never changes the primary decoder.","",
           "## 60-epoch fixed-prefix results", "",
           "| Dataset | K | Correlation exact | Previous unbounded | Calibrated unbounded | Calibrated bounded | Calibrated unbounded cosine | Calibrated unbounded error | Calibrated bounded saturation |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag in ("A","B"):
        for k in KS:
            rows={ (r["method"],r["bounded"]):r for r in epochs if r["dataset"]==tag and r["K"]==k and r["epoch"]==b2.EPOCHS and r["order"] in ("none","random_seed0") }
            c=rows[("Original Correlation",False)];p=rows[("Previous Shared-Spike Timing",False)]
            u=rows[("Calibrated Shared-Spike Timing",False)];b=rows[("Calibrated Shared-Spike Timing",True)]
            lines.append(f"| {tag} | {k} | {c['exact_count']}/{k} | {p['exact_count']}/{k} | {u['exact_count']}/{k} | {b['exact_count']}/{k} | {u['cosine']:.6f} | {u['normalized_fro_error']:.6f} | {b['saturated_synapse_percent']:.1f}% |")
    lines += ["", "All rows use **fixed prefix subsets**, not all-combination averages. K=10 is the complete dataset, so original correlation baselines are A=4/10 and B=2/10. `csv/association_results.csv` lists each digit; `csv/bit_scores_and_margins.csv` retains every final score and signed margin.","",
              "## Correlation reproduction and clipping", ""]
    for tag in ("A","B"):
        rows={ (r["method"],r["bounded"]):r for r in epochs if r["dataset"]==tag and r["K"]==10 and r["epoch"]==b2.EPOCHS and r["order"] in ("none","random_seed0") }
        c=rows[("Original Correlation",False)];u=rows[("Calibrated Shared-Spike Timing",False)];b=rows[("Calibrated Shared-Spike Timing",True)]
        lines.append(f"- Dataset {tag}, K=10: correlation exact {c['exact_count']}/10, strictly-positive-margin {c['strictly_positive_margin_count']}/10; calibrated unbounded exact {u['exact_count']}/10, strictly-positive-margin {u['strictly_positive_margin_count']}/10, scale {u['scale']:.9f}, cosine {u['cosine']:.9f}, normalized error {u['normalized_fro_error']:.9f}, max residual {u['max_abs_weight_error']:.9g}. Calibrated bounded exact {b['exact_count']}/10, saturation {b['saturated_synapse_percent']:.1f}%, final-epoch clipping attempts {b['clipping_events']}, suppressed updates {b['suppressed_updates']}, sign reversals {b['sign_reversals']}.")
        order_sub=[r for r in orders if r["dataset"]==tag and r["K"]==10 and r["method"]=="Calibrated Shared-Spike Timing" and r["bounded"]]
        lines.append("- Learning order, dataset "+tag+", K=10: "+"; ".join(f"{r['order']} exact {r['exact_count']}/10, max |W−W_random| {r['max_abs_weight_difference_from_random']:.9g}" for r in order_sub)+".")
    lines += ["", "When the four source-measured updates are not exactly equal, the unclipped matrix need not be precisely a scalar multiple of correlation even though all signs match. Clipping adds a separate nonlinear distortion and can suppress later updates; no sign reversal was observed. The learned raw W has a +0.05 offset, so only W_final−W_initial is used for static signed readout and matrix comparison.","",
              "## Output threshold boundary", ""]
    for tag in ("A","B"):
        for method,bounded in (("Original Correlation",False),("Calibrated Shared-Spike Timing",False),("Calibrated Shared-Spike Timing",True)):
            successes=[r["digit"] for r in associations if r["dataset"]==tag and r["K"]==10 and r["method"]==method and r["bounded"]==bounded and r["exact"]]
            lines.append(f"- Dataset {tag}, K=10, {method}{' / bounded' if bounded else ''}: exact digits {successes}.")
    for tag in ("A","B"):
        near=[r for r in bits if r["dataset"]==tag and r["K"]==10 and r["method"]=="Calibrated Shared-Spike Timing" and not r["bounded"] and r["near_zero"]]
        lines.append(f"- Dataset {tag}, calibrated unbounded K=10: {len(near)} output scores satisfy |score|<1e−10. Their exact values and targets are in `csv/bit_scores_and_margins.csv`.")
    boundary=[r for r in bits if r["dataset"]=="A" and r["K"]==10 and r["digit"]==0 and r["output_neuron"]==13]
    lines.append("- Dataset A digit 0/output 13 actual scores: "+"; ".join(f"{r['method']}{' bounded' if r['bounded'] else ''} {r['score']:+.9g}" for r in boundary)+". Target sign is −1.")
    lines += ["", "Dataset A's pure-correlation digit 0/output 13 score is exactly zero, and the primary rule maps it to −1. A scaled floating-point STDP matrix may yield a tiny signed nonzero instead; exact decoder recall and strict-positive-margin recall are reported separately. Tiny-score discrepancies must not be interpreted as evidence of a meaningful learning advantage or loss.","",
              "## Scope and suitability", "",
              "This verifies teacher-event timing and repeated spike-pair updates through the original kernel, not spontaneous teacher posts or LIF recall. A nearly correlation-proportional unclipped matrix is a useful **event-level control** for later local plasticity studies; using it as a full SNN baseline still requires checking that prescribed teacher events can be implemented and that clipping/threshold boundaries are handled consistently. No new learning rule, architecture, error feedback or noisy cue was introduced.","",
              "## Reproduce", "", "```bash", "python3 my_research/experiments/diagnostics/shared_spike_timing_calibration.py \\",
              "  --dataset-a local_stdp_audit_dataset_handoff_20261020/dataset_A.npz \\",
              "  --dataset-b local_stdp_audit_dataset_handoff_20261020/dataset_B.npz", "```", ""]
    (out/"summary.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset-a",type=Path,required=True)
    parser.add_argument("--dataset-b",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    out=args.out or ROOT/"results"/"shared_spike_timing_calibration"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True,exist_ok=False)
    for d in ("scripts","config","csv","npz","figures"):(out/d).mkdir()
    shutil.copy2(__file__,out/"scripts"/Path(__file__).name)
    shutil.copy2(Path(prior.__file__),out/"scripts"/Path(prior.__file__).name)
    b2=prior.source_module()
    assert b2.DT==0.1 and b2.EPOCHS==60 and b2.W_MIN==-1.5 and b2.W_MAX==1.5
    datasets={"A":prior.read_dataset(args.dataset_a),"B":prior.read_dataset(args.dataset_b)}
    np.savez_compressed(out/"npz"/"datasets.npz",
                        A_X_bipolar=datasets["A"][0],A_Y_bipolar=datasets["A"][1],A_X_binary=datasets["A"][2],
                        B_X_bipolar=datasets["B"][0],B_Y_bipolar=datasets["B"][1],B_X_binary=datasets["B"][2])
    proposed,proposed_rows,proposed_metric,chosen,selected_rows,selected_metric,candidates=calibration_search(b2)
    prior.write_csv(out/"csv"/"timing_candidates.csv",candidates)
    prior.write_csv(out/"csv"/"proposed_four_case_updates.csv",proposed_rows)
    prior.write_csv(out/"csv"/"selected_four_case_updates.csv",selected_rows)
    assert chosen["pre_plus_ms"]==3.0 and chosen["pre_minus_ms"]==7.0
    single_presentation_matrices(b2,datasets,chosen,out)
    presentation_orders(b2,out)
    previous=dict(pre_plus_ms=2.0,pre_minus_ms=7.0,post_plus_ms=4.5,
                  post_minus_early_ms=1.0,post_minus_late_ms=8.0)
    config=dict(git_revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip(),
                original_source=str(prior.SOURCE.relative_to(ROOT)),
                original_source_sha256=hashlib.sha256(prior.SOURCE.read_bytes()).hexdigest(),
                datasets={tag:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for tag,p in (("A",args.dataset_a),("B",args.dataset_b))},
                proposed_timing=proposed,selected_timing=chosen,previous_timing=previous,
                timing_search=dict(pre_plus_ms=[3.0],pre_minus_ms=[7.0],post_plus_ms=[4.8,5.2,0.1],
                                   post_minus_early_ms=[1.0,1.6,0.1],post_minus_late_ms=[8.4,9.0,0.1],
                                   objective="minimize relative four-case magnitude spread; maximize minimum magnitude for ties",
                                   dataset_recall_used=False,candidate_count=len(candidates)),
                K=list(KS),subsets="fixed prefix 0..K-1",epochs=b2.EPOCHS,seed=SEED,
                initial_weight=b2.W_INIT,A_plus=b2.A_PLUS,A_minus=b2.A_MINUS,tau_stdp_ms=b2.TAU_STDP,
                DT_ms=b2.DT,T_max_ms=b2.T_MAX,original_bounds=[b2.W_MIN,b2.W_MAX],
                natural_posts="excluded; teacher-event lists only",
                decoder="score=X_bipolar@(W_final-W_initial); +1 iff score>0",
                near_zero_tolerance=NEAR_ZERO_TOL)
    (out/"config"/"experiment_config.json").write_text(json.dumps(config,indent=2)+"\n")
    # Prerequisites complete before running 60-epoch training.
    assert all(r["sign_correct"] for r in selected_rows)
    epochs,associations,bits,orders=run_learning(b2,datasets,previous,chosen,out)
    make_figures(b2,proposed_rows,selected_rows,candidates,epochs,out)
    write_summary(b2,proposed,proposed_metric,chosen,selected_metric,epochs,associations,bits,orders,out)
    archive=out.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():z.write(p,p.relative_to(out.parent))
    print(json.dumps(dict(result_directory=str(out),archive=str(archive),
                          proposed_spread_percent=100*proposed_metric["relative_spread"],
                          selected_spread_percent=100*selected_metric["relative_spread"],
                          selected_timing=chosen,
                          k10={tag:{key:next(r["exact_count"] for r in epochs if r["dataset"]==tag and r["K"]==10 and r["epoch"]==b2.EPOCHS and r["method"]==label and r["bounded"]==bounded and r["order"] in ("none","random_seed0"))
                                     for key,label,bounded in (("original_correlation","Original Correlation",False),
                                                               ("previous_unbounded","Previous Shared-Spike Timing",False),
                                                               ("calibrated_unbounded","Calibrated Shared-Spike Timing",False),
                                                               ("calibrated_bounded","Calibrated Shared-Spike Timing",True))}
                               for tag in ("A","B")}),indent=2))


if __name__=="__main__":main()
