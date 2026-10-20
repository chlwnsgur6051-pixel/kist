"""Bounded weight-dependent STDP mathematical simulation; no original code edits.

Run from heteroassociation/: python my_research/experiments/diagnostics/weight_dependent_stdp.py
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_weight_dependent_stdp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "Ver관리" / "2. temporal"
sys.path.insert(0, str(SOURCE))
import stdp_temporal_symmetric_teacher_only as original

ROOT = PROJECT / "results" / "weight_dependent_stdp"
OUT = ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
CSV = OUT / "csv"
NPZ = OUT / "npz"
FIG = OUT / "figures"
K_VALUES = (1, 2, 3, 5, 7, 10)
MU_VALUES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
ORDER_SEED = 20261019
EPOCHS = int(original.EPOCHS)
INIT = float(original.W_INIT)
BASE_LO = float(original.W_MIN)
BASE_HI = float(original.W_MAX)
DT = float(original.DT)
BASE_A = float(original.A_PLUS * np.exp(-(original.T_TEACH-original.T_ON)/original.TAU_STDP))
X01 = original.data.make_digit_dataset()[0].reshape(10, 15).astype(int)
Y01 = original.data.make_letter_dataset()[0].reshape(10, 15).astype(int)
XB = 2*X01-1
YB = 2*Y01-1


def save_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subsets_for(k):
    return np.array(list(itertools.combinations(range(10), k)), dtype=int)


def signs_for(subsets):
    # Axis: subset, association, input neuron, output neuron. Each event is local.
    return (XB[subsets][:, :, :, None] * YB[subsets][:, :, None, :]).astype(np.int8)


def original_schedule(k):
    random = np.random.RandomState(0)  # original wrapper resets seed 0 per output
    return np.array([random.permutation(k) for _ in range(EPOCHS)], dtype=int)


def verify_timing_and_dataset():
    assert X01.shape == Y01.shape == (10, 15)
    assert X01[9, 12:15].tolist() == [0,0,1]
    timing_rows = []
    for digit in range(10):
        for out in range(15):
            target = int(Y01[digit,out])
            teach = "fire" if target else "timing_ltd"
            pre = original.encode(X01[digit].astype(float), teach=teach)
            for i in range(15):
                dt = (original.STEP_TEACH-int(pre[i]))*DT
                timing_sign = int(np.sign(dt))
                expected = int(XB[digit,i]*YB[digit,out])
                assert timing_sign == expected and abs(abs(dt)-2.5)<1e-10
                timing_rows.append(dict(digit=digit,output_idx=out,input_idx=i,
                                        input_bit=int(X01[digit,i]),target_bit=target,
                                        pre_time_ms=float(pre[i]*DT),teacher_time_ms=float(original.T_TEACH),
                                        signed_delta_t_ms=dt,timing_direction=timing_sign,
                                        bipolar_outer_product_sign=expected))
    save_csv(CSV/"timing_direction_validation.csv",timing_rows)
    expected={1:100.0,2:100.0,3:64.44444444444444,5:37.77777777777778,7:25.833333333333332,10:10.0}
    baseline={}
    for k in K_VALUES:
        subsets=subsets_for(k)
        corr=np.einsum("nki,nkj->nij",XB[subsets],YB[subsets])
        scores=np.einsum("nki,nij->nkj",XB[subsets],corr)
        pred=np.where(scores>0,1,-1)
        exact=100*float(np.mean(np.all(pred==YB[subsets],axis=2)))
        assert abs(exact-expected[k])<1e-9,(k,exact)
        baseline[k]=dict(exact_percent=exact,bit_percent=100*float(np.mean(pred==YB[subsets])),
                         subsets=len(subsets),associations=len(subsets)*k)
    return baseline


def train_batch(subsets, mu, a_plus=BASE_A, a_minus=BASE_A,
                lo=BASE_LO, hi=BASE_HI, schedule=None, epochs=EPOCHS):
    """Vectorized independent synapse-local updates across subset/order batches."""
    subsets=np.asarray(subsets,dtype=int)
    if subsets.ndim==1: subsets=subsets[None,:]
    n,k=subsets.shape
    signs=signs_for(subsets)
    weight=np.full((n,15,15),INIT,dtype=np.float64)
    clips=np.zeros(n,dtype=np.int64)
    batch_index=np.arange(n)
    if schedule is None:
        schedule=original_schedule(k) if epochs==EPOCHS else np.arange(k,dtype=int)[None,:]
    schedule=np.asarray(schedule,dtype=int)
    assert schedule.ndim in (2,3)
    assert schedule.shape[-1]==k
    if schedule.ndim==2:
        assert schedule.shape[0]==epochs
    else:
        assert schedule.shape==(n,epochs,k)
    for ep in range(epochs):
        for pos in range(k):
            if schedule.ndim==2:
                sign=signs[:,schedule[ep,pos]]
            else:
                sign=signs[batch_index,schedule[:,ep,pos]]
            if mu==0:
                proposed=weight+np.where(sign>0,a_plus,-a_minus)
            else:
                u=(weight-lo)/(hi-lo)
                proposed=weight+np.where(sign>0,a_plus*np.power(1-u,mu),
                                          -a_minus*np.power(u,mu))
            clips+=np.count_nonzero((proposed<lo)|(proposed>hi),axis=(1,2))
            np.clip(proposed,lo,hi,out=weight)
    return weight,clips


def evaluate(subsets, weight, lo=BASE_LO, hi=BASE_HI):
    delta=weight-INIT
    scores=np.einsum("nki,nij->nkj",XB[subsets],delta)
    raw_scores=np.einsum("nki,nij->nkj",XB[subsets],weight)
    target=YB[subsets]
    pred=np.where(scores>0,1,-1)
    raw_pred=np.where(raw_scores>0,1,-1)
    correct=pred==target
    exact=np.all(correct,axis=2)
    margins=target*scores
    saturated=np.isclose(weight,lo,atol=1e-12)|np.isclose(weight,hi,atol=1e-12)
    return dict(scores=scores,raw_scores=raw_scores,pred=pred,raw_pred=raw_pred,
                correct=correct,exact=exact,margins=margins,
                saturation=np.mean(saturated,axis=(1,2)))


def aggregate(k, mu, stage, setting, subsets, weight, clips, evaluated,
              reference=None, a_plus=BASE_A, a_minus=BASE_A, lo=BASE_LO, hi=BASE_HI):
    result=dict(stage=stage,setting=setting,k=k,mu=mu,subset_count=len(subsets),
        association_evaluations=len(subsets)*k,exact_count=int(np.sum(evaluated["exact"])),
        exact_accuracy_percent=100*float(np.mean(evaluated["exact"])),
        bit_accuracy_percent=100*float(np.mean(evaluated["correct"])),
        raw_weight_exact_accuracy_percent=100*float(np.mean(np.all(evaluated["raw_pred"]==YB[subsets],axis=2))),
        mean_signed_margin=float(np.mean(evaluated["margins"])),
        minimum_signed_margin=float(np.min(evaluated["margins"])),
        weight_saturation_percent=100*float(np.mean(evaluated["saturation"])),
        clipped_update_events=int(np.sum(clips)),
        weight_mean=float(np.mean(weight)),weight_std=float(np.std(weight)),
        weight_min=float(np.min(weight)),weight_max=float(np.max(weight)),
        a_plus=float(a_plus),a_minus=float(a_minus),bound_min=float(lo),bound_max=float(hi),
        epochs=int(EPOCHS if stage!="one_update" else 1))
    if reference is not None:
        base_ok=reference["correct"]
        result.update(corrected_output_bits=int(np.sum(~base_ok & evaluated["correct"])),
                      newly_incorrect_output_bits=int(np.sum(base_ok & ~evaluated["correct"])),
                      improved_associations=int(np.sum(~reference["exact"] & evaluated["exact"])),
                      newly_failed_associations=int(np.sum(reference["exact"] & ~evaluated["exact"])))
    else:
        result.update(corrected_output_bits=0,newly_incorrect_output_bits=0,
                      improved_associations=0,newly_failed_associations=0)
    return result


def main_sweep():
    rows=[]; margins=[]; associations=[]; snapshots={}; one_update=[]
    archive=PROJECT/"my_research/results/diagnosis_multipair_crosstalk/full_20261007_111051_279590"
    archived={}
    if (archive/"pair_count_summary.csv").exists():
        with (archive/"pair_count_summary.csv").open(newline="") as stream:
            archived={int(r["k"]):r for r in csv.DictReader(stream)}
    checks=[]
    for k in K_VALUES:
        subsets=subsets_for(k)
        corr=np.einsum("nki,nkj->nij",XB[subsets],YB[subsets])
        corrscores=np.einsum("nki,nij->nkj",XB[subsets],corr)
        corrpred=np.where(corrscores>0,1,-1)
        additive,add_clips=train_batch(subsets,0.0)
        add_eval=evaluate(subsets,additive)
        if k in archived:
            old=int(archived[k]["B_exact_associations"])
            now=int(np.sum(add_eval["exact"]))
            assert old==now,(k,old,now)
            selected={1:(1,),3:(0,1,2),5:(0,1,2,3,4),10:tuple(range(10))}[k]
            old_path=archive/"selected_weights"/f"K{k}_{'-'.join(map(str,selected))}"/"weights_learned.npy"
            index=next(i for i,idx in enumerate(subsets) if tuple(idx)==selected)
            old_weight=np.load(old_path)
            weight_match=bool(np.array_equal(old_weight,additive[index]))
            assert weight_match,(k,old_path)
            checks.append(dict(k=k,archived_additive_exact=old,math_additive_exact=now,
                               exact_match=True,selected_subset="-".join(map(str,selected)),
                               selected_raw_weight_exact_match=weight_match,
                               selected_raw_weight_max_abs_difference=float(np.max(np.abs(old_weight-additive[index])))))
        for mu in MU_VALUES:
            if mu==0:
                weight,clips=additive,add_clips
                ev=add_eval
            else:
                weight,clips=train_batch(subsets,mu)
                ev=evaluate(subsets,weight)
            item=aggregate(k,mu,"mu_sweep","original_B2_schedule",subsets,weight,clips,ev,add_eval)
            item["original_correlation_exact_accuracy_percent"]=100*float(np.mean(np.all(corrpred==YB[subsets],axis=2)))
            item["exact_change_vs_additive_pp"]=item["exact_accuracy_percent"]-100*float(np.mean(add_eval["exact"]))
            rows.append(item)
            if mu in (0.0,0.5,1.0,2.0):
                ow,oc=train_batch(subsets,mu,epochs=1)
                oe=evaluate(subsets,ow)
                one_update.append(aggregate(k,mu,"one_update","ascending_once",subsets,ow,oc,oe))
            for n,idx in enumerate(subsets):
                sub="-".join(map(str,idx))
                for local,digit in enumerate(idx):
                    associations.append(dict(k=k,subset=sub,digit=int(digit),letter=chr(65+digit),mu=mu,
                        target="".join(map(str,(YB[digit]>0).astype(int))),
                        prediction="".join(map(str,(ev["pred"][n,local]>0).astype(int))),
                        additive_prediction="".join(map(str,(add_eval["pred"][n,local]>0).astype(int))),
                        exact=int(ev["exact"][n,local]),additive_exact=int(add_eval["exact"][n,local]),
                        bit_accuracy_percent=100*float(np.mean(ev["correct"][n,local])),
                        corrected_output_bits=int(np.sum(~add_eval["correct"][n,local]&ev["correct"][n,local])),
                        newly_incorrect_output_bits=int(np.sum(add_eval["correct"][n,local]&~ev["correct"][n,local])),
                        minimum_signed_margin=float(np.min(ev["margins"][n,local]))))
                    for j in range(15):
                        margins.append(dict(k=k,subset=sub,digit=int(digit),output_idx=j,mu=mu,
                            target_sign=int(YB[digit,j]),
                            additive_score=float(add_eval["scores"][n,local,j]),
                            weight_dependent_score=float(ev["scores"][n,local,j]),
                            raw_weight_score=float(ev["raw_scores"][n,local,j]),
                            additive_signed_margin=float(add_eval["margins"][n,local,j]),
                            signed_output_margin=float(ev["margins"][n,local,j]),
                            margin_change=float(ev["margins"][n,local,j]-add_eval["margins"][n,local,j]),
                            additive_correct=int(add_eval["correct"][n,local,j]),
                            weight_dependent_correct=int(ev["correct"][n,local,j]),
                            corrected=int(not add_eval["correct"][n,local,j] and ev["correct"][n,local,j]),
                            newly_incorrect=int(add_eval["correct"][n,local,j] and not ev["correct"][n,local,j])))
            snapshots[(k,mu)]=dict(subsets=subsets.copy(),weight=weight.copy(),clips=clips.copy(),
                                    scores=ev["scores"].copy(),margins=ev["margins"].copy(),
                                    exact=ev["exact"].copy(),saturation=ev["saturation"].copy(),
                                    correlation=corr.copy())
            print(f"main sweep K={k} mu={mu:g} exact={item['exact_accuracy_percent']:.2f}%",flush=True)
        np.savez_compressed(NPZ/f"main_weights_K{k}.npz",
            subsets=subsets,mus=np.array(MU_VALUES),W_correlation=corr,
            W_initial=np.full_like(additive,INIT),
            W_raw=np.stack([snapshots[k,mu]["weight"] for mu in MU_VALUES]),
            W_effective=np.stack([snapshots[k,mu]["weight"]-INIT for mu in MU_VALUES]),
            scores=np.stack([snapshots[k,mu]["scores"] for mu in MU_VALUES]),
            signed_margins=np.stack([snapshots[k,mu]["margins"] for mu in MU_VALUES]),
            clip_events=np.stack([snapshots[k,mu]["clips"] for mu in MU_VALUES]))
    save_csv(CSV/"baseline_reproduction.csv",checks)
    save_csv(CSV/"accuracy_results.csv",rows)
    save_csv(CSV/"one_update_results.csv",one_update)
    save_csv(CSV/"association_results.csv",associations)
    save_csv(CSV/"output_margins.csv",margins)
    return rows,snapshots,checks,associations,margins


def unique_orders(k, subset_index, count=20):
    if k<=3:
        return list(itertools.permutations(range(k)))
    base=[tuple(range(k)),tuple(reversed(range(k)))]
    seen=set(base)
    rng=np.random.default_rng(ORDER_SEED+1000*k+subset_index)
    while len(base)<count:
        candidate=tuple(int(x) for x in rng.permutation(k))
        if candidate not in seen:
            base.append(candidate);seen.add(candidate)
    return base


def learning_order_study(snapshots):
    order_rows=[];aggregate_rows=[];example_rows=[]
    for k in K_VALUES:
        subsets=subsets_for(k)
        orders=[unique_orders(k,n) for n in range(len(subsets))]
        order_count=len(orders[0]);assert all(len(x)==order_count and len(set(x))==order_count for x in orders)
        expanded=np.repeat(subsets,order_count,axis=0)
        schedules=np.array([np.tile(order,(EPOCHS,1)) for per_subset in orders for order in per_subset])
        for mu in MU_VALUES:
            weight,clips=train_batch(expanded,mu,schedule=schedules)
            ev=evaluate(expanded,weight)
            view_w=weight.reshape(len(subsets),order_count,15,15)
            view_score=ev["scores"].reshape(len(subsets),order_count,k,15)
            view_margin=ev["margins"].reshape(len(subsets),order_count,k,15)
            view_exact=ev["exact"].reshape(len(subsets),order_count,k)
            view_bit=ev["correct"].reshape(len(subsets),order_count,k,15)
            baseline=view_w[:,0:1]
            wdiff=np.max(np.abs(view_w-baseline),axis=(2,3))
            np.savez_compressed(NPZ/f"order_weights_K{k}_mu{mu:g}.npz",
                subsets=subsets,orders=np.array(orders),mu=mu,W_raw=view_w,
                W_effective=view_w-INIT,scores=view_score,signed_margins=view_margin,
                exact=view_exact,correct_bits=view_bit,clip_events=clips.reshape(len(subsets),order_count))
            rates=np.mean(view_exact,axis=2)
            bitrates=np.mean(view_bit,axis=(2,3))
            for n,idx in enumerate(subsets):
                sub="-".join(map(str,idx))
                for oi,order in enumerate(orders[n]):
                    label="ascending" if oi==0 else "descending" if oi==1 else "permutation"
                    order_rows.append(dict(k=k,subset=sub,mu=mu,order_index=oi,order_type=label,
                        order_digits="-".join(str(idx[p]) for p in order),epochs=EPOCHS,
                        exact_count=int(np.sum(view_exact[n,oi])),association_count=k,
                        exact_accuracy_percent=100*float(rates[n,oi]),
                        bit_accuracy_percent=100*float(bitrates[n,oi]),
                        mean_signed_margin=float(np.mean(view_margin[n,oi])),
                        minimum_signed_margin=float(np.min(view_margin[n,oi])),
                        maximum_weight_difference_from_ascending=float(wdiff[n,oi]),
                        differing_synapses_from_ascending=int(np.count_nonzero(np.abs(view_w[n,oi]-view_w[n,0])>1e-10)),
                        clipped_update_events=int(clips[n*order_count+oi])))
            aggregate_rows.append(dict(k=k,mu=mu,subset_count=len(subsets),unique_orders_per_subset=order_count,
                order_evaluations=len(subsets)*order_count,
                ascending_exact_accuracy_percent=100*float(np.mean(rates[:,0])),
                descending_exact_accuracy_percent=100*float(np.mean(rates[:,1])) if order_count>1 else 100*float(np.mean(rates[:,0])),
                mean_exact_accuracy_percent=100*float(np.mean(rates)),
                overall_exact_std_percent=100*float(np.std(rates)),
                mean_within_subset_exact_std_percent=100*float(np.mean(np.std(rates,axis=1))),
                mean_within_subset_exact_range_percent=100*float(np.mean(np.ptp(rates,axis=1))),
                mean_bit_accuracy_percent=100*float(np.mean(bitrates)),
                mean_within_subset_bit_std_percent=100*float(np.mean(np.std(bitrates,axis=1))),
                fraction_order_runs_with_weight_difference=float(np.mean(wdiff>1e-10)),
                maximum_weight_difference=float(np.max(wdiff))))
            # One concrete non-ascending run with largest exact-rate difference.
            difference=np.abs(rates-rates[:,0:1]); n,oi=np.unravel_index(np.argmax(difference),difference.shape)
            if oi==0 and order_count>1:
                n,oi=np.unravel_index(np.argmax(wdiff[:,1:]),wdiff[:,1:].shape);oi+=1
            for i in range(15):
                for j in range(15):
                    before=float(view_w[n,0,i,j]);after=float(view_w[n,oi,i,j])
                    example_rows.append(dict(k=k,mu=mu,subset="-".join(map(str,subsets[n])),
                        comparison_order_index=int(oi),comparison_order="-".join(str(subsets[n,p]) for p in orders[n][oi]),
                        ascending_exact_percent=100*float(rates[n,0]),comparison_exact_percent=100*float(rates[n,oi]),
                        input_idx=i,output_idx=j,ascending_raw_weight=before,comparison_raw_weight=after,
                        difference=after-before))
            print(f"order study K={k} mu={mu:g} orders={order_count}",flush=True)
    save_csv(CSV/"learning_order_results.csv",order_rows)
    save_csv(CSV/"learning_order_summary.csv",aggregate_rows)
    save_csv(CSV/"learning_order_synapse_examples.csv",example_rows)
    return aggregate_rows


def score_setting(rows,mu,stage,setting):
    selected=[r for r in rows if r["mu"]==mu and r["stage"]==stage and r["setting"]==setting and r["k"] in (3,5,7,10)]
    assert len(selected)==4
    return float(np.mean([r["exact_accuracy_percent"] for r in selected]))


def parameter_study(main_rows):
    # Fixed staged candidate set; selection uses equal-K mean exact accuracy.
    base=np.mean([r["exact_accuracy_percent"] for r in main_rows if r["mu"]==0 and r["k"] in (3,5,7,10)])
    scored=[]
    for mu in MU_VALUES[1:]:
        value=np.mean([r["exact_accuracy_percent"] for r in main_rows if r["mu"]==mu and r["k"] in (3,5,7,10)])
        scored.append((float(value-base),mu))
    promising=[mu for gain,mu in sorted(scored,key=lambda x:(-x[0],x[1])) if gain>0][:2]
    if not promising:
        save_csv(CSV/"parameter_selection.csv",[dict(stage="mu_selection",baseline_equal_K_mean=base,
                   promising_mu="none",reason="No positive equal-K improvement")])
        return [],[],promising
    chosen=[0.0]+promising
    selection=[];rows=[];association_rows=[]
    selection.append(dict(stage="mu_selection",baseline_equal_K_mean_percent=base,
                          selected_mu=";".join(map(str,promising)),rule="top two positive equal-K mean exact gains across K=3,5,7,10"))

    def run(stage,setting,mu,a_plus,a_minus,lo,hi):
        per_k=[]
        for k in (3,5,7,10):
            subsets=subsets_for(k)
            weight,clips=train_batch(subsets,mu,a_plus,a_minus,lo,hi)
            ev=evaluate(subsets,weight,lo,hi)
            item=aggregate(k,mu,stage,setting,subsets,weight,clips,ev,
                           a_plus=a_plus,a_minus=a_minus,lo=lo,hi=hi)
            rows.append(item);per_k.append(item)
            for n,idx in enumerate(subsets):
                for local,digit in enumerate(idx):
                    association_rows.append(dict(stage=stage,setting=setting,k=k,mu=mu,
                        subset="-".join(map(str,idx)),digit=int(digit),exact=int(ev["exact"][n,local]),
                        bit_accuracy_percent=100*float(np.mean(ev["correct"][n,local])),
                        prediction="".join(map(str,(ev["pred"][n,local]>0).astype(int))),
                        target="".join(map(str,Y01[digit]))))
            if k==10:
                np.savez_compressed(NPZ/f"parameter_K10_{stage}_{setting}_mu{mu:g}.npz",
                    subsets=subsets,mu=mu,W_initial=np.full_like(weight,INIT),W_raw=weight,
                    W_effective=weight-INIT,scores=ev["scores"],margins=ev["margins"],clip_events=clips)
        return float(np.mean([r["exact_accuracy_percent"] for r in per_k]))

    lr_factors=(0.5,1.0,2.0)
    lr_scores=[]
    for factor in lr_factors:
        setting=f"rate_{factor:g}x"
        for mu in chosen:
            value=run("learning_rate",setting,mu,BASE_A*factor,BASE_A*factor,BASE_LO,BASE_HI)
            if mu in promising:lr_scores.append((value,mu,factor))
    best_lr=max(lr_scores,key=lambda x:(x[0],-x[2],-x[1]))
    selected_factor=best_lr[2]
    selection.append(dict(stage="learning_rate_selection",baseline_equal_K_mean_percent=base,
                          selected_mu=best_lr[1],selected_factor=selected_factor,
                          rule="max equal-K mean exact across selected mu and rate factors; tie lower factor"))

    bounds=(0.75,1.5,3.0)
    bound_scores=[]
    for bound in bounds:
        setting=f"symmetric_bound_{bound:g}"
        for mu in chosen:
            value=run("weight_bound",setting,mu,BASE_A*selected_factor,BASE_A*selected_factor,-bound,bound)
            if mu in promising:bound_scores.append((value,mu,bound))
    best_bound=max(bound_scores,key=lambda x:(x[0],-x[2],-x[1]))
    selected_bound=best_bound[2]
    selection.append(dict(stage="weight_bound_selection",baseline_equal_K_mean_percent=base,
                          selected_mu=best_bound[1],selected_factor=selected_factor,
                          selected_bound=selected_bound,
                          rule="max equal-K mean exact among selected mu; tie tighter bound"))

    # Ratio is conditional on a positive gain over the matched additive control.
    compare_setting=f"symmetric_bound_{selected_bound:g}"
    matched_additive=score_setting(rows,0.0,"weight_bound",compare_setting)
    matched_best=max(score_setting(rows,mu,"weight_bound",compare_setting) for mu in promising)
    if matched_best>matched_additive:
        for ratio in (0.8,1.0,1.2):
            setting=f"ratio_{ratio:g}"
            plus=BASE_A*selected_factor*np.sqrt(ratio)
            minus=BASE_A*selected_factor/np.sqrt(ratio)
            for mu in chosen:
                run("ltp_ltd_ratio",setting,mu,plus,minus,-selected_bound,selected_bound)
        selection.append(dict(stage="ratio_test",baseline_equal_K_mean_percent=matched_additive,
                              selected_mu=";".join(map(str,promising)),selected_factor=selected_factor,
                              selected_bound=selected_bound,rule="ratios 0.8,1.0,1.2; A+*A- fixed"))
    else:
        selection.append(dict(stage="ratio_test_skipped",baseline_equal_K_mean_percent=matched_additive,
                              selected_mu=";".join(map(str,promising)),selected_factor=selected_factor,
                              selected_bound=selected_bound,rule="no matched additive improvement"))
    save_csv(CSV/"parameter_selection.csv",selection)
    save_csv(CSV/"parameter_sweep_results.csv",rows)
    save_csv(CSV/"parameter_association_results.csv",association_rows)
    return rows,selection,promising


def plots(main_rows,snapshots,order_rows,param_rows,promising):
    by={(r["k"],r["mu"]):r for r in main_rows}
    ks=(3,5,7,10)
    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for ax,k in zip(axes.flat,ks):
        ax.plot(MU_VALUES,[by[k,mu]["exact_accuracy_percent"] for mu in MU_VALUES],"o-",label="Weight-dependent STDP")
        ax.axhline(by[k,0.0]["exact_accuracy_percent"],color="tab:orange",ls="--",label="Additive STDP")
        ax.axhline(by[k,0.0]["original_correlation_exact_accuracy_percent"],color="grey",ls=":",label="Original Correlation")
        ax.set(title=f"Number of Stored Associations = {k}",xlabel="Weight Dependence Exponent",ylabel="Exact Recall Accuracy (%)",ylim=(0,105));ax.legend(fontsize=7)
    fig.suptitle("Exact Recall Accuracy vs Weight Dependence Exponent; Original 60-Epoch Schedule")
    fig.savefig(FIG/"exact_accuracy_vs_weight_dependence_exponent.png",dpi=160);plt.close(fig)

    candidate=promising[0] if promising else 1.0
    pos=np.arange(len(ks));fig,ax=plt.subplots(figsize=(9,4))
    ax.bar(pos-.2,[by[k,0]["original_correlation_exact_accuracy_percent"] for k in ks],.2,label="Original Correlation")
    ax.bar(pos,[by[k,0]["exact_accuracy_percent"] for k in ks],.2,label="Additive STDP")
    ax.bar(pos+.2,[by[k,candidate]["exact_accuracy_percent"] for k in ks],.2,label=f"Weight-dependent STDP (μ={candidate:g})")
    ax.set_xticks(pos,ks);ax.set(xlabel="Number of Stored Associations",ylabel="Exact Recall Accuracy (%)",title="Original Correlation vs Matched Bounded STDP")
    ax.legend();fig.tight_layout();fig.savefig(FIG/"additive_vs_weight_dependent_accuracy.png",dpi=160);plt.close(fig)

    for stage,parameter,label,file in [("learning_rate","a_plus","Learning Rate (Effective Update Magnitude)","accuracy_vs_learning_rate.png"),
                                        ("weight_bound","bound_max","Weight Bound (Symmetric Magnitude)","accuracy_vs_weight_bound.png")]:
        filtered=[r for r in param_rows if r["stage"]==stage]
        if not filtered:continue
        fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
        for ax,k in zip(axes.flat,ks):
            for mu in (0.0,candidate):
                rs=sorted([r for r in filtered if r["k"]==k and r["mu"]==mu],key=lambda x:x[parameter])
                ax.plot([r[parameter] for r in rs],[r["exact_accuracy_percent"] for r in rs],"o-",label="Additive STDP" if mu==0 else f"Weight-dependent STDP μ={mu:g}")
            ax.set(title=f"Number of Stored Associations = {k}",xlabel=label,ylabel="Exact Recall Accuracy (%)",ylim=(0,105));ax.legend(fontsize=7)
        fig.suptitle(f"Exact Recall Accuracy vs {label}; Other Parameters Fixed by Stage")
        fig.savefig(FIG/file,dpi=160);plt.close(fig)

    order_by={(r["k"],r["mu"]):r for r in order_rows}
    fig,ax=plt.subplots(figsize=(9,5))
    for mu in (0.0,candidate):
        means=[order_by[k,mu]["mean_exact_accuracy_percent"] for k in K_VALUES]
        spread=[order_by[k,mu]["mean_within_subset_exact_std_percent"] for k in K_VALUES]
        ax.errorbar(K_VALUES,means,yerr=spread,marker="o",capsize=4,
                    label="Additive STDP" if mu==0 else f"Weight-dependent STDP μ={mu:g}")
    ax.set(xlabel="Number of Stored Associations",ylabel="Exact Recall Accuracy (%)",
           title="Learning Order Sensitivity: Mean and Mean Within-Subset Standard Deviation",ylim=(0,105))
    ax.legend();fig.tight_layout();fig.savefig(FIG/"learning_order_sensitivity.png",dpi=160);plt.close(fig)

    for k in (3,5,7,10):
        subset=tuple(range(k));n=np.where(np.all(snapshots[k,0]["subsets"]==subset,axis=1))[0][0]
        corr=snapshots[k,0]["correlation"][n]
        additive=snapshots[k,0]["weight"][n]-INIT
        dependent=snapshots[k,candidate]["weight"][n]-INIT
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
        for ax,arr,title in zip(axes,(corr,additive,dependent),
                                ("Original Correlation","Additive STDP Effective Weight",f"Weight-dependent STDP Effective Weight (μ={candidate:g})")):
            vmax=max(float(np.max(np.abs(arr))),1e-12)
            im=ax.imshow(arr,cmap="coolwarm",vmin=-vmax,vmax=vmax,aspect="auto")
            ax.set(title=title,xlabel="Output Neuron Index",ylabel="Input Neuron Index")
            fig.colorbar(im,ax=ax,fraction=.046)
        fig.suptitle(f"Weight Matrices; K={k}, Prefix Subset, Original 60-Epoch Schedule")
        fig.savefig(FIG/f"weight_heatmap_K{k}.png",dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for ax,k in zip(axes.flat,ks):
        a=snapshots[k,0]["weight"].ravel()
        b=snapshots[k,candidate]["weight"].ravel()
        ax.hist(a,bins=35,alpha=.55,label="Additive STDP")
        ax.hist(b,bins=35,alpha=.55,label=f"Weight-dependent STDP μ={candidate:g}")
        ax.set(title=f"Number of Stored Associations = {k}",xlabel="Raw Synaptic Weight",ylabel="Synapse Count")
        ax.legend(fontsize=7)
    fig.suptitle("Weight Distribution and Saturation; Original Bounds ±1.5")
    fig.savefig(FIG/"weight_distribution_and_saturation.png",dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for ax,k in zip(axes.flat,ks):
        ax.plot(MU_VALUES,[by[k,mu]["corrected_output_bits"] for mu in MU_VALUES],"o-",label="Corrected Output Bits")
        ax.plot(MU_VALUES,[by[k,mu]["newly_incorrect_output_bits"] for mu in MU_VALUES],"s--",label="Newly Incorrect Output Bits")
        ax.set(title=f"Number of Stored Associations = {k}",xlabel="Weight Dependence Exponent",ylabel="Output Bit Count")
        ax.legend(fontsize=7)
    fig.suptitle("Output Bit Changes Relative to Matched Additive STDP")
    fig.savefig(FIG/"corrected_and_newly_incorrect_bits.png",dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for ax,k in zip(axes.flat,ks):
        before=snapshots[k,0]["margins"].ravel()
        after=snapshots[k,candidate]["margins"].ravel()
        ax.scatter(before,after,s=4,alpha=.25)
        lo=min(before.min(),after.min());hi=max(before.max(),after.max())
        ax.plot([lo,hi],[lo,hi],color="black",ls="--",label="Unchanged Margin")
        ax.set(title=f"Number of Stored Associations = {k}",xlabel="Additive Signed Output Margin",ylabel=f"Weight-dependent Signed Output Margin (μ={candidate:g})")
    fig.suptitle("Signed Output Margin Comparison Across All Stored Associations")
    fig.savefig(FIG/"signed_output_margin_comparison.png",dpi=160);plt.close(fig)
    return candidate


def report(baseline,main_rows,order_rows,param_rows,selection,promising,checks,associations,candidate):
    lookup={(r["k"],r["mu"]):r for r in main_rows}
    result_table="\n".join(f"| {k} | {baseline[k]['exact_percent']:.2f} | {lookup[k,0]['exact_accuracy_percent']:.2f} | {lookup[k,candidate]['exact_accuracy_percent']:.2f} | {lookup[k,candidate]['bit_accuracy_percent']:.2f} | {lookup[k,0]['weight_saturation_percent']:.2f} | {lookup[k,candidate]['weight_saturation_percent']:.2f} |" for k in K_VALUES)
    selected_table="\n".join(f"| {k} | {max((r for r in main_rows if r['k']==k),key=lambda r:r['exact_accuracy_percent'])['mu']:g} | {max(r['exact_accuracy_percent'] for r in main_rows if r['k']==k):.2f} |" for k in K_VALUES)
    order_lookup={(r["k"],r["mu"]):r for r in order_rows}
    order_table="\n".join(f"| {k} | {order_lookup[k,0]['mean_within_subset_exact_std_percent']:.3f} | {order_lookup[k,candidate]['mean_within_subset_exact_std_percent']:.3f} | {order_lookup[k,0]['fraction_order_runs_with_weight_difference']:.3f} | {order_lookup[k,candidate]['fraction_order_runs_with_weight_difference']:.3f} |" for k in K_VALUES)
    b10=lookup[10,0];d10=lookup[10,candidate]
    improved=[r for r in associations if r["k"]==10 and r["mu"]==candidate and r["exact"] and not r["additive_exact"]]
    worsened=[r for r in associations if r["k"]==10 and r["mu"]==candidate and not r["exact"] and r["additive_exact"]]
    tradeoff_rows=[]
    for k in (3,5,7,10):
        subset_cases=[r for r in associations if r["k"]==k and r["mu"]==candidate]
        gained=[r for r in subset_cases if r["exact"] and not r["additive_exact"]]
        lost=[r for r in subset_cases if not r["exact"] and r["additive_exact"]]
        tradeoff_rows.append(dict(k=k,improved_count=len(gained),worsened_count=len(lost),
                                  improved_digits=sorted(set(r["digit"] for r in gained)),
                                  worsened_digits=sorted(set(r["digit"] for r in lost))))
    tradeoff_table="\n".join(f"| {r['k']} | {r['improved_count']} | {', '.join(map(str,r['improved_digits'])) or 'none'} | {r['worsened_count']} | {', '.join(map(str,r['worsened_digits'])) or 'none'} |" for r in tradeoff_rows)
    parameter_best=None
    if param_rows:
        groups=defaultdict(list)
        for row in param_rows:groups[(row["stage"],row["setting"],row["mu"])].append(row)
        candidates=[]
        for key,rows in groups.items():
            if key[2]==0:continue
            if len(rows)==4:candidates.append((float(np.mean([r["exact_accuracy_percent"] for r in rows])),key))
        parameter_best=max(candidates,default=None)
    parameter_comparison=[]
    if param_rows:
        grouped=defaultdict(dict)
        for row in param_rows:
            grouped[(row["stage"],row["setting"])][(row["mu"],row["k"])]=row
        for (stage,setting),items in sorted(grouped.items()):
            additive_mean=float(np.mean([items[0.0,k]["exact_accuracy_percent"] for k in (3,5,7,10)]))
            for mu in promising:
                value=float(np.mean([items[mu,k]["exact_accuracy_percent"] for k in (3,5,7,10)]))
                parameter_comparison.append(dict(stage=stage,setting=setting,mu=mu,
                    additive_equal_K_mean_percent=additive_mean,dependent_equal_K_mean_percent=value,
                    change_percentage_points=value-additive_mean,
                    additive_K10_exact_count=items[0.0,10]["exact_count"],
                    dependent_K10_exact_count=items[mu,10]["exact_count"]))
    save_csv(CSV/"parameter_matched_comparison.csv",parameter_comparison)
    full_source=PROJECT/"Ver관리/2. temporal/stdp_temporal_symmetric_teacher_only.py"
    text=f"""# Weight-dependent STDP mathematical experiment

**Scope.** Feedforward 15×15, current digit 0–9→letter A–J patterns, bipolar representation, static `score=X_b@(W_raw−0.05)` and `score>0` → +1. No LIF or noisy cues. This is a **mathematical sequential update simulation**, not a run of the original spike-event STDP simulator. Original unbounded correlation is reported separately. K=1,2,3,5,7,10 use all 10,45,120,252,120,1 subsets respectively. Every result is association-weighted within K. Files in `csv/` and `npz/` preserve raw scores, margins, weights, subset and order identifiers.

## Timing and baseline verification

Original B2 `encode()` and teacher time 4.5 ms were called for all 2,250 synapses in the dataset: pre 2 ms gives LTP, pre 7 ms LTD, and every direction equals `x_b*y_b`. Details: `csv/timing_direction_validation.csv`. The original trainer uses `A±=0.02`, `tau_STDP=2 ms`, hence event magnitude `A=0.02 exp(-2.5/2)={BASE_A:.12f}`, 60 epochs, initial raw W=0.05, bounds [-1.5,1.5], and `np.random.seed(0)` reset for each output column. The mathematical simulator uses exactly the same per-epoch permutation sequence and teacher-event direction, while skipping membrane dynamics that do not enter the original updater's selected teacher event.

Pure correlation `X_b.T@Y_b` reproduced exact accuracy {', '.join(f'K={k}: {baseline[k]["exact_percent"]:.2f}%' for k in K_VALUES)}. At μ=0, the bounded sequential rule is the matched Additive STDP control; archived full-sweep exact counts matched for K=1,3,5,10: {checks}. K=2 and K=7 have no archived B2 full-sweep reference in that run. Raw W and effective ΔW are separate arrays in `npz/main_weights_K*.npz`. The primary static readout uses ΔW to remove the uniform 0.05 offset; raw-W readout accuracy is recorded as a sensitivity comparison. One update per association is separately evaluated in `csv/one_update_results.csv`; it is not conflated with 60-epoch results.

## Original schedule, matched bounds and learning magnitude

| K | Original Correlation Exact (%) | Additive STDP Exact (%) | Weight-dependent STDP μ={candidate:g} Exact (%) | μ={candidate:g} Bit (%) | Additive Saturation (%) | μ={candidate:g} Saturation (%) |
|---:|---:|---:|---:|---:|---:|---:|
{result_table}

Best observed μ for each K within the prespecified sweep (exploratory, same data):

| K | Best observed μ | Exact Recall Accuracy (%) |
|---:|---:|---:|
{selected_table}

`csv/accuracy_results.csv` records every μ, clipping event count, bounded weight distribution, bit accuracy and corrected/new output bits against μ=0. `csv/association_results.csv` names every improved/worsened digit for each subset; `csv/output_margins.csv` gives all output-bit signed margins and predictions. A positive signed margin alone is not used as the tie decision: score zero is decoded −1 exactly as in the original correlation baseline.

Exact-association tradeoffs for μ={candidate:g} at the original schedule (counts refer to subset–digit evaluations, so a digit may occur in multiple subsets):

| K | Newly successful evaluations | Digit IDs | Newly failed evaluations | Digit IDs |
|---:|---:|---|---:|---|
{tradeoff_table}

## Learning order sensitivity

K≤3: all unique association orders. K=5,7,10: exactly 20 unique orders per subset, including ascending and descending, with the remaining orders generated by seed {ORDER_SEED}. Each fixed order is repeated for 60 epochs; this differs intentionally from the original reshuffled-per-epoch schedule. All μ values, including μ=0, use the same orders and epochs. Full W and per-bit signed margins for every order are in `npz/order_weights_K*_mu*.npz`; `csv/learning_order_results.csv` and `csv/learning_order_summary.csv` give accuracy, mean/min margins, clipping and differences from ascending. A representative 225-synapse order comparison per K and μ is in `csv/learning_order_synapse_examples.csv`.

| K | Additive Mean Within-Subset Exact SD (pp) | μ={candidate:g} Mean Within-Subset Exact SD (pp) | Additive Runs with Changed W | μ={candidate:g} Runs with Changed W |
|---:|---:|---:|---:|---:|
{order_table}

## Staged parameter study

Selection criterion is the unweighted mean of exact accuracy percentages at K=3,5,7,10, so K=10 is not the sole driver. Positive-gain μ values selected from the prespecified sweep: {promising or 'none'}. Equal A± learning-rate factors 0.5,1,2 were tested first at original bounds; then symmetric bounds ±0.75,±1.5,±3 at the selected rate. LTP/LTD ratio 0.8,1,1.2 was tested only if the selected equal-rate setting beat its **matched μ=0 additive** control; geometric mean of A+ and A− was held fixed. `csv/parameter_selection.csv` records every stage and `csv/parameter_sweep_results.csv` preserves all tested K values, including μ=0 controls. This is same-dataset exploratory selection, not a general optimum. Highest equal-K mean among tested parameter conditions: {parameter_best}.

`csv/parameter_matched_comparison.csv` places every tested weight-dependent condition beside its μ=0 control at the **same** rate, bounds and A+/A− ratio. The ratio 0.8 setting changes the additive baseline itself; a K=10 success under that ratio is not evidence that weight dependence alone caused it.

## Direct answers and limitations

1. **Additive baseline:** μ=0 matches the bounded additive rule and archived B2 static exact counts where available. Pure unbounded correlation matches all supplied reference rates.
2. **Accuracy improvement:** See table above. Any μ improvement must be compared with μ=0 at the same rate, bounds, initial W and schedule; original correlation is a separate unbounded reference.
3. **Effective μ:** The sweep and staged comparisons identify observed values, without validation on held-out patterns.
4. **Saturation:** The table and `accuracy_results.csv` measure terminal-bound ratio and attempted clipped updates separately. Lower saturation alone does not establish better recall.
5. **Order:** The order table gives within-subset variability and fraction of runs whose W differs from ascending; a best order is not treated as an improvement claim.
6. **Association tradeoffs at K=10, μ={candidate:g}:** improved exact digits {[r['digit'] for r in improved]}, worsened exact digits {[r['digit'] for r in worsened]}; subset-level tradeoffs at K<10 are in `association_results.csv`.
7. **K=10:** matched additive {b10['exact_count']}/10 versus μ={candidate:g} {d10['exact_count']}/10 at original settings; this is one extra exact digit in the only K=10 subset, not evidence of a statistically reliable or broadly meaningful gain. Parameter-stage K=10 results are all listed in `parameter_sweep_results.csv`.
8. **Simulator feasibility:** This rule is synapse-local and retains timing-based direction, but the original `stdp_update` magnitude expression at `{full_source.relative_to(PROJECT)}` would have to change. The simulator's natural spikes, teacher timing, and clipping could interact with that rule; this mathematical accuracy does not prove actual spike-event performance or biological plausibility. No original file was changed.
"""
    (OUT/"summary.md").write_text(text,encoding="utf-8")


def main():
    for folder in (CSV,NPZ,FIG):folder.mkdir(parents=True,exist_ok=False)
    baseline=verify_timing_and_dataset()
    main_rows,snapshots,checks,associations,margins=main_sweep()
    order_rows=learning_order_study(snapshots)
    param_rows,selection,promising=parameter_study(main_rows)
    candidate=plots(main_rows,snapshots,order_rows,param_rows,promising)
    source_paths=[PROJECT/"binary_character.py",SOURCE/"stdp_temporal_symmetric_teacher_only.py",
                  SOURCE/"b2_full_output.py",SOURCE/"correlation_pair_count_sweep.py"]
    config=dict(status="success",command="python my_research/experiments/diagnostics/weight_dependent_stdp.py",
        run_directory=OUT.name,python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,
        git_revision=subprocess.check_output(["git","-C",str(PROJECT),"rev-parse","HEAD"],text=True).strip(),
        source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in source_paths},
        experiment_script_sha256=digest(Path(__file__)),
        dataset=dict(digit_patterns=X01.tolist(),letter_patterns=Y01.tolist(),mapping=[f"{i}->{chr(65+i)}" for i in range(10)]),
        representation="bipolar 2*x-1; W[input_idx,output_idx]",
        counts=K_VALUES,mu_values=MU_VALUES,order_seed=ORDER_SEED,
        original_stdp=dict(initial_raw_weight=INIT,bounds=[BASE_LO,BASE_HI],epochs=EPOCHS,
            A_plus=original.A_PLUS,A_minus=original.A_MINUS,tau_stdp_ms=original.TAU_STDP,
            early_pre_ms=original.T_ON,late_pre_ms=original.T_OFF,teacher_ms=original.T_TEACH,
            effective_one_event_magnitude=BASE_A,original_per_epoch_order_seed=0),
        primary_readout="XB @ (W_raw - W_INIT) > 0, else -1; zero ties negative",
        raw_weight_readout="XB @ W_raw > 0, separate sensitivity",
        original_schedule="np.random.RandomState(0).permutation(K) once per epoch, same sequence for all output columns",
        order_study="fixed unique permutation repeated every epoch; K<=3 exhaustive, K>=5 20 unique per subset",
        one_update="one ascending presentation per association, separate exploratory control",
        parameter_stages="mu; equal-rate factors [0.5,1,2]; symmetric bounds [0.75,1.5,3]; ratio [0.8,1,1.2] conditional")
    (OUT/"config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    shutil.copyfile(__file__,OUT/"run_weight_dependent_stdp.py")
    report(baseline,main_rows,order_rows,param_rows,selection,promising,checks,associations,candidate)
    with zipfile.ZipFile(str(OUT)+".zip","w",zipfile.ZIP_DEFLATED) as z:
        for path in OUT.rglob("*"):
            if path.is_file():z.write(path,path.relative_to(ROOT))
    print(json.dumps(dict(output=str(OUT),zip=str(OUT)+".zip",baseline=baseline,
        selected_mu=promising,figure_mu=candidate,
        original_k10=[r for r in main_rows if r["k"]==10],
        order_k10=[r for r in order_rows if r["k"]==10 and r["mu"] in (0,candidate)],
        parameter_stages=selection),ensure_ascii=False,indent=2))


if __name__=="__main__":
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/"failure.json").write_text(json.dumps(dict(status="failed",error=str(exc),traceback=traceback.format_exc()),indent=2),encoding="utf-8")
        raise
