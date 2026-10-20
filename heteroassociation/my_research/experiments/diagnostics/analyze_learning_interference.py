"""One-step learning benefit, interference, feasible intervals, and sequential controls.

Run from repository root:
  python my_research/experiments/diagnostics/analyze_learning_interference.py
No STDP/LIF source is modified; all recall here is static bipolar matrix recall.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_learning_interference")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_association_strength import (X,Y,G,OUTERS,PROJECT,REF,baseline_check,
                                           load_verified_dataset,unique_orders)

ROOT=PROJECT/"results"/"association_learning_interference"
KS=(3,5,7,10)
GRID=(0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0)
BOUNDS=((0.1,4.0),(0.5,2.0))
EPS=1e-4
ORACLE_NUMERIC_GUARD=1e-9  # interior of exact score-zero preservation boundary
SEED=20261019
POLICIES=("equal","first_strong_rest_weak","negative_conflict_down_eta2",
          "minimum_margin_delta1","retrospective_oracle_all_correct_bits",
          "retrospective_oracle_exact_associations")


def writer(path,fields):
    f=path.open("w",newline="",encoding="utf-8")
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    return f,w


def save_csv(path,rows):
    rows=list(rows)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def rank(a):
    a=np.asarray(a,dtype=float);order=np.argsort(a,kind="mergesort");r=np.empty(len(a))
    i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        r[order[i:j]]=(i+1+j)/2;i=j
    return r


def spearman(x,y):
    a,b=rank(x),rank(y)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def assess(ids,W):
    ids=np.asarray(ids,dtype=int)
    if len(ids)==0:
        z=np.empty((0,15),dtype=float)
        return z,np.empty((0,15),dtype=bool),np.empty(0,dtype=bool)
    scores=X[ids]@W
    margin=Y[ids]*scores
    correct=(scores>0)==(Y[ids]>0)  # -1 target at score=0 is exact-correct
    return margin,correct,np.all(correct,axis=1)


def bound_for_bits(prev_margin,contribution,mask,robust,previous_targets=None):
    if len(prev_margin)==0:return math.inf,0
    adverse=mask & (contribution<0)
    if robust:
        adverse &= prev_margin>EPS
        upper=(prev_margin[adverse]-EPS)/(-contribution[adverse])
    else:
        upper=prev_margin[adverse]/(-contribution[adverse])
        # score=0 is wrong for a +1 target: its exact-preservation limit is open.
        if previous_targets is not None and len(upper):
            positive_target=previous_targets[adverse]>0
            upper=np.where(positive_target,np.nextafter(upper,-math.inf),upper)
    return (float(np.min(upper)) if len(upper) else math.inf,int(np.sum(adverse)))


def step_info(W,previous,p):
    """Return current-only features and retrospective references separately."""
    before=W.copy()
    current_margin=Y[p]*(X[p]@before)
    current_correct=((X[p]@before)>0)==(Y[p]>0)
    required=max(0.,float((EPS-current_margin.min())/15))
    S=OUTERS[p]
    alignment=float(np.sum(before*S))
    assert np.isclose(alignment,float(np.sum(current_margin)))
    neg_conflict=float(np.maximum(0,-before*S).sum())
    norm=float(np.linalg.norm(before))
    total_abs=float(np.abs(before).sum())
    conflict_fraction=neg_conflict/(total_abs+1e-12) if total_abs else 0.
    prev_margin,prev_correct,prev_exact=assess(previous,before)
    if previous:
        contribution=G[np.asarray(previous),p,None]*Y[np.asarray(previous)]*Y[p]
    else:contribution=np.empty((0,15),dtype=float)
    assert contribution.shape==prev_margin.shape
    targets=Y[np.asarray(previous,dtype=int)]
    all_exact,all_n=bound_for_bits(prev_margin,contribution,prev_correct,False,targets)
    all_robust,_=bound_for_bits(prev_margin,contribution,prev_correct,True)
    full_mask=np.broadcast_to(prev_exact[:,None],prev_correct.shape) & prev_correct
    full_exact,full_n=bound_for_bits(prev_margin,contribution,full_mask,False,targets)
    full_robust,_=bound_for_bits(prev_margin,contribution,full_mask,True)
    return dict(before=before,current_margin=current_margin,current_correct=current_correct,
                required=required,alignment=alignment,negative_conflict=neg_conflict,
                conflict_fraction=conflict_fraction,weight_norm=norm,total_abs_weight=total_abs,
                prev_margin=prev_margin,prev_correct=prev_correct,prev_exact=prev_exact,
                contribution=contribution,
                safe_all_exact=all_exact,safe_all_robust=all_robust,
                safe_full_exact=full_exact,safe_full_robust=full_robust,
                adverse_all_count=all_n,adverse_full_count=full_n)


def feasible(required,safe,lo,hi):
    return max(lo,required)<=min(hi,safe)


def oracle_feasible(required,safe,lo,hi):
    """Require a tiny interior interval to make score>0 stable in floating point."""
    guarded=safe-ORACLE_NUMERIC_GUARD if math.isfinite(safe) else safe
    return feasible(required,guarded,lo,hi)


def counterfactual(info,previous,p,alpha):
    assert alpha>0
    W=info["before"]+alpha*OUTERS[p]
    current_after=Y[p]*(X[p]@W)
    prev_after,prev_ok_after,prev_ex_after=assess(previous,W)
    before_m=info["prev_margin"];before_ok=info["prev_correct"];before_ex=info["prev_exact"]
    assert np.allclose(current_after,info["current_margin"]+15*alpha,atol=1e-10)
    assert np.allclose(prev_after,before_m+alpha*info["contribution"],atol=1e-10)
    current_ok=((X[p]@W)>0)==(Y[p]>0)
    return dict(current_margin_after=current_after,current_ok_after=current_ok,
                current_exact_after=bool(np.all(current_ok)),
                current_robust_after=bool(np.all(current_after>=EPS-1e-12)),
                current_corrected_bits=int(np.sum(~info["current_correct"] & current_ok)),
                current_new_errors=int(np.sum(info["current_correct"] & ~current_ok)),
                prev_margin_after=prev_after,prev_ok_after=prev_ok_after,prev_ex_after=prev_ex_after,
                previous_bits_lost=int(np.sum(before_ok & ~prev_ok_after)),
                previous_bits_recovered=int(np.sum(~before_ok & prev_ok_after)),
                previous_associations_forgotten=int(np.sum(before_ex & ~prev_ex_after)),
                previous_associations_recovered=int(np.sum(~before_ex & prev_ex_after)))


def oracle_alpha(info,previous,p,bounds,criterion):
    """Full-history, fixed lexicographic objective; never reads future pairs.

    If robust current learning can preserve the protected previous correct bits,
    choose its smallest feasible alpha. Otherwise minimize protected bit losses,
    then maximize current robust bits, then minimize forgotten exact associations,
    then choose the smallest alpha. This branch uses past targets nontrivially.
    """
    lo,hi=bounds
    full_only=criterion=="exact_associations"
    safe=info["safe_full_exact"] if full_only else info["safe_all_exact"]
    if oracle_feasible(info["required"],safe,lo,hi):
        selected=float(max(lo,info["required"]))
        assert selected<=hi and selected<=safe-ORACLE_NUMERIC_GUARD
        return selected
    thresholds=(EPS-info["current_margin"])/15
    candidates={float(lo),float(hi)}
    for t in thresholds:
        if lo<=t<=hi:candidates.add(float(np.nextafter(t,math.inf)))
    req=max(lo,info["required"])
    if req<=hi:candidates.add(float(req))
    # Loss count changes only at earlier correct-bit crossing strengths.
    prior_mask=np.broadcast_to(info["prev_exact"][:,None],info["prev_correct"].shape) & info["prev_correct"] if full_only else info["prev_correct"]
    adverse=prior_mask & (info["contribution"]<0)
    boundaries=info["prev_margin"][adverse]/(-info["contribution"][adverse])
    for t in boundaries:
        if lo<=t<=hi:
            candidates.add(float(t))
            candidates.add(float(np.nextafter(t,-math.inf)))
            candidates.add(float(np.nextafter(t,math.inf)))
    candidates=[a for a in candidates if lo<=a<=hi and a>0]
    selected=None;best=None
    for a in sorted(candidates):
        after=counterfactual(info,previous,p,a)
        robust_bits=int(np.sum(after["current_margin_after"]>=EPS-1e-12))
        lost=int(np.sum(prior_mask & ~after["prev_ok_after"]))
        objective=(lost,-robust_bits,after["previous_associations_forgotten"],a)
        if best is None or objective<best:best=objective;selected=a
    assert selected is not None
    return selected


def choose_policy(info,previous,p,step,cumulative,policy,bounds):
    lo,hi=bounds
    if policy=="equal":alpha=1.
    elif policy=="first_strong_rest_weak":alpha=2. if step==0 else .5
    elif policy=="negative_conflict_down_eta2":alpha=math.exp(-2*info["conflict_fraction"])
    elif policy=="minimum_margin_delta1":alpha=max(1.,max(0.,(1-info["current_margin"].min())/15))
    elif policy=="retrospective_oracle_all_correct_bits":
        alpha=oracle_alpha(info,previous,p,bounds,"all_correct_bits")
    elif policy=="retrospective_oracle_exact_associations":
        alpha=oracle_alpha(info,previous,p,bounds,"exact_associations")
    else:raise ValueError(policy)
    return float(np.clip(alpha,lo,hi))


def rollout(ids,order,policy,bounds):
    W=np.zeros((15,15),dtype=float)
    previous=[];steps=[];alpha_by_digit=np.zeros(10,dtype=float)
    for step,p in enumerate(order):
        info=step_info(W,previous,p)
        a=choose_policy(info,previous,p,step,float(alpha_by_digit.sum()),policy,bounds)
        effect=counterfactual(info,previous,p,a)
        W+=a*OUTERS[p]
        alpha_by_digit[p]=a
        previous.append(p)
        margins,ok,exact=assess(previous,W)
        steps.append(dict(step=step+1,digit=p,alpha=a,current_exact=int(effect["current_exact_after"]),
                          current_robust=int(effect["current_robust_after"]),
                          previous_bits_lost=effect["previous_bits_lost"],
                          previous_bits_recovered=effect["previous_bits_recovered"],
                          forgotten=effect["previous_associations_forgotten"],
                          recovered=effect["previous_associations_recovered"],
                          exact_count=int(exact.sum()),bit_accuracy=float(ok.mean()),
                          min_margin=float(margins.min()),
                          exact_digits="-".join(str(q) for q,z in zip(previous,exact) if z),
                          required_alpha=info["required"],safe_full_exact=info["safe_full_exact"]))
    final_m,final_ok,final_exact=assess(ids,W)
    return dict(W=W,alpha=alpha_by_digit,steps=steps,exact=final_exact,ok=final_ok,
                margins=final_m)


def analyze(out,pilot=False):
    load_verified_dataset()
    baseline=baseline_check()
    assert {r["k"]:r["exact_accuracy"] for r in baseline if r["k"] in KS}=={3:232/360,5:476/1260,7:217/840,10:.1}
    for d in (out,out/"scripts",out/"config",out/"csv",out/"npz"):
        d.mkdir(parents=True,exist_ok=False if d==out else True)
    shutil.copy2(__file__,out/"scripts"/Path(__file__).name)
    save_csv(out/"csv"/"baseline_verification.csv",baseline)
    config=dict(dataset_archive=str(REF),archive_sha256=hashlib.sha256(REF.read_bytes()).hexdigest(),
                dataset_source="binary_character.py",dataset_source_sha256=hashlib.sha256((PROJECT/"binary_character.py").read_bytes()).hexdigest(),
                X=X.tolist(),Y=Y.tolist(),k=list(KS),candidate_alpha=list(GRID),bounds=[list(b) for b in BOUNDS],
                epsilon=EPS,seed=SEED,pilot=pilot,readout="score>0 -> +1; score<=0 -> -1",
                oracle_numeric_guard=ORACLE_NUMERIC_GUARD,
                policy_parameters=dict(first_strong=2.,rest_weak=.5,negative_conflict_eta=2.,minimum_margin_delta=1.),
                oracle_objective="if robust current learning can preserve protected prior correct bits, choose smallest feasible alpha; otherwise minimize protected bit losses, maximize current robust bit count, minimize forgotten exact associations, then smallest alpha",
                oracle_candidates="bounds, epsilon current-bit crossings, prior protected-bit loss crossings and adjacent floating-point values",
                git_revision=subprocess.run(["git","rev-parse","HEAD"],cwd=PROJECT,capture_output=True,text=True).stdout.strip(),
                numpy=np.__version__)
    (out/"config"/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    step_fields=["k","subset","order_id","order","step","digit","previous_count",
                 "before_exact","before_incorrect_bits","before_min_margin","before_mean_margin",
                 "alpha_required","already_exact","already_robust","current_min_margin_after_alpha1",
                 "alignment","negative_weight_conflict","conflict_fraction","weight_norm","total_abs_weight",
                 "cumulative_alpha_before","safe_all_exact","safe_all_robust","safe_full_exact","safe_full_robust",
                 "adverse_all_count","adverse_full_count","vulnerable_previous_bits",
                 "feasible_all_exact_0_1_4","feasible_full_exact_0_1_4",
                 "feasible_all_robust_0_1_4","feasible_full_robust_0_1_4",
                 "feasible_all_robust_0_5_2","feasible_full_robust_0_5_2",
                 "gap_all_robust","gap_full_robust","alpha1_previous_bits_lost",
                 "alpha1_previous_associations_forgotten","alpha1_previous_bits_recovered"]
    candidate_fields=["k","subset","order_id","order","step","digit","alpha",
                      "before_current_exact","before_current_min_margin","before_current_incorrect_bits",
                      "alpha_required_robust","after_current_exact","after_current_robust",
                      "after_current_min_margin","margin_gain","current_bits_corrected","current_new_errors",
                      "previous_bits_lost","previous_bits_recovered","previous_associations_forgotten",
                      "previous_associations_recovered","previous_bit_accuracy_delta",
                      "previous_exact_accuracy_delta","previous_min_margin_after",
                      "previous_margin_change_min","previous_margin_change_mean","previous_margin_change_max",
                      "positive_contributions","negative_contributions","zero_contributions",
                      "data_use"]
    bit_fields=["k","subset","order_id","order","step","current_digit","previous_digit","output_neuron",
                "target_bit","before_margin","before_exact_correct","before_robust_correct",
                "signed_contribution_per_unit_alpha","classification",
                "alpha_loss_boundary","alpha1_after_margin","alpha1_after_correct","data_use"]
    run_fields=["k","subset","order_id","order","bounds","policy","exact_count","exact_accuracy",
                "bit_accuracy","mean_minimum_margin","minimum_margin","total_alpha","alpha_by_digit_json",
                "exact_digits","forgotten_total","recovered_total","previous_bits_lost_total",
                "previous_bits_recovered_total","current_exact_after_count","weight_norm"]
    transition_fields=["k","subset","order_id","order","bounds","policy","step","digit","alpha",
                       "current_exact","current_robust","previous_bits_lost","previous_bits_recovered",
                       "forgotten","recovered","exact_count","bit_accuracy","min_margin","exact_digits",
                       "required_alpha","safe_full_exact"]
    files=[]
    for name,fields in (("equal_step_features.csv",step_fields),("counterfactual_candidates.csv",candidate_fields),
                        ("previous_bit_contributions.csv",bit_fields),("policy_runs.csv",run_fields),
                        ("policy_transitions.csv",transition_fields)):
        f,w=writer(out/"csv"/name,fields);files.append(f)
        if name.startswith("equal_step"):sw=w
        elif name.startswith("counterfactual"):cw=w
        elif name.startswith("previous_bit"):bw=w
        elif name.startswith("policy_runs"):rw=w
        else:tw=w
    step_rows=[];policy_rows=[]
    candidate_agg=defaultdict(lambda:defaultdict(list))
    transition_agg=defaultdict(lambda:defaultdict(list))
    representatives={}
    k10_candidate_scatter=[]
    checked_identities=0
    k10_weights=[];k10_weight_labels=[]
    try:
        for k in KS:
            subsets=list(combinations(range(10),k))
            if pilot:subsets=subsets[:2] if k!=10 else subsets
            for ids in subsets:
                sub="-".join(map(str,ids));orders=unique_orders(ids,k)
                if pilot:orders=orders[:2]
                for order_id,order in enumerate(orders):
                    order_str="-".join(map(str,order))
                    W=np.zeros((15,15),dtype=float);previous=[]
                    for step,p in enumerate(order):
                        info=step_info(W,previous,p)
                        before_m=info["prev_margin"];before_ok=info["prev_correct"]
                        before_ex=info["prev_exact"]
                        c=info["contribution"]
                        assert np.allclose(info["current_margin"]+15, Y[p]*(X[p]@(W+OUTERS[p])))
                        alpha1=counterfactual(info,previous,p,1.)
                        checked_identities+=1
                        safeall=info["safe_all_robust"];safefull=info["safe_full_robust"]
                        req=info["required"]
                        step_row=dict(k=k,subset=sub,order_id=order_id,order=order_str,step=step+1,digit=p,
                                      previous_count=len(previous),before_exact=int(np.all(info["current_correct"])),
                                      before_incorrect_bits=int(np.sum(~info["current_correct"])),
                                      before_min_margin=float(info["current_margin"].min()),
                                      before_mean_margin=float(info["current_margin"].mean()),alpha_required=req,
                                      already_exact=int(np.all(info["current_correct"])),
                                      already_robust=int(np.all(info["current_margin"]>=EPS)),
                                      current_min_margin_after_alpha1=float(info["current_margin"].min()+15),
                                      alignment=info["alignment"],negative_weight_conflict=info["negative_conflict"],
                                      conflict_fraction=info["conflict_fraction"],weight_norm=info["weight_norm"],
                                      total_abs_weight=info["total_abs_weight"],cumulative_alpha_before=float(step),
                                      safe_all_exact=info["safe_all_exact"],safe_all_robust=safeall,
                                      safe_full_exact=info["safe_full_exact"],safe_full_robust=safefull,
                                      adverse_all_count=info["adverse_all_count"],adverse_full_count=info["adverse_full_count"],
                                      vulnerable_previous_bits=int(np.sum(before_ok & (before_m>0) & (before_m<=2))),
                                      feasible_all_exact_0_1_4=int(feasible(req,info["safe_all_exact"],.1,4)),
                                      feasible_full_exact_0_1_4=int(feasible(req,info["safe_full_exact"],.1,4)),
                                      feasible_all_robust_0_1_4=int(feasible(req,safeall,.1,4)),
                                      feasible_full_robust_0_1_4=int(feasible(req,safefull,.1,4)),
                                      feasible_all_robust_0_5_2=int(feasible(req,safeall,.5,2)),
                                      feasible_full_robust_0_5_2=int(feasible(req,safefull,.5,2)),
                                      gap_all_robust=max(0.,req-safeall) if math.isfinite(safeall) else 0.,
                                      gap_full_robust=max(0.,req-safefull) if math.isfinite(safefull) else 0.,
                                      alpha1_previous_bits_lost=alpha1["previous_bits_lost"],
                                      alpha1_previous_associations_forgotten=alpha1["previous_associations_forgotten"],
                                      alpha1_previous_bits_recovered=alpha1["previous_bits_recovered"])
                        sw.writerow(step_row);step_rows.append(step_row)
                        for h,q in enumerate(previous):
                            for j in range(15):
                                cm=float(c[h,j]);m=float(before_m[h,j])
                                after=float(alpha1["prev_margin_after"][h,j])
                                assert abs(after-(m+cm))<1e-10
                                bw.writerow(dict(k=k,subset=sub,order_id=order_id,order=order_str,step=step+1,
                                                 current_digit=p,previous_digit=q,output_neuron=j,
                                                 target_bit=int(Y[q,j]>0),before_margin=m,
                                                 before_exact_correct=int(before_ok[h,j]),
                                                 before_robust_correct=int(m>=EPS),
                                                 signed_contribution_per_unit_alpha=cm,
                                                 classification="positive" if cm>0 else ("negative" if cm<0 else "zero"),
                                                 alpha_loss_boundary=m/(-cm) if before_ok[h,j] and cm<0 else "",
                                                 alpha1_after_margin=after,
                                                 alpha1_after_correct=int(alpha1["prev_ok_after"][h,j]),
                                                 data_use="retrospective_full_history_only"))
                        for a in GRID:
                            e=counterfactual(info,previous,p,a)
                            changed=e["prev_margin_after"]-before_m
                            cand=dict(k=k,subset=sub,order_id=order_id,order=order_str,step=step+1,digit=p,alpha=a,
                                      before_current_exact=int(np.all(info["current_correct"])),
                                      before_current_min_margin=float(info["current_margin"].min()),
                                      before_current_incorrect_bits=int(np.sum(~info["current_correct"])),
                                      alpha_required_robust=req,after_current_exact=int(e["current_exact_after"]),
                                      after_current_robust=int(e["current_robust_after"]),
                                      after_current_min_margin=float(e["current_margin_after"].min()),
                                      margin_gain=15*a,current_bits_corrected=e["current_corrected_bits"],
                                      current_new_errors=e["current_new_errors"],
                                      previous_bits_lost=e["previous_bits_lost"],
                                      previous_bits_recovered=e["previous_bits_recovered"],
                                      previous_associations_forgotten=e["previous_associations_forgotten"],
                                      previous_associations_recovered=e["previous_associations_recovered"],
                                      previous_bit_accuracy_delta=float((e["prev_ok_after"].sum()-before_ok.sum())/(15*len(previous))) if previous else 0.,
                                      previous_exact_accuracy_delta=float((e["prev_ex_after"].sum()-before_ex.sum())/len(previous)) if previous else 0.,
                                      previous_min_margin_after=float(e["prev_margin_after"].min()) if previous else "",
                                      previous_margin_change_min=float(changed.min()) if previous else "",
                                      previous_margin_change_mean=float(changed.mean()) if previous else "",
                                      previous_margin_change_max=float(changed.max()) if previous else "",
                                      positive_contributions=int(np.sum(c>0)),negative_contributions=int(np.sum(c<0)),
                                      zero_contributions=int(np.sum(c==0)),
                                      data_use="retrospective_counterfactual_not_online_decision")
                            cw.writerow(cand)
                            ca=candidate_agg[(k,a)]
                            for key in ("after_current_exact","current_bits_corrected","previous_bits_lost",
                                        "previous_bits_recovered","previous_associations_forgotten",
                                        "previous_associations_recovered"):
                                ca[key].append(cand[key])
                            if k==10:
                                k10_candidate_scatter.append((a,cand["current_bits_corrected"],cand["previous_bits_lost"]))
                            category=None
                            if previous and a>=1.0:
                                if cand["after_current_exact"] and not cand["before_current_exact"]:
                                    if cand["previous_associations_forgotten"]>0:category="benefit_with_forgetting"
                                    elif cand["previous_bits_lost"]==0:category="benefit_without_loss"
                                if (a>1 and cand["current_bits_corrected"]==alpha1["current_corrected_bits"]
                                        and cand["previous_bits_lost"]>=alpha1["previous_bits_lost"]+2):
                                    category=category or "extra_damage_little_benefit"
                                if (cand["after_current_exact"] and not cand["before_current_exact"]
                                        and cand["previous_associations_recovered"]>0):
                                    category=category or "current_and_previous_recovery"
                            if category and category not in representatives:
                                worst=np.argwhere(info["prev_correct"] & ~e["prev_ok_after"])
                                sample=None
                                if len(worst):
                                    h,j=map(int,worst[0]);sample=dict(previous_digit=previous[h],output_neuron=j,
                                         previous_margin_before=float(before_m[h,j]),
                                         signed_contribution_per_alpha=float(c[h,j]),
                                         previous_margin_after=float(e["prev_margin_after"][h,j]))
                                representatives[category]=dict(**cand,example_lost_bit=sample)
                        W+=OUTERS[p]  # equal-alpha trajectory; candidates did not mutate W
                        previous.append(p)
                        assert np.array_equal(W,np.sum(OUTERS[np.asarray(previous)],axis=0))
                    # Complete rollouts start again from W=0 for every policy.
                    for bounds in BOUNDS:
                        bound=f"{bounds[0]}-{bounds[1]}"
                        for policy in POLICIES:
                            rr=rollout(ids,order,policy,bounds)
                            row=dict(k=k,subset=sub,order_id=order_id,order=order_str,bounds=bound,policy=policy,
                                     exact_count=int(rr["exact"].sum()),exact_accuracy=float(np.mean(rr["exact"])),
                                     bit_accuracy=float(np.mean(rr["ok"])),
                                     mean_minimum_margin=float(np.mean(rr["margins"].min(axis=1))),
                                     minimum_margin=float(rr["margins"].min()),total_alpha=float(rr["alpha"].sum()),
                                     alpha_by_digit_json=json.dumps(rr["alpha"].tolist()),
                                     exact_digits="-".join(str(q) for q,z in zip(ids,rr["exact"]) if z),
                                     forgotten_total=sum(s["forgotten"] for s in rr["steps"]),
                                     recovered_total=sum(s["recovered"] for s in rr["steps"]),
                                     previous_bits_lost_total=sum(s["previous_bits_lost"] for s in rr["steps"]),
                                     previous_bits_recovered_total=sum(s["previous_bits_recovered"] for s in rr["steps"]),
                                     current_exact_after_count=sum(s["current_exact"] for s in rr["steps"]),
                                     weight_norm=float(np.linalg.norm(rr["W"])))
                            rw.writerow(row);policy_rows.append(row)
                            for s in rr["steps"]:
                                trans=dict(k=k,subset=sub,order_id=order_id,order=order_str,bounds=bound,policy=policy,**s)
                                tw.writerow(trans)
                                ta=transition_agg[(k,bound,policy,s["step"])]
                                for key in ("forgotten","recovered","exact_count","previous_bits_lost","current_exact"):
                                    ta[key].append(s[key])
                            if k==10:
                                k10_weights.append(rr["W"].astype(np.float32))
                                k10_weight_labels.append((order_id,bound,policy))
            print(f"Finished K={k}: diagnostic steps={len(step_rows)}, policy runs={len(policy_rows)}",flush=True)
    finally:
        for f in files:f.close()
    if k10_weights:
        np.savez_compressed(out/"npz"/"k10_policy_weights.npz",weights=np.stack(k10_weights),
                            order_id=np.array([r[0] for r in k10_weight_labels]),
                            bounds=np.array([r[1] for r in k10_weight_labels]),
                            policy=np.array([r[2] for r in k10_weight_labels]))
    np.savez_compressed(out/"npz"/"dataset_and_outer_products.npz",X=X,Y=Y,G=G,outer_products=OUTERS)
    return dict(baseline=baseline,steps=step_rows,candidate_agg=candidate_agg,
                policies=policy_rows,transition_agg=transition_agg,representatives=representatives,
                k10_candidate_scatter=k10_candidate_scatter,checked_identities=checked_identities,
                pilot=pilot,config=config)


def summarize(out,state):
    steps=state["steps"];policies=state["policies"]
    candidate_rows=[]
    for (k,a),values in sorted(state["candidate_agg"].items()):
        candidate_rows.append(dict(k=k,alpha=a,n=len(values["after_current_exact"]),
                                   current_exact_rate=float(np.mean(values["after_current_exact"])),
                                   mean_current_bits_corrected=float(np.mean(values["current_bits_corrected"])),
                                   mean_previous_bits_lost=float(np.mean(values["previous_bits_lost"])),
                                   mean_previous_bits_recovered=float(np.mean(values["previous_bits_recovered"])),
                                   mean_previous_associations_forgotten=float(np.mean(values["previous_associations_forgotten"])),
                                   mean_previous_associations_recovered=float(np.mean(values["previous_associations_recovered"]))))
    save_csv(out/"csv"/"candidate_alpha_summary.csv",candidate_rows)
    feasibility=[]
    for k in KS:
        rows=[r for r in steps if r["k"]==k and r["previous_count"]>0]
        for bound in BOUNDS:
            lo,hi=bound
            for criterion in ("all","full"):
                for standard in ("exact","robust"):
                    safe_key=f"safe_{criterion}_{standard}"
                    safe=np.array([r[safe_key] for r in rows],dtype=float)
                    req=np.array([r["alpha_required"] for r in rows])
                    flags=np.array([feasible(r["alpha_required"],r[safe_key],lo,hi) for r in rows])
                    finite=safe[np.isfinite(safe)]
                    feasibility.append(dict(k=k,bounds=f"{lo}-{hi}",criterion=criterion,
                                            standard=standard,n=len(rows),feasible_count=int(flags.sum()),
                                            feasible_fraction=float(flags.mean()),infeasible_count=int((~flags).sum()),
                                            required_median=float(np.median(req)),required_p90=float(np.quantile(req,.9)),
                                            safe_finite_median=float(np.median(finite)) if len(finite) else "",
                                            safe_infinite_fraction=float(np.mean(~np.isfinite(safe))),
                                            no_feasible_due_to_required_above_max=int(np.sum(req>hi)),
                                            no_feasible_due_to_safe_below_min=int(np.sum(safe<lo)),
                                            no_feasible_due_to_required_above_safe=int(np.sum(req>safe))))
    save_csv(out/"csv"/"feasibility_summary.csv",feasibility)
    feature_names=("before_min_margin","before_mean_margin","before_incorrect_bits","alignment",
                   "negative_weight_conflict","conflict_fraction","weight_norm","step","cumulative_alpha_before")
    outcomes=("alpha1_previous_bits_lost","alpha1_previous_associations_forgotten",
              "safe_all_robust","safe_full_robust","feasible_all_robust_0_1_4","feasible_full_robust_0_1_4")
    corr=[]
    for k in KS:
        rows=[r for r in steps if r["k"]==k and r["previous_count"]>0]
        for feature in feature_names:
            for outcome in outcomes:
                corr.append(dict(k=k,n=len(rows),feature=feature,outcome=outcome,
                                 spearman_r=spearman([r[feature] for r in rows],[r[outcome] for r in rows]),
                                 note="descriptive; repeated orders/subsets from one dataset"))
    save_csv(out/"csv"/"current_information_spearman.csv",corr)
    groups=defaultdict(list)
    for r in policies:groups[(r["k"],r["bounds"],r["policy"])].append(r)
    policy_summary=[]
    for (k,bound,policy),rows in sorted(groups.items()):
        exact=np.array([r["exact_accuracy"] for r in rows]);bit=np.array([r["bit_accuracy"] for r in rows])
        policy_summary.append(dict(k=k,bounds=bound,policy=policy,n_runs=len(rows),
                                   exact_mean=float(exact.mean()),exact_median=float(np.median(exact)),
                                   exact_std=float(exact.std()),exact_min=float(exact.min()),exact_max=float(exact.max()),
                                   exact_p10=float(np.quantile(exact,.1)),exact_p90=float(np.quantile(exact,.9)),
                                   bit_mean=float(bit.mean()),mean_min_margin=float(np.mean([r["mean_minimum_margin"] for r in rows])),
                                   forgotten_mean=float(np.mean([r["forgotten_total"] for r in rows])),
                                   recovered_mean=float(np.mean([r["recovered_total"] for r in rows])),
                                   previous_bits_lost_mean=float(np.mean([r["previous_bits_lost_total"] for r in rows])),
                                   total_alpha_mean=float(np.mean([r["total_alpha"] for r in rows]))))
    save_csv(out/"csv"/"policy_summary.csv",policy_summary)
    trans_summary=[]
    for (k,bound,policy,step),v in sorted(state["transition_agg"].items()):
        trans_summary.append(dict(k=k,bounds=bound,policy=policy,step=step,n=len(v["forgotten"]),
                                  forgotten_mean=float(np.mean(v["forgotten"])),
                                  recovered_mean=float(np.mean(v["recovered"])),
                                  exact_count_mean=float(np.mean(v["exact_count"])),
                                  previous_bits_lost_mean=float(np.mean(v["previous_bits_lost"])),
                                  current_exact_rate=float(np.mean(v["current_exact"]))))
    save_csv(out/"csv"/"policy_transition_summary.csv",trans_summary)
    reps=list(state["representatives"].values())
    if reps:
        save_csv(out/"csv"/"representative_steps.csv",
                 [dict(category=category,**r,example_lost_bit_json=json.dumps(r["example_lost_bit"]))
                  for category,r in state["representatives"].items()])
    return dict(candidate=candidate_rows,feasibility=feasibility,correlations=corr,
                policy=policy_summary,transitions=trans_summary,representatives=state["representatives"])


def make_figures(out,state,summary):
    figs=out/"figures";figs.mkdir()
    cand=summary["candidate"];feas=summary["feasibility"];policy=summary["policy"]
    trans=summary["transitions"];steps=state["steps"]
    def save(fig,name):
        fig.tight_layout();fig.savefig(figs/name,dpi=145);plt.close(fig)
    def candidate_curve(field,ylabel,title,name):
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
        for ax,k in zip(axes.flat,KS):
            rr=sorted((r for r in cand if r["k"]==k),key=lambda r:r["alpha"])
            ax.plot([r["alpha"] for r in rr],[r[field] for r in rr],marker="o")
            ax.axvline(1,color="black",ls="--",lw=.7,label="Equal α=1")
            ax.set(title=f"K={k}",xlabel="Association Learning Strength α",ylabel=ylabel)
            ax.grid(alpha=.2)
        fig.suptitle(title+" | One-Step Counterfactuals from Equal-Alpha W_before")
        save(fig,name)
    candidate_curve("current_exact_rate","Current Exact Recall Fraction",
                    "Current Recall Accuracy vs Association Learning Strength",
                    "current_recall_vs_learning_strength.png")
    candidate_curve("mean_previous_bits_lost","Mean Previous Correct Bits Lost",
                    "Previous-Memory Bit Losses vs Association Learning Strength",
                    "previous_bit_losses_vs_learning_strength.png")
    fig,ax=plt.subplots(figsize=(8,5))
    points=np.array(state["k10_candidate_scatter"])
    im=ax.scatter(points[:,1],points[:,2],c=points[:,0],s=8,alpha=.25,cmap="viridis")
    fig.colorbar(im,ax=ax,label="Candidate Learning Strength α")
    ax.set(title="K=10: Self-Memory Gain vs Previous-Memory Interference",
           xlabel="Current Output Bits Corrected",ylabel="Previously Correct Output Bits Lost")
    save(fig,"self_gain_vs_previous_interference.png")
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,k in zip(axes.flat,KS):
        rr=[r for r in steps if r["k"]==k and r["previous_count"]>0]
        finite=[r for r in rr if math.isfinite(r["safe_full_robust"])]
        ax.scatter([r["alpha_required"] for r in finite],[r["safe_full_robust"] for r in finite],s=6,alpha=.2)
        ax.plot([0,4],[0,4],color="black",ls="--",lw=.7,label="required = safe")
        ax.set(xlim=(0,4.1),ylim=(0,4.1),title=f"K={k}; infinite safe bounds omitted ({len(rr)-len(finite)})",
               xlabel="Required Alpha (ε=1e-4)",ylabel="Safe Alpha Upper Bound (Prior Exact Memories)")
        ax.grid(alpha=.2)
    fig.suptitle("Required Alpha vs Safe Alpha Upper Bound | Equal-Alpha Trajectories")
    save(fig,"required_vs_safe_alpha.png")
    fig,ax=plt.subplots(figsize=(8,5))
    xx=np.arange(len(KS));width=.35
    for offset,criterion,label in ((-width/2,"all","All Correct Bits"),(width/2,"full","Prior Exact Associations")):
        vals=[next(r["feasible_fraction"] for r in feas if r["k"]==k and r["bounds"]=="0.1-4.0"
                   and r["criterion"]==criterion and r["standard"]=="robust") for k in KS]
        ax.bar(xx+offset,100*np.array(vals),width,label=label)
    ax.set(xticks=xx,xticklabels=KS,xlabel="Number of Stored Associations K",
           ylabel="Feasible Learning Steps (%)",ylim=(0,105),
           title="Fraction with Feasible Alpha Interval | ε=1e-4, Bounds 0.1–4.0")
    ax.legend();save(fig,"feasible_interval_fraction.png")
    fig,ax=plt.subplots(figsize=(8,5))
    for policy_name in POLICIES:
        rr=sorted((r for r in trans if r["k"]==10 and r["bounds"]=="0.5-2.0"
                   and r["policy"]==policy_name),key=lambda r:r["step"])
        ax.plot([r["step"] for r in rr],[r["forgotten_mean"] for r in rr],marker="o",label=policy_name)
    ax.set(title="K=10: Forgetting vs Number of Stored Associations | Bounds 0.5–2.0",
           xlabel="Number of Learned Associations",ylabel="Mean Previously Exact Associations Forgotten")
    ax.legend(fontsize=6,ncol=2);save(fig,"forgetting_vs_stored_associations.png")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    rr=[r for r in steps if r["k"]==10 and r["previous_count"]>0]
    for ax,field,label in ((axes[0],"before_min_margin","Current Minimum Signed Margin"),
                           (axes[1],"conflict_fraction","Negative Weight Conflict Fraction")):
        ax.scatter([r[field] for r in rr],[r["alpha1_previous_bits_lost"] for r in rr],s=8,alpha=.3)
        ax.set(xlabel=label,ylabel="Previous Correct Bits Lost at α=1")
        ax.grid(alpha=.2)
    fig.suptitle("K=10: Current Weight Features vs Actual Previous-Memory Losses")
    save(fig,"weight_features_vs_previous_losses.png")
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharey=True)
    for ax,k in zip(axes.flat,KS):
        rr=[next(r for r in policy if r["k"]==k and r["bounds"]=="0.5-2.0"
                 and r["policy"]==name) for name in POLICIES]
        ax.bar(range(len(POLICIES)),[100*r["exact_mean"] for r in rr])
        ax.axhline(100*rr[0]["exact_mean"],color="black",ls="--",lw=.8,label="Equal α")
        ax.set(xticks=range(len(POLICIES)),xticklabels=[x.replace("retrospective_oracle_","Oracle ").replace("negative_conflict_down_eta2","Conflict down") for x in POLICIES],
               title=f"K={k}",ylabel="Mean Exact Recall Accuracy (%)")
        ax.tick_params(axis="x",rotation=45,labelsize=7);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Exact Recall Accuracy by Learning Policy | Same Orders, Bounds 0.5–2.0")
    save(fig,"exact_accuracy_by_learning_policy.png")
    fig,ax=plt.subplots(figsize=(10,5))
    for name in POLICIES:
        vals=[100*r["exact_accuracy"] for r in state["policies"] if r["k"]==10 and r["bounds"]=="0.5-2.0" and r["policy"]==name]
        ax.plot(np.sort(vals),label=name)
    ax.set(title="K=10 Learning Order Sensitivity | Sorted Accuracy Across 100 Orders",
           xlabel="Order Rank (sorted within policy)",ylabel="Exact Recall Accuracy (%)")
    ax.legend(fontsize=7);save(fig,"k10_learning_order_sensitivity.png")
    reps=summary["representatives"]
    rep=reps.get("benefit_with_forgetting") or reps.get("extra_damage_little_benefit") or next(iter(reps.values()),None)
    fig,ax=plt.subplots(figsize=(8,4.5))
    if rep and rep["example_lost_bit"]:
        b=rep["example_lost_bit"]
        ax.bar(["Before update","After update"],[b["previous_margin_before"],b["previous_margin_after"]],color=["tab:blue","tab:red"])
        ax.axhline(0,color="black",lw=.7)
        ax.set(title=f"Representative Signed Margin Change: K={rep['k']}, step {rep['step']}, α={rep['alpha']}",
               ylabel="Previous Output Bit Signed Margin (score units)")
        ax.text(.5,.9,f"Digit {b['previous_digit']}, bit {b['output_neuron']}; contribution {b['signed_contribution_per_alpha']:+.1f}/α",
                transform=ax.transAxes,ha="center")
    else:
        ax.text(.5,.5,"No previous-bit loss example found",transform=ax.transAxes,ha="center")
        ax.set(title="Representative Signed Margin Changes")
    save(fig,"representative_signed_margin_changes.png")


def report(out,state,s):
    def p(k,policy,bound="0.5-2.0"):
        return next(r for r in s["policy"] if r["k"]==k and r["bounds"]==bound and r["policy"]==policy)
    def f(k,criterion,standard,bound="0.1-4.0"):
        return next(r for r in s["feasibility"] if r["k"]==k and r["bounds"]==bound
                    and r["criterion"]==criterion and r["standard"]==standard)
    k10=[r for r in state["steps"] if r["k"]==10 and r["previous_count"]>0]
    required=np.array([r["alpha_required"] for r in k10])
    finite_safe=np.array([r["safe_full_robust"] for r in k10 if math.isfinite(r["safe_full_robust"])])
    policy_lines=[]
    for name in POLICIES:
        r=p(10,name)
        policy_lines.append(f"| {name} | {100*r['exact_mean']:.2f} | {100*r['exact_median']:.2f} | {100*r['exact_min']:.0f}–{100*r['exact_max']:.0f} | {100*r['bit_mean']:.2f} | {r['forgotten_mean']:.2f} |")
    c1=p(10,"negative_conflict_down_eta2");equal=p(10,"equal")
    oa=p(10,"retrospective_oracle_all_correct_bits");of=p(10,"retrospective_oracle_exact_associations")
    reps=s["representatives"]
    rep_lines=[]
    for category in ("benefit_without_loss","benefit_with_forgetting","extra_damage_little_benefit","current_and_previous_recovery"):
        r=reps.get(category)
        if r:
            b=r["example_lost_bit"]
            detail=(f" Previous digit {b['previous_digit']} bit {b['output_neuron']}: margin {b['previous_margin_before']:+.3f} + {r['alpha']}×({b['signed_contribution_per_alpha']:+.3f}) = {b['previous_margin_after']:+.3f}." if b else "")
            rep_lines.append(f"- **{category}:** K={r['k']}, subset {r['subset']}, order {r['order']}, step {r['step']} (digit {r['digit']}), α={r['alpha']}; current corrected bits {r['current_bits_corrected']}, previous bit losses {r['previous_bits_lost']}, forgotten exact associations {r['previous_associations_forgotten']}, recovered exact associations {r['previous_associations_recovered']}.{detail}")
        else:rep_lines.append(f"- **{category}:** No matching event in the evaluated equal-alpha counterfactual grid.")
    factors=[]
    for feature in ("before_min_margin","before_incorrect_bits","alignment","negative_weight_conflict","conflict_fraction","weight_norm","step"):
        r=next(r for r in s["correlations"] if r["k"]==10 and r["feature"]==feature and r["outcome"]=="alpha1_previous_bits_lost")
        factors.append(f"| {feature} | {r['spearman_r'] if r['spearman_r'] is not None else 'undefined'} |")
    report=f"""# Self-Memory Gain and Interference During Sequential Learning

## Protocol and checks
The bipolar 15×15 digit 0–9 → letter A–J dataset exactly matches `association_alpha_milp_reference_20261019.zip`. Pure correlation reproduced K=3 **64.44%**, K=5 **37.78%**, K=7 **25.83%**, K=10 **10.00%** before experiments. Noisy cues, LIF simulation, simulator modification and post-training transformation were excluded. All K=3 subsets and permutations, all K=5/7 subsets with 20 unique seeded orders, and K=10 with 100 unique orders were evaluated. Order lists come from the prior online study and include ascending/descending for K>3.

Each candidate α in `{GRID}` starts from the **same equal-alpha W_before**; only after all counterfactuals does the diagnostic path apply α=1. Every candidate verified `m_current_after=m_current_before+15α` and `m_previous_after=m_previous_before+α·(x_q·x_p)y_qy_p`; {state['checked_identities']} step identities were checked. `csv/previous_bit_contributions.csv` records every earlier q/output bit's signed coefficient and α=1 margin; `csv/counterfactual_candidates.csv` records all candidate outcomes. These retrospective calculations did not select alpha in the online policies.

Exact correctness uses `score>0→+1`, `score≤0→−1`. Thus score zero is exact-correct for a −1 target but has signed margin zero and is **not ε-robust**. Robust feasibility uses ε={EPS}; exact-preservation bounds also include zero-margin correct bits. The smallest *additional* robust strength is `max(0,(ε−minimum current signed margin)/15)`. If already robust, this equals 0; a positive alpha_min is an imposed experimental bound, not a learning necessity. Candidate-grid results can exceed this analytical minimum because the grid is discrete.

## Required versus safe learning strength
For K=10 equal-alpha trajectories after at least one previous memory, median required α was **{np.median(required):.3f}**, 90th percentile **{np.quantile(required,.9):.3f}**. The median *finite* robust upper bound preserving prior exact associations was **{np.median(finite_safe):.3f}**; an infinite bound means no currently robust correct bit has adverse signed contribution. `csv/feasibility_summary.csv` reports finite/infinite distributions and infeasibility reasons.

| K | All previously correct bits feasible, α∈[0.1,4] | Previously exact associations feasible, α∈[0.1,4] | Previously exact associations feasible, α∈[0.5,2] |
|---:|---:|---:|---:|
{chr(10).join(f"| {k} | {100*f(k,'all','robust')['feasible_fraction']:.2f}% | {100*f(k,'full','robust')['feasible_fraction']:.2f}% | {100*f(k,'full','robust','0.5-2.0')['feasible_fraction']:.2f}% |" for k in KS)}

These fractions exclude the first step, where there is no previous memory. They describe **local feasibility at the fixed W_before**, not the architecture's global capacity. The separate exact-preservation fractions are in CSV; they can differ from ε-robust fractions when a negative-target score ties at zero. A non-feasible interval can arise because required α exceeds the allowed maximum, a safe upper bound is below the allowed minimum, or required α exceeds the safe bound. All three counts and required–safe gaps are saved.

## Full sequential rollouts (same orders; bounds 0.5–2.0)
The fixed schedule is α=2 for the first pair and α=0.5 thereafter. Current minimum-margin uses δ=1. Previous negative-conflict down uses η=2, as in the earlier online experiment. Both retrospective oracles use only the present and **previous** pairs, never future pairs. If robust current learning can preserve (a) all previously correct bits or (b) bits in previously exact associations, the oracle chooses the smallest such α. For this *selection* it stays {ORACLE_NUMERIC_GUARD:g} inside the analytical exact score-zero boundary to avoid floating-point ambiguity; the analytical safe bounds in CSV remain unguarded. If this is infeasible, it first minimizes losses of those protected bits, then maximizes current robust bit count, then minimizes forgotten exact associations, then chooses the smallest α among deterministic margin-crossing candidates. The infeasible branch can decline to learn the current memory to protect earlier ones. These full-history analytical references require stored past targets.

| K=10 policy | Mean exact (%) | Median (%) | Range (%) | Mean bit (%) | Mean forgotten transitions |
|---|---:|---:|---:|---:|---:|
{chr(10).join(policy_lines)}

Equal α=1 is {100*equal['exact_mean']:.2f}%; prior conflict-down η=2 is {100*c1['exact_mean']:.2f}%; the all-correct-bit oracle is {100*oa['exact_mean']:.2f}% and prior-exact-association oracle {100*of['exact_mean']:.2f}%. These are means across **100 orders of the same K=10 subset**, not new datasets. The oracle's larger information set does not guarantee a globally optimal final W: its fixed objective is myopic and it cannot read future pairs. `csv/policy_runs.csv` and `csv/policy_transitions.csv` retain every order, alpha, association list, forgetting and recovery; K=10 final W is in `npz/k10_policy_weights.npz`.

## Representative observed changes
{chr(10).join(rep_lines)}

## Current-information features and STDP feasibility
For K=10 equal-alpha steps, descriptive Spearman ρ with previous correct-bit losses caused by α=1:

| Observable before update | Spearman ρ |
|---|---:|
{chr(10).join(factors)}

All observations reuse ten patterns and overlapping subsets/orders; these coefficients are **not independent-sample significance tests**. Current minimum/mean target-signed margins and incorrect-bit count need comparison with the current target. Weight alignment is the sum of current signed margins, so it is not an independent feature. Conflict, weight norm, step index and cumulative α require aggregation or scalar memory but no past pair identity. The exact per-bit interference coefficients and safe upper bound require prior x and y or replay; W_before alone generally cannot identify which prior memory is vulnerable. The oracle's full history is not synapse-local STDP information. An actual spike-event implementation would additionally need a mechanism to aggregate across synapses/outputs and broadcast a positive association-level α; this study implements neither mechanism nor a biological learning rule.

The observed constraints separate current-memory gain from prior-memory damage; a higher final exact count may redistribute errors. No single policy should be selected solely from the highest K=10 score on this fixed dataset. Further research should focus on whether a measurable, low-information proxy for vulnerable prior bits exists, rather than assume full-history safe bounds are locally available. `config/config.json` records source hashes, dataset, seed, bounds, epsilon and the fixed oracle objective. Run `python my_research/experiments/diagnostics/analyze_learning_interference.py` from the repository root.
"""
    (out/"summary.md").write_text(report,encoding="utf-8")


def main():
    out=ROOT/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pilot="--pilot" in sys.argv
    state=analyze(out,pilot)
    s=summarize(out,state)
    if not pilot:make_figures(out,state,s)
    report(out,state,s)
    metrics=dict(status="pilot_success" if pilot else "success",checked_step_identities=state["checked_identities"],
                 step_count=len(state["steps"]),policy_run_count=len(state["policies"]),
                 representative_categories=list(s["representatives"]),
                 baseline=state["baseline"])
    (out/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n")
    archive=out.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for file in sorted(out.rglob("*")):
            if file.is_file():z.write(file,arcname=out.name+"/"+str(file.relative_to(out)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({"result_dir":str(out),"zip":str(archive),**metrics},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
