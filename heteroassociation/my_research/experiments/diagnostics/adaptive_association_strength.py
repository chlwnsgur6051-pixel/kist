"""Online association-strength analysis using only (x_p, y_p, W_before).

Run: python my_research/experiments/diagnostics/adaptive_association_strength.py
Outputs a new timestamped directory in results/adaptive_association_strength/.
No STDP/LIF simulation is run; archived MILP alpha is evaluated offline only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_adaptive_association_strength")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
import binary_character as data  # noqa: E402

REF = PROJECT / "association_alpha_milp_reference_20261019.zip"
ROOT = PROJECT / "results" / "adaptive_association_strength"
K_VALUES = (1, 2, 3, 5, 7, 10)
BOUNDS = ((0.5, 2.0), (0.1, 4.0))
DELTAS = (0.01, 0.5, 1.0, 2.0)
ETAS = (0.5, 1.0, 2.0, 4.0)
SEED = 20261019
X = 2*data.make_digit_dataset()[0].reshape(10,15).astype(np.int64)-1
Y = 2*data.make_letter_dataset()[0].reshape(10,15).astype(np.int64)-1
G = X@X.T
OUTERS = np.einsum("pi,pj->pij",X,Y).astype(float)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    rows=list(rows)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)


def evaluate(ids, weight):
    ids=np.asarray(ids,dtype=int)
    scores=X[ids]@weight
    ok=(scores>0)==(Y[ids]>0)
    exact=np.all(ok,axis=1)
    signed=Y[ids]*scores
    return scores,ok,exact,signed


def load_verified_dataset():
    with zipfile.ZipFile(REF) as z:
        d=np.load(io.BytesIO(z.read("dataset.npz")))
        a=np.load(io.BytesIO(z.read("optimized_alpha_reference.npz")))
        if not (np.array_equal(d["X_bipolar"],X) and np.array_equal(d["Y_bipolar"],Y)):
            raise RuntimeError("Reference archive dataset does not match binary_character.py")
        if not (np.array_equal(a["X"],X) and np.array_equal(a["Y"],Y)):
            raise RuntimeError("Archived alpha vectors use a different dataset")
        oracle=np.asarray(a["alpha_milp"],dtype=float)
        _,_,ex,_=evaluate(range(10),X.T@(oracle[:,None]*Y))
        if int(ex.sum())!=6:
            raise RuntimeError("Archived MILP vector failed its 6/10 reproduction check")
    return oracle


def baseline_check():
    expected={1:1.0,2:1.0,3:232/360,5:476/1260,7:217/840,10:0.1}
    rows=[]
    for k in K_VALUES:
        exact_n=total=bit_n=0
        for ids in combinations(range(10),k):
            _,ok,ex,_=evaluate(ids,np.sum(OUTERS[list(ids)],axis=0))
            exact_n+=int(ex.sum()); total+=k; bit_n+=int(ok.sum())
        value=exact_n/total
        assert abs(value-expected[k])<1e-12,(k,value,expected[k])
        rows.append(dict(k=k,subset_count=math.comb(10,k),association_count=total,
                         exact_count=exact_n,exact_accuracy=value,bit_accuracy=bit_n/(total*15)))
    return rows


def unique_orders(ids,k):
    ids=tuple(ids)
    if k<=3:
        return list(permutations(ids))
    needed=100 if k==10 else 20
    chosen=[ids,ids[::-1]]
    seen=set(chosen)
    digest=int(hashlib.sha256((str(ids)+str(SEED)).encode()).hexdigest()[:16],16)
    rng=np.random.default_rng(digest)
    while len(chosen)<needed:
        order=tuple(map(int,rng.permutation(ids)))
        if order not in seen:
            chosen.append(order);seen.add(order)
    return chosen


def specs():
    result=[("equal",None),("random",None),("fixed_increasing",None),("fixed_decreasing",None)]
    result += [("margin_minimum",d) for d in DELTAS]
    for name in ("alignment_conflict_up","alignment_aligned_up","negative_conflict_down","negative_conflict_up"):
        result += [(name,eta) for eta in ETAS]
    assert len(result)==24
    return result


def choose_alpha(rule,param,lo,hi,step,k,m_min,alignment_norm,conflict_norm,random_alpha):
    if rule=="equal": value=1.0
    elif rule=="random": value=random_alpha
    elif rule=="fixed_increasing": value=lo+(hi-lo)*(step/(k-1) if k>1 else .5)
    elif rule=="fixed_decreasing": value=hi-(hi-lo)*(step/(k-1) if k>1 else .5)
    elif rule=="margin_minimum": value=max(1.0,max(0,(param-m_min)/15))
    elif rule=="alignment_conflict_up": value=np.exp(-param*alignment_norm)
    elif rule=="alignment_aligned_up": value=np.exp(param*alignment_norm)
    elif rule=="negative_conflict_down": value=np.exp(-param*conflict_norm)
    elif rule=="negative_conflict_up": value=np.exp(param*conflict_norm)
    else: raise ValueError(rule)
    return float(np.clip(value,lo,hi))


def simulate(ids,order,rule,param,bounds,random_values,record_steps=False):
    lo,hi=bounds
    W=np.zeros((15,15),dtype=float)
    alpha=np.zeros(10,dtype=float)
    prev_exact=np.zeros(10,dtype=bool)
    previously=[]
    step_rows=[]
    first_features=[]
    forgotten_total=recovered_total=0
    for step,p in enumerate(order):
        before=W.copy()
        S=OUTERS[p]
        score_before=X[p]@before
        margins_before=Y[p]*score_before
        m_min=float(margins_before.min())
        alignment=float(np.sum(before*S))
        assert np.isclose(alignment,float(np.sum(margins_before)))
        cumulative=float(alpha.sum())
        alignment_norm=alignment/(225*max(1,cumulative))
        conflict=float(np.maximum(0,-before*S).sum())
        total_weight=float(np.abs(before).sum())
        conflict_norm=conflict/(total_weight+1e-12) if total_weight else 0.0
        strength=choose_alpha(rule,param,lo,hi,step,len(ids),m_min,
                              alignment_norm,conflict_norm,random_values[step])
        assert strength>0
        update=strength*S
        assert np.array_equal(np.sign(update),S)
        W=before+update
        alpha[p]=strength
        previously.append(p)
        scores,ok,ex,margins=evaluate(previously,W)
        old=np.array(previously[:-1],dtype=int)
        forgotten=int(np.sum(prev_exact[old] & ~ex[:-1])) if len(old) else 0
        recovered=int(np.sum(~prev_exact[old] & ex[:-1])) if len(old) else 0
        retained=int(np.sum(prev_exact[old] & ex[:-1])) if len(old) else 0
        forgotten_total+=forgotten;recovered_total+=recovered
        prev_exact[np.asarray(previously,dtype=int)]=ex
        if step==0 or record_steps:
            first_features.append(dict(step=step+1,digit=p,alpha=strength,alignment=alignment,
                                       alignment_normalized=alignment_norm,negative_conflict=conflict,
                                       conflict_normalized=conflict_norm,m_min_before=m_min,
                                       cumulative_alpha_before=cumulative))
        step_rows.append(dict(step=step+1,learned_digit=p,alpha=strength,
                              current_exact=int(ex[-1]),current_correct_bits=int(ok[-1].sum()),
                              exact_count=int(ex.sum()),bit_accuracy=float(ok.mean()),
                              forgotten=forgotten,recovered=recovered,retained=retained,
                              minimum_signed_margin=float(margins.min()),
                              exact_digits="-".join(map(str,[q for q,v in zip(previously,ex) if v])),
                              exact_mask=sum((1<<q) for q,v in zip(previously,ex) if v)))
    final_score,final_ok,final_ex,final_margin=evaluate(ids,W)
    equal_score,equal_ok,equal_ex,equal_margin=evaluate(ids,np.sum(OUTERS[list(ids)],axis=0))
    assert np.allclose(W,np.einsum("p,pij->ij",alpha,OUTERS))
    if rule=="equal":
        assert np.array_equal(final_ex,equal_ex)
        assert np.array_equal(final_ok,equal_ok)
    return dict(W=W,alpha=alpha,steps=step_rows,features=first_features,
                final_score=final_score,final_ok=final_ok,final_ex=final_ex,final_margin=final_margin,
                equal_ok=equal_ok,equal_ex=equal_ex,equal_margin=equal_margin,
                forgotten_total=forgotten_total,recovered_total=recovered_total)


def quantile_summary(values):
    v=np.asarray(values,dtype=float)
    return dict(mean=float(v.mean()),median=float(np.median(v)),std=float(v.std()),
                min=float(v.min()),max=float(v.max()),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)))


def plot_results(out,agg,k10rows,feature_rows,step_summary,representative):
    figs=out/"figures"; figs.mkdir()
    primary=[r for r in agg if r["bounds"]=="0.5-2.0" and r["parameter"] in ("", 1.0)]
    names=["equal","random","fixed_increasing","fixed_decreasing","margin_minimum",
           "alignment_conflict_up","alignment_aligned_up","negative_conflict_down","negative_conflict_up"]
    short={"equal":"Equal","random":"Random","fixed_increasing":"Increase schedule","fixed_decreasing":"Decrease schedule",
           "margin_minimum":"Minimum margin","alignment_conflict_up":"Conflict↑ (B1)",
           "alignment_aligned_up":"Aligned↑ (B2)","negative_conflict_down":"Conflict↓ (C1)",
           "negative_conflict_up":"Conflict↑ (C2)"}
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharey=True)
    for ax,k in zip(axes.flat,(3,5,7,10)):
        rr={r["rule"]:r for r in primary if r["k"]==k}
        vals=[100*rr[n]["exact_mean"] for n in names]
        ax.bar(range(len(names)),vals,color=["gray"]+(["tab:blue"]*3)+(["tab:orange"]*5))
        ax.axhline(100*rr["equal"]["exact_mean"],color="black",ls="--",lw=.8,label="Equal α")
        ax.set(xticks=range(len(names)),xticklabels=[short[n] for n in names],title=f"K={k}",ylim=(0,105))
        ax.tick_params(axis="x",rotation=55,labelsize=8)
        ax.grid(axis="y",alpha=.2)
    fig.supylabel("Exact Recall Accuracy (%)")
    fig.suptitle("Exact Recall Accuracy by Learning Rule | α Bounds 0.5–2.0, Parameter 1")
    fig.tight_layout();fig.savefig(figs/"exact_accuracy_by_learning_rule.png",dpi=145);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for ax,k in zip(axes.flat,(3,5,7,10)):
        for rule in ("margin_minimum","alignment_conflict_up","alignment_aligned_up","negative_conflict_down","negative_conflict_up"):
            rows=sorted((r for r in agg if r["k"]==k and r["bounds"]=="0.5-2.0" and r["rule"]==rule),key=lambda r:float(r["parameter"]))
            ax.plot([float(r["parameter"]) for r in rows],[100*r["exact_mean"] for r in rows],marker="o",label=short[rule])
        baseline=next(r for r in agg if r["k"]==k and r["bounds"]=="0.5-2.0" and r["rule"]=="equal")
        ax.axhline(100*baseline["exact_mean"],color="black",ls="--",label="Equal α")
        ax.set(title=f"K={k}",xlabel="Learning Strength Parameter (δ for A, η for B/C)",ylabel="Mean Exact Recall Accuracy (%)")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Exact Recall Accuracy vs Learning Strength Parameter | Bounds 0.5–2.0")
    fig.tight_layout();fig.savefig(figs/"accuracy_vs_learning_strength_parameter.png",dpi=145);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,5))
    data=[];labels=[]
    for n in names:
        rr=[r for r in k10rows if r["bounds"]=="0.5-2.0" and r["rule"]==n and r["parameter"] in ("",1.0)]
        data.append([100*float(r["exact_accuracy"]) for r in rr]);labels.append(short[n])
    ax.boxplot(data,tick_labels=labels,showmeans=True)
    ax.axhline(10,color="black",ls="--",label="Equal α")
    ax.set(title="K=10: Exact Recall Accuracy Across 100 Learning Orders | Bounds 0.5–2.0",
           ylabel="Exact Recall Accuracy (%)",xlabel="Learning Rule")
    ax.tick_params(axis="x",rotation=45);ax.legend();fig.tight_layout();fig.savefig(figs/"k10_accuracy_across_learning_orders.png",dpi=145);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,5))
    arr=np.stack([representative[n]["alpha"] for n in names])
    im=ax.imshow(arr,aspect="auto",cmap="viridis",vmin=.5,vmax=2)
    ax.set(xticks=range(10),yticks=range(len(names)),yticklabels=[short[n] for n in names],
           xlabel="Digit Association",ylabel="Learning Rule",
           title="Alpha Assigned to Each Association | K=10, Ascending Order, Bounds 0.5–2.0")
    fig.colorbar(im,ax=ax,label="Association Learning Strength α")
    fig.tight_layout();fig.savefig(figs/"alpha_by_association.png",dpi=145);plt.close(fig)

    for kind,file,rules,xkey,xlabel in [
        ("alignment","alignment_vs_selected_alpha.png",("alignment_conflict_up","alignment_aligned_up"),"alignment_normalized","Normalized Mean Weight Alignment"),
        ("conflict","conflict_vs_selected_alpha.png",("negative_conflict_down","negative_conflict_up"),"conflict_normalized","Negative Weight Conflict Fraction")]:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
        for ax,rule in zip(axes,rules):
            rr=[r for r in feature_rows if r["rule"]==rule and r["bounds"]=="0.5-2.0" and r["parameter"]==1.0]
            ax.scatter([r[xkey] for r in rr],[r["alpha"] for r in rr],s=5,alpha=.25)
            ax.axhline(1,color="black",ls="--",lw=.8)
            ax.set(title=short[rule],xlabel=xlabel,ylabel="Selected Association Strength α")
            ax.grid(alpha=.2)
        fig.suptitle(f"K=10: {kind.title()} vs Selected Alpha | 100 Orders, η=1, Bounds 0.5–2.0")
        fig.tight_layout();fig.savefig(figs/file,dpi=145);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    for n in names:
        rr=sorted((r for r in step_summary if r["k"]==10 and r["bounds"]=="0.5-2.0" and r["rule"]==n and r["parameter"] in ("",1.0)),key=lambda r:r["step"])
        ax.plot([r["step"] for r in rr],[r["forgotten_mean"] for r in rr],marker="o",label=short[n])
    ax.set(title="K=10: Previously Recalled Associations Forgotten After Each Update",
           xlabel="Number of Learned Associations",ylabel="Mean Forgotten Associations per Order")
    ax.legend(ncol=3,fontsize=7);fig.tight_layout();fig.savefig(figs/"forgotten_after_each_update.png",dpi=145);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    for n in names:
        rr=sorted((r for r in step_summary if r["k"]==10 and r["bounds"]=="0.5-2.0" and r["rule"]==n and r["parameter"] in ("",1.0)),key=lambda r:r["step"])
        ax.plot([r["step"] for r in rr],[r["exact_mean"] for r in rr],marker="o",label=short[n])
    ax.set(title="K=10: Retained Associations vs Number Learned | 100 Orders",
           xlabel="Number of Learned Associations",ylabel="Mean Currently Recalled Associations")
    ax.legend(ncol=3,fontsize=7);fig.tight_layout();fig.savefig(figs/"retained_associations_by_step.png",dpi=145);plt.close(fig)

    equal=representative["equal"]["W"]
    adaptive=representative["alignment_conflict_up"]["W"]
    vmax=max(abs(equal).max(),abs(adaptive).max())
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),sharex=True,sharey=True)
    for ax,w,title in zip(axes,(equal,adaptive),("Equal α=1","Online Alignment Conflict↑, η=1")):
        im=ax.imshow(w,cmap="RdBu_r",vmin=-vmax,vmax=vmax)
        ax.set(title=title,xlabel="Output Neuron",ylabel="Input Neuron")
    fig.colorbar(im,ax=axes,label="Weight (arbitrary units)",shrink=.8)
    fig.suptitle("K=10: Equal Alpha vs Adaptive Alpha Weight Matrices | Ascending Order")
    fig.savefig(figs/"equal_vs_adaptive_weight_matrices.png",dpi=145,bbox_inches="tight");plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    for name in ("equal","alignment_conflict_up","alignment_aligned_up","negative_conflict_down","negative_conflict_up"):
        m=representative[name]["final_margin"]
        ax.plot(range(10),m.min(axis=1),marker="o",label=short[name])
    ax.axhline(0,color="black",lw=.8)
    ax.set(title="K=10: Minimum Signed Output Margin | Ascending Order, η=1, Bounds 0.5–2.0",
           xlabel="Digit Association",ylabel="Minimum Signed Output Margin (score units)")
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figs/"signed_output_margin_changes.png",dpi=145);plt.close(fig)


def execute(out,pilot=False):
    start=time.time()
    oracle=load_verified_dataset()
    base=baseline_check()  # fail before any parameter experiment
    for d in (out,out/"scripts",out/"config",out/"csv",out/"npz"):
        d.mkdir(parents=True,exist_ok=False if d==out else True)
    shutil.copy2(__file__,out/"scripts"/Path(__file__).name)
    write_csv(out/"csv"/"baseline_accuracy.csv",base)
    cfg=dict(seed=SEED,dataset_source=str(REF),dataset_sha256=sha(REF),source_dataset="binary_character.py",
             source_dataset_sha256=sha(PROJECT/"binary_character.py"),
             dataset_match_verified=True,X_bipolar=X.tolist(),Y_bipolar=Y.tolist(),
             k_values=K_VALUES,bounds=BOUNDS,deltas=DELTAS,etas=ETAS,
             orders={"K1-3":"all permutations","K5-7":20,"K10":100},
             pilot=pilot,recall="score > 0 -> +1, otherwise -1",no_clipping=True,
             random_alpha="independent seeded uniform within bounds for each subset/order/position",
             equal_baseline=[r for r in base],
             git_revision=subprocess.run(["git","rev-parse","HEAD"],cwd=PROJECT,capture_output=True,text=True).stdout.strip(),
             python=sys.version,numpy=np.__version__)
    (out/"config"/"config.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n")
    spec=specs()
    run_fields=["run_id","k","subset","order_id","order","bounds","rule","parameter",
                "exact_count","exact_accuracy","bit_accuracy","min_signed_margin","mean_signed_margin",
                "normalized_min_margin","total_alpha","alpha_by_digit_json","exact_digits",
                "rescued_vs_equal","lost_vs_equal","corrected_output_bits","newly_incorrect_output_bits",
                "forgotten_total","recovered_total","weight_frobenius_difference_from_correlation",
                "step_exact_mask_json","step_bit_accuracy_json","step_min_margin_json"]
    step_fields=["run_id","k","subset","order_id","order","bounds","rule","parameter",
                 "step","learned_digit","alpha","current_exact","current_correct_bits","exact_count",
                 "bit_accuracy","forgotten","recovered","retained","minimum_signed_margin","exact_digits","exact_mask"]
    feature_fields=["run_id","order_id","order","bounds","rule","parameter","step","digit","alpha",
                    "alignment","alignment_normalized","negative_conflict","conflict_normalized",
                    "m_min_before","cumulative_alpha_before"]
    group=defaultdict(lambda:defaultdict(list))
    steps_group=defaultdict(lambda:defaultdict(list))
    k10rows=[]; feature_rows=[]; representative={}
    k10_weights=[];k10_weight_labels=[]
    run_id=0
    with (out/"csv"/"run_results.csv").open("w",newline="",encoding="utf-8") as rf, \
         (out/"csv"/"intermediate_recall.csv").open("w",newline="",encoding="utf-8") as sf, \
         (out/"csv"/"k10_online_features.csv").open("w",newline="",encoding="utf-8") as ff:
        rw=csv.DictWriter(rf,fieldnames=run_fields);rw.writeheader()
        sw=csv.DictWriter(sf,fieldnames=step_fields);sw.writeheader()
        fw=csv.DictWriter(ff,fieldnames=feature_fields);fw.writeheader()
        for k in K_VALUES:
            subsets=list(combinations(range(10),k))
            if pilot:
                subsets=subsets[:2] if k!=10 else subsets
            for ids in subsets:
                orders=unique_orders(ids,k)
                if pilot and k>3:
                    orders=orders[:3]
                sub="-".join(map(str,ids))
                for order_id,order in enumerate(orders):
                    order_str="-".join(map(str,order))
                    for lo,hi in BOUNDS:
                        bound=f"{lo}-{hi}"
                        digest=int(hashlib.sha256((str((ids,order,lo,hi,SEED))+"random").encode()).hexdigest()[:16],16)
                        random_values=np.random.default_rng(digest).uniform(lo,hi,size=k)
                        for rule,param in spec:
                            run_id+=1
                            result=simulate(ids,order,rule,param,(lo,hi),random_values,record_steps=(k==10))
                            ex=result["final_ex"];ok=result["final_ok"];margin=result["final_margin"]
                            equal_ex=result["equal_ex"];equal_ok=result["equal_ok"]
                            rescued=[q for q,z,e in zip(ids,ex,equal_ex) if z and not e]
                            lost=[q for q,z,e in zip(ids,ex,equal_ex) if not z and e]
                            corrected=int(np.sum(ok & ~equal_ok));damaged=int(np.sum(~ok & equal_ok))
                            alpha=result["alpha"]
                            gkey=(k,bound,rule,"" if param is None else float(param))
                            exact_rate=float(np.mean(ex));bit_rate=float(np.mean(ok))
                            group[gkey]["exact"].append(exact_rate)
                            group[gkey]["bit"].append(bit_rate)
                            group[gkey]["forgotten"].append(result["forgotten_total"])
                            group[gkey]["recovered"].append(result["recovered_total"])
                            group[gkey]["rescued"].append(len(rescued))
                            group[gkey]["lost"].append(len(lost))
                            group[gkey]["total_alpha"].append(float(alpha.sum()))
                            row=dict(run_id=run_id,k=k,subset=sub,order_id=order_id,order=order_str,bounds=bound,
                                     rule=rule,parameter="" if param is None else float(param),
                                     exact_count=int(ex.sum()),exact_accuracy=exact_rate,bit_accuracy=bit_rate,
                                     min_signed_margin=float(margin.min()),mean_signed_margin=float(margin.mean()),
                                     normalized_min_margin=float(margin.min()*k/alpha.sum()),total_alpha=float(alpha.sum()),
                                     alpha_by_digit_json=json.dumps(alpha.tolist()),
                                     exact_digits="-".join(map(str,[q for q,z in zip(ids,ex) if z])),
                                     rescued_vs_equal="-".join(map(str,rescued)),lost_vs_equal="-".join(map(str,lost)),
                                     corrected_output_bits=corrected,newly_incorrect_output_bits=damaged,
                                     forgotten_total=result["forgotten_total"],recovered_total=result["recovered_total"],
                                     weight_frobenius_difference_from_correlation=float(np.linalg.norm(result["W"]-np.sum(OUTERS[list(ids)],axis=0))),
                                     step_exact_mask_json=json.dumps([s["exact_mask"] for s in result["steps"]]),
                                     step_bit_accuracy_json=json.dumps([s["bit_accuracy"] for s in result["steps"]]),
                                     step_min_margin_json=json.dumps([s["minimum_signed_margin"] for s in result["steps"]]))
                            rw.writerow(row)
                            for s in result["steps"]:
                                sw.writerow({k2:row[k2] for k2 in ("run_id","k","subset","order_id","order","bounds","rule","parameter")} | s)
                                steps_group[gkey+(s["step"],)]["exact"].append(s["exact_count"])
                                steps_group[gkey+(s["step"],)]["forgotten"].append(s["forgotten"])
                                steps_group[gkey+(s["step"],)]["recovered"].append(s["recovered"])
                                steps_group[gkey+(s["step"],)]["retained"].append(s["retained"])
                            if k==10:
                                k10rows.append(row)
                                for f in result["features"]:
                                    frow={k2:row[k2] for k2 in ("run_id","order_id","order","bounds","rule","parameter")} | f
                                    fw.writerow(frow);feature_rows.append(frow)
                                k10_weights.append(result["W"].astype(np.float32))
                                k10_weight_labels.append((run_id,rule,bound,"" if param is None else float(param),order_id))
                                if order_id==0 and bound=="0.5-2.0" and (param is None or param==1.0):
                                    representative[rule]=result
                # all parameter rules shared exactly the same order list
            rf.flush();sf.flush();ff.flush()
            print(f"Completed K={k}, runs={run_id}, elapsed={time.time()-start:.1f}s",flush=True)
    agg=[]
    for (k,bound,rule,param),v in sorted(group.items()):
        e=quantile_summary(v["exact"]);b=quantile_summary(v["bit"])
        agg.append(dict(k=k,bounds=bound,rule=rule,parameter=param,n_runs=len(v["exact"]),
                        exact_mean=e["mean"],exact_median=e["median"],exact_std=e["std"],
                        exact_min=e["min"],exact_max=e["max"],exact_p10=e["p10"],exact_p90=e["p90"],
                        bit_mean=b["mean"],bit_std=b["std"],
                        forgotten_mean=float(np.mean(v["forgotten"])),recovered_mean=float(np.mean(v["recovered"])),
                        rescued_vs_equal_mean=float(np.mean(v["rescued"])),lost_vs_equal_mean=float(np.mean(v["lost"])),
                        total_alpha_mean=float(np.mean(v["total_alpha"]))))
    write_csv(out/"csv"/"aggregate_accuracy.csv",agg)
    ss=[]
    for (k,bound,rule,param,step),v in sorted(steps_group.items()):
        ss.append(dict(k=k,bounds=bound,rule=rule,parameter=param,step=step,n=len(v["exact"]),
                       exact_mean=float(np.mean(v["exact"])),forgotten_mean=float(np.mean(v["forgotten"])),
                       recovered_mean=float(np.mean(v["recovered"])),retained_mean=float(np.mean(v["retained"]))))
    write_csv(out/"csv"/"intermediate_summary.csv",ss)
    if k10_weights:
        np.savez_compressed(out/"npz"/"k10_final_weights.npz",weights=np.stack(k10_weights),
                            run_ids=np.array([r[0] for r in k10_weight_labels],dtype=int),
                            rule=np.array([r[1] for r in k10_weight_labels]),
                            bounds=np.array([r[2] for r in k10_weight_labels]),
                            parameter=np.array([str(r[3]) for r in k10_weight_labels]),
                            order_id=np.array([r[4] for r in k10_weight_labels],dtype=int))
    np.savez_compressed(out/"npz"/"reference_and_dataset.npz",X=X,Y=Y,G=G,outer_products=OUTERS,
                        milp_alpha=oracle,milp_weight=X.T@(oracle[:,None]*Y))
    if not pilot:
        plot_results(out,agg,k10rows,feature_rows,ss,representative)
    return dict(config=cfg,baseline=base,aggregate=agg,k10rows=k10rows,run_count=run_id,
                duration_seconds=time.time()-start,pilot=pilot)


def report(out,state):
    agg=state["aggregate"]
    def get(k,rule,bound="0.5-2.0",param=""):
        return next(r for r in agg if r["k"]==k and r["bounds"]==bound and r["rule"]==rule and r["parameter"]==param)
    equal=get(10,"equal")
    primary=[]
    for rule,param in [("equal",""),("random",""),("fixed_increasing",""),("fixed_decreasing",""),
                       ("margin_minimum",1.0),("alignment_conflict_up",1.0),("alignment_aligned_up",1.0),
                       ("negative_conflict_down",1.0),("negative_conflict_up",1.0)]:
        r=get(10,rule,param=param)
        primary.append(f"| {rule} | {param} | {100*r['exact_mean']:.2f} | {100*r['exact_median']:.2f} | {100*r['exact_p10']:.2f}–{100*r['exact_p90']:.2f} | {100*r['bit_mean']:.2f} | {r['forgotten_mean']:.2f} |")
    adaptive=[r for r in agg if r["k"]==10 and r["rule"] in ("margin_minimum","alignment_conflict_up","alignment_aligned_up","negative_conflict_down","negative_conflict_up")]
    best=max(adaptive,key=lambda r:(r["exact_mean"],-r["exact_std"]))
    strongest_conflict=max((get(10,n,param=1.0) for n in ("alignment_conflict_up","negative_conflict_up")),key=lambda r:r["exact_mean"])
    reduce_conflict=max((get(10,n,param=1.0) for n in ("alignment_aligned_up","negative_conflict_down")),key=lambda r:r["exact_mean"])
    summary=f"""# Online Adaptive Association Learning Strength

## Setup and validity
The archived `association_alpha_milp_reference_20261019.zip` bipolar X/Y arrays exactly match `binary_character.py` and use digit 0–9 → letter A–J. Archived target-informed MILP α reproduces 6/10 exact and is stored for **offline reference only**; no online rule reads it. `config/config.json` records hashes, patterns, fixed seed, bounds, parameter grid, and order sampling. No noisy cues, LIF, clipping, nonlinear transform, hidden layer, or simulator modification. Static decision: score>0 means +1, tie means −1.

The α=1 all-subset reproduction check passed **before** sweeps: K=1 100%, K=2 100%, K=3 64.44%, K=5 37.78%, K=7 25.83%, K=10 10.00%. K≤3 used every permutation per subset; K=5/7 used 20 unique seeded orders including ascending and descending; K=10 used 100 unique seeded orders for its one subset. Every rule used identical orders. Each random control drew one independent seeded uniform α per position within the same bounds. The fixed schedules depend on position only.

At every step the selector used the *current* x, y and a copy of W_before. Signed-margin Rule A uses `max(1,(δ−min(y·xW_before))/15)` before clipping, so it cannot actively downweight. Alignment normalization uses `sum(W_before·outer(x,y))/(225·max(1,cumulative_alpha_before))`; this equals the sum of signed current-association margins divided by the denominator. Negative conflict uses `sum(max(0,−W_before·outer(x,y)))/(sum(abs(W_before))+1e−12)`, zero when W_before=0. B/C exponentials preserve positive α. Every update was `α·outer(x,y)`, and sign preservation was asserted. No final sum(α)=K constraint was imposed; per-run total α and K/total-α-normalized margins are recorded separately.

## Results: K=10 across 100 learning orders, bounds 0.5–2.0, predefined parameter 1
| Rule | δ/η | Mean exact (%) | Median exact (%) | 10th–90th (%) | Mean bit (%) | Mean forgotten transitions |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(primary)}

Across the **exploratory grid**, the largest K=10 *mean across orders* was {100*best['exact_mean']:.2f}% for `{best['rule']}` at parameter {best['parameter']}, bounds {best['bounds']}; range {100*best['exact_min']:.0f}–{100*best['exact_max']:.0f}% and standard deviation {100*best['exact_std']:.2f} percentage points. This was selected retrospectively from the same ten associations and is not held-out evidence. The equal baseline is {100*equal['exact_mean']:.2f}% for every order. All parameter settings and order-level outcomes are in `csv/aggregate_accuracy.csv` and `csv/run_results.csv`.

For the predefined conservative η=1 comparison, reinforcing conflict gives at most {100*strongest_conflict['exact_mean']:.2f}% mean exact among B1/C2; reducing strength under conflict gives at most {100*reduce_conflict['exact_mean']:.2f}% among B2/C1. Compare both with random and position-only controls in the table; the full flexible-bound results are in CSV. These are observations for this fixed dataset, not evidence that one direction is universally beneficial. `csv/intermediate_recall.csv` records every update's new-memory recall, prior-memory forgetting/recovery, exact mask and bit accuracy. `csv/k10_online_features.csv` records the actual pre-update alignment/conflict/α for all K=10 runs. `npz/k10_final_weights.npz` stores all K=10 final matrices; other runs' exact W can be reconstructed from dataset outer products and `alpha_by_digit_json`. Global MILP α was not used in any selector.

## Mechanism, controls, and limitations
Association identity, output target and W_before are mathematically sufficient for these rules, but computing a shared association-level α requires aggregation across 225 synapses. Rule A further requires the *minimum* of 15 target-signed output scores; reading output and comparing against the present target entails nonlocal supervised feedback in an actual SNN. Alignment and negative conflict also aggregate synaptic products, so they are not individually synapse-local STDP updates. Their scalar α could modulate local update magnitude **if** an additional aggregation/modulation mechanism were supplied; this was not implemented or biologically validated. The sign of each local update remains the bipolar outer-product sign in this mathematical model, not a measured spike-event outcome.

Intermediate evaluation was retrospective and never fed back into alpha selection. A single K=10 subset with 100 orders tests order sensitivity, not independent dataset generalization. Parameter selection on this same dataset can overfit. The MILP 60% is target-informed, bound- and normalization-specific and is not a universal upper bound. Differences between adaptive, random and fixed schedules do not alone establish a causal rule; the recorded α assignments and margins support further scrutiny. This experiment does not justify implementing any particular rule in the original spike-event simulator without additional validation.

Run: `python my_research/experiments/diagnostics/adaptive_association_strength.py`. Each run makes a new timestamped directory and ZIP. Full result status and duration are in `metrics.json`.
"""
    (out/"summary.md").write_text(summary,encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pilot",action="store_true",help="small correctness/runtime pilot; not a full analysis")
    args=ap.parse_args()
    out=ROOT/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    state=execute(out,args.pilot)
    report(out,state)
    metrics=dict(status="pilot_success" if args.pilot else "success",run_count=state["run_count"],
                 duration_seconds=state["duration_seconds"],baseline=state["baseline"],
                 k10_order_count=3 if args.pilot else 100)
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    archive=out.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for file in sorted(out.rglob("*")):
            if file.is_file(): z.write(file,arcname=out.name+"/"+str(file.relative_to(out)))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    print(json.dumps({"result_dir":str(out),"zip":str(archive),**metrics},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
