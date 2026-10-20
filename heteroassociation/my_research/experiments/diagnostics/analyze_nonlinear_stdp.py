"""Mathematical local-update and nonlinear-correlation diagnostics.

Run from heteroassociation/: python my_research/experiments/diagnostics/analyze_nonlinear_stdp.py
This never changes or runs the original B2 spike-event trainer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_nonlinear_stdp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
import binary_character as data

K_VALUES = (1, 2, 3, 5, 7, 10)
GAMMAS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
SEED = 0
ROOT = PROJECT / "results" / "nonlinear_stdp_analysis"
OUT = ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
FIG = OUT / "figures"
CSV = OUT / "csv"
NPZ = OUT / "npz"
X01 = data.make_digit_dataset()[0].reshape(10, 15).astype(np.int64)
Y01 = data.make_letter_dataset()[0].reshape(10, 15).astype(np.int64)
XB = 2 * X01 - 1
YB = 2 * Y01 - 1


def f(z, gamma):
    return np.sign(z) * np.abs(z) ** gamma


def inverse_f(w, gamma):
    return np.sign(w) * np.abs(w) ** (1 / gamma)


def local_update(w, s, gamma):
    """Elementwise synapse-local update: each element reads only itself and timing sign."""
    delta_w = f(inverse_f(w, gamma) + s, gamma) - w
    return w + delta_w


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_write(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing_sign_matrix(digit):
    """Original B2 teacher-only target-zero timing inversion, teacher at 4.5 ms."""
    pre = np.where(X01[digit, :, None] == Y01[digit, None, :], 2.0, 7.0)
    teacher = 4.5
    direction = np.where(teacher - pre > 0, 1, -1)
    assert np.array_equal(direction, np.outer(XB[digit], YB[digit]))
    return direction


def sequential(indices, gamma, order):
    weight = np.zeros((15, 15), dtype=float)
    for digit in order:
        assert digit in indices
        weight = local_update(weight, timing_sign_matrix(digit), gamma)
    return weight


def accuracy(pred, target):
    return int(np.all(pred == target, axis=1).sum()), int(np.sum(pred == target))


def preflight():
    assert X01.shape == Y01.shape == (10, 15)
    assert X01[9, 12:15].tolist() == [0, 0, 1]
    assert np.array_equal(X01, data.make_digit_dataset()[0].reshape(10, 15))
    assert np.array_equal(Y01, data.make_letter_dataset()[0].reshape(10, 15))
    for digit in range(10):
        timing_sign_matrix(digit)
    expected = {3: (64.44444444444444, 86.11111111111111),
                5: (37.77777777777778, 44.285714285714285),
                7: (25.833333333333332, 30.952380952380953),
                10: (10.0, 10.0)}
    found = {}
    for k, pair in expected.items():
        rates = []
        for gamma in (1.0, 2.0):
            exact = total = 0
            for indices in combinations(range(10), k):
                x, y = XB[list(indices)], YB[list(indices)]
                w = f(x.T @ y, gamma)
                pred = np.where(x @ w > 0, 1, -1)
                exact += np.all(pred == y, axis=1).sum()
                total += k
            rates.append(100 * exact / total)
        assert np.allclose(rates, pair, atol=1e-10), (k, rates, pair)
        found[k] = rates
    return found


def matrix_comparison():
    rng = np.random.default_rng(SEED)
    rows = []
    for k in K_VALUES:
        indices = tuple(range(k))
        corr = XB[list(indices)].T @ YB[list(indices)]
        for gamma in (1.0, 2.0):
            target = f(corr, gamma)
            reverse = tuple(reversed(indices))
            shuffled = tuple(int(v) for v in rng.permutation(indices))
            orders = [("ascending", indices), ("reverse", reverse), ("seeded_permutation", shuffled)]
            matrices = {}
            for order_name, order in orders:
                weight = sequential(indices, gamma, order)
                matrices[order_name] = weight
                difference = weight - target
                rows.append(dict(k=k, subset="-".join(map(str, indices)),gamma=gamma,
                    order=order_name,order_digits="-".join(map(str,order)),
                    max_absolute_error=float(np.max(np.abs(difference))),
                    normalized_frobenius_error=float(np.linalg.norm(difference)/np.linalg.norm(target)),
                    cosine_similarity=float(np.vdot(weight,target)/(np.linalg.norm(weight)*np.linalg.norm(target))),
                    max_order_difference=float(np.max(np.abs(weight-matrices["ascending"]))) if order_name!="ascending" else 0.0))
                np.savez_compressed(NPZ/f"sequential_K{k}_gamma{gamma:g}_{order_name}.npz",
                    indices=np.array(indices),order=np.array(order),W_corr=corr,W_target=target,
                    W_sequential=weight,W_difference=difference)
            assert max(np.max(np.abs(mat-target)) for mat in matrices.values()) < 1e-10
            if gamma == 2.0:
                fig, axes = plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
                items=[(target,"Target Nonlinear Weight"),
                       (matrices["ascending"],"Sequential Weight Update"),
                       (matrices["ascending"]-target,"Weight Difference")]
                for ax,(arr,title) in zip(axes,items):
                    v=max(float(np.max(np.abs(arr))),1e-12)
                    image=ax.imshow(arr,cmap="coolwarm",vmin=-v,vmax=v,aspect="auto")
                    ax.set(xlabel="Output Neuron Index",ylabel="Input Neuron Index",title=title)
                    fig.colorbar(image,ax=ax,fraction=.047)
                fig.suptitle(f"Sequential Nonlinear Update vs Target, K={k}, Exponent=2, No Clipping")
                fig.savefig(FIG/f"sequential_weight_comparison_K{k}.png",dpi=160)
                plt.close(fig)
    write_csv(CSV/"sequential_verification.csv",rows)
    return rows


def sweep():
    summary = defaultdict(lambda: dict(subsets=0,associations=0,correct_bits=0,exact=0,
                                      corrected_bits=0,newly_incorrect_bits=0,
                                      improved_associations=0,newly_failed_associations=0))
    subset_rows=[]
    association_rows=[]
    mechanism_rows=[]
    distribution=Counter()
    representative={}
    output_path=CSV/"output_bit_results.csv"
    fields=["k","subset","digit","letter","output_idx","gamma","target_sign",
            "baseline_score","nonlinear_score","baseline_prediction","nonlinear_prediction",
            "baseline_signed_margin","signed_margin","margin_change","baseline_correct",
            "nonlinear_correct","corrected_output_bit","newly_incorrect_output_bit"]
    with output_path.open("w",newline="",encoding="utf-8") as handle:
        bit_writer=csv.DictWriter(handle,fieldnames=fields)
        bit_writer.writeheader()
        for k in K_VALUES:
            for indices in combinations(range(10),k):
                subset="-".join(map(str,indices))
                idx=list(indices)
                x,y=XB[idx],YB[idx]
                corr=x.T@y
                base_score=x@corr
                base_pred=np.where(base_score>0,1,-1)
                base_ok=base_pred==y
                base_exact=np.all(base_ok,axis=1)
                n_ltp=(k+corr)//2
                n_ltd=(k-corr)//2
                for magnitude in np.unique(np.abs(corr)):
                    positions=np.abs(corr)==magnitude
                    for sign in (-1,0,1):
                        selected=positions & (np.sign(corr)==sign)
                        count=int(selected.sum())
                        if count:
                            distribution[(k,int(magnitude),sign,int((k+sign*magnitude)//2),
                                          int((k-sign*magnitude)//2))]+=count
                for gamma in GAMMAS:
                    weight=f(corr,gamma)
                    scores=x@weight
                    pred=np.where(scores>0,1,-1)
                    ok=pred==y
                    exact=np.all(ok,axis=1)
                    corrected=(~base_ok)&ok
                    new=base_ok&(~ok)
                    item=summary[(k,gamma)]
                    item["subsets"]+=1
                    item["associations"]+=k
                    item["correct_bits"]+=int(ok.sum())
                    item["exact"]+=int(exact.sum())
                    item["corrected_bits"]+=int(corrected.sum())
                    item["newly_incorrect_bits"]+=int(new.sum())
                    item["improved_associations"]+=int(np.sum((~base_exact)&exact))
                    item["newly_failed_associations"]+=int(np.sum(base_exact&(~exact)))
                    subset_rows.append(dict(k=k,subset=subset,gamma=gamma,
                        exact_count=int(exact.sum()),association_count=k,
                        exact_accuracy_percent=100*float(exact.mean()),
                        bit_accuracy_percent=100*float(ok.mean()),
                        corrected_output_bits=int(corrected.sum()),
                        newly_incorrect_output_bits=int(new.sum()),
                        minimum_signed_margin=float(np.min(y*scores))))
                    for local,digit in enumerate(indices):
                        association_rows.append(dict(k=k,subset=subset,digit=digit,letter=chr(65+digit),
                            gamma=gamma,target_bits="".join(str(int(v>0)) for v in y[local]),
                            prediction_bits="".join(str(int(v>0)) for v in pred[local]),
                            exact=int(exact[local]),bit_accuracy_percent=100*float(ok[local].mean()),
                            baseline_exact=int(base_exact[local]),
                            corrected_output_bits=int(corrected[local].sum()),
                            newly_incorrect_output_bits=int(new[local].sum()),
                            minimum_signed_margin=float(np.min(y[local]*scores[local])),
                            mean_signed_margin=float(np.mean(y[local]*scores[local]))))
                        for j in range(15):
                            margin0=float(y[local,j]*base_score[local,j])
                            margin=float(y[local,j]*scores[local,j])
                            bit_writer.writerow(dict(k=k,subset=subset,digit=digit,letter=chr(65+digit),
                                output_idx=j,gamma=gamma,target_sign=int(y[local,j]),
                                baseline_score=float(base_score[local,j]),nonlinear_score=float(scores[local,j]),
                                baseline_prediction=int(base_pred[local,j]),nonlinear_prediction=int(pred[local,j]),
                                baseline_signed_margin=margin0,signed_margin=margin,
                                margin_change=margin-margin0,baseline_correct=int(base_ok[local,j]),
                                nonlinear_correct=int(ok[local,j]),corrected_output_bit=int(corrected[local,j]),
                                newly_incorrect_output_bit=int(new[local,j])))
                    if gamma == 2.0 and k in (3,5,7,10):
                        for local,digit in enumerate(indices):
                            for j in range(15):
                                signed_baseline=y[local,j]*x[local]*corr[:,j]
                                signed_new=y[local,j]*x[local]*weight[:,j]
                                delta_contrib=signed_new-signed_baseline
                                unanimous=np.abs(corr[:,j])==k
                                status=("corrected" if corrected[local,j] else
                                        "newly_incorrect" if new[local,j] else
                                        "stable_correct" if ok[local,j] else "stable_incorrect")
                                row=dict(k=k,subset=subset,digit=digit,output_idx=j,status=status,
                                    target_sign=int(y[local,j]),baseline_score=float(base_score[local,j]),
                                    nonlinear_score=float(scores[local,j]),
                                    baseline_signed_margin=float(np.sum(signed_baseline)),
                                    nonlinear_signed_margin=float(np.sum(signed_new)),
                                    margin_change=float(np.sum(delta_contrib)),
                                    unanimous_synapse_count=int(np.sum(unanimous)),
                                    unanimous_margin_change=float(np.sum(delta_contrib[unanimous])),
                                    nonunanimous_margin_change=float(np.sum(delta_contrib[~unanimous])),
                                    positive_contribution_gain=float(np.sum(delta_contrib[delta_contrib>0])),
                                    negative_contribution_loss=float(np.sum(delta_contrib[delta_contrib<0])))
                                mechanism_rows.append(row)
                                if k in (3,10) and status in ("corrected","newly_incorrect"):
                                    key=(k,status)
                                    # Largest absolute margin shift, deterministic tie via first encountered.
                                    if key not in representative or abs(row["margin_change"])>abs(representative[key]["row"]["margin_change"]):
                                        representative[key]=dict(row=row,indices=indices,x=x[local].copy(),
                                            y=int(y[local,j]),corr=corr[:,j].copy(),new=weight[:,j].copy(),
                                            n_ltp=n_ltp[:,j].copy(),n_ltd=n_ltd[:,j].copy())
            print(f"Exponent sweep complete: K={k}",flush=True)
    write_csv(CSV/"subset_results.csv",subset_rows)
    write_csv(CSV/"association_results.csv",association_rows)
    write_csv(CSV/"bit_mechanism_gamma2.csv",mechanism_rows)
    dist_rows=[]
    for (k,magnitude,sign,nplus,nminus),count in sorted(distribution.items()):
        dist_rows.append(dict(k=k,absolute_weight=magnitude,weight_sign=sign,
                              ltp_agreement_count=nplus,ltd_agreement_count=nminus,
                              majority_agreement_count=max(nplus,nminus),synapse_subset_count=count))
    write_csv(CSV/"synapse_weight_distribution.csv",dist_rows)
    summary_rows=[]
    for k in K_VALUES:
        base=summary[(k,1.0)]
        baseline_exact=100*base["exact"]/base["associations"]
        baseline_bit=100*base["correct_bits"]/(base["associations"]*15)
        for gamma in GAMMAS:
            s=summary[(k,gamma)]
            exact=100*s["exact"]/s["associations"]
            bit=100*s["correct_bits"]/(s["associations"]*15)
            summary_rows.append(dict(k=k,gamma=gamma,subset_count=s["subsets"],
                evaluated_associations=s["associations"],exact_count=s["exact"],
                exact_accuracy_percent=exact,bit_accuracy_percent=bit,
                exact_change_percentage_points=exact-baseline_exact,
                bit_change_percentage_points=bit-baseline_bit,
                corrected_output_bits=s["corrected_bits"],
                newly_incorrect_output_bits=s["newly_incorrect_bits"],
                net_corrected_bits=s["corrected_bits"]-s["newly_incorrect_bits"],
                improved_associations=s["improved_associations"],
                newly_failed_associations=s["newly_failed_associations"]))
    write_csv(CSV/"exponent_accuracy.csv",summary_rows)
    mechanism_summary=[]
    groups=defaultdict(list)
    for row in mechanism_rows:
        groups[(row["k"],row["status"])].append(row)
    for (k,status),rows in sorted(groups.items()):
        mechanism_summary.append(dict(k=k,status=status,count=len(rows),
            mean_margin_change=float(np.mean([r["margin_change"] for r in rows])),
            mean_unanimous_margin_change=float(np.mean([r["unanimous_margin_change"] for r in rows])),
            mean_nonunanimous_margin_change=float(np.mean([r["nonunanimous_margin_change"] for r in rows])),
            mean_positive_contribution_gain=float(np.mean([r["positive_contribution_gain"] for r in rows])),
            mean_negative_contribution_loss=float(np.mean([r["negative_contribution_loss"] for r in rows]))))
    write_csv(CSV/"mechanism_summary.csv",mechanism_summary)
    rep_rows=[]
    for (k,status),case in representative.items():
        row=case["row"]
        for i in range(15):
            base_signed=int(case["y"]*case["x"][i]*case["corr"][i])
            nonlinear_signed=float(case["y"]*case["x"][i]*case["new"][i])
            rep_rows.append(dict(k=k,status=status,subset=row["subset"],digit=row["digit"],
                output_idx=row["output_idx"],input_idx=i,input_sign=int(case["x"][i]),
                target_sign=case["y"],baseline_weight=int(case["corr"][i]),
                nonlinear_weight=float(case["new"][i]),ltp_agreement_count=int(case["n_ltp"][i]),
                ltd_agreement_count=int(case["n_ltd"][i]),
                baseline_signed_contribution=base_signed,
                nonlinear_signed_contribution=nonlinear_signed,
                signed_contribution_change=nonlinear_signed-base_signed))
    write_csv(CSV/"representative_synapse_contributions.csv",rep_rows)
    return summary_rows,mechanism_rows,mechanism_summary,dist_rows,representative


def plot_sweep(rows, mechanism, distribution, representative):
    by={(r["k"],r["gamma"]):r for r in rows}
    fig,axes=plt.subplots(3,2,figsize=(11,12),sharex=True,constrained_layout=True)
    for ax,k in zip(axes.flat,K_VALUES):
        exact=[by[k,g]["exact_accuracy_percent"] for g in GAMMAS]
        bit=[by[k,g]["bit_accuracy_percent"] for g in GAMMAS]
        ax.plot(GAMMAS,exact,"o-",label="Exact Recall Accuracy (%)")
        ax.plot(GAMMAS,bit,"s--",label="Bit Accuracy (%)")
        ax.axvline(1,color="grey",ls=":",label="Correlation Baseline" if k==1 else None)
        ax.set(title=f"Number of Stored Associations = {k}",xlabel="Nonlinear Exponent",ylabel="Accuracy (%)",ylim=(0,105))
        ax.legend(fontsize=8)
    fig.suptitle("Static Bipolar Recall Across All Association Subsets")
    fig.savefig(FIG/"exponent_accuracy_curves.png",dpi=160);plt.close(fig)

    best={}
    for k in K_VALUES:
        best_val=max(by[k,g]["exact_accuracy_percent"] for g in GAMMAS)
        tied=[g for g in GAMMAS if abs(by[k,g]["exact_accuracy_percent"]-best_val)<1e-12]
        best[k]=(min(tied),best_val,tied)
    pos=np.arange(len(K_VALUES));fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(pos-.18,[by[k,1.0]["exact_accuracy_percent"] for k in K_VALUES],.36,label="Correlation Baseline (Exponent=1)")
    ax.bar(pos+.18,[best[k][1] for k in K_VALUES],.36,label="Best Observed Exponent")
    ax.set_xticks(pos,[f"{k}\nγ={best[k][0]:g}" for k in K_VALUES])
    ax.set(xlabel="Number of Stored Associations / Best Observed Exponent",ylabel="Exact Recall Accuracy (%)",
           title="Baseline vs Best Observed Exponent on Evaluated Subsets",ylim=(0,105))
    ax.legend();fig.tight_layout();fig.savefig(FIG/"best_exponent_comparison.png",dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for ax,k in zip(axes.flat,(3,5,7,10)):
        counts=Counter()
        for row in distribution:
            if row["k"]==k:counts[row["absolute_weight"]]+=row["synapse_subset_count"]
        ax.bar(sorted(counts),[counts[v] for v in sorted(counts)])
        ax.set(title=f"Number of Stored Associations = {k}",xlabel="Absolute Correlation Weight",ylabel="Synapse-Subset Count")
    fig.suptitle("Weight Magnitude Distribution Over All Subsets")
    fig.savefig(FIG/"weight_magnitude_distribution.png",dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for ax,k in zip(axes.flat,(3,5,7,10)):
        vals=[r["margin_change"] for r in mechanism if r["k"]==k]
        ax.hist(vals,bins=40,color="tab:blue")
        ax.axvline(0,color="black",ls="--")
        ax.set(title=f"Number of Stored Associations = {k}",xlabel="Signed Output Margin Change (Exponent 2 − 1)",ylabel="Output Bit Count")
    fig.suptitle("Nonlinear Transformation and Signed Output Margins")
    fig.savefig(FIG/"signed_margin_change.png",dpi=160);plt.close(fig)

    fig,axes=plt.subplots(3,2,figsize=(11,12),sharex=True,constrained_layout=True)
    for ax,k in zip(axes.flat,K_VALUES):
        ax.plot(GAMMAS,[by[k,g]["corrected_output_bits"] for g in GAMMAS],"o-",label="Corrected Output Bits")
        ax.plot(GAMMAS,[by[k,g]["newly_incorrect_output_bits"] for g in GAMMAS],"s--",label="Newly Incorrect Output Bits")
        ax.set(title=f"Number of Stored Associations = {k}",xlabel="Nonlinear Exponent",ylabel="Output Bit Count")
        ax.legend(fontsize=8)
    fig.suptitle("Output Bit Changes Relative to Exponent 1 Across All Subsets")
    fig.savefig(FIG/"corrected_vs_new_errors.png",dpi=160);plt.close(fig)

    keys=[(3,"corrected"),(3,"newly_incorrect"),(10,"corrected"),(10,"newly_incorrect")]
    selected=[key for key in keys if key in representative]
    if selected:
        fig,axes=plt.subplots(len(selected),1,figsize=(11,3*len(selected)),constrained_layout=True,squeeze=False)
        for ax,key in zip(axes.flat,selected):
            c=representative[key];row=c["row"]
            baseline=c["y"]*c["x"]*c["corr"]
            nonlinear=c["y"]*c["x"]*c["new"]
            xi=np.arange(15)
            ax.bar(xi-.2,baseline,.4,label="Exponent 1 Signed Contribution")
            ax.bar(xi+.2,nonlinear,.4,label="Exponent 2 Signed Contribution")
            ax.set_xticks(xi)
            ax.set(xlabel="Input Synapse Index",ylabel="Signed Output Margin Contribution",
                   title=f"K={key[0]}, Digit {row['digit']}, Output {row['output_idx']}, {key[1].replace('_',' ').title()}")
            ax.legend(fontsize=8)
        fig.suptitle("Actual Synapse Contributions to Selected Output Bits")
        fig.savefig(FIG/"representative_synapse_contributions.png",dpi=160);plt.close(fig)
    return best


def feasibility():
    # B2 1-pair observed scale from prior raw weight verification, not fitted here.
    b2_scale=0.34380575623222825
    rows=[]
    for k in K_VALUES:
        idx=list(range(k));corr=XB[idx].T@YB[idx]
        target=f(corr,2.0)
        mapped=0.05+b2_scale*target
        rows.append(dict(k=k,subset="-".join(map(str,idx)),gamma=2,
            candidate_weight_max_abs=float(np.max(np.abs(target))),
            candidate_unit_weight_outside_B2_bounds=int(np.sum((target < -1.5)|(target > 1.5))),
            candidate_unit_weight_outside_B2_ratio=float(np.mean((target < -1.5)|(target > 1.5))),
            b2_offset=0.05,b2_one_pair_delta_scale=b2_scale,
            scaled_candidate_min=float(mapped.min()),scaled_candidate_max=float(mapped.max()),
            scaled_candidate_outside_B2_bounds=int(np.sum((mapped < -1.5)|(mapped > 1.5))),
            scaled_candidate_outside_B2_ratio=float(np.mean((mapped < -1.5)|(mapped > 1.5)))))
    write_csv(CSV/"existing_stdp_feasibility.csv",rows)
    return rows


def main():
    for p in (FIG,CSV,NPZ):p.mkdir(parents=True,exist_ok=False)
    baseline=preflight()
    sequence=matrix_comparison()
    summary_rows,mechanism,mechanism_summary,distribution,representative=sweep()
    assert sum(r["subset_count"] for r in summary_rows if r["gamma"]==1.0)==548
    assert sum(r["evaluated_associations"] for r in summary_rows if r["gamma"]==1.0)==2570
    best=plot_sweep(summary_rows,mechanism,distribution,representative)
    feas=feasibility()
    source=[PROJECT/"binary_character.py",PROJECT/"Ver관리/2. temporal/stdp_temporal_symmetric_teacher_only.py",
            PROJECT/"Ver관리/2. temporal/correlation_pair_count_sweep.py"]
    config=dict(status="success",command="python my_research/experiments/diagnostics/analyze_nonlinear_stdp.py",
        date_local=OUT.name,seed=SEED,gammas=GAMMAS,counts=K_VALUES,
        subset_selection="all itertools.combinations(range(10), K) for exponent sweep; prefix subset for sequential check",
        association_mapping=[f"{i}->{chr(65+i)}" for i in range(10)],
        digit_patterns=X01.tolist(),letter_patterns=Y01.tolist(),
        representation="bipolar 2*bit-1",weight_orientation="input neuron index x output neuron index",
        tie_rule="score > 0 => +1; score <= 0 => -1",
        exact_metric="all 15 output signs equal target for one stored association",
        sequential="w0=0; no clipping; one unit signed update per association; target-zero timing inversion retained; no epochs",
        gamma2_mechanism="all K=3,5,7,10 subset-association-output bits; unanimous synapse means abs(Wcorr)==K",
        original_B2=dict(initial_weight=0.05,epochs=60,weight_bounds=[-1.5,1.5],
            pre_early_ms=2,pre_late_ms=7,teacher_ms=4.5,A_plus=0.02,A_minus=0.02,tau_stdp_ms=2),
        source_sha256={str(p.relative_to(PROJECT)):sha(p) for p in source},
        script_sha256=sha(Path(__file__)),git_revision=subprocess.check_output(["git","-C",str(PROJECT),"rev-parse","HEAD"],text=True).strip(),
        python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__)
    json_write(OUT/"config.json",config)
    source_copy=OUT/"run_nonlinear_stdp_analysis.py"
    shutil.copyfile(__file__,source_copy)
    summary_lookup={(r["k"],r["gamma"]):r for r in summary_rows}
    rates="\n".join(f"| {k} | {summary_lookup[k,1.0]['exact_accuracy_percent']:.2f} | {summary_lookup[k,2.0]['exact_accuracy_percent']:.2f} | {best[k][0]:g} | {best[k][1]:.2f} | {', '.join(map(str,best[k][2]))} |" for k in K_VALUES)
    strongest=max(sequence,key=lambda r:r["max_absolute_error"])
    mech="\n".join(f"| {k} | {summary_lookup[k,2.0]['corrected_output_bits']} | {summary_lookup[k,2.0]['newly_incorrect_output_bits']} | {summary_lookup[k,2.0]['exact_change_percentage_points']:+.2f} |" for k in (3,5,7,10))
    unanimous_rates={}
    for k in (3,5,7,10):
        relevant=[r for r in distribution if r["k"]==k]
        unanimous_rates[k]=100*sum(r["synapse_subset_count"] for r in relevant if r["absolute_weight"]==k)/sum(r["synapse_subset_count"] for r in relevant)
    illustrative=[]
    for key in ((3,"corrected"),(3,"newly_incorrect"),(10,"corrected"),(10,"newly_incorrect")):
        if key not in representative:continue
        row=representative[key]["row"]
        illustrative.append(f"- K={key[0]} {key[1]}: subset {row['subset']}, digit {row['digit']}, output {row['output_idx']}; signed margin {row['baseline_signed_margin']:+g} → {row['nonlinear_signed_margin']:+g}; unanimous-synapse change {row['unanimous_margin_change']:+g}, other-synapse change {row['nonunanimous_margin_change']:+g}. Full 15-input decomposition: `csv/representative_synapse_contributions.csv`.")
    text=f"""# Nonlinear Weight-Dependent STDP Candidate: Mathematical and Static-Recall Analysis

This is a **mathematical candidate constructed to reproduce a target matrix**, not an observed biological plasticity rule or a run of the original spike-event STDP simulator. No LIF recall, noisy cues, pseudoinverse, or learned decoder were used. Feedforward 15×15 architecture, existing 0–9→A–J dataset, bipolar signs, and `score>0` decision are fixed. Run command, hashes, seed and full patterns are in `config.json`.

## Sequential local update

`f(z)=sign(z)|z|^γ`, `inverse_f(w)=sign(w)|w|^(1/γ)`, `Δw=f(inverse_f(w)+s)−w`. B2 teacher-only timing convention maps matching bipolar input/output bits to early pre-spike 2 ms and LTP; differing bits to late pre-spike 7 ms and LTD, with teacher at 4.5 ms. Thus each event sign is `s=x_b*y_b`. We applied one unit event per association, from effective weight zero, without clipping. γ=1 recovers additive correlation; γ=2 recovers `sign(Wcorr)|Wcorr|²`. Ascending, reverse and seeded permutations were checked for K=1,2,3,5,7,10. Maximum matrix error across these checks: {strongest['max_absolute_error']:.3g}; `csv/sequential_verification.csv` and `npz/` preserve target, learned candidate and differences. Order independence here follows the no-clipping one-update-per-pair mathematical protocol; it is not a claim about the original 60-epoch B2 trainer.

## Exponent sweep: all subsets, association-weighted exact rate (%)

| K | Baseline γ=1 | γ=2 | Best observed γ (smallest tie) | Best exact (%) | All exact-rate ties |
|---:|---:|---:|---:|---:|---|
{rates}

Existing γ=1/2 reference values for K=3,5,7,10 were reproduced exactly: {baseline}. K=1 uses 10 subsets, K=2 45, K=3 120, K=5 252, K=7 120, K=10 1. Exact rate weights every stored association equally; `csv/exponent_accuracy.csv` also reports bit accuracy, changes from γ=1 and corrected/new output bits. Best observed γ is selected *after* viewing these subsets and is not an out-of-sample optimum.

## Mechanism at γ=2 relative to γ=1

| K | Corrected output bits | Newly incorrect output bits | Exact change (percentage points) |
|---:|---:|---:|---:|
{mech}

For each synapse, `Wcorr` equals LTP count minus LTD count. For K=3, |W|=3 means all three updates agreed; γ=2 changes magnitude 3→9 while |W|=1 remains 1. The actual signed margin contribution of input synapse i to target output j is `y_b[j]*x_b[i]*W[i,j]`; `csv/bit_mechanism_gamma2.csv` separates unanimous and other synapses for *every* stored output bit in K=3,5,7,10. `csv/representative_synapse_contributions.csv` gives individual 15-synapse decompositions for selected flips:

"""+"\n".join(illustrative)+f"""

An exponent can increase some correct signed contributions and also magnify wrong-direction contributions. The `csv/mechanism_summary.csv` and signed-margin figures quantify both; K=10's unchanged 1/10 exact rate is a net outcome, not evidence that all margins are unchanged. Association overlap and weight signs determine which output bits improve or deteriorate; stronger magnitude alone does not guarantee a correct target sign.

Direct observations: At K=3, γ=2 corrects 196 output bits and creates zero new errors. A representative correction (subset 0-8-9, digit 9, output 7) changes signed margin −1→+17; the +18 shift is entirely from unanimous-synapse contributions. Across all subset-synapse positions, the fraction with unanimous direction falls from {unanimous_rates[3]:.2f}% at K=3 to {unanimous_rates[10]:.2f}% at K=10. At K=10, γ=2 corrects 7 bits but creates 4 errors; a representative new error (digit 2, output 11) changes margin +2→−4. Neither selected K=10 corrected nor newly incorrect example has a unanimous synapse contribution. The number of *exactly recalled associations* remains 1/10, because bit-level gains need not complete an entire 15-bit output and can coexist with new failures. These are observations from this fixed dataset, not a general causal law.

## Feasibility in the existing B2 simulator

The timing **direction** can be preserved: teacher at 4.5 ms with target-zero timing inversion yields the needed ± sign. The proposed **magnitude** can be evaluated locally from current synaptic effective weight and event direction. To apply it, `stdp_update` in `Ver관리/2. temporal/stdp_temporal_symmetric_teacher_only.py` would need a different magnitude formula; `train()` also sets `w=0.05`, loops 60 epochs and clips each update. `b2_full_output.py` only drives the 15 independent output columns and resets seed. No original file was changed.

The mathematical state starts at effective zero, so the 0.05 baseline must be removed when updating and restored for actual stored W; this is distinct from transforming raw W. A unit update repeated 60 epochs yields `f(60*Wcorr)`, not `f(Wcorr)`, unless each presentation is normalized (e.g. 1/60 in the inverse-transformed state). Actual B2 timing gives an exponential magnitude of about 0.00573 for each event, not unit ±1, and extra natural spike events could alter an event-driven implementation. Current [-1.5,1.5] bounds clip candidate γ=2 weights in several K values; `csv/existing_stdp_feasibility.csv` quantifies both unit target and a clearly labeled 0.05+observed-single-pair-scale mapping. Clipping destroys the exact no-clipping order-independence guarantee, and multiple events or different presentation order can then matter. Mathematical compatibility is established; biological plausibility and actual spike-event implementation remain untested.

## Files and limitations

`csv/output_bit_results.csv` contains scores, target, prediction and signed margin for every evaluated output bit and exponent. `csv/subset_results.csv` and `csv/association_results.csv` retain grouping. `csv/synapse_weight_distribution.csv` records direction-agreement counts. `npz/sequential_*.npz` retains target and candidate matrices. All plots in `figures/` can be regenerated from these numbers. This is a retrospective sweep over a fixed ten-pattern dataset; no independent test patterns or noise trials were assessed. Figure titles and axes name the changed quantity and evaluated condition.
"""
    (OUT/"summary.md").write_text(text,encoding="utf-8")
    with zipfile.ZipFile(str(OUT)+".zip","w",zipfile.ZIP_DEFLATED) as archive:
        for path in OUT.rglob("*"):
            if path.is_file():archive.write(path,path.relative_to(ROOT))
    print(json.dumps(dict(output=str(OUT),zip=str(OUT)+".zip",baseline_reproduction=baseline,
                          best={str(k):dict(gamma=best[k][0],exact_percent=best[k][1],ties=best[k][2]) for k in K_VALUES},
                          max_sequential_error=strongest["max_absolute_error"],
                          feasibility=feas),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        json_write(OUT/"failure.json",dict(status="failed",error=str(exc),traceback=traceback.format_exc()))
        raise
