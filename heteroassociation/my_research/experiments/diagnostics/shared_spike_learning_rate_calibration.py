"""Learning-rate calibration for shared-spike teacher-only B2 STDP.

K=10 runs call the unmodified source stdp_update at every presentation.
All-subset runs replay cached source-function event deltas with the same
per-presentation clipping; equivalence is checked against direct source runs.
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
import shared_spike_timing_calibration as calibration


ROOT=Path(__file__).resolve().parents[3]
RATES=(0.0200,0.0100,0.0075,0.0050)
KS=(1,2,3,5,7,10)
CHECKPOINTS=(1,5,10,20,30,40,50,60)
SEED=0
NEAR_ZERO_TOL=1e-10
TIMING=dict(pre_plus_ms=3.0,pre_minus_ms=7.0,post_plus_ms=5.0,
            post_minus_early_ms=1.3,post_minus_late_ms=8.7)


def key_rate(rate):
    return f"{rate:.4f}".replace(".","p")


def metrics(x,y,w,reference):
    effective=w-0.05
    scores,pred,signed,exact,strict=calibration.margins_and_recall(x,y,effective)
    result=dict(exact_count=int(exact.sum()),exact_percent=100*float(exact.mean()),
                bit_percent=100*float(np.mean(pred==y)),
                strictly_positive_margin_count=int(strict.sum()),
                min_signed_margin=float(signed.min()),
                near_zero_score_count=int(np.sum(np.abs(scores)<NEAR_ZERO_TOL)),
                weight_min=float(w.min()),weight_max=float(w.max()),
                weight_norm=float(np.linalg.norm(effective)),
                **prior.similarity(effective,reference))
    return result,scores,pred,signed,exact,strict


def rate_sweep(b2,datasets,out):
    epoch_rows,association_rows,bit_rows,run_status=[],[],[],[]
    histories={}
    for tag,(x,y,x01) in datasets.items():
        corr=x.T@y
        for rate in RATES:
            b2.A_PLUS=b2.A_MINUS=rate  # isolated imported module; source file unchanged
            history,clip,suppressed,reversals=calibration.train(
                b2,x,y,x01,tuple(range(10)),TIMING,True,"random_seed0")
            assert np.all(reversals==0)
            histories[(tag,rate)]=(history,clip,suppressed)
            np.savez_compressed(out/"npz"/f"K10_{tag}_rate_{key_rate(rate)}.npz",
                                W_initial=history[0],W_final=history[-1],
                                W_effective=history[-1]-history[0],W_by_epoch=history,
                                W_correlation=corr,clipping_events_by_epoch=clip,
                                suppressed_updates_by_epoch=suppressed,
                                sign_reversals_by_epoch=reversals)
            cumulative_clip=0
            cumulative_suppressed=0
            for ep,w in enumerate(history):
                cumulative_clip+=int(clip[ep])
                cumulative_suppressed+=int(suppressed[ep])
                sat=100*float(np.mean((np.abs(w-b2.W_MIN)<=1e-12)|
                                      (np.abs(w-b2.W_MAX)<=1e-12)))
                m,scores,pred,signed,exact,strict=metrics(x,y,w,corr)
                epoch_rows.append(dict(dataset=tag,K=10,learning_rate=rate,epoch=ep,
                                       cumulative_presentations=ep*10,
                                       nominal_rate_times_epoch=rate*ep,
                                       clipping_events=int(clip[ep]),cumulative_clipping_events=cumulative_clip,
                                       suppressed_updates=int(suppressed[ep]),
                                       cumulative_suppressed_updates=cumulative_suppressed,
                                       saturated_synapse_percent=sat,sign_reversals=int(reversals[ep]),**m))
                if ep in CHECKPOINTS:
                    for p in range(10):
                        association_rows.append(dict(dataset=tag,K=10,learning_rate=rate,epoch=ep,
                                                     digit=p,letter=chr(ord('A')+p),
                                                     exact=bool(exact[p]),strictly_positive_margin=bool(strict[p]),
                                                     bit_percent=100*float(np.mean(pred[p]==y[p])),
                                                     minimum_signed_margin=float(signed[p].min()),
                                                     mean_signed_margin=float(signed[p].mean()),
                                                     target_bits="".join(map(str,((y[p]+1)//2).tolist())),
                                                     predicted_bits="".join(map(str,((pred[p]+1)//2).tolist()))))
                        for j in range(15):
                            bit_rows.append(dict(dataset=tag,K=10,learning_rate=rate,epoch=ep,
                                                 digit=p,output_neuron=j,target_sign=int(y[p,j]),
                                                 score=float(scores[p,j]),prediction_sign=int(pred[p,j]),
                                                 signed_margin=float(signed[p,j]),
                                                 correct=bool(pred[p,j]==y[p,j]),
                                                 near_zero=bool(abs(scores[p,j])<NEAR_ZERO_TOL)))
            run_status.append(dict(dataset=tag,learning_rate=rate,
                                   cumulative_clipping_events=int(np.sum(clip)),
                                   cumulative_suppressed_updates=int(np.sum(suppressed)),
                                   final_saturation_percent=epoch_rows[-1]["saturated_synapse_percent"],
                                   no_clipping=bool(np.sum(clip)==0 and epoch_rows[-1]["saturated_synapse_percent"]==0)))
    prior.write_csv(out/"csv"/"epoch_metrics.csv",epoch_rows)
    prior.write_csv(out/"csv"/"association_margin_by_epoch.csv",association_rows)
    prior.write_csv(out/"csv"/"bit_scores_by_epoch.csv",bit_rows)
    prior.write_csv(out/"csv"/"k10_learning_rate_sweep.csv",[r for r in epoch_rows if r["epoch"]==60])
    prior.write_csv(out/"csv"/"clipping_status.csv",run_status)
    return histories,epoch_rows,association_rows,bit_rows,run_status


def cached_event_matrices(b2,datasets):
    delta={}
    for tag,(x,y,x01) in datasets.items():
        mats=[]
        for p in range(10):
            pre=calibration.event_vector(b2,x01[p],TIMING)
            mat=np.column_stack([calibration.event_delta(b2,pre,
                    calibration.teacher_steps(b2,int(y[p,j]),TIMING)) for j in range(15)])
            assert np.array_equal(np.sign(mat),np.outer(x[p],y[p]))
            mats.append(mat)
        delta[tag]=np.stack(mats)
    return delta


def replay_subset(b2,ids,delta):
    w=np.full((15,15),b2.W_INIT,dtype=float)
    history=[w.copy()]
    rng=np.random.RandomState(SEED)
    clipping=0
    for ep in range(b2.EPOCHS):
        for p in rng.permutation(ids):
            prospective=w+delta[p]
            clipping+=int(np.sum((prospective<b2.W_MIN)|(prospective>b2.W_MAX)))
            w=np.clip(prospective,b2.W_MIN,b2.W_MAX)
        history.append(w.copy())
    return np.stack(history),clipping


def verify_replay(b2,datasets,delta,selected_rate,histories):
    errors=[]
    for tag,(x,y,x01) in datasets.items():
        cached,_=replay_subset(b2,tuple(range(10)),delta[tag])
        original=histories[(tag,selected_rate)][0]
        errors.append(float(np.max(np.abs(cached-original))))
        assert np.allclose(cached,original,atol=1e-11)
        for ids in ((0,1,2),(2,5,9)):
            cached,_=replay_subset(b2,ids,delta[tag])
            direct=calibration.train(b2,x,y,x01,ids,TIMING,True,"random_seed0")[0]
            errors.append(float(np.max(np.abs(cached-direct))))
            assert np.allclose(cached,direct,atol=1e-11)
    return max(errors)


def all_subset_study(b2,datasets,selected_rate,histories,out):
    b2.A_PLUS=b2.A_MINUS=selected_rate
    delta=cached_event_matrices(b2,datasets)
    replay_error=verify_replay(b2,datasets,delta,selected_rate,histories)
    rows,summary=[] ,[]
    for tag,(x,y,_) in datasets.items():
        for k in KS:
            weights=[];subsets=[];stats=[]
            for ids in itertools.combinations(range(10),k):
                idx=list(ids);xx=x[idx];yy=y[idx];corr=xx.T@yy
                trajectory,clipping=replay_subset(b2,ids,delta[tag])
                learned=trajectory[-1]
                m,score,pred,signed,exact,strict=metrics(xx,yy,learned,corr)
                _,cp,_,ce,cs=calibration.margins_and_recall(xx,yy,corr)
                sat=100*float(np.mean((np.abs(learned-b2.W_MIN)<=1e-12)|
                                      (np.abs(learned-b2.W_MAX)<=1e-12)))
                row=dict(dataset=tag,K=k,subset_ids=";".join(map(str,ids)),learning_rate=selected_rate,
                         correlation_exact_count=int(ce.sum()),
                         correlation_exact_percent=100*float(ce.mean()),
                         correlation_bit_percent=100*float(np.mean(cp==yy)),
                         correlation_strict_positive_margin_count=int(cs.sum()),
                         stdp_recalled_ids=";".join(str(ids[q]) for q in range(k) if exact[q]),
                         correlation_recalled_ids=";".join(str(ids[q]) for q in range(k) if ce[q]),
                         clipping_events=clipping,saturated_synapse_percent=sat,**m)
                rows.append(row);stats.append(row)
                weights.append(learned);subsets.append(ids)
            np.savez_compressed(out/"npz"/f"all_subsets_{tag}_K{k}.npz",
                                subset_ids=np.asarray(subsets,dtype=int),W_final=np.stack(weights),
                                W_initial=np.full((15,15),b2.W_INIT),
                                X_bipolar=x,Y_bipolar=y)
            summary.append(dict(dataset=tag,K=k,subset_count=len(stats),learning_rate=selected_rate,
                                correlation_mean_exact_percent=float(np.mean([r["correlation_exact_percent"] for r in stats])),
                                stdp_mean_exact_percent=float(np.mean([r["exact_percent"] for r in stats])),
                                correlation_mean_bit_percent=float(np.mean([r["correlation_bit_percent"] for r in stats])),
                                stdp_mean_bit_percent=float(np.mean([r["bit_percent"] for r in stats])),
                                subsets_with_clipping=sum(r["clipping_events"]>0 for r in stats),
                                max_clipping_events=max(r["clipping_events"] for r in stats),
                                max_saturation_percent=max(r["saturated_synapse_percent"] for r in stats)))
    prior.write_csv(out/"csv"/"all_subset_results.csv",rows)
    prior.write_csv(out/"csv"/"all_subset_summary.csv",summary)
    return summary,replay_error


def make_figures(epoch_rows,selected_rate,out):
    fdir=out/"figures"
    final=[r for r in epoch_rows if r["epoch"]==60]
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,tag in zip(axes,("A","B")):
        rows=sorted([r for r in final if r["dataset"]==tag],key=lambda r:r["learning_rate"])
        ax.plot([r["learning_rate"] for r in rows],[r["saturated_synapse_percent"] for r in rows],marker="o")
        ax.axvline(selected_rate,color="tab:green",ls="--",label="Selected no-clipping rate")
        ax.set(title=f"Dataset {tag}, K=10, 60 Epochs",xlabel="A_plus = A_minus")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel("Weight Saturation (%)")
    fig.suptitle("Learning Rate vs Weight Saturation");fig.tight_layout();fig.savefig(fdir/"learning_rate_vs_saturation.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,tag,baseline in zip(axes,("A","B"),(40,20)):
        rows=sorted([r for r in final if r["dataset"]==tag],key=lambda r:r["learning_rate"])
        ax.plot([r["learning_rate"] for r in rows],[r["exact_percent"] for r in rows],marker="o",label="Bounded Shared-Spike STDP")
        ax.axhline(baseline,color="black",ls=":",label="Original Correlation")
        ax.axvline(selected_rate,color="tab:green",ls="--",label="Selected no-clipping rate")
        ax.set(title=f"Dataset {tag}, K=10, 60 Epochs",xlabel="A_plus = A_minus",ylim=(0,100))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel("Exact Recall Accuracy (%)")
    fig.suptitle("Learning Rate vs Static Recall Accuracy");fig.tight_layout();fig.savefig(fdir/"learning_rate_vs_accuracy.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,tag in zip(axes,("A","B")):
        for rate in RATES:
            rows=[r for r in epoch_rows if r["dataset"]==tag and r["learning_rate"]==rate]
            ax.plot([r["epoch"] for r in rows],[r["normalized_fro_error"] for r in rows],label=f"A={rate:.4f}")
        ax.set(title=f"Dataset {tag}, K=10",xlabel="Learning Epoch")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel("Normalized Frobenius Error")
    fig.suptitle("Weight Error vs Learning Epoch");fig.tight_layout();fig.savefig(fdir/"weight_error_by_epoch.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ri,tag in enumerate(("A","B")):
        with np.load(out/"npz"/f"K10_{tag}_rate_{key_rate(selected_rate)}.npz") as d:
            corr=d["W_correlation"];effective=d["W_effective"]
        scale=prior.similarity(effective,corr)["scale"]
        for ci,(mat,title) in enumerate(((scale*corr,"Scaled Correlation"),(effective,"Learned Effective Weight"),(effective-scale*corr,"Weight Difference"))):
            lim=max(np.max(np.abs(scale*corr)),np.max(np.abs(effective))) if ci<2 else max(np.max(np.abs(mat)),1e-12)
            axes[ri,ci].imshow(mat,cmap="coolwarm",vmin=-lim,vmax=lim)
            axes[ri,ci].set(title=f"Dataset {tag}: {title}",xlabel="Output neuron",ylabel="Input neuron")
    fig.suptitle(f"K=10, A_plus=A_minus={selected_rate:.4f}, 60 Epochs, Original Bounds")
    fig.tight_layout();fig.savefig(fdir/"correlation_vs_learned_weight.png",dpi=150);plt.close(fig)


def write_summary(selected_rate,epoch_rows,associations,bit_rows,status,subset_summary,replay_error,out):
    lines=["# Shared-Spike STDP Learning Rate Calibration","",
           "The original B2 `stdp_update` was called at every K=10 presentation. Only its imported module parameters `A_PLUS=A_MINUS` were varied; original simulator files were not edited. Teacher post events were scheduled, and natural postsynaptic spikes were excluded from plasticity.","",
           "## Fixed protocol","",
           "- Shared pre +1@3 ms, −1@7 ms; teacher target +1@5 ms, target −1@1.3/8.7 ms. One pre-event vector is used for every output column. Initial raw weight 0.05, bounds [−1.5,+1.5], 60 epochs, seed 0 and the same per-epoch permutation for every rate.",
           "- Rates tested: 0.0200, 0.0100, 0.0075, 0.0050 with equal LTP/LTD amplitude. Timing, tau, all other parameters and target mapping were fixed. Static readout is `X_bipolar @ (W_final−W_initial)` with +1 iff score>0.",
           "- Selected rate is the **largest rate with zero clipping attempts and zero final saturation in both A and B at K=10**, chosen without looking at recall accuracy. Epoch checkpoints 1,5,10,20,30,40,50,60 come from the same trajectory.","",
           "## K=10 final measurements","",
           "| Dataset | Rate | Exact | Bit accuracy | Saturated | Cumulative clipping | Suppressed updates | Cosine | Normalized error | Scale |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag in ("A","B"):
        for rate in RATES:
            r=next(z for z in epoch_rows if z["dataset"]==tag and z["learning_rate"]==rate and z["epoch"]==60)
            lines.append(f"| {tag} | {rate:.4f} | {r['exact_count']}/10 | {r['bit_percent']:.2f}% | {r['saturated_synapse_percent']:.1f}% | {r['cumulative_clipping_events']} | {r['cumulative_suppressed_updates']} | {r['cosine']:.6f} | {r['normalized_fro_error']:.6f} | {r['scale']:.6f} |")
    lines += ["",f"Selected no-clipping rate: **{selected_rate:.4f}**.","",
              "## Same-trajectory epoch effects","",
              "`csv/epoch_metrics.csv` contains all 61 epoch states for every rate. `csv/association_margin_by_epoch.csv` and `csv/bit_scores_by_epoch.csv` retain the requested checkpoints, including each digit's minimum signed margin. The best-fit scale/weight norm track cumulative update magnitude. Clipping and suppressed-update counters isolate saturation. The four-case timing imbalance remains fixed at 0.458653%; reducing rate scales it but cannot remove it. Near-zero scores (|score|<1e−10) are labeled for numerical interpretation only; the primary `score>0` decision is unchanged.","",
              "## All-subset check at selected rate","",
              f"Every combination for K={list(KS)} was replayed using source-function event deltas and the original per-presentation clipping. A direct source-call trajectory matched replay for both K=10 datasets and two smaller representative subsets per dataset (maximum absolute discrepancy {replay_error:.3g}); this fast replay changes no update rule. Fixed-prefix K=10 runs remain direct source calls.","",
              "| Dataset | K | Subsets | Correlation mean exact | STDP mean exact | Subsets clipped |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in subset_summary:
        lines.append(f"| {r['dataset']} | {r['K']} | {r['subset_count']} | {r['correlation_mean_exact_percent']:.3f}% | {r['stdp_mean_exact_percent']:.3f}% | {r['subsets_with_clipping']} |")
    lines += ["", "The all-subset mean is distinct from a fixed-prefix result. Each association subset uses the same digit-letter pairs for correlation and STDP; all raw subset outcomes and final matrices are saved in CSV/NPZ.","",
              "## K=10 score-boundary diagnostic","",]
    for tag in ("A","B"):
        for rate in RATES:
            r=next(z for z in epoch_rows if z["dataset"]==tag and z["learning_rate"]==rate and z["epoch"]==60)
            recalled=[a["digit"] for a in associations if a["dataset"]==tag and a["learning_rate"]==rate and a["epoch"]==60 and a["exact"]]
            lines.append(f"- Dataset {tag}, rate {rate:.4f}: exact {r['exact_count']}/10, strictly-positive-margin {r['strictly_positive_margin_count']}/10, |score|<1e−10 bit count {r['near_zero_score_count']}, recalled digits {recalled}.")
    boundary=[r for r in bit_rows if r["dataset"]=="A" and r["digit"]==0 and r["output_neuron"]==13 and r["epoch"]==60]
    lines.append("- Dataset A digit 0/output 13 (target −1; pure-correlation score exactly 0): "+"; ".join(f"A={r['learning_rate']:.4f}, score={r['score']:+.9g}, signed margin={r['signed_margin']:+.9g}" for r in boundary)+". This threshold boundary can change a bit decision under tiny score shifts; it is reported separately from clipping and structural weight error.")
    lines += ["", "## Interpretation and scope", "",
              "Clipping prevention, near-proportional weight geometry, static recall reproduction and capacity improvement are separate findings. Reproducing correlation recall is **not** evidence of memory capacity beyond correlation. The baseline is suitable as a controlled teacher-event update reference if the selected rate has no clipping across the tested subsets; it is not a validated natural-post or LIF memory model. No noisy cues, error feedback or new learning rule were used.","",
              "## Reproduce", "", "```bash",
              "python3 my_research/experiments/diagnostics/shared_spike_learning_rate_calibration.py \\",
              "  --dataset-a local_stdp_audit_dataset_handoff_20261020/dataset_A.npz \\",
              "  --dataset-b local_stdp_audit_dataset_handoff_20261020/dataset_B.npz", "```", ""]
    (out/"summary.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset-a",type=Path,required=True)
    parser.add_argument("--dataset-b",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    out=args.out or ROOT/"results"/"shared_spike_learning_rate_calibration"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True,exist_ok=False)
    for d in ("scripts","config","csv","npz","figures"):(out/d).mkdir()
    for script in (__file__,prior.__file__,calibration.__file__):
        shutil.copy2(script,out/"scripts"/Path(script).name)
    b2=prior.source_module()
    assert (b2.W_INIT,b2.W_MIN,b2.W_MAX,b2.EPOCHS,b2.DT)==(0.05,-1.5,1.5,60,0.1)
    datasets={"A":prior.read_dataset(args.dataset_a),"B":prior.read_dataset(args.dataset_b)}
    np.savez_compressed(out/"npz"/"datasets.npz",
                        A_X_bipolar=datasets["A"][0],A_Y_bipolar=datasets["A"][1],A_X_binary=datasets["A"][2],
                        B_X_bipolar=datasets["B"][0],B_Y_bipolar=datasets["B"][1],B_X_binary=datasets["B"][2])
    config=dict(git_revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip(),
                original_source=str(prior.SOURCE.relative_to(ROOT)),
                original_source_sha256=hashlib.sha256(prior.SOURCE.read_bytes()).hexdigest(),
                dataset_files={tag:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                               for tag,p in (("A",args.dataset_a),("B",args.dataset_b))},
                learning_rates=list(RATES),selected_rate_rule="largest rate with zero K10 clipping and saturation on both datasets",
                timing=TIMING,seed=SEED,epochs=b2.EPOCHS,initial_weight=b2.W_INIT,
                weight_bounds=[b2.W_MIN,b2.W_MAX],tau_stdp_ms=b2.TAU_STDP,DT_ms=b2.DT,
                K=list(KS),epoch_checkpoints=list(CHECKPOINTS),
                natural_posts="excluded from plasticity",decoder="score>0 => +1, else -1",
                near_zero_tolerance=NEAR_ZERO_TOL,
                subset_replay="cached deltas computed with original stdp_update; clipped after each presentation; validated against direct source calls")
    (out/"config"/"experiment_config.json").write_text(json.dumps(config,indent=2)+"\n")
    histories,epoch_rows,associations,bit_rows,status=rate_sweep(b2,datasets,out)
    eligible=[rate for rate in RATES if all(s["no_clipping"] for s in status if s["learning_rate"]==rate)]
    if not eligible:
        (out/"summary.md").write_text("No tested rate prevented clipping at K=10 in both datasets; full subset study was not run. See K10 CSV files.\n")
        print(json.dumps(dict(result_directory=str(out),status="no eligible no-clipping rate")))
        return
    selected_rate=max(eligible)
    subset_summary,replay_error=all_subset_study(b2,datasets,selected_rate,histories,out)
    make_figures(epoch_rows,selected_rate,out)
    write_summary(selected_rate,epoch_rows,associations,bit_rows,status,subset_summary,replay_error,out)
    archive=out.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():z.write(p,p.relative_to(out.parent))
    print(json.dumps(dict(result_directory=str(out),archive=str(archive),selected_rate=selected_rate,
                          all_subset_counts={str(r["K"]):r["subset_count"] for r in subset_summary if r["dataset"]=="A"},
                          k10={tag:{str(rate):next(r["exact_count"] for r in epoch_rows if r["dataset"]==tag and r["learning_rate"]==rate and r["epoch"]==60) for rate in RATES} for tag in ("A","B")}),indent=2))


if __name__=="__main__":main()
