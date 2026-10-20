"""Audit archived predecessor and recorded B2 synapse events without editing originals.

Run: MPLCONFIGDIR=/tmp/mpl_baseline python my_research/experiments/diagnostics/verify_baseline.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_baseline")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import binary_character as data

OLD = ROOT / "Ver관리/1. stdp_dev/results/contrastive_optimal_weight/contrastive_quadratic.json"
ONE = ROOT / "my_research/results/diagnosis_1pair_digit1_20261007_091816_264144"
MULTI = ROOT / "my_research/results/diagnosis_multipair_crosstalk/full_20261007_111051_279590"
OUT = ROOT / "my_research/results/baseline_verification" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUT.mkdir(parents=True)
(OUT / "figures").mkdir()
(OUT / "raw").mkdir()


def write_csv(name, rows):
    rows = list(rows)
    if not rows:
        return
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate(pred, y):
    return int(np.all(pred == y, axis=1).sum()), float(np.mean(pred == y))


def similarity(a, b):
    c = float(np.vdot(a, b) / np.vdot(a, a))
    r = b - c * a
    return dict(c=c, pearson=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                cosine=float(np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
                sign_agreement=float(np.mean(np.sign(a) == np.sign(b))),
                max_abs_residual=float(np.max(np.abs(r))), rmse=float(np.sqrt(np.mean(r*r))),
                relative_frobenius_error=float(np.linalg.norm(r) / np.linalg.norm(b)))


X = data.make_digit_dataset()[0].reshape(10, 15).astype(np.int64)
Y = data.make_letter_dataset()[0].reshape(10, 15).astype(np.int64)
Xold = X.copy()
Xold[9, 12:15] = 1  # Explicitly documented pre-rank-fix final row 111.
archive = json.loads(OLD.read_text())


def predecessor(x):
    # Exact archived algorithm: linear 0/1 input plus bias, bipolar teacher current,
    # Moore-Penrose solution, clip 20, one LIF step, threshold 0.8, count >= 1.
    phi = np.c_[x.astype(np.float32), np.ones(10, dtype=np.float32)]
    teacher = (2 * Y - 1).astype(np.float32)
    wb = np.linalg.pinv(phi).astype(np.float32) @ teacher
    wb = np.clip(wb, -20, 20)
    current = phi @ wb
    pred = (current >= np.float32(0.8)).astype(np.int64)
    return wb, current, pred


wb_old, current_old, pred_old = predecessor(Xold)
wb_current, current_current, pred_current = predecessor(X)
arch_pred = np.array([np.array(r["pred"]).ravel() for r in archive["pair_rows"]])
archive_match = bool(np.array_equal(pred_old, arch_pred))
archive_ex, archive_ba = evaluate(arch_pred, Y)
old_ex, old_ba = evaluate(pred_old, Y)
current_pred_ex, current_pred_ba = evaluate(pred_current, Y)
np.save(OUT / "raw/predecessor_old_weight_bias.npy", wb_old)
np.save(OUT / "raw/predecessor_old_input.npy", Xold)
np.save(OUT / "raw/predecessor_current_weight_bias.npy", wb_current)
np.save(OUT / "raw/current_input.npy", X)
np.save(OUT / "raw/target.npy", Y)


def correlation(x):
    xb, yb = 2*x-1, 2*Y-1
    w = xb.T @ yb
    scores = xb @ w
    return w, scores, (scores > 0).astype(np.int64)


wc_old, scores_old, corr_old = correlation(Xold)
wc_current, scores_current, corr_current = correlation(X)
corr_old_ex, corr_old_ba = evaluate(corr_old, Y)
corr_current_ex, corr_current_ba = evaluate(corr_current, Y)
common_old = (current_old > 0).astype(np.int64)
common_current = (current_current > 0).astype(np.int64)
common_old_ex, common_old_ba = evaluate(common_old, Y)
common_current_ex, common_current_ba = evaluate(common_current, Y)
np.save(OUT / "raw/correlation_old_weight.npy", wc_old)
np.save(OUT / "raw/correlation_current_weight.npy", wc_current)

pair_rows = []
for i in range(10):
    for label, p in [("archived_predecessor", arch_pred), ("reproduced_predecessor_old", pred_old),
                     ("predecessor_current", pred_current), ("correlation_old", corr_old),
                     ("correlation_current", corr_current),
                     ("predecessor_old_common_score_gt0", common_old),
                     ("predecessor_current_common_score_gt0", common_current)]:
        pair_rows.append(dict(experiment=label, digit=i, letter=chr(65+i),
                              input_bits="".join(map(str, (Xold if "old" in label or label == "archived_predecessor" else X)[i])),
                              target="".join(map(str, Y[i])), prediction="".join(map(str, p[i])),
                              exact=int(np.array_equal(p[i], Y[i])), bit_accuracy=float(np.mean(p[i] == Y[i]))))
write_csv("reproduction_pair_predictions.csv", pair_rows)

initial = np.load(ONE / "weights_initial.npy")
learned = np.load(ONE / "weights_learned.npy")
delta = np.load(ONE / "weights_delta.npy")
assert np.allclose(learned-initial, delta, atol=1e-13)
assert delta.shape == (15, 15)
theory = np.outer(2*X[1]-1, 2*Y[1]-1)
sim1 = similarity(theory, delta)
raw_rows = []
for i in range(15):
    for j in range(15):
        raw_rows.append(dict(input_neuron_index=i, output_neuron_index=j, input_bit=int(X[1,i]),
                             target_bit=int(Y[1,j]), bipolar_input=int(2*X[1,i]-1),
                             bipolar_target=int(2*Y[1,j]-1), theoretical_correlation_sign=int(theory[i,j]),
                             initial_weight=float(initial[i,j]), final_weight=float(learned[i,j]),
                             delta_weight=float(delta[i,j]), expected_stdp_direction="LTP" if theory[i,j]>0 else "LTD",
                             actual_stdp_direction="LTP" if delta[i,j]>0 else "LTD" if delta[i,j]<0 else "zero"))
write_csv("single_pair_synapse_raw.csv", raw_rows)

events = read_csv(ONE / "stdp_events.csv")
assert len(events) == 15*15*60
selected = {}
for a,b in [(1,1),(1,0),(0,1),(0,0)]:
    selected[(a,b)] = next((i,j) for i in range(15) for j in range(15) if X[1,i]==a and Y[1,j]==b)
update_rows, trajectory_rows = [], []
for (a,b),(i,j) in selected.items():
    evs = sorted((e for e in events if int(e["input_idx"])==i and int(e["output_idx"])==j), key=lambda e:int(e["epoch"]))
    assert len(evs)==60
    for e in evs:
        epoch = int(e["epoch"])+1
        row = dict(epoch=epoch, pair="1->B", input_index=i, output_index=j, input_bit=a, target_bit=b,
                   pre_time_ms=float(e["pre_time_ms"]), teacher_post_time_ms=float(e["teacher_time_ms"]),
                   delta_t_ms=float(e["teacher_time_ms"])-float(e["pre_time_ms"]),
                   delta_w=float(e["actual_step_delta"]), weight_before=float(e["weight_before"]),
                   weight_after=float(e["weight_after"]))
        trajectory_rows.append(row)
        if epoch <= 5 or epoch == 60:
            update_rows.append(row)
    assert np.isclose(float(evs[-1]["weight_after"]), learned[i,j])
write_csv("representative_synapse_updates.csv", update_rows)
write_csv("representative_weight_trajectories.csv", trajectory_rows)

for name, arr, title in [("theoretical_correlation_heatmap.png", theory, "Digit 1→B bipolar outer product"),
                         ("learned_stdp_delta_heatmap.png", delta, "B2 learned ΔW"),
                         ("residual_heatmap.png", delta-sim1["c"]*theory, "ΔW − c Wcorr")]:
    plt.figure(figsize=(6,5)); plt.imshow(arr,cmap="coolwarm",vmin=-max(abs(arr.min()),abs(arr.max())),vmax=max(abs(arr.min()),abs(arr.max())))
    plt.colorbar(); plt.xlabel("Output neuron index");plt.ylabel("Input neuron index");plt.title(title);plt.tight_layout()
    plt.savefig(OUT/"figures"/name,dpi=160);plt.close()
plt.figure(figsize=(5,4));plt.scatter(theory.ravel(),delta.ravel(),s=12);plt.xlabel("Wcorr");plt.ylabel("B2 ΔW");plt.title("All 225 synapses");plt.tight_layout();plt.savefig(OUT/"figures/correlation_vs_stdp_scatter.png",dpi=160);plt.close()
plt.figure(figsize=(7,4))
for (a,b),(i,j) in selected.items():
    es=[r for r in trajectory_rows if r["input_index"]==i and r["output_index"]==j]
    plt.plot([0]+[e["epoch"] for e in es],[initial[i,j]]+[e["weight_after"] for e in es],label=f"x={a}, y={b} ({i},{j})")
plt.xlabel("Epoch");plt.ylabel("Weight");plt.title("B2 teacher-only synapse trajectories");plt.legend();plt.tight_layout();plt.savefig(OUT/"figures/representative_weight_trajectories.png",dpi=160);plt.close()

write_csv("matrix_similarity.csv", [dict(pair="1->B",**sim1, clipping_count=int(np.sum((learned <= -1.5+1e-10)|(learned >= 1.5-1e-10))))])
multi_rows=[]
for k, subset in [(1,"1"),(3,"0-1-2"),(5,"0-1-2-3-4"),(10,"0-1-2-3-4-5-6-7-8-9")]:
    d=MULTI/"selected_weights"/f"K{k}_{subset}"
    a=np.load(d/"weights_correlation.npy");b=np.load(d/"weights_delta.npy");w=np.load(d/"weights_learned.npy")
    idx=[int(z) for z in subset.split("-")]
    p=((2*X[idx]-1)@b > 0).astype(int)
    ex,ba=evaluate(p,Y[idx]); ss=similarity(a,b)
    # Read recorded clip events, including transient clips, from the original sweep.
    diag=next(r for r in read_csv(MULTI/"weight_diagnostics.csv") if int(r["k"])==k and r["subset"]==subset)
    multi_rows.append(dict(k=k,subset=subset,static_exact_count=ex,static_bit_accuracy=ba,
                           clipping_count=int(diag["clip_events"]),bound_weight_count=int(diag["bound_weight_count"]),
                           weight_min=float(w.min()),weight_max=float(w.max()),**ss))
write_csv("multipair_weight_comparison.csv",multi_rows)

comparison=[]
for label,dataset,repr_,learn,recall,ex,ba in [
    ("Predecessor linear contrastive pinv / one-step LIF / pre-rank-fix dataset","pre-rank-fix digit 9 111","0/1 with bias","pseudoinverse bipolar teacher current","one-step LIF threshold 0.8",old_ex,old_ba),
    ("Direct bipolar correlation / static score>0 / pre-rank-fix dataset","pre-rank-fix digit 9 111","bipolar","outer-product sum","static score>0",corr_old_ex,corr_old_ba),
    ("Predecessor linear contrastive pinv / one-step LIF / current rank-10 dataset","current digit 9 001","0/1 with bias","pseudoinverse bipolar teacher current","one-step LIF threshold 0.8",current_pred_ex,current_pred_ba),
    ("Direct bipolar correlation / static score>0 / current rank-10 dataset","current digit 9 001","bipolar","outer-product sum","static score>0",corr_current_ex,corr_current_ba),
    ("Predecessor linear contrastive pinv / common static score>0 / pre-rank-fix dataset","pre-rank-fix digit 9 111","0/1 with bias","pseudoinverse bipolar teacher current","static score>0",common_old_ex,common_old_ba),
    ("Predecessor linear contrastive pinv / common static score>0 / current rank-10 dataset","current digit 9 001","0/1 with bias","pseudoinverse bipolar teacher current","static score>0",common_current_ex,common_current_ba)]:
    comparison.append(dict(experiment=label,dataset=dataset,representation=repr_,learning=learn,recall=recall,K=10,exact=f"{ex}/10",bit_accuracy=ba))
write_csv("experiment_comparison.csv",comparison)

hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"binary_character.py",OLD,ROOT/"Ver관리/0. origin/contrastive_optimal_weight.py",ROOT/"Ver관리/2. temporal/stdp_temporal_symmetric_teacher_only.py"]}
metrics=dict(status="success", archived_exact=archive_ex,archived_bit_accuracy=archive_ba,reproduction_exact=old_ex,
             reproduction_bit_accuracy=old_ba, archived_pair_predictions_match=archive_match,
             correlation_old_exact=corr_old_ex,correlation_current_exact=corr_current_ex,
             common_score_gt0_old_predecessor_exact=common_old_ex,
             common_score_gt0_current_predecessor_exact=common_current_ex,
             predecessor_current_exact=current_pred_ex,single_pair_similarity=sim1,multipair=multi_rows,
             ranks=dict(current_binary=int(np.linalg.matrix_rank(X)),old_binary=int(np.linalg.matrix_rank(Xold)),
                        current_augmented=int(np.linalg.matrix_rank(np.c_[X,np.ones(10)])),
                        old_augmented=int(np.linalg.matrix_rank(np.c_[Xold,np.ones(10)]))),
             source_sha256=hashes,python=sys.version,numpy=np.__version__)
write_json("metrics.json",metrics)
write_json("config.json",dict(command="MPLCONFIGDIR=/tmp/mpl_baseline python my_research/experiments/diagnostics/verify_baseline.py",
                              predecessor_basis="linear",predecessor_teacher_currents=[-1,1],predecessor_bias=True,
                              predecessor_weight_clip=20,predecessor_threshold=0.8,predecessor_steps=1,
                              predecessor_count_threshold=1,correlation_tie="score=0 -> 0",
                              b2_source_run=str(ONE.relative_to(ROOT)),multipair_source_run=str(MULTI.relative_to(ROOT))))

setup=f"""# Experiment setup

Dataset: `binary_character.py` (SHA256 `{hashes['binary_character.py']}`), 10 digit and 10 letter 5×3 binary patterns. Current digit 9 ends in `001`; documented pre-fix digit 9 ends in `111`. Current raw input rank {metrics['ranks']['current_binary']}; pre-fix rank {metrics['ranks']['old_binary']}.

Association: 0→A, 1→B, 2→C, 3→D, 4→E, 5→F, 6→G, 7→H, 8→I, 9→J. All 10 share one W in K=10 comparisons.

Exact patterns (row-major, 15 bits):

| Pair | Digit | Letter |
|---|---|---|
"""+"\n".join(f"| {i}→{chr(65+i)} | {''.join(map(str,X[i]))} | {''.join(map(str,Y[i]))} |" for i in range(10))+f"""

Predecessor: linear raw 0/1 features plus bias, teacher current `2Y−1`, Moore–Penrose pseudoinverse, clip ±20, one LIF step with threshold 0.8, tau 0.02 s, dt 0.001 s, reset 0, count ≥1. Since initial membrane is 0 and there is one step, prediction is equivalent to `XW+b≥0.8`. Archived JSON reports feature rank 9 and matches the pre-fix digit 9.

Current correlation: `Xb=2X−1`, `Yb=2Y−1`, `W=Xb.T@Yb`; predict 1 iff `Xb@W>0` (tie=0), no bias. B2: 2/7 ms input spikes, teacher 4.5 ms only during learning, target-0 timing inversion, 60 epochs, initial weight 0.05, A±=0.02, tau_STDP=2 ms, bounds ±1.5. B2 static diagnostics read `Xb@ΔW>0`. Original LIF any-spike readout is separate and not used for raw-weight similarity.

Exact association means all 15 output bits match. Bit accuracy is correct bits divided by all evaluated bits. Inputs/outputs and raw weights are in `raw/` and the CSV files.
"""
(OUT/"experiment_setup.md").write_text(setup,encoding="utf-8")
trace=f"""# Provenance of the alleged 6/10

The archived `Ver관리/1. stdp_dev/results/contrastive_optimal_weight/contrastive_quadratic.json` records exact_count=6, mean_bit_acc=0.9, `basis=linear`, feature_dim=15, augmented rank=9. The run name `contrastive_quadratic` is misleading: its actual configuration is linear. Original algorithm is in `Ver관리/0. origin/contrastive_optimal_weight.py`; archive does not retain a command line or W. Successes in archived JSON: {', '.join(r['pair'] for r in archive['pair_rows'] if r['exact'])}. This is 10 associations in one pseudoinverse W, not 10 independent one-pair trainings.

Separately, `Ver관리/2. temporal/results_pair_count_sweep/run.log` says perfect subsets 6/10 for K=1. Those are ten independently trained B2 networks; this is a different 6/10 and cannot be interpreted as shared K=10 recall.

Current dataset has rank 10 after digit-9 final row `111`→`001`; archived rank 9 matches the documented older version. The archived JSON does not contain raw input arrays, so older version identification relies on rank, code comment, and pairwise prediction reproduction, not a retained input hash.
"""
(OUT/"predecessor_6of10_trace.md").write_text(trace,encoding="utf-8")
report=f"""# Baseline verification report

Original PyTorch script could not run because PyTorch is unavailable in this environment. A NumPy wrapper executes the archived linear pseudoinverse formula and identical one-step decision rule, using the documented pre-fix digit 9. Archived prediction matrix reproduced exactly: **{archive_match}**. Archived {archive_ex}/10 ({archive_ba:.4f} bit accuracy); wrapper {old_ex}/10 ({old_ba:.4f}). See `reproduction_pair_predictions.csv` for all 10 target/prediction rows and `raw/` for W, input, target.

Original conditions: predecessor pre-fix = {old_ex}/10; current bipolar correlation = {corr_current_ex}/10. Common pre-fix dataset with original decision rules: predecessor {old_ex}/10 vs correlation {corr_old_ex}/10. Common current dataset with original decision rules: predecessor {current_pred_ex}/10 vs correlation {corr_current_ex}/10. Additional common `score>0` decision: pre-fix predecessor {common_old_ex}/10 vs correlation {corr_old_ex}/10; current predecessor {common_current_ex}/10 vs correlation {corr_current_ex}/10. The latter is a diagnostic, not the archived predecessor evaluation. These comparisons isolate dataset and threshold differences as far as possible, but representation, bias, and learning rule still differ. Both use the same 15-bit target and exact metric.

Digit 1→B B2 raw ΔW: c={sim1['c']:.12g}, Pearson={sim1['pearson']:.12g}, cosine={sim1['cosine']:.12g}, sign agreement={sim1['sign_agreement']:.6f}, relative Frobenius error={sim1['relative_frobenius_error']:.3g}, RMSE={sim1['rmse']:.3g}. All 225 synapses are in `single_pair_synapse_raw.csv`. Representative first five and last epoch updates are in `representative_synapse_updates.csv`, and all 60 points in `representative_weight_trajectories.csv`. These are instrumented actual calls to the original STDP updater from the previous diagnosis; that run verified instrumented versus uninstrumented final weights. This script reuses the recorded events rather than re-running B2. No 1-pair clipping.

Multi-pair selected archived W comparisons:

| K | subset | c | sign agreement | relative error | clip events | static exact |
|---:|---|---:|---:|---:|---:|
"""+"\n".join(f"| {r['k']} | {r['subset']} | {r['c']:.6g} | {r['sign_agreement']:.4f} | {r['relative_frobenius_error']:.4f} | {r['clipping_count']} | {r['static_exact_count']}/{r['k']} |" for r in multi_rows)+"""

These are representative existing subsets, not means of the full 383-subset sweep. Clip events include transient update clipping. K=5/10 deviation is consistent with clipping but this table alone does not apportion every residual to clipping versus order effects.

## Direct answers

**Q1.** The archived predecessor 6/10 is linear contrastive pseudoinverse of 0/1 digit features plus bias to bipolar teacher currents, one-step LIF threshold 0.8 on the pre-rank-fix dataset. The `contrastive_quadratic` run name conflicts with the JSON's `basis=linear`; the JSON is authoritative for this run.

**Q2.** No. The current 1/10 is a direct bipolar outer-product correlation memory with static score>0 and rank-10 digit data. They differ in learning construction, representation, bias/decision rule, and dataset version.

**Q3.** The most direct demonstrated difference is learning/readout method: on the same pre-fix dataset, predecessor {old_ex}/10 versus correlation {corr_old_ex}/10; on the same current dataset, predecessor {current_pred_ex}/10 versus correlation {corr_current_ex}/10. Dataset rank change is an additional difference, quantified above.

**Q4.** Yes for Digit 1→B: all 225 ΔW signs agree with theoretical bipolar outer product, c={sim1['c']:.12g}, relative error {sim1['relative_frobenius_error']:.3g}; raw epoch updates show all four bit combinations.

**Q5.** K=1 and selected K=3 retain proportionality to numerical precision. Selected K=5 and K=10 depart, with recorded clipping. See `multipair_weight_comparison.csv`; full-subset onset varies and is not inferred from these four subsets alone.

**Q6.** The narrow supported statement is that B2's 1-pair synaptic update implements the intended four sign directions and produces ΔW proportional to the outer product; this is backed by raw initial/final weights, 60 epoch trajectories and actual updater events. It does not imply multi-pair recall is correct or that the temporal LIF readout succeeds.

Limitations: PyTorch original-script numerical reproduction was unavailable; NumPy float32 pseudoinverse is equivalent mathematically and was checked against archived predictions. The archived JSON does not preserve original W or input array. No STDP/LIF parameters were tuned.
"""
(OUT/"reproduction_report.md").write_text(report,encoding="utf-8")
(OUT/"README.md").write_text("# Baseline verification\n\nRun: `MPLCONFIGDIR=/tmp/mpl_baseline python my_research/experiments/diagnostics/verify_baseline.py` from project root. This timestamped folder contains predecessor reproduction, pair predictions, raw STDP synapses, matrix comparisons, setup, figures and report.\n",encoding="utf-8")
zip_path=OUT.with_suffix(".zip")
with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file(): z.write(p,p.relative_to(OUT.parent))
print(json.dumps(dict(output=str(OUT),zip=str(zip_path),archived_match=archive_match,old=old_ex,current_correlation=corr_current_ex,single_pair=sim1,multipair=multi_rows),ensure_ascii=False,indent=2))
