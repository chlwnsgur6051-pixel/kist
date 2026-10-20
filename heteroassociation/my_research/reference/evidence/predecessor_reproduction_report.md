# Baseline verification report

Original PyTorch script could not run because PyTorch is unavailable in this environment. A NumPy wrapper executes the archived linear pseudoinverse formula and identical one-step decision rule, using the documented pre-fix digit 9. Archived prediction matrix reproduced exactly: **True**. Archived 6/10 (0.9000 bit accuracy); wrapper 6/10 (0.9000). See `reproduction_pair_predictions.csv` for all 10 target/prediction rows and `raw/` for W, input, target.

Original conditions: predecessor pre-fix = 6/10; current bipolar correlation = 1/10. Common pre-fix dataset with original decision rules: predecessor 6/10 vs correlation 3/10. Common current dataset with original decision rules: predecessor 10/10 vs correlation 1/10. Additional common `score>0` decision: pre-fix predecessor 6/10 vs correlation 3/10; current predecessor 10/10 vs correlation 1/10. The latter is a diagnostic, not the archived predecessor evaluation. These comparisons isolate dataset and threshold differences as far as possible, but representation, bias, and learning rule still differ. Both use the same 15-bit target and exact metric.

Digit 1→B B2 raw ΔW: c=0.343805756232, Pearson=1, cosine=1, sign agreement=1.000000, relative Frobenius error=1.15e-16, RMSE=3.97e-17. All 225 synapses are in `single_pair_synapse_raw.csv`. Representative first five and last epoch updates are in `representative_synapse_updates.csv`, and all 60 points in `representative_weight_trajectories.csv`. These are instrumented actual calls to the original STDP updater from the previous diagnosis; that run verified instrumented versus uninstrumented final weights. This script reuses the recorded events rather than re-running B2. No 1-pair clipping.

Multi-pair selected archived W comparisons:

| K | subset | c | sign agreement | relative error | clip events | static exact |
|---:|---|---:|---:|---:|---:|
| 1 | 1 | 0.343806 | 1.0000 | 0.0000 | 0 | 1/1 |
| 3 | 0-1-2 | 0.343806 | 1.0000 | 0.0000 | 0 | 3/3 |
| 5 | 0-1-2-3-4 | 0.334763 | 1.0000 | 0.0581 | 436 | 0/5 |
| 10 | 0-1-2-3-4-5-6-7-8-9 | 0.254982 | 1.0000 | 0.2435 | 8931 | 1/10 |

These are representative existing subsets, not means of the full 383-subset sweep. Clip events include transient update clipping. K=5/10 deviation is consistent with clipping but this table alone does not apportion every residual to clipping versus order effects.

## Direct answers

**Q1.** The archived predecessor 6/10 is linear contrastive pseudoinverse of 0/1 digit features plus bias to bipolar teacher currents, one-step LIF threshold 0.8 on the pre-rank-fix dataset. The `contrastive_quadratic` run name conflicts with the JSON's `basis=linear`; the JSON is authoritative for this run.

**Q2.** No. The current 1/10 is a direct bipolar outer-product correlation memory with static score>0 and rank-10 digit data. They differ in learning construction, representation, bias/decision rule, and dataset version.

**Q3.** The most direct demonstrated difference is learning/readout method: on the same pre-fix dataset, predecessor {old_ex}/10 versus correlation {corr_old_ex}/10; on the same current dataset, predecessor {current_pred_ex}/10 versus correlation {corr_current_ex}/10. Dataset rank change is an additional difference, quantified above.

**Q4.** Yes for Digit 1→B: all 225 ΔW signs agree with theoretical bipolar outer product, c={sim1['c']:.12g}, relative error {sim1['relative_frobenius_error']:.3g}; raw epoch updates show all four bit combinations.

**Q5.** K=1 and selected K=3 retain proportionality to numerical precision. Selected K=5 and K=10 depart, with recorded clipping. See `multipair_weight_comparison.csv`; full-subset onset varies and is not inferred from these four subsets alone.

**Q6.** The narrow supported statement is that B2's 1-pair synaptic update implements the intended four sign directions and produces ΔW proportional to the outer product; this is backed by raw initial/final weights, 60 epoch trajectories and actual updater events. It does not imply multi-pair recall is correct or that the temporal LIF readout succeeds.

Limitations: PyTorch original-script numerical reproduction was unavailable; NumPy float32 pseudoinverse is equivalent mathematically and was checked against archived predictions. The archived JSON does not preserve original W or input array. No STDP/LIF parameters were tuned.
