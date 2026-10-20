# Shared-Spike STDP Timing Calibration

This is an event-timing calibration of the **unmodified** B2 `stdp_update`. Timing selection used only four-case update magnitude and sign, never Dataset A/B recall. Teacher events were prescribed; natural output spikes and LIF recall were not tested.

## Timing and calibration

- Proposed: {'pre_plus_ms': 3.0, 'pre_minus_ms': 7.0, 'post_plus_ms': 5.0, 'post_minus_early_ms': 1.3, 'post_minus_late_ms': 8.7}; measured four-case relative spread 0.458653%, minimum |Δw| 0.007357589.
- Selected from a predeclared 245-point grid with pre times fixed at 3/7 ms, target+ post 4.8–5.2 ms, target− early 1.0–1.6 ms and late 8.4–9.0 ms (0.1-ms steps): {'pre_plus_ms': 3.0, 'pre_minus_ms': 7.0, 'post_plus_ms': 5.0, 'post_minus_early_ms': 1.3, 'post_minus_late_ms': 8.7}.
- Measured selected spread 0.458653%, minimum |Δw| 0.007357589; all four signs correct. Objective: minimize `(max|Δw|−min|Δw|)/mean|Δw|`, then maximize minimum |Δw|. No dataset score entered this choice.
- All times are on the 0.1-ms grid within the 0–10-ms window; input timing is independent of output target and one 15-element pre-event vector is passed to every output column.
- 60 epochs, W_initial=0.05, A_plus=A_minus=0.02, tau=2.0 ms, original bounds [-1.5,1.5] or no-clipping control; random permutation seed 0. K=[1,2,3,5,7,10] uses one fixed prefix subset 0..K−1 per K, identical across methods. Ascending/descending controls test bounded calibrated timing.
- Primary static decoder: `score = X_bipolar @ (W_final−W_initial)`; `score>0` predicts +1, otherwise −1. Strict positive signed-margin recall and |score|<1e−10 are separate diagnostics; this tolerance never changes the primary decoder.

## 60-epoch fixed-prefix results

| Dataset | K | Correlation exact | Previous unbounded | Calibrated unbounded | Calibrated bounded | Calibrated unbounded cosine | Calibrated unbounded error | Calibrated bounded saturation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 0.999998 | 0.001837 | 0.0% |
| A | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 0.999998 | 0.001938 | 0.0% |
| A | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.999998 | 0.002190 | 0.0% |
| A | 5 | 5/5 | 3/5 | 5/5 | 5/5 | 0.999997 | 0.002238 | 2.2% |
| A | 7 | 5/7 | 1/7 | 5/7 | 5/7 | 0.999997 | 0.002461 | 19.1% |
| A | 10 | 4/10 | 2/10 | 4/10 | 2/10 | 0.999995 | 0.003027 | 28.9% |
| B | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 0.999998 | 0.001837 | 0.0% |
| B | 2 | 2/2 | 2/2 | 2/2 | 2/2 | 0.999998 | 0.001938 | 0.0% |
| B | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.999998 | 0.002190 | 0.0% |
| B | 5 | 5/5 | 4/5 | 5/5 | 5/5 | 0.999998 | 0.002189 | 1.8% |
| B | 7 | 6/7 | 5/7 | 6/7 | 7/7 | 0.999997 | 0.002357 | 16.0% |
| B | 10 | 2/10 | 2/10 | 2/10 | 2/10 | 0.999996 | 0.002931 | 28.4% |

All rows use **fixed prefix subsets**, not all-combination averages. K=10 is the complete dataset, so original correlation baselines are A=4/10 and B=2/10. `csv/association_results.csv` lists each digit; `csv/bit_scores_and_margins.csv` retains every final score and signed margin.

## Correlation reproduction and clipping

- Dataset A, K=10: correlation exact 4/10, strictly-positive-margin 3/10; calibrated unbounded exact 4/10, strictly-positive-margin 4/10, scale 0.441752975, cosine 0.999995418, normalized error 0.003027298, max residual 0.0124199518. Calibrated bounded exact 2/10, saturation 28.9%, final-epoch clipping attempts 573, suppressed updates 472, sign reversals 0.
- Learning order, dataset A, K=10: random_seed0 exact 2/10, max |W−W_random| 0; ascending exact 2/10, max |W−W_random| 0.0147828244; descending exact 2/10, max |W−W_random| 0.014749001.
- Dataset B, K=10: correlation exact 2/10, strictly-positive-margin 2/10; calibrated unbounded exact 2/10, strictly-positive-margin 2/10, scale 0.441794979, cosine 0.999995705, normalized error 0.002930998, max residual 0.0121679321. Calibrated bounded exact 2/10, saturation 28.4%, final-epoch clipping attempts 504, suppressed updates 424, sign reversals 0.
- Learning order, dataset B, K=10: random_seed0 exact 2/10, max |W−W_random| 0; ascending exact 2/10, max |W−W_random| 0.0147828244; descending exact 2/10, max |W−W_random| 0.014749001.

When the four source-measured updates are not exactly equal, the unclipped matrix need not be precisely a scalar multiple of correlation even though all signs match. Clipping adds a separate nonlinear distortion and can suppress later updates; no sign reversal was observed. The learned raw W has a +0.05 offset, so only W_final−W_initial is used for static signed readout and matrix comparison.

## Output threshold boundary

- Dataset A, K=10, Original Correlation: exact digits [0, 1, 2, 4].
- Dataset A, K=10, Calibrated Shared-Spike Timing: exact digits [0, 1, 2, 4].
- Dataset A, K=10, Calibrated Shared-Spike Timing / bounded: exact digits [2, 4].
- Dataset B, K=10, Original Correlation: exact digits [1, 6].
- Dataset B, K=10, Calibrated Shared-Spike Timing: exact digits [1, 6].
- Dataset B, K=10, Calibrated Shared-Spike Timing / bounded: exact digits [1, 6].
- Dataset A, calibrated unbounded K=10: 0 output scores satisfy |score|<1e−10. Their exact values and targets are in `csv/bit_scores_and_margins.csv`.
- Dataset B, calibrated unbounded K=10: 0 output scores satisfy |score|<1e−10. Their exact values and targets are in `csv/bit_scores_and_margins.csv`.
- Dataset A digit 0/output 13 actual scores: Previous Shared-Spike Timing -4.8642983; Calibrated Shared-Spike Timing -0.0304410582; Previous Shared-Spike Timing bounded -4.36493259; Calibrated Shared-Spike Timing bounded +2.29249591; Original Correlation +0. Target sign is −1.

Dataset A's pure-correlation digit 0/output 13 score is exactly zero, and the primary rule maps it to −1. A scaled floating-point STDP matrix may yield a tiny signed nonzero instead; exact decoder recall and strict-positive-margin recall are reported separately. Tiny-score discrepancies must not be interpreted as evidence of a meaningful learning advantage or loss.

## Scope and suitability

This verifies teacher-event timing and repeated spike-pair updates through the original kernel, not spontaneous teacher posts or LIF recall. A nearly correlation-proportional unclipped matrix is a useful **event-level control** for later local plasticity studies; using it as a full SNN baseline still requires checking that prescribed teacher events can be implemented and that clipping/threshold boundaries are handled consistently. No new learning rule, architecture, error feedback or noisy cue was introduced.

## Reproduce

```bash
python3 my_research/experiments/diagnostics/shared_spike_timing_calibration.py \
  --dataset-a local_stdp_audit_dataset_handoff_20261020/dataset_A.npz \
  --dataset-b local_stdp_audit_dataset_handoff_20261020/dataset_B.npz
```
