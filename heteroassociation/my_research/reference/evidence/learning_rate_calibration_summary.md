# Shared-Spike STDP Learning Rate Calibration

The original B2 `stdp_update` was called at every K=10 presentation. Only its imported module parameters `A_PLUS=A_MINUS` were varied; original simulator files were not edited. Teacher post events were scheduled, and natural postsynaptic spikes were excluded from plasticity.

## Fixed protocol

- Shared pre +1@3 ms, −1@7 ms; teacher target +1@5 ms, target −1@1.3/8.7 ms. One pre-event vector is used for every output column. Initial raw weight 0.05, bounds [−1.5,+1.5], 60 epochs, seed 0 and the same per-epoch permutation for every rate.
- Rates tested: 0.0200, 0.0100, 0.0075, 0.0050 with equal LTP/LTD amplitude. Timing, tau, all other parameters and target mapping were fixed. Static readout is `X_bipolar @ (W_final−W_initial)` with +1 iff score>0.
- Selected rate is the **largest rate with zero clipping attempts and zero final saturation in both A and B at K=10**, chosen without looking at recall accuracy. Epoch checkpoints 1,5,10,20,30,40,50,60 come from the same trajectory.

## K=10 final measurements

| Dataset | Rate | Exact | Bit accuracy | Saturated | Cumulative clipping | Suppressed updates | Cosine | Normalized error | Scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.0200 | 2/10 | 85.33% | 28.9% | 11566 | 10162 | 0.959340 | 0.282252 | 0.288617 |
| A | 0.0100 | 4/10 | 85.33% | 4.9% | 872 | 852 | 0.997708 | 0.067665 | 0.213141 |
| A | 0.0075 | 4/10 | 85.33% | 0.0% | 0 | 0 | 0.999995 | 0.003027 | 0.165657 |
| A | 0.0050 | 4/10 | 85.33% | 0.0% | 0 | 0 | 0.999995 | 0.003027 | 0.110438 |
| B | 0.0200 | 2/10 | 85.33% | 28.4% | 9209 | 8114 | 0.963856 | 0.266424 | 0.308464 |
| B | 0.0100 | 2/10 | 86.00% | 2.7% | 469 | 449 | 0.998612 | 0.052679 | 0.216120 |
| B | 0.0075 | 2/10 | 86.00% | 0.0% | 0 | 0 | 0.999996 | 0.002931 | 0.165673 |
| B | 0.0050 | 2/10 | 86.00% | 0.0% | 0 | 0 | 0.999996 | 0.002931 | 0.110449 |

Selected no-clipping rate: **0.0075**.

## Same-trajectory epoch effects

`csv/epoch_metrics.csv` contains all 61 epoch states for every rate. `csv/association_margin_by_epoch.csv` and `csv/bit_scores_by_epoch.csv` retain the requested checkpoints, including each digit's minimum signed margin. The best-fit scale/weight norm track cumulative update magnitude. Clipping and suppressed-update counters isolate saturation. The four-case timing imbalance remains fixed at 0.458653%; reducing rate scales it but cannot remove it. Near-zero scores (|score|<1e−10) are labeled for numerical interpretation only; the primary `score>0` decision is unchanged.

## All-subset check at selected rate

Every combination for K=[1, 2, 3, 5, 7, 10] was replayed using source-function event deltas and the original per-presentation clipping. A direct source-call trajectory matched replay for both K=10 datasets and two smaller representative subsets per dataset (maximum absolute discrepancy 9.16e-16); this fast replay changes no update rule. Fixed-prefix K=10 runs remain direct source calls.

| Dataset | K | Subsets | Correlation mean exact | STDP mean exact | Subsets clipped |
|---|---:|---:|---:|---:|---:|
| A | 1 | 10 | 100.000% | 100.000% | 0 |
| A | 2 | 45 | 100.000% | 100.000% | 0 |
| A | 3 | 120 | 90.556% | 90.556% | 0 |
| A | 5 | 252 | 54.683% | 54.683% | 0 |
| A | 7 | 120 | 36.548% | 36.548% | 0 |
| A | 10 | 1 | 40.000% | 40.000% | 0 |
| B | 1 | 10 | 100.000% | 100.000% | 0 |
| B | 2 | 45 | 100.000% | 100.000% | 0 |
| B | 3 | 120 | 96.944% | 96.944% | 0 |
| B | 5 | 252 | 70.397% | 70.397% | 0 |
| B | 7 | 120 | 46.429% | 46.429% | 0 |
| B | 10 | 1 | 20.000% | 20.000% | 0 |

The all-subset mean is distinct from a fixed-prefix result. Each association subset uses the same digit-letter pairs for correlation and STDP; all raw subset outcomes and final matrices are saved in CSV/NPZ.

## K=10 score-boundary diagnostic

- Dataset A, rate 0.0200: exact 2/10, strictly-positive-margin 2/10, |score|<1e−10 bit count 0, recalled digits [2, 4].
- Dataset A, rate 0.0100: exact 4/10, strictly-positive-margin 4/10, |score|<1e−10 bit count 0, recalled digits [0, 1, 2, 4].
- Dataset A, rate 0.0075: exact 4/10, strictly-positive-margin 4/10, |score|<1e−10 bit count 0, recalled digits [0, 1, 2, 4].
- Dataset A, rate 0.0050: exact 4/10, strictly-positive-margin 4/10, |score|<1e−10 bit count 0, recalled digits [0, 1, 2, 4].
- Dataset B, rate 0.0200: exact 2/10, strictly-positive-margin 2/10, |score|<1e−10 bit count 0, recalled digits [1, 6].
- Dataset B, rate 0.0100: exact 2/10, strictly-positive-margin 2/10, |score|<1e−10 bit count 0, recalled digits [1, 6].
- Dataset B, rate 0.0075: exact 2/10, strictly-positive-margin 2/10, |score|<1e−10 bit count 0, recalled digits [1, 6].
- Dataset B, rate 0.0050: exact 2/10, strictly-positive-margin 2/10, |score|<1e−10 bit count 0, recalled digits [1, 6].
- Dataset A digit 0/output 13 (target −1; pure-correlation score exactly 0): A=0.0200, score=+2.29249591, signed margin=-2.29249591; A=0.0100, score=-0.0152205291, signed margin=+0.0152205291; A=0.0075, score=-0.0114153968, signed margin=+0.0114153968; A=0.0050, score=-0.00761026456, signed margin=+0.00761026456. This threshold boundary can change a bit decision under tiny score shifts; it is reported separately from clipping and structural weight error.

## Interpretation and scope

Clipping prevention, near-proportional weight geometry, static recall reproduction and capacity improvement are separate findings. Reproducing correlation recall is **not** evidence of memory capacity beyond correlation. The baseline is suitable as a controlled teacher-event update reference if the selected rate has no clipping across the tested subsets; it is not a validated natural-post or LIF memory model. No noisy cues, error feedback or new learning rule were used.

## Reproduce

```bash
python3 my_research/experiments/diagnostics/shared_spike_learning_rate_calibration.py \
  --dataset-a local_stdp_audit_dataset_handoff_20261020/dataset_A.npz \
  --dataset-b local_stdp_audit_dataset_handoff_20261020/dataset_B.npz
```
