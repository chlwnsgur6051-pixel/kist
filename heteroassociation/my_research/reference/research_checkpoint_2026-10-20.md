# Research checkpoint — 2026-10-20

## A. Research objective

Study STDP-based heteroassociative memory within a fixed single-layer 15-input ×
15-output feedforward network. Ten digit inputs are paired by row with letter
targets A–J. This checkpoint preserves verified baselines; it proposes no new
learning rule and makes no capacity claim beyond measured static recall.

## B. Architecture and constraints

Inputs and targets are bipolar (−1,+1). One 15×15 matrix connects the input
and output populations. In the calibrated teacher-event wrapper, every output
column receives the same input-neuron spike times: +1 at 3 ms, −1 at 7 ms.
Target +1 has a prescribed teacher spike at 5 ms; target −1 has teacher spikes
at 1.3 and 8.7 ms. The unchanged B2 `stdp_update` determines LTP/LTD from
event timing. Natural output spikes do not enter plasticity in this wrapper.

## C. Confirmed datasets

All datasets contain 10×15 binary input and target arrays and their exact
bipolar transforms; each has bipolar input rank 10. The association order is
0→A through 9→J. [Bitmap panels](validation/20261020_151508_517997/reference_bitmaps.png)
show the frozen inputs and targets. Source and frozen SHA-256 hashes are in
`datasets/manifest.json`. A/B are fixed supplied datasets. Their target arrays
match each other but differ in 50 elements from the *current* Original dataset
targets; substituting `binary_character.py` targets for A/B would be invalid.

| Dataset | Mean absolute input inner product | K=10 correlation exact | Rank |
|---|---:|---:|---:|
| Original | 6.556 | 1/10 | 10 |
| A | 4.378 | 4/10 | 10 |
| B | 3.533 | 2/10 | 10 |

Every-subset Original Correlation exact recall rates, in K order 1,2,3,5,7,10:
Original 100, 100, 64.444, 37.778, 25.833, 10%; A 100, 100, 90.556,
54.683, 36.548, 40%; B 100, 100, 96.944, 70.397, 46.429, 20%. These are
mean association accuracies within each subset; [baseline_metrics.csv](validation/20261020_151508_517997/baseline_metrics.csv)
also records bit accuracy and minimum signed margins. Pseudoinverse `pinv(X)@Y`
recalls 10/10 on each dataset at threshold zero; this is a mathematical
reference, not an STDP result. Dataset A's K=10 correlation recalls four digits
under `score>0`, but only three have strictly positive margin across all bits.
Its digit 0 / output 13 score is exactly zero; the two criteria must stay
distinct. All exact bit scores are preserved in the validation CSVs.

## D. Confirmed baseline model

**Calibrated Shared-Spike Additive STDP:** B2 kernel, A_plus=A_minus=0.0075,
tau=2 ms, DT=0.1 ms, initial raw weight 0.05, bounds [−1.5,1.5], 60 epochs,
seed 0, a seeded permutation each epoch. The shared input event vector is
reused for all 15 output columns. Evaluation uses
`X_bipolar @ (W_final−W_initial)` and +1 iff score>0. The executable wrapper is
`my_research/reference/verify_checkpoint.py`, using the original kernel and
`my_research/experiments/diagnostics/shared_spike_timing_calibration.py`.

On A/B, the new direct verification again found zero clipping and saturation,
exact recall 4/10 and 2/10, cosine to correlation 0.9999954 and 0.9999957,
and normalized scaled weight error 0.0030273 and 0.0029310. The prior
learning-rate study selected 0.0075 as the largest tested rate with no K=10
clipping on *both* A/B; its all-subset evaluation matched correlation exact
and bit accuracy for every A/B subset. The four update magnitudes retain a
0.458653% relative spread, so learned weights are approximately, not exactly,
proportional to correlation.

**New Original-dataset check:** this exact configuration obtained 1/10 static
exact (digit 7), matching correlation exact, but incurred **75 clipping
attempts and one saturated synapse**. Cosine was 0.9998074 and normalized
scaled error 0.0196242. It is therefore *not* a zero-clipping baseline on the
Original dataset. This result was measured on 2026-10-20 and must not be
described as part of the earlier A/B validation.

## E. Key experiments and findings

- The historical B2 teacher-only model used target-dependent input timing;
  the local-feasibility audit identified that as inconsistent with one input
  neuron's shared outgoing spike train.
- The shared-spike timing study verified all four bipolar LTP/LTD signs using
  the original kernel and reduced update-magnitude spread to 0.458653%.
- The learning-rate study found 0.02 clips at K=10; 0.0075 avoids clipping on
  both A/B and reproduces static correlation recall across all tested subsets.
- Weighted-correlation MILP achieved 6/10 on the Original dataset using all
  targets, as an offline reference. Online and weight-dependent mathematical
  studies are exploratory; they do not validate a spike-event local rule.
- Earlier Original-dataset LIF failures remain separate from this new
  shared-spike teacher-event static baseline.

## F. Validation boundary

| Finding | Status |
|---|---|
| Shared presynaptic spike consistency | Validated in wrapper |
| Four bipolar LTP/LTD signs | Validated with original B2 kernel |
| Nearly matched update magnitudes | Validated; 0.458653% residual spread |
| Repeated teacher-event STDP learning | Validated on A/B; Original tested separately with clipping |
| Correlation matrix approximation | Validated on A/B; Original differs more due to clipping |
| Static recall | Validated on A/B and newly measured on Original |
| Natural postsynaptic spike learning | Not validated |
| LIF recall with this shared-spike model | Not validated |
| Biological plausibility | Not established |
| Generalization to unseen datasets | Not established |

## G. Open research questions

How does multi-association crosstalk constrain capacity? Can synaptic traces
or locally available learning history improve recall without global feedback?
Can actual postsynaptic activity replace prescribed teacher events? Does the
new shared-spike model support reliable LIF recall? These remain questions,
not outcomes of this checkpoint.

## Reproduction record

Executed `python3 my_research/reference/verify_checkpoint.py --freeze --dataset all`.
The first validation directory recorded an assertion failure because Original
had clipping; no source or result was overwritten. The verifier was corrected
to record this dataset-specific discrepancy while retaining the zero-clipping
assertions for A/B. The successful run was
`python3 my_research/reference/verify_checkpoint.py --dataset all`, timestamped
2026-10-20 in [verification.json](validation/20261020_151508_517997/verification.json).
Checks passed for shapes, source equality, hashes, rank, every-subset
correlation rates, K=10 pseudoinverse, shared pre-event encoding, four signs,
direct 60-epoch B2 updates, and A/B zero clipping. Environment: Python with
NumPy 2.2.6 and Matplotlib 3.10.9. The direct scripts operate within the
existing architecture and never change the original B2 source.
