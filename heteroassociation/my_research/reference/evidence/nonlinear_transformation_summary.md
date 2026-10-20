# Nonlinear Weight-Dependent STDP Candidate: Mathematical and Static-Recall Analysis

This is a **mathematical candidate constructed to reproduce a target matrix**, not an observed biological plasticity rule or a run of the original spike-event STDP simulator. No LIF recall, noisy cues, pseudoinverse, or learned decoder were used. Feedforward 15×15 architecture, existing 0–9→A–J dataset, bipolar signs, and `score>0` decision are fixed. Run command, hashes, seed and full patterns are in `config.json`.

## Sequential local update

`f(z)=sign(z)|z|^γ`, `inverse_f(w)=sign(w)|w|^(1/γ)`, `Δw=f(inverse_f(w)+s)−w`. B2 teacher-only timing convention maps matching bipolar input/output bits to early pre-spike 2 ms and LTP; differing bits to late pre-spike 7 ms and LTD, with teacher at 4.5 ms. Thus each event sign is `s=x_b*y_b`. We applied one unit event per association, from effective weight zero, without clipping. γ=1 recovers additive correlation; γ=2 recovers `sign(Wcorr)|Wcorr|²`. Ascending, reverse and seeded permutations were checked for K=1,2,3,5,7,10. Maximum matrix error across these checks: 0; `csv/sequential_verification.csv` and `npz/` preserve target, learned candidate and differences. Order independence here follows the no-clipping one-update-per-pair mathematical protocol; it is not a claim about the original 60-epoch B2 trainer.

## Exponent sweep: all subsets, association-weighted exact rate (%)

| K | Baseline γ=1 | γ=2 | Best observed γ (smallest tie) | Best exact (%) | All exact-rate ties |
|---:|---:|---:|---:|---:|---|
| 1 | 100.00 | 100.00 | 0.5 | 100.00 | 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0 |
| 2 | 100.00 | 100.00 | 0.5 | 100.00 | 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0 |
| 3 | 64.44 | 86.11 | 2.5 | 90.28 | 2.5, 3.0, 4.0 |
| 5 | 37.78 | 44.29 | 4 | 47.94 | 4.0 |
| 7 | 25.83 | 30.95 | 4 | 34.52 | 4.0 |
| 10 | 10.00 | 10.00 | 0.5 | 10.00 | 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 |

Existing γ=1/2 reference values for K=3,5,7,10 were reproduced exactly: {3: [np.float64(64.44444444444444), np.float64(86.11111111111111)], 5: [np.float64(37.77777777777778), np.float64(44.285714285714285)], 7: [np.float64(25.833333333333332), np.float64(30.952380952380953)], 10: [np.float64(10.0), np.float64(10.0)]}. K=1 uses 10 subsets, K=2 45, K=3 120, K=5 252, K=7 120, K=10 1. Exact rate weights every stored association equally; `csv/exponent_accuracy.csv` also reports bit accuracy, changes from γ=1 and corrected/new output bits. Best observed γ is selected *after* viewing these subsets and is not an out-of-sample optimum.

## Mechanism at γ=2 relative to γ=1

| K | Corrected output bits | Newly incorrect output bits | Exact change (percentage points) |
|---:|---:|---:|---:|
| 3 | 196 | 0 | +21.67 |
| 5 | 441 | 108 | +6.51 |
| 7 | 309 | 83 | +5.12 |
| 10 | 7 | 4 | +0.00 |

For each synapse, `Wcorr` equals LTP count minus LTD count. For K=3, |W|=3 means all three updates agreed; γ=2 changes magnitude 3→9 while |W|=1 remains 1. The actual signed margin contribution of input synapse i to target output j is `y_b[j]*x_b[i]*W[i,j]`; `csv/bit_mechanism_gamma2.csv` separates unanimous and other synapses for *every* stored output bit in K=3,5,7,10. `csv/representative_synapse_contributions.csv` gives individual 15-synapse decompositions for selected flips:

- K=3 corrected: subset 0-8-9, digit 9, output 7; signed margin -1 → +17; unanimous-synapse change +18, other-synapse change +0. Full 15-input decomposition: `csv/representative_synapse_contributions.csv`.
- K=10 corrected: subset 0-1-2-3-4-5-6-7-8-9, digit 1, output 13; signed margin -4 → +16; unanimous-synapse change +0, other-synapse change +20. Full 15-input decomposition: `csv/representative_synapse_contributions.csv`.
- K=10 newly_incorrect: subset 0-1-2-3-4-5-6-7-8-9, digit 2, output 11; signed margin +2 → -4; unanimous-synapse change +0, other-synapse change -6. Full 15-input decomposition: `csv/representative_synapse_contributions.csv`.

An exponent can increase some correct signed contributions and also magnify wrong-direction contributions. The `csv/mechanism_summary.csv` and signed-margin figures quantify both; K=10's unchanged 1/10 exact rate is a net outcome, not evidence that all margins are unchanged. Association overlap and weight signs determine which output bits improve or deteriorate; stronger magnitude alone does not guarantee a correct target sign.

Direct observations: At K=3, γ=2 corrects 196 output bits and creates zero new errors. A representative correction (subset 0-8-9, digit 9, output 7) changes signed margin −1→+17; the +18 shift is entirely from unanimous-synapse contributions. Across all subset-synapse positions, the fraction with unanimous direction falls from 31.70% at K=3 to 0.44% at K=10. At K=10, γ=2 corrects 7 bits but creates 4 errors; a representative new error (digit 2, output 11) changes margin +2→−4. Neither selected K=10 corrected nor newly incorrect example has a unanimous synapse contribution. The number of *exactly recalled associations* remains 1/10, because bit-level gains need not complete an entire 15-bit output and can coexist with new failures. These are observations from this fixed dataset, not a general causal law.

## Feasibility in the existing B2 simulator

The timing **direction** can be preserved: teacher at 4.5 ms with target-zero timing inversion yields the needed ± sign. The proposed **magnitude** can be evaluated locally from current synaptic effective weight and event direction. To apply it, `stdp_update` in `Ver관리/2. temporal/stdp_temporal_symmetric_teacher_only.py` would need a different magnitude formula; `train()` also sets `w=0.05`, loops 60 epochs and clips each update. `b2_full_output.py` only drives the 15 independent output columns and resets seed. No original file was changed.

The mathematical state starts at effective zero, so the 0.05 baseline must be removed when updating and restored for actual stored W; this is distinct from transforming raw W. A unit update repeated 60 epochs yields `f(60*Wcorr)`, not `f(Wcorr)`, unless each presentation is normalized (e.g. 1/60 in the inverse-transformed state). Actual B2 timing gives an exponential magnitude of about 0.00573 for each event, not unit ±1, and extra natural spike events could alter an event-driven implementation. Current [-1.5,1.5] bounds clip candidate γ=2 weights in several K values; `csv/existing_stdp_feasibility.csv` quantifies both unit target and a clearly labeled 0.05+observed-single-pair-scale mapping. Clipping destroys the exact no-clipping order-independence guarantee, and multiple events or different presentation order can then matter. Mathematical compatibility is established; biological plausibility and actual spike-event implementation remain untested.

## Files and limitations

`csv/output_bit_results.csv` contains scores, target, prediction and signed margin for every evaluated output bit and exponent. `csv/subset_results.csv` and `csv/association_results.csv` retain grouping. `csv/synapse_weight_distribution.csv` records direction-agreement counts. `npz/sequential_*.npz` retains target and candidate matrices. All plots in `figures/` can be regenerated from these numbers. This is a retrospective sweep over a fixed ten-pattern dataset; no independent test patterns or noise trials were assessed. Figure titles and axes name the changed quantity and evaluated condition.
