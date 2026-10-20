# Experiment registry

The paths below are relative to `heteroassociation/`. Summaries copied into
`my_research/reference/evidence/` are the compact Git archive; full local
timestamped outputs remain at the indicated original paths. Unless stated
otherwise, mathematical experiments use the current Original dataset from
`binary_character.py`, stored-input static `score>0` recall, and no LIF.

| Experiment / date | Source | Dataset and parameters | Main result / output | Status |
|---|---|---|---|---|
| Original B2 teacher-only / pre-2026-10-20 | `Ver관리/2. temporal/stdp_temporal_symmetric_teacher_only.py` | Original; target-dependent timing, original B2 settings | Historical comparator; input timing can depend on target | Pending Verification for shared-spike feasibility |
| Predecessor 6/10 trace / 2026-10-08 | `my_research/experiments/diagnostics/verify_baseline.py` | Pre-rank-fix digits; linear contrastive pseudoinverse, one-step LIF | Archived 6/10 reproduced by NumPy wrapper; `my_research/results/baseline_verification/20261008_150339_603921/` | Mathematical Analysis |
| Original Correlation / 2026-10-20 | `my_research/reference/verify_checkpoint.py` | Original/A/B; all subsets K=1,2,3,5,7,10; `X.T@Y` | K=10 1/10, 4/10, 2/10; `my_research/reference/validation/20261020_151508_517997/` | Validated Baseline |
| Pseudoinverse / 2026-10-20 | `my_research/reference/verify_checkpoint.py` | Original/A/B; K=10; `pinv(X)@Y` | 10/10 each; mathematical reference, not STDP | Validated Baseline |
| Shared-Spike STDP Timing Calibration / 2026-10-20 | `my_research/experiments/diagnostics/shared_spike_timing_calibration.py` | A/B; pre 3/7 ms, post 5 or 1.3/8.7 ms; A=0.02 | Four signs correct, magnitude spread 0.458653%; `results/shared_spike_timing_calibration/20261020_144958_149043/` | Validated Baseline (event timing) |
| Shared-Spike Learning Rate Calibration / 2026-10-20 | `my_research/experiments/diagnostics/shared_spike_learning_rate_calibration.py` | A/B; A=0.0075, 60 epochs, bounds ±1.5, seed 0 | Zero K=10 clipping; A 4/10, B 2/10; every subset matched correlation; `results/shared_spike_learning_rate_calibration/20261020_150253_577134/` | Validated Baseline (teacher-event/static) |
| Original-dataset shared-spike check / 2026-10-20 | `my_research/reference/verify_checkpoint.py` | Original; same calibrated settings | 1/10, 75 clipping attempts; `my_research/reference/validation/20261020_151508_517997/` | Negative Result for zero-clipping transfer |
| Weighted Correlation / MILP alpha / 2026-10-19 | `my_research/experiments/diagnostics/analyze_association_strength.py` | Original; weighted outer products; offline MILP reference | Equal 1/10, MILP 6/10; `results/association_strength_analysis/20261019_162228_767845/` | Mathematical Analysis |
| Weight-dependent STDP / 2026-10-19 | `my_research/experiments/diagnostics/weight_dependent_stdp.py` | Original; sequential mathematical updates, matched additive control | Exploratory K=10 matched additive 1/10 versus μ=2 2/10; `results/weight_dependent_stdp/20261019_145405_819046/` | Exploratory Result |
| Association-strength factors / 2026-10-19 | `my_research/experiments/diagnostics/analyze_association_strength.py` | Original; one-alpha sensitivity, target compatibility | Single-alpha change reached at most 3/10 on K=10 grid; same output as weighted-correlation row | Mathematical Analysis |
| Adaptive association strength / 2026-10-19 | `my_research/experiments/diagnostics/adaptive_association_strength.py` | Original; online mathematical updates, 100 K=10 orders | Conflict-down η=2 exploratory mean 19.4%, shuffled-alpha control 19.73%; `results/adaptive_association_strength/20261019_175320_098182/` | Exploratory Result |
| Self-memory vs interference / 2026-10-20 | `my_research/experiments/diagnostics/analyze_learning_interference.py` | Original; counterfactual α and full-history oracle | K=10 local robust feasibility for exact associations 49.78% of later steps under [0.1,4]; `results/association_learning_interference/20261020_094045_632808/` | Mathematical Analysis |
| Nonlinear transformation / 2026-10-19 | `my_research/experiments/diagnostics/analyze_nonlinear_stdp.py` | Original; static exponent sweep and mathematical sequential candidate | γ=2 improves K=3 but not K=10 exact; simulator implementation and biological plausibility unverified; `results/nonlinear_stdp_analysis/20261019_135231_363889/` | Exploratory Result |
| LIF recall with calibrated shared-spike baseline | Existing LIF in original project | Calibrated A/B/Original | Not run in this checkpoint | Pending Verification |

All reported results are scoped to their own dataset and decoder. In
particular, MILP uses global target information, mathematical update
simulations are not spike-event training, and the trusted calibrated model
uses prescribed teacher events rather than natural post-spikes.
