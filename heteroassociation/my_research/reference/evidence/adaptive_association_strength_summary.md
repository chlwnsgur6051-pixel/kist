# Online Adaptive Association Learning Strength

## Setup and validity
The archived `association_alpha_milp_reference_20261019.zip` bipolar X/Y arrays exactly match `binary_character.py` and use digit 0–9 → letter A–J. Archived target-informed MILP α reproduces 6/10 exact and is stored for **offline reference only**; no online rule reads it. `config/config.json` records hashes, patterns, fixed seed, bounds, parameter grid, and order sampling. No noisy cues, LIF, clipping, nonlinear transform, hidden layer, or simulator modification. Static decision: score>0 means +1, tie means −1.

The α=1 all-subset reproduction check passed **before** sweeps: K=1 100%, K=2 100%, K=3 64.44%, K=5 37.78%, K=7 25.83%, K=10 10.00%. K≤3 used every permutation per subset; K=5/7 used 20 unique seeded orders including ascending and descending; K=10 used 100 unique seeded orders for its one subset. Every rule used identical orders. Each random control drew one independent seeded uniform α per position within the same bounds. The fixed schedules depend on position only.

At every step the selector used the *current* x, y and a copy of W_before. Signed-margin Rule A uses `max(1,(δ−min(y·xW_before))/15)` before clipping, so it cannot actively downweight. Alignment normalization uses `sum(W_before·outer(x,y))/(225·max(1,cumulative_alpha_before))`; this equals the sum of signed current-association margins divided by the denominator. Negative conflict uses `sum(max(0,−W_before·outer(x,y)))/(sum(abs(W_before))+1e−12)`, zero when W_before=0. B/C exponentials preserve positive α. Every update was `α·outer(x,y)`, and sign preservation was asserted. No final sum(α)=K constraint was imposed; per-run total α and K/total-α-normalized margins are recorded separately.

## Results: K=10 across 100 learning orders, bounds 0.5–2.0, predefined parameter 1
| Rule | δ/η | Mean exact (%) | Median exact (%) | 10th–90th (%) | Mean bit (%) | Mean forgotten transitions |
|---|---:|---:|---:|---:|---:|---:|
| equal |  | 10.00 | 10.00 | 10.00–10.00 | 78.00 | 5.14 |
| random |  | 15.40 | 20.00 | 0.00–30.00 | 78.24 | 4.00 |
| fixed_increasing |  | 16.70 | 20.00 | 9.00–30.00 | 78.28 | 6.17 |
| fixed_decreasing |  | 15.80 | 20.00 | 10.00–20.00 | 78.19 | 3.19 |
| margin_minimum | 1.0 | 11.90 | 10.00 | 0.00–20.00 | 78.62 | 6.99 |
| alignment_conflict_up | 1.0 | 10.90 | 10.00 | 10.00–10.00 | 78.93 | 4.67 |
| alignment_aligned_up | 1.0 | 15.70 | 20.00 | 10.00–20.00 | 78.26 | 4.87 |
| negative_conflict_down | 1.0 | 18.60 | 20.00 | 10.00–20.00 | 78.41 | 4.19 |
| negative_conflict_up | 1.0 | 13.30 | 10.00 | 10.00–20.00 | 80.03 | 4.96 |

Across the **exploratory grid**, the largest K=10 *mean across orders* was 19.40% for `negative_conflict_down` at parameter 2.0, bounds 0.5-2.0; range 10–30% and standard deviation 6.45 percentage points. This was selected retrospectively from the same ten associations and is not held-out evidence. The equal baseline is 10.00% for every order. All parameter settings and order-level outcomes are in `csv/aggregate_accuracy.csv` and `csv/run_results.csv`.

For the predefined conservative η=1 comparison, reinforcing conflict gives at most 13.30% mean exact among B1/C2; reducing strength under conflict gives at most 18.60% among B2/C1. Compare both with random and position-only controls in the table; the full flexible-bound results are in CSV. These are observations for this fixed dataset, not evidence that one direction is universally beneficial. `csv/intermediate_recall.csv` records every update's new-memory recall, prior-memory forgetting/recovery, exact mask and bit accuracy. `csv/k10_online_features.csv` records the actual pre-update alignment/conflict/α for all K=10 runs. `npz/k10_final_weights.npz` stores all K=10 final matrices; other runs' exact W can be reconstructed from dataset outer products and `alpha_by_digit_json`. Global MILP α was not used in any selector.

## Mechanism, controls, and limitations
Association identity, output target and W_before are mathematically sufficient for these rules, but computing a shared association-level α requires aggregation across 225 synapses. Rule A further requires the *minimum* of 15 target-signed output scores; reading output and comparing against the present target entails nonlocal supervised feedback in an actual SNN. Alignment and negative conflict also aggregate synaptic products, so they are not individually synapse-local STDP updates. Their scalar α could modulate local update magnitude **if** an additional aggregation/modulation mechanism were supplied; this was not implemented or biologically validated. The sign of each local update remains the bipolar outer-product sign in this mathematical model, not a measured spike-event outcome.

Intermediate evaluation was retrospective and never fed back into alpha selection. A single K=10 subset with 100 orders tests order sensitivity, not independent dataset generalization. Parameter selection on this same dataset can overfit. The MILP 60% is target-informed, bound- and normalization-specific and is not a universal upper bound. Differences between adaptive, random and fixed schedules do not alone establish a causal rule; the recorded α assignments and margins support further scrutiny. This experiment does not justify implementing any particular rule in the original spike-event simulator without additional validation.

Run: `python my_research/experiments/diagnostics/adaptive_association_strength.py`. Each run makes a new timestamped directory and ZIP. Full result status and duration are in `metrics.json`.


## Additional offline controls: alpha assignment and recency
The post-hoc control permuted each online run's **already chosen** alpha values across its ten presentation positions ten times. It did not choose online alpha, alter learning, or inspect future patterns during the original run. For the predefined C1 (negative conflict down, η=1, bounds 0.5–2.0), online mean exact was 18.60%; shuffled assignment mean was 16.95% over 1000 post-hoc permutations. `csv/posthoc_shuffle_summary.csv` gives every adaptive setting. This comparison preserves the alpha multiset but randomizes which association receives each value; any difference is descriptive because permutations are not independent datasets.

For the retrospectively best mean C1 η=2, the online mean was 19.40% and shuffled mean was 19.73%; its online assignment did not outperform the shuffled assignment. Thus the grid's highest mean is not evidence that the rule assigns alpha to the right association.

For the same C1 setting, final exact recall by presentation position averaged 22.33% over positions 1–3 and 20.00% over positions 8–10. `csv/recency_by_position.csv` reports every position/rule; these rates test for recency association, not a causal mechanism. `npz/k10_signed_margin_trajectories.npz` stores all 15 target-signed output margins for each presented digit after every K=10 update, with NaN before presentation. These traces are retrospective and were not fed to alpha selection.

Equal α=1 recalls only digit 7 at K=10. Under predefined C1 η=1, the most frequently rescued digit was 6 (exact in 55.0% of orders); previously exact digit 7 was lost in 0.0% of orders. `csv/k10_association_outcomes.csv` gives every digit/rule/parameter, including mean assigned alpha and mean presentation position. This separates gains in other memories from loss of the original success.

### Across pair counts: conservative bounds, predefined parameter 1
| K | Equal α exact % | Random α exact % | Fixed increasing exact % | Minimum margin exact % | Negative conflict down exact % |
|---:|---:|---:|---:|---:|---:|
| 1 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 2 | 100.00 | 87.78 | 66.67 | 100.00 | 88.89 |
| 3 | 64.44 | 63.24 | 59.31 | 75.00 | 63.06 |
| 5 | 37.78 | 36.14 | 35.38 | 38.07 | 35.65 |
| 7 | 25.83 | 25.76 | 25.18 | 23.04 | 26.20 |
| 10 | 10.00 | 15.40 | 16.70 | 11.90 | 18.60 |

No single online rule improves every K. In particular, minimum-margin strengthening raises K=3 from 64.44% to 75.00% but lowers K=7 from 25.83% to 23.04%; conflict-down improves K=10 while reducing K=2 from 100% to 88.89%. These are association-weighted means over all evaluated subset/order runs at each K, not an average of separate datasets. A single K=10 order can reach 40% under some grid settings, including a fixed schedule; this is an observed maximum, not typical performance.
