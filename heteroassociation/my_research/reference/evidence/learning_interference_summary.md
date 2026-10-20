# Self-Memory Gain and Interference During Sequential Learning

## Protocol and checks
The bipolar 15×15 digit 0–9 → letter A–J dataset exactly matches `association_alpha_milp_reference_20261019.zip`. Pure correlation reproduced K=3 **64.44%**, K=5 **37.78%**, K=7 **25.83%**, K=10 **10.00%** before experiments. Noisy cues, LIF simulation, simulator modification and post-training transformation were excluded. All K=3 subsets and permutations, all K=5/7 subsets with 20 unique seeded orders, and K=10 with 100 unique orders were evaluated. Order lists come from the prior online study and include ascending/descending for K>3.

Each candidate α in `(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)` starts from the **same equal-alpha W_before**; only after all counterfactuals does the diagnostic path apply α=1. Every candidate verified `m_current_after=m_current_before+15α` and `m_previous_after=m_previous_before+α·(x_q·x_p)y_qy_p`; 45160 step identities were checked. `csv/previous_bit_contributions.csv` records every earlier q/output bit's signed coefficient and α=1 margin; `csv/counterfactual_candidates.csv` records all candidate outcomes. These retrospective calculations did not select alpha in the online policies.

Exact correctness uses `score>0→+1`, `score≤0→−1`. Thus score zero is exact-correct for a −1 target but has signed margin zero and is **not ε-robust**. Robust feasibility uses ε=0.0001; exact-preservation bounds also include zero-margin correct bits. The smallest *additional* robust strength is `max(0,(ε−minimum current signed margin)/15)`. If already robust, this equals 0; a positive alpha_min is an imposed experimental bound, not a learning necessity. Candidate-grid results can exceed this analytical minimum because the grid is discrete.

## Required versus safe learning strength
For K=10 equal-alpha trajectories after at least one previous memory, median required α was **1.267**, 90th percentile **3.267**. The median *finite* robust upper bound preserving prior exact associations was **1.111**; an infinite bound means no currently robust correct bit has adverse signed contribution. `csv/feasibility_summary.csv` reports finite/infinite distributions and infeasibility reasons.

| K | All previously correct bits feasible, α∈[0.1,4] | Previously exact associations feasible, α∈[0.1,4] | Previously exact associations feasible, α∈[0.5,2] |
|---:|---:|---:|---:|
| 3 | 81.53% | 81.53% | 79.72% |
| 5 | 57.39% | 66.25% | 62.19% |
| 7 | 44.69% | 59.30% | 53.84% |
| 10 | 31.67% | 49.78% | 44.56% |

These fractions exclude the first step, where there is no previous memory. They describe **local feasibility at the fixed W_before**, not the architecture's global capacity. The separate exact-preservation fractions are in CSV; they can differ from ε-robust fractions when a negative-target score ties at zero. A non-feasible interval can arise because required α exceeds the allowed maximum, a safe upper bound is below the allowed minimum, or required α exceeds the safe bound. All three counts and required–safe gaps are saved.

## Full sequential rollouts (same orders; bounds 0.5–2.0)
The fixed schedule is α=2 for the first pair and α=0.5 thereafter. Current minimum-margin uses δ=1. Previous negative-conflict down uses η=2, as in the earlier online experiment. Both retrospective oracles use only the present and **previous** pairs, never future pairs. If robust current learning can preserve (a) all previously correct bits or (b) bits in previously exact associations, the oracle chooses the smallest such α. For this *selection* it stays 1e-09 inside the analytical exact score-zero boundary to avoid floating-point ambiguity; the analytical safe bounds in CSV remain unguarded. If this is infeasible, it first minimizes losses of those protected bits, then maximizes current robust bit count, then minimizes forgotten exact associations, then chooses the smallest α among deterministic margin-crossing candidates. The infeasible branch can decline to learn the current memory to protect earlier ones. These full-history analytical references require stored past targets.

| K=10 policy | Mean exact (%) | Median (%) | Range (%) | Mean bit (%) | Mean forgotten transitions |
|---|---:|---:|---:|---:|---:|
| equal | 10.00 | 10.00 | 10–10 | 78.00 | 5.14 |
| first_strong_rest_weak | 15.70 | 10.00 | 10–30 | 75.92 | 1.34 |
| negative_conflict_down_eta2 | 19.40 | 20.00 | 10–30 | 78.55 | 3.04 |
| minimum_margin_delta1 | 11.90 | 10.00 | 0–30 | 78.62 | 7.00 |
| retrospective_oracle_all_correct_bits | 15.10 | 10.00 | 10–30 | 79.25 | 5.71 |
| retrospective_oracle_exact_associations | 18.20 | 20.00 | 0–30 | 79.59 | 6.15 |

Equal α=1 is 10.00%; prior conflict-down η=2 is 19.40%; the all-correct-bit oracle is 15.10% and prior-exact-association oracle 18.20%. These are means across **100 orders of the same K=10 subset**, not new datasets. The oracle's larger information set does not guarantee a globally optimal final W: its fixed objective is myopic and it cannot read future pairs. `csv/policy_runs.csv` and `csv/policy_transitions.csv` retain every order, alpha, association list, forgetting and recovery; K=10 final W is in `npz/k10_policy_weights.npz`.

## Representative observed changes
- **benefit_without_loss:** K=3, subset 0-1-2, order 0-1-2, step 2 (digit 1), α=1.0; current corrected bits 11, previous bit losses 0, forgotten exact associations 0, recovered exact associations 0.
- **benefit_with_forgetting:** K=3, subset 0-1-2, order 0-1-2, step 2 (digit 1), α=3.0; current corrected bits 11, previous bit losses 9, forgotten exact associations 1, recovered exact associations 0. Previous digit 0 bit 0: margin +15.000 + 3.0×(-5.000) = +0.000.
- **extra_damage_little_benefit:** K=5, subset 0-1-2-3-4, order 3-2-0-1-4, step 4 (digit 1), α=3.0; current corrected bits 11, previous bit losses 2, forgotten exact associations 0, recovered exact associations 0. Previous digit 0 bit 5: margin +15.000 + 3.0×(-5.000) = +0.000.
- **current_and_previous_recovery:** K=5, subset 0-1-2-3-5, order 0-1-2-3-5, step 5 (digit 5), α=1.0; current corrected bits 6, previous bit losses 2, forgotten exact associations 0, recovered exact associations 1. Previous digit 3 bit 5: margin +10.000 + 1.0×(-11.000) = -1.000.

## Current-information features and STDP feasibility
For K=10 equal-alpha steps, descriptive Spearman ρ with previous correct-bit losses caused by α=1:

| Observable before update | Spearman ρ |
|---|---:|
| before_min_margin | -0.38785813787171664 |
| before_incorrect_bits | -0.15453409644668634 |
| alignment | 0.43683475635245417 |
| negative_weight_conflict | 0.23463486897656757 |
| conflict_fraction | -0.35363357853172034 |
| weight_norm | 0.4917758536590956 |
| step | 0.5429043792664239 |

All observations reuse ten patterns and overlapping subsets/orders; these coefficients are **not independent-sample significance tests**. Current minimum/mean target-signed margins and incorrect-bit count need comparison with the current target. Weight alignment is the sum of current signed margins, so it is not an independent feature. Conflict, weight norm, step index and cumulative α require aggregation or scalar memory but no past pair identity. The exact per-bit interference coefficients and safe upper bound require prior x and y or replay; W_before alone generally cannot identify which prior memory is vulnerable. The oracle's full history is not synapse-local STDP information. An actual spike-event implementation would additionally need a mechanism to aggregate across synapses/outputs and broadcast a positive association-level α; this study implements neither mechanism nor a biological learning rule.

The observed constraints separate current-memory gain from prior-memory damage; a higher final exact count may redistribute errors. No single policy should be selected solely from the highest K=10 score on this fixed dataset. Further research should focus on whether a measurable, low-information proxy for vulnerable prior bits exists, rather than assume full-history safe bounds are locally available. `config/config.json` records source hashes, dataset, seed, bounds, epsilon and the fixed oracle objective. Run `python my_research/experiments/diagnostics/analyze_learning_interference.py` from the repository root.


## Oracle branch audit
`csv/k10_oracle_decisions.csv` replays every K=10 oracle decision and verifies feasible-branch preservation and current ε-robust recall. In infeasible branches, the fixed objective may keep older bits correct at the cost of the current memory. This explains why extra history does not imply a higher final exact count.
| Bounds | Oracle | Infeasible steps / 1,000 | Current not ε-robust in infeasible steps | Infeasible steps with protected-bit loss |
|---|---|---:|---:|---:|
| 0.1-4.0 | retrospective_oracle_all_correct_bits | 697 | 417 | 667 |
| 0.1-4.0 | retrospective_oracle_exact_associations | 571 | 328 | 472 |
| 0.5-2.0 | retrospective_oracle_all_correct_bits | 698 | 414 | 670 |
| 0.5-2.0 | retrospective_oracle_exact_associations | 574 | 330 | 475 |

The two oracle variants selected different α values in 173/200 matched order/bound runs (`csv/k10_oracle_variant_differences.csv`). Neither is a future-aware global optimizer.

For K=10 equal-alpha trajectories, the finite exact-preservation upper bound for **all previously correct bits** has median α=0.551 (10th–90th percentile 0.077–2.400); 633/900 steps have that upper bound at or below α=1. This boundary varies with current associations and prior margins, so large alpha alone is not a complete explanation of forgetting.
A numerical recovery example from that step is previous digit 0, output 7: signed margin -8.000 + 1×(+9.000) = +1.000; the bit changes from incorrect to correct.
The K=10 full-history oracles did **not** outperform the predefined negative-conflict-down online rule in mean exact recall (15.10%/18.20% versus 19.40%). This does not show that previous-pair information has no value: the oracle objective is myopic, bounds restrict alpha, and future associations are unknown. The conflict-down η=2 rule remains a candidate for further diagnostic work, but these data do not justify implementing it in the spike-event simulator. A practical implementation would require population-level aggregation and broadcast of association strength; exact safety bounds additionally require prior-pair storage or replay.
