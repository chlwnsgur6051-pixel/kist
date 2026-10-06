# Experiment history

The source files and compact artifacts below were copied from `Ver관리/2. temporal/`. Values are historical records as found in their files; they were not regenerated during setup. “Reported in request” means present in the handoff note and should be treated as a reported value, not independently validated by the current setup.

| Experiment | Purpose / implementation | Historical artifact result |
|---|---|---|
| Initial baseline | `stdp_temporal.py` plus `baseline_full_output.py`; 2/7 ms encoding, target 1 teacher spike, target 0 shunt | `results_full_baseline/metrics.json`: 0/10 exact, bit accuracy 0.6667. This agrees with the reported baseline. |
| B1 timing LTD | `stdp_temporal_timing_ltd.py` plus `b1_full_output.py`; target-0 input timing intervention and teacher event | `results_full_B1/metrics.json`: 0/10 exact, bit accuracy 0.7667. |
| B2 symmetric | `stdp_temporal_symmetric.py` plus `b2_full_output.py` imports the teacher-only source; B2 variant source exists separately | `results_full_B2/metrics.json`: 0/10 exact, bit accuracy 0.7533. The wrapper code imports `stdp_temporal_symmetric_teacher_only.py`, so verify provenance before attributing this folder to the symmetric source variant. |
| B2 teacher-only | `stdp_temporal_symmetric_teacher_only.py` plus `b2_full_output.py`; target-specific reversed timing for target 0 and teacher event for both classes | `results_full_B2_teacher_only/metrics.json`: 0/10 exact, bit accuracy 0.7400; matches the reported 10-pair value. |
| B2 pair-count sweep, version 1 | `pair_count_sweep.py` writes `subset_results.csv` and `summary.json` | Summary: 1-pair 6/10 perfect subsets; 3-pair mean exact rate 0.1028; 5-pair 0.0238; 7-pair 0.0179; 10-pair 0/10 exact and 0.7400 bit accuracy. |
| B2 pair-count sweep, version 2 | Same script path in the current checkout writes `subset_results2.csv` and `summary2.json`; it imports `stdp_temporal_symmetric.py`, unlike version 1's teacher-only import | Summary2: 1-pair 6/10; 3-pair mean exact rate 0.1167; 5-pair 0.0579; 7-pair 0.0583; 10-pair 0/10 exact and 0.7533 bit accuracy. The two versions are intentionally kept separate. |
| Pure bipolar correlation sweep | `correlation_pair_count_sweep.py`, static `W=XᵀY`, sign threshold | `summary.json`: 1-pair 10/10; 10-pair 1/10 exact and 0.7800 bit accuracy. This is not an STDP result and has no LIF recall. |
| Crosstalk diagnosis | `diagnose_crosstalk.py` computes bipolar correlation scores/margins from the teacher-only source's dataset | Source copied. No separate output artifact was present in the selected historical result folders. |

The legacy pair-count sweep source files themselves contain `main()` blocks that enumerate every subset and write into paths beside the source. Do not run them as part of setup or use them as a lightweight smoke check.

Other files in `Ver관리/2. temporal/` include latency experiments, visualization products, paper PDFs, and unrelated weighted/gradient explorations. They were not copied because they do not implement the specifically requested direct-spike full-output baseline/B1/B2/correlation investigation. Their existence remains visible in the original location.

The copied `README.md` is retained as historical context. It describes latency-coded single-output experiments whose named files are not all part of the selected source set; use source code and the manifest above to establish what each copied model actually does.
