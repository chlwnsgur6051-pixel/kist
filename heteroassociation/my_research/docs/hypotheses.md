# Hypotheses and discriminating evidence

## H1: learning/recall representation mismatch

Target-dependent timing changes may create useful STDP updates, while recall returns to the standard early/late input code. Since LIF integrates synaptic input over time and prediction is “any spike means 1,” learned positive weights driven by late input could cause false positives.

**Evidence to collect:** per-association learned weight matrix; actual LIF membrane traces and spike times; false-positive/false-negative counts; compare LIF output with an explicitly labeled static readout of the same weights. Keep weights fixed for recall comparisons.

**Supports H1:** a weight matrix separates targets under a suitable static readout, while actual LIF traces show threshold crossings for incorrect target-zero cues under the existing recall code.

**Weakens H1:** both static and LIF recall fail on the same bits, or learned weights do not consistently distinguish target classes.

## H2: STDP learning produces an inadequate matrix

Pair-based event updates, sequential presentation, clipping, and interference may fail to encode enough information even before LIF readout is considered.

**Evidence to collect:** compare trained weights and their margins against the dataset targets without changing the learning procedure; compare with a pure bipolar correlation matrix only as a static mathematical reference, never as STDP performance.

**Supports H2:** the trained matrix lacks target-aligned margins under reasonable readouts and differs materially from the reference where errors occur.

## H3: memory load drives crosstalk

More stored pairs may increase cross-association interference. Existing pair-count summaries show a degradation pattern in both STDP variants and the correlation comparator, but the variant/version distinction must be respected.

**Evidence to collect:** subset-level results under the same selected subsets and seeds, maintaining separate STDP and correlation reports.

## Order of investigation

First recover provenance and inspect saved historical weights/predictions. Then isolate whether errors appear in the learned matrix or only in spike dynamics. Consider encoding, STDP window, weight dependence, sparse/ON-OFF representations, threshold, or inhibition only after the relevant bottleneck is supported by evidence. Test one modification at a time.
