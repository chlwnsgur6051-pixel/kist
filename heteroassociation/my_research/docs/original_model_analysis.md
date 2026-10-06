# Original model analysis

This analysis is based on direct inspection of the copied source files. Line numbers refer to those copies, whose contents match the originals at setup time. No experiment was run to produce this analysis.

## Dataset and network

`legacy/shared/binary_character.py` defines 5×3 bitmap patterns and `make_digit_dataset()` / `make_letter_dataset()` (original `binary_character.py`, pattern dictionary around lines 7–115; dataset functions around lines 117–128). The full-output wrappers independently train one 15-input, one-output model for each of the 15 letter pixels, then assemble the predictions. The resulting matrix therefore contains 15 learned output vectors, each with 15 weights; it is not a single jointly simulated 15-neuron population. See `baseline_full_output.py` lines 20–40 and 43–100, with corresponding B1/B2 wrappers using the same pattern.

## Input encoding and LIF dynamics

The initial baseline encodes input bit 1 at 2 ms and bit 0 at 7 ms (`stdp_temporal.py` lines 41–45, 73–75). The simulation uses `dt=0.1 ms`, a membrane decay `exp(-dt / tau_m)` with `tau_m=3 ms`, threshold 1, and reset 0 (lines 42–49, 66–70, 78–95). At each time step, the weights of inputs spiking at that time are summed and added to the decayed membrane potential. A threshold crossing emits a postsynaptic spike and resets the voltage.

Recall runs the input without a teacher; any output spike means bit 1, and no spike means bit 0 (`stdp_temporal.py` lines 111–118). Thus a single event at any time in the simulation window decides the bit.

## Teacher handling and learning loop

In the initial baseline, a target-1 trial forces the output membrane above threshold at `T_TEACH=3 ms`. For target 0, `shunt` caps the voltage below threshold on every step. Training presents pairs in a random permutation each epoch for 60 epochs, updates after each presentation, and starts each output's weights at 0.05 (`stdp_temporal.py` lines 50–58, 121–134). Because shunting prevents postsynaptic spikes, target-0 presentations pass an empty postsynaptic event list to the STDP update and therefore produce no pair updates. This is directly visible in `simulate` and `train` (lines 78–108, 121–134).

B1 retains the 2 ms / 7 ms recall encoding, but target-0 learning input is remapped to 4 ms for bit 1 and 0 ms for bit 0; target-0 presentations also force a teacher spike at 3 ms. The B1 timing map, teacher, and training loop are in `stdp_temporal_timing_ltd.py` lines 59–64, 79–114, and 140–153.

B2 uses the normal encoding for recall and target 1. For target 0 learning it swaps the input bit timing: bit 1 is late and bit 0 early (`stdp_temporal_symmetric_teacher_only.py` lines 79–103). Both target classes force a postsynaptic spike at the teacher time 4.5 ms before STDP update (`simulate`, lines 106–134; `train`, lines 154–162). With `Δt = t_post - t_pre`, each pre/post pair gets `+A+ exp(-Δt/τ)` when `Δt>0` and `-A- exp(Δt/τ)` when `Δt<0`; equal-time pairs get no update. The inspected defaults are `A+=A-=0.02`, `τ=2 ms`, initial weight 0.05, and clipping to [-1.5, 1.5] (lines 50–58, 124–134). The timing-based update has a bipolar-correlation-like direction, but is not algebraically identical to the correlation rule: event times, exponential factors, no-update equal-time pairs, clipping, and sequential updates matter.

`stdp_temporal_symmetric.py` is a related B2 variant with additional natural-spike diagnostics and a `recall_peak` helper; `stdp_temporal_symmetric_teacher_only.py` is the source imported by `b2_full_output.py` and the original crosstalk diagnostic.

## Full-output evaluation

Each wrapper selects one output index, resets NumPy's seed to 0, trains that scalar output, and predicts all cues. It stacks the 15 vectors into a `(n_pairs, 15)` matrix, then reports exact-pattern and bit accuracy (`baseline_full_output.py` lines 20–40 and 43–119). No false-positive/false-negative counts are included in these historical metrics. The `predictions.npy`, `targets.npy`, and `weights.npy` files preserve the matrices/vectors used by each full-output summary.

## Pure correlation comparator

`correlation_pair_count_sweep.py` maps 0/1 to -1/+1, forms `W = XᵀY`, computes `score = XW`, and thresholds at score > 0 (lines 49–127). This is a static bipolar correlation memory, not STDP and not LIF spike recall. It is useful as an algebraic comparator for crosstalk and capacity, with this distinction explicit in every report.

## Potential failure mechanisms (hypotheses)

- The baseline target-0 path receives no STDP update because it has no postsynaptic event.
- B1/B2 use learning-time timing manipulations, while recall always uses the standard 2 ms / 7 ms code. The LIF neuron integrates positive and negative weight contributions at different times and labels any threshold crossing as 1; this creates a plausible learning/recall mismatch, especially for late input events.
- Sequential learning, bounded weights, and the shared association set can produce interference even when each pair's local update direction is reasonable.
- The full-output wrapper trains each output bit independently, so cross-output competition or inhibition is absent.

These are mechanisms suggested by source inspection, not a complete causal diagnosis. Distinguishing a poor learned matrix from a recall-dynamics mismatch requires controlled analysis of the saved weights and actual LIF traces.
