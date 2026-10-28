# Methods

Coverage: full: 500 epochs and 10 independent seeds per condition; 4 jitter sources and σ=0, 0.1, 0.2, 0.5 ms. Part A additionally tests σ=1.0 ms with 100,000 continuous-time pair samples per source/σ.

Datasets: archived Original and Modified only. Binary inputs and original letter targets unchanged. Effective weights start at zero; association order 0→9; no clipping, backpropagation, or explicit error gate. Teacher-free jitter-free Clocked LIF evaluates each epoch and final weight.

Independent Gaussian timing errors are sampled per stream/presentation. Input errors are per active presynaptic neuron, shared by its 15 outgoing synapses; prediction and teacher errors are each one clock displacement per presentation. Continuous samples are rounded to nearest 0.1 ms using NumPy rint (ties to even). The applied timing errors are recorded. No event is clamped, reordered, or resampled. The isolated adapter uses the original Euler integration order: input current, one-step teacher current, membrane update, synaptic-current decay, threshold test, reset and 1-ms refractory.

If teacher step is at/before prediction readout, the presentation is logged and omitted from plasticity to prevent target leakage. Input spikes at/after prediction are processed at their actual gridded times and counted. Negative or late event times extend the initially quiescent trial window. These policy counts appear per run. Part A keeps unrounded Gaussian timing, including order violations, as an independent kernel diagnostic.

For each actual pre/post pair, the unchanged centered window is evaluated at the observed grid timing. Correct-11 residual plasticity is recorded separately. At σ=0, archived exact recall is compared at every epoch and archived weights are compared at epochs 1, 10, 20, ... 500; a discrepancy invalidates baseline equivalence. The numerical weight tolerance is 1e-12.

The saved `representative_trial_replay_traces.npz` independently replays the first three Original/combined/σ=0.5/seed-0 presentations from their recorded gridded event schedule. It stores all 15 neuron voltages, currents, actual post-event masks, and all 225 per-synapse prediction, teacher, and net updates. The Digit 1 replay is checked against the membrane trace and update matrix saved during the batched run. This replay does not generate new jitter or retrain the model.

Software and source hashes are in `source_manifest.json`.
