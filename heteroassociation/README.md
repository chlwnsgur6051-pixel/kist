# STDP Hetero-associative Memory

Spiking neural network research on storing digit-to-letter bitmap associations with spike-timing-dependent plasticity (STDP) and recalling them through LIF output dynamics.

## Project map

- `binary_character.py`: 5×3 digit and letter bitmap dataset.
- `Ver관리/`: chronological research code, models, experiments, and historical outputs. Treat these sources as research records; preserve original files when developing new methods.
- `my_research/`: organized research workspace, selected legacy copies, analysis, experiment protocol, and future experiment locations.
- `my_research/docs/source_manifest.md`: provenance and hashes for copied legacy material.

## Current research focus

Determine whether recall failures come from the learned synaptic weights or from the LIF recall dynamics and the timing representation. The working constraint is direct input-output association with actual spike events and classical pair-based STDP as the starting point. No performance target is claimed as achieved until reproduced from a recorded run.

## Getting started

See [`my_research/README.md`](my_research/README.md) for the workspace layout and environment, and [`my_research/docs/experiment_protocol.md`](my_research/docs/experiment_protocol.md) before running experiments. Historical results are evidence from earlier runs; they are not regenerated automatically.

## Git workflow

The local checkout is inside an existing workspace repository. Instructions for synchronizing this project to its standalone GitHub repository are in [`my_research/docs/git_workflow.md`](my_research/docs/git_workflow.md).
