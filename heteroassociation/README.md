# STDP Hetero-associative Memory

Spiking neural network research on storing digit-to-letter bitmap associations with spike-timing-dependent plasticity (STDP) and recalling them through LIF output dynamics.

## Project map

- `binary_character.py`: 5×3 digit and letter bitmap dataset.
- [`research/`](research/README.md): validated, independently documented models and their reproducibility checks.
- `Ver관리/`: chronological research code, models, experiments, and historical outputs. Treat these sources as research records; preserve original files when developing new methods.
- `my_research/`: organized research workspace, selected legacy copies, analysis, experiment protocol, and future experiment locations.
- `my_research/docs/source_manifest.md`: provenance and hashes for copied legacy material.

## Current research focus

Two validated 15×15 feedforward models are preserved separately. The [clocked LIF + STDP model](research/clocked_stdp_heteroassociation/README.md) reaches 10/10 teacher-free recall on Original and Modified K=10. The [continuous-state Natural LIF + all-to-all STDP model](research/continuous_stdp_heteroassociation/README.md) also reaches 10/10 with signed weights on those datasets using learning multiplier 1. Its Original nonnegative-weight condition reaches 4/10. See each model's configuration, result tables, and limitations before comparing them; their neuron dynamics and learning protocols differ.

## Getting started

See [`my_research/README.md`](my_research/README.md) for the workspace layout and environment, and [`my_research/docs/experiment_protocol.md`](my_research/docs/experiment_protocol.md) before running experiments. Historical results are evidence from earlier runs; they are not regenerated automatically.

## Git workflow

The local checkout is inside an existing workspace repository. Instructions for synchronizing this project to its standalone GitHub repository are in [`my_research/docs/git_workflow.md`](my_research/docs/git_workflow.md).
