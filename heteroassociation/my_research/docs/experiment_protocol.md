# Experiment protocol

No experiment was executed during workspace initialization. Use this protocol when a specific experiment is authorized.

## Fixed task definition

- Dataset: the ten ordered 5×3 digit-to-letter pairs from the versioned `binary_character.py` copy.
- Main architecture: 15 input neurons, direct synapses to 15 output LIF neurons; no hidden layer.
- Primary success measure: exact 15-bit recall per association.
- Also report bit accuracy, false positives, false negatives, and per-output accuracy.
- Run actual LIF spike recall for STDP models. Label static correlation readout separately.

## Record for every run

Create a unique `results/<experiment>_YYYYMMDD_HHMMSS/` directory. Save the source revision/hash, config, dataset version, pair count and exact subset, random seed(s), epoch count, encoding times, STDP parameters, weight initialization and bounds, LIF parameters, teacher behavior, recall rule, metrics, predictions, weights, success/failure status, and a short interpretation with limitations. Never overwrite an existing run.

Keep model-weight arrays and other bulky generated tensors local unless a specific experiment request asks to version them. Version the source, config, compact metrics, and report so the research remains understandable without checkpoint files.

## Fair comparison

Keep baseline and each modification in separate experiment configurations. Hold the dataset, association subset/order, seeds, and evaluation constant when comparing one changed factor. If a method is stochastic, repeat across declared seeds and report each run plus aggregates. Do not tune on evaluation outcomes without marking the result as exploratory.

## Learning versus recall

Document teacher events and any target-dependent input timing used during training separately from cue encoding at recall. Record whether recall used a teacher, inhibition, a static threshold, or the actual LIF event rule. A forced teacher spike is a learning signal; it must not be counted as recall output.

## Failure handling

Keep failed runs. Record exceptions, interrupted runs, and incomplete artifacts with status and cause. Do not remove poor outcomes or change the exact-recall criterion after seeing results.
