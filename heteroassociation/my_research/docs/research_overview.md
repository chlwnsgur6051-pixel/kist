# Research overview

## Goal

Develop a direct `Input X → STDP synapse W → LIF Output Y` hetero-associative memory. The network should store multiple input-output associations in one shared weight matrix and recall an output from an input cue using actual spikes and output-neuron dynamics.

The immediate research goal is to identify, with reproducible evidence, why the current system fails and then find a grounded improvement. The near-term performance target is exact recall for at least 5 of 10 associations; this is a goal, not an achieved result.

## Dataset

The current dataset contains ten 5×3 binary digit patterns (0–9) and ten 5×3 binary letter patterns (A–J), paired in order. Each input and output has 15 bits. The pattern definitions are in `legacy/shared/binary_character.py`, copied from the original root `binary_character.py`.

The source comments indicate that digit 9's final row was changed from `111` to `001` to increase the digit matrix rank. Treat this as part of the dataset version and do not silently replace it.

## Constraints

- Use actual spike events and LIF dynamics for the core method.
- Keep classical pair-based STDP as the initial learning rule.
- Keep direct input-output connections; hidden layers and backpropagation are outside the current scope.
- Consider changes to encoding or the STDP rule only as separately controlled experiments.
- Consider output inhibition only when evidence points to a need.

## Evaluation

- Exact recall accuracy: fraction of associations whose 15 output bits all match.
- Bit accuracy: correct output bits divided by all output bits.
- False positives: predicted 1 where target is 0.
- False negatives: predicted 0 where target is 1.
- Per-output accuracy may help localize weak output pixels.

Exact recall is the main success measure. Static score baselines and LIF spike recall must be labeled separately.
