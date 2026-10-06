# my_research project rules

These rules apply to all work under `my_research/`.

1. Do not modify original research source files. Preserve legacy copies as provenance.
2. Write new experiments and research code inside `my_research/`.
3. Keep baselines separate from each modification, with separate configs and result directories.
4. Record experiment conditions, code revision, dataset, and random seeds.
5. Never change evaluation criteria to make results look better. Use exact pattern recall and bit accuracy consistently, and report false positives and false negatives when relevant.
6. Keep failed and incomplete experiment records with their failure status and explanation.
7. Describe learning and recall as separate stages, including any teacher signals used only during learning.
8. Do not describe pure correlation results as STDP performance.
9. Distinguish actual LIF spike recall from static score/readout results.
10. Change one variable at a time when testing a research hypothesis where practical.
11. Keep every experiment reproducible from recorded code, configuration, data, seed, and dependencies.
12. Separate facts verified from code/data from hypotheses and interpretations.
13. Before changing architecture, gather evidence about the current bottleneck.
14. Do not overwrite historical results. Create a new timestamped result directory for every run.
15. Do not execute an experiment unless the user specifically asks for it.

The repository root is an existing Git repository above this folder. Do not create a nested Git repository or rewrite repository history.
