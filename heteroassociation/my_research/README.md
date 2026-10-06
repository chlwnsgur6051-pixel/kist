# STDP Hetero-associative Memory research workspace

This folder is an isolated working area for studying direct input-output associations in a spiking neural network. The original project and its historical results remain in place. Selected sources and compact result artifacts are copied under `legacy/` with their original paths recorded in `docs/source_manifest.md`.

## Layout

- `legacy/temporal/`: selected original temporal baseline, B1/B2, teacher-only, sweep, and diagnostic sources. These files are preserved without edits.
- `legacy/shared/`: copied dataset definition. `legacy/binary_character.py` is a compatibility shim for the legacy scripts' historical import path.
- `legacy/historical_results/`: compact JSON, CSV, and log artifacts, with separate directories for the two pair-count sweep versions. Small NPY weight/prediction arrays remain local and are excluded from Git.
- `src/`: destination for new, clearly separated data, model, learning, recall, and evaluation modules.
- `experiments/`: future baseline and diagnostic experiment entry points.
- `results/`: new experiment results. Never overwrite historical artifacts.
- `docs/`: research context, source analysis, history, protocol, hypotheses, and provenance.
- `tests/`: reserved for meaningful project tests when requested/needed.

## Environment

The inspected active environment is Python 3.10.21, NumPy 2.2.6, and Matplotlib 3.10.9. Pandas and pytest are not installed. No dependency installation was performed. See `requirements.txt` for the minimal plotting/data dependencies.

The copied scripts retain their original fixed output paths beside the source. Their training entry points were not run during setup. Before a future authorized reproduction, direct output to a new timestamped directory under `results/`; do not run a copied entry point unchanged if its fixed destination could overwrite an earlier run. For example, the historical script entry point is `baseline_full_output.py` in `legacy/temporal/`.

See `docs/experiment_protocol.md` before any future experiment.

## Git

The enclosing Git repository is rooted at the parent `projects/` directory. This folder intentionally has no nested `.git`; the existing repository state and unrelated working changes are preserved.
