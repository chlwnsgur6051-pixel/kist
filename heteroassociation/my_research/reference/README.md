# Fixed heteroassociative-memory references

This directory freezes three 10-pair, 15-input × 15-output bipolar datasets and
the **Calibrated Shared-Spike Additive STDP** teacher-event baseline. The old B2
source remains at `Ver관리/2. temporal/stdp_temporal_symmetric_teacher_only.py`;
the wrapper calls its unchanged `stdp_update` function.

From `heteroassociation/`, with Python, NumPy and Matplotlib installed:

```bash
python3 my_research/reference/verify_checkpoint.py --dataset A
python3 my_research/reference/verify_checkpoint.py --dataset B
python3 my_research/reference/verify_checkpoint.py --dataset Original
```

The commands each create a timestamped directory under `validation/`. Run
`python3 my_research/reference/verify_checkpoint.py --dataset all` for one
combined check. `--freeze` was used once to copy the original bitmap definitions
and supplied A/B NPZ values into `datasets/`; existing frozen files are checked
element by element and never overwritten. The verifier checks frozen file hashes,
shape, bipolar conversion, rank, every-subset correlation accuracy, K=10
pseudoinverse accuracy, shared pre-spike timing, four event-update signs, and a
direct 60-epoch bounded call to the original B2 `stdp_update` for each dataset.

Dataset A/B target bitmaps in the supplied NPZ **differ from the current
`binary_character.py` letter bitmaps**. The frozen A/B `Y_binary` arrays are
exact inverses of their supplied `Y_bipolar` arrays; no replacement letter
patterns were inferred. The positional association labels remain 0→A through
9→J. `datasets/manifest.json` records the source and frozen SHA-256 hashes.

Primary static evaluation is `score=X_bipolar @ W`, prediction +1 only when
`score>0`. Correlation uses `W=X.T@Y`; pseudoinverse uses `W=pinv(X)@Y` and is
a mathematical reference. Shared-spike STDP uses `W=W_final−W_initial` for
readout. Natural postsynaptic spike learning and LIF recall are outside this
validated baseline. The calibrated setting was selected on A/B; the Original
dataset's new limited verification found clipping and is separately labeled in
the checkpoint report.

`evidence/` contains compact copies of selected historical summaries and
numeric references. Historical full sweeps remain in their original local
timestamped `results/` folders and are intentionally not duplicated in Git.
