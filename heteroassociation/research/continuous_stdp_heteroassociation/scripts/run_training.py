"""Run the preserved continuous STDP source with the bundled references.

Numba is optional for this wrapper. The validated run used the Python fallback,
which changes execution speed only. The lambda=0.1 source is archived separately.
"""
from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/continuous_stdp_experiment.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Write a fresh replay here; archived results are left untouched.")
    parser.add_argument("--gap-ms", type=float, default=5.,
                        help="Original validated gap is 5 ms. Other values are unvalidated.")
    args = parser.parse_args()
    if args.gap_ms != 5.:
        parser.error("This archived reference runner accepts only the validated 5 ms gap.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be new or empty.")
    if importlib.util.find_spec("numba") is None:
        shim = types.ModuleType("numba")
        shim.njit = lambda function: function
        sys.modules["numba"] = shim
        print("Numba unavailable; using the verified pure-Python execution path.", flush=True)
    sys.argv = [str(SOURCE), "--gap-ms", "5", "--out-dir", str(args.output_dir)]
    runpy.run_path(str(SOURCE), run_name="__main__")


if __name__ == "__main__":
    main()
