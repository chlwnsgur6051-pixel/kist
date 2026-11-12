"""Write SHA-256, size and provenance for every archived file (artifact_manifest.csv)."""
from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = "results/spike_timing_correction_baseline_v2_validation"
SKIP_DIRS = {"build", "runs", "__pycache__"}


def provenance(rel):
    if rel.startswith("datasets/"):
        return f"byte-identical copy of ../baseline_v1/{rel}"
    if rel.startswith("results/validation/"):
        return f"copy of {VALIDATION}/csv/{Path(rel).name}"
    if rel.startswith("results/csv/") or rel.startswith("results/checkpoints/"):
        return f"5000-epoch reference runs of {VALIDATION} (group 'reference', candidate models)"
    if rel in ("src/stc_core.c", "src/reference_model.py"):
        return f"copy of {VALIDATION}/scripts/{Path(rel).name}"
    if rel == "src/stc_model.py":
        return f"{VALIDATION}/scripts/stc_model.py with archive-relative paths and STEP_V2/EXP_V2 added"
    return "written for this archive"


def main():
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "."], cwd=ROOT,
                            capture_output=True, text=True)
    if listed.returncode == 0 and listed.stdout.strip():
        files = sorted(ROOT / line for line in listed.stdout.splitlines())
    else:
        files = sorted(p for p in ROOT.rglob("*") if p.is_file()
                       and not SKIP_DIRS & set(p.relative_to(ROOT).parts) and p.suffix != ".pyc")
    rows = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel == "artifact_manifest.csv" or not path.is_file():
            continue
        data = path.read_bytes()
        rows.append(dict(archived_path=rel, sha256=hashlib.sha256(data).hexdigest(),
                         size_bytes=len(data), provenance=provenance(rel)))
    with open(ROOT / "artifact_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} files -> artifact_manifest.csv")


if __name__ == "__main__":
    main()
