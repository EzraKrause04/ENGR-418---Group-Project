"""Reproducibility pass (build step 13): clean clone -> make setup -> make all -> compare.

    make repro                                  # clone HEAD, run everything, compare with this checkout
    make repro ARGS="--keep"                    # keep the clone (its path is printed)
    python scripts/repro_check.py --data DIR    # take the data inputs from another folder

Steps:

1. ``git clone`` the committed HEAD of this repository into a temporary
   folder. Uncommitted changes are *not* included, and the script says so.
2. Data inputs that a clean clone needs but that aren't committed are copied
   in, and listed: ``data/raw/*.jpg``, ``data/cup_log.csv``,
   ``data/circles.csv``, ``labels/rater_*.csv`` and the embedding cache
   ``features/emb_*``. They come from this checkout, or from ``--data``.
   Nothing under ``data/raw_full`` is ever copied.
3. Run ``make setup``, then ``make all`` in the clone, logging to
   ``<clone>/make_all.log``.
4. Check that every expected table, figure and record exists, and compare each
   CSV in ``results/tables/`` with this checkout's copy (or ``--compare DIR``):
   numbers must agree to within ``--rtol`` / ``--atol``.

Exit code: 0 if make all succeeded, nothing expected is missing, and every
compared table matches; 1 otherwise. The clone is deleted unless ``--keep``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ["data/raw/*.jpg", "data/cup_log.csv", "data/circles.csv", "labels/rater_*.csv", "features/emb_*"]
TABLES = ["T1_dataset", "agreement", "E0_agreement", "E1_feature_validity", "T3_task_a", "T4_task_b", "T4b_foam_gain",
          "T5_ablation", "E4_importance", "E5_pourer", "leakage_check", "gonogo_classifier", "gonogo_features"]
FIGURES = ["F1_pipeline", "F2_examples", "F3_agreement", "F4_feature_validity", "F5_task_a", "F6_confusion",
           "F7_importance", "F8_features_by_milk", "gonogo_by_milk", "gonogo_by_dropin", "gonogo_overall"]
OTHER = ["results/env.txt", "results/runs/latest.json", "models/final.json", "features/features.csv"]


def sh(cmd: list[str], cwd: Path, log: Path | None = None, timeout: int | None = None) -> int:
    env = {**os.environ}
    env.pop("VIRTUAL_ENV", None)  # the clone must use its own .venv
    if log is None:
        return subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout).returncode
    with log.open("w") as fh:
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode


def copy_inputs(src: Path, dst: Path, force: bool) -> list[str]:
    copied = []
    for pattern in INPUTS:
        for f in sorted(src.glob(pattern)):
            target = dst / f.relative_to(src)
            if force or not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                copied.append(str(f.relative_to(src)))
    return copied


def compare_tables(new: Path, ref: Path, rtol: float, atol: float) -> tuple[list[str], list[str]]:
    """(lines describing differences, names compared)."""
    diffs, compared = [], []
    for f in sorted(new.glob("*.csv")):
        r = ref / f.name
        if not r.exists():
            continue
        compared.append(f.stem)
        a, b = pd.read_csv(f), pd.read_csv(r)
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            diffs.append(f"{f.name}: shape/columns differ ({a.shape} vs {b.shape})")
            continue
        for c in a.columns:
            x, y = a[c], b[c]
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                xv, yv = x.to_numpy(float), y.to_numpy(float)
                same = np.isclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True)
                if not same.all():
                    d = np.nanmax(np.abs(xv[~same] - yv[~same]))
                    diffs.append(f"{f.name}: column {c} differs (max abs diff {d:.3g})")
            elif not x.astype(str).equals(y.astype(str)):
                diffs.append(f"{f.name}: column {c} differs")
    return diffs, compared


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder to take the data inputs from (default: this checkout)")
    ap.add_argument("--compare", help="folder whose results/tables to compare with (default: this checkout)")
    ap.add_argument("--keep", action="store_true", help="keep the clone")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--timeout", type=int, default=3 * 3600, help="seconds allowed for make all")
    args = ap.parse_args(argv)

    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("NOTE: this checkout has uncommitted changes; the clone uses the committed HEAD only.")
    tmp = Path(tempfile.mkdtemp(prefix="latte-repro-"))
    clone = tmp / "latte-coach"
    ok = True
    try:
        print(f"Cloning HEAD into {clone} ...", flush=True)
        if sh(["git", "clone", "--quiet", str(ROOT), str(clone)], cwd=tmp) != 0:
            print("ERROR: git clone failed", file=sys.stderr)
            return 1
        src = Path(args.data).resolve() if args.data else ROOT
        copied = copy_inputs(src, clone, force=bool(args.data))
        print(f"Copied {len(copied)} data input(s) the clone didn't have"
              + (f" from {src}" if args.data else " (not committed)") + ".")
        if not list((clone / "data" / "raw").glob("*.jpg")):
            print("ERROR: no photos in data/raw: nothing to reproduce.", file=sys.stderr)
            return 1

        t0 = time.time()
        print("make setup ...", flush=True)
        if sh(["make", "setup"], cwd=clone, log=tmp / "make_setup.log") != 0:
            print(f"ERROR: make setup failed; see {tmp / 'make_setup.log'}", file=sys.stderr)
            args.keep = True
            return 1
        print(f"make all ... (log: {clone / 'make_all.log'})", flush=True)
        rc = sh(["make", "all"], cwd=clone, log=clone / "make_all.log", timeout=args.timeout)
        took = time.time() - t0
        print(f"make setup + make all: exit {rc} in {took / 60:.1f} min")
        if rc != 0:
            ok = False
            tail = (clone / "make_all.log").read_text().splitlines()[-15:]
            print("make all FAILED. Last lines of the log:\n  " + "\n  ".join(tail))

        missing = [f"results/tables/{t}.csv" for t in TABLES if not (clone / "results" / "tables" / f"{t}.csv").exists()]
        missing += [f"results/figures/{f}.png" for f in FIGURES if not (clone / "results" / "figures" / f"{f}.png").exists()]
        missing += [o for o in OTHER if not (clone / o).exists()]
        readme = (clone / "README.md").read_text()
        if "No results yet" in readme:
            missing.append("README Results block")
        if missing:
            ok = False
            print("Missing after make all (see the log for the skip reasons):\n  " + "\n  ".join(missing))
        else:
            print(f"All {len(TABLES)} tables, {len(FIGURES)} figures and the records were regenerated.")

        ref = Path(args.compare).resolve() if args.compare else ROOT
        diffs, compared = compare_tables(clone / "results" / "tables", ref / "results" / "tables", args.rtol, args.atol)
        if compared:
            print(f"Compared {len(compared)} tables with {ref / 'results' / 'tables'}: "
                  + ("all identical within tolerance." if not diffs else f"{len(diffs)} difference(s):"))
            for d in diffs:
                print(f"  {d}")
            ok = ok and not diffs
        else:
            print(f"No tables in {ref / 'results' / 'tables'} to compare with.")
        print("REPRODUCIBLE" if ok else "NOT REPRODUCED: see above.")
        return 0 if ok else 1
    finally:
        if args.keep or not ok:
            print(f"Kept the clone at {clone}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
