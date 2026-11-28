"""Reproducibility pass (build step 13): clean clone -> make setup -> make all -> compare.

    make repro                                   # clone HEAD, run everything, compare with this checkout
    make repro REPRO="--keep"                    # keep the clone (its path is printed)
    python scripts/repro_check.py --data DIR     # inputs from another folder (compared with DIR by default)

Steps:

1. ``git clone`` the committed HEAD of this repository into a temporary
   folder. Uncommitted changes are *not* included, and the script says so.
2. Inputs. A clean clone should hold everything except the photos'
   originals. ``data/raw/*.jpg`` is copied in if it isn't committed. The
   other inputs that belong in the repo (``data/cup_log.csv``,
   ``data/circles.csv``, ``labels/rater_*.csv``) must be committed: if one
   exists here but not in HEAD, the check fails ("not committed") unless
   ``--allow-uncommitted-data`` is given. The embedding cache
   (``features/emb_*``) is copied only with ``--with-cache``, so by default
   the clone recomputes the embeddings. With ``--data DIR`` every input is
   taken from DIR. Nothing under ``data/raw_full`` is ever copied.
3. Run ``make setup``, then ``make all``, in the clone, logging to
   ``<clone>/make_all.log``. Variables from the outer ``make repro`` call are
   not passed down to the clone.
4. Check that every expected table, figure and record exists. Then compare
   each CSV in ``results/tables/`` with the reference (this checkout, ``--data
   DIR`` or ``--compare DIR``): numbers must agree to within ``--rtol`` /
   ``--atol``, and a table present on only one side counts as a difference.

Result line and exit code:

    REPRODUCIBLE                 make all succeeded, nothing missing, every table matches   (exit 0)
    REGENERATED, NOT COMPARED    make all succeeded, nothing missing, no reference tables    (exit 0)
    NOT REPRODUCED               anything else                                              (exit 1)

make all stops at the first failing target. ``make leakcheck`` fails, with
exit 2, when a model beats chance on shuffled labels. That is deliberate:
results that might leak must not flow into the final models and figures.
The clone is kept whenever the check doesn't succeed, or with ``--keep``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = "data/raw/*.jpg"
REPO_INPUTS = ["data/cup_log.csv", "data/circles.csv", "labels/rater_*.csv"]
CACHE = "features/emb_*"
TABLES = ["T1_dataset", "agreement", "E0_agreement", "E1_feature_validity", "T3_task_a", "T4_task_b", "T4b_foam_gain",
          "T5_ablation", "E4_importance", "E5_pourer", "leakage_check", "gonogo_classifier", "gonogo_features"]
FIGURES = ["F1_pipeline", "F2_examples", "F3_agreement", "F4_feature_validity", "F5_task_a", "F6_confusion",
           "F7_importance", "F8_features_by_milk", "gonogo_by_milk", "gonogo_by_dropin", "gonogo_overall"]
OTHER = ["results/env.txt", "results/runs/latest.json", "models/final.json", "features/features.csv"]
MAKE_VARS = ("MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "MAKELEVEL", "MAKEFILES", "VIRTUAL_ENV")


def clean_env() -> dict[str, str]:
    """The environment for the clone: nothing from an outer make call or an activated venv."""
    return {k: v for k, v in os.environ.items() if k not in MAKE_VARS}


def sh(cmd: list[str], cwd: Path, log: Path, timeout: int | None = None) -> int:
    """Run cmd in its own process group; on timeout, kill the whole group (make and its children)."""
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=cwd, env=clean_env(), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise


def gather_inputs(src: Path, clone: Path, from_data: bool, with_cache: bool, allow_uncommitted: bool) -> tuple[list[str], list[str]]:
    """Copy inputs into the clone. Returns (copied, not_committed)."""
    copied, not_committed = [], []
    patterns = [PHOTOS, *REPO_INPUTS] + ([CACHE] if with_cache else [])
    for pattern in patterns:
        for f in sorted(src.glob(pattern)):
            rel = f.relative_to(src)
            target = clone / rel
            if target.exists() and not from_data:
                continue
            if pattern in REPO_INPUTS and not from_data and not allow_uncommitted:
                not_committed.append(str(rel))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            copied.append(str(rel))
    return copied, not_committed


def compare_tables(new: Path, ref: Path, rtol: float, atol: float) -> tuple[list[str], list[str]]:
    """(differences, tables compared). A table on only one side is a difference."""
    diffs, compared = [], []
    new_names = {f.stem for f in new.glob("*.csv")}
    ref_names = {f.stem for f in ref.glob("*.csv")} if ref.exists() else set()
    if not ref_names:
        return [], []
    for name in sorted(new_names ^ ref_names):
        diffs.append(f"{name}.csv: only in {'the clone' if name in new_names else 'the reference'}")
    for name in sorted(new_names & ref_names):
        compared.append(name)
        a, b = pd.read_csv(new / f"{name}.csv"), pd.read_csv(ref / f"{name}.csv")
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            diffs.append(f"{name}.csv: shape/columns differ ({a.shape} vs {b.shape})")
            continue
        for c in a.columns:
            x, y = a[c], b[c]
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
                xv, yv = x.to_numpy(float), y.to_numpy(float)
                same = np.isclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True)
                if not same.all():
                    d = np.nanmax(np.abs(xv[~same] - yv[~same]))
                    diffs.append(f"{name}.csv: column {c} differs (max abs diff {d:.3g})")
            elif not x.astype(str).equals(y.astype(str)):
                diffs.append(f"{name}.csv: column {c} differs")
    return diffs, compared


def stopped_at(log: Path) -> str:
    m = re.findall(r"make(?:\[\d+\])?: \*\*\* \[([^\]]+)\]", log.read_text()) if log.exists() else []
    return m[-1] if m else "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder to take all data inputs from (default: this checkout)")
    ap.add_argument("--compare", help="folder whose results/tables to compare with (default: --data, else this checkout)")
    ap.add_argument("--with-cache", action="store_true", help="also copy the embedding cache (default: recompute)")
    ap.add_argument("--allow-uncommitted-data", action="store_true",
                    help="copy cup_log/circles/labels that aren't committed instead of failing")
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
    ok = False
    try:
        print(f"Cloning HEAD into {clone} ...", flush=True)
        if subprocess.run(["git", "clone", "--quiet", str(ROOT), str(clone)], cwd=tmp, env=clean_env()).returncode != 0:
            print("ERROR: git clone failed", file=sys.stderr)
            return 1
        src = Path(args.data).resolve() if args.data else ROOT
        copied, not_committed = gather_inputs(src, clone, bool(args.data), args.with_cache, args.allow_uncommitted_data)
        if not_committed:
            print("ERROR: these inputs exist here but are not committed, so a clean clone wouldn't have them:\n  "
                  + "\n  ".join(not_committed) + "\nCommit them, or pass --allow-uncommitted-data.", file=sys.stderr)
            return 1
        print(f"Copied {len(copied)} input(s) into the clone" + (f" from {src}" if args.data else " (not committed)")
              + ("" if args.with_cache else "; the embedding cache is recomputed") + ".")
        if not list((clone / "data" / "raw").glob("*.jpg")):
            print("ERROR: no photos in data/raw: nothing to reproduce.", file=sys.stderr)
            return 1

        t0 = time.time()
        print("make setup ...", flush=True)
        if sh(["make", "setup", "ARGS="], cwd=clone, log=tmp / "make_setup.log") != 0:
            print(f"ERROR: make setup failed; see {tmp / 'make_setup.log'}", file=sys.stderr)
            return 1
        log = clone / "make_all.log"
        print(f"make all ... (log: {log})", flush=True)
        try:
            rc = sh(["make", "all", "ARGS="], cwd=clone, log=log, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"ERROR: make all timed out after {args.timeout} s (all its processes were stopped). Log: {log}",
                  file=sys.stderr)
            return 1
        print(f"make setup + make all: exit {rc} in {(time.time() - t0) / 60:.1f} min")
        good = rc == 0
        if not good:
            tail = log.read_text().splitlines()[-15:]
            print(f"make all STOPPED at '{stopped_at(log)}'. Last lines of the log:\n  " + "\n  ".join(tail))

        missing = [f"results/tables/{t}.csv" for t in TABLES if not (clone / "results" / "tables" / f"{t}.csv").exists()]
        missing += [f"results/figures/{f}.png" for f in FIGURES if not (clone / "results" / "figures" / f"{f}.png").exists()]
        missing += [o for o in OTHER if not (clone / o).exists()]
        if "No results yet" in (clone / "README.md").read_text():
            missing.append("README Results block")
        if missing:
            good = False
            print("Missing after make all (skipped for lack of data, or never reached if make all stopped):\n  "
                  + "\n  ".join(missing))
        else:
            print(f"All {len(TABLES)} tables, {len(FIGURES)} figures and the records were regenerated.")

        ref = Path(args.compare).resolve() if args.compare else src
        diffs, compared = compare_tables(clone / "results" / "tables", ref / "results" / "tables", args.rtol, args.atol)
        if compared or diffs:
            print(f"Compared {len(compared)} tables with {ref / 'results' / 'tables'}: "
                  + ("all identical within tolerance." if not diffs else f"{len(diffs)} difference(s):"))
            for d in diffs:
                print(f"  {d}")
            ok = good and not diffs
            print("REPRODUCIBLE" if ok else "NOT REPRODUCED: see above.")
        else:
            ok = good
            print(f"No reference tables in {ref / 'results' / 'tables'}." + (
                " REGENERATED, NOT COMPARED: make all reran everything from a clean clone, but there was nothing to "
                "compare with (pass --compare DIR, or run make all here first)." if ok else " NOT REPRODUCED: see above."))
        return 0 if ok else 1
    finally:
        if args.keep or not ok:
            print(f"Kept the clone at {clone}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
