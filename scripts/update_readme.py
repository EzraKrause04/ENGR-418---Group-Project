"""Fill the README's Results section from results/tables/ (build step 12).

    make readme          # also part of make all

Only the text between ``<!-- results:start -->`` and ``<!-- results:end -->``
in README.md is replaced. It holds numbers and tables copied from the
generated CSVs (agreement, T3, T4, leakage check) plus which run they came
from, and nothing interpretive: Key findings and Limitations are the students'
to write (brief, sections 2 and 15). Each item is checked against the current
data, the same fingerprint make train records, and is marked "out of date"
rather than shown when it no longer matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latte.io import DEFAULT_CONFIG, cfg_path, load_config

START, END = "<!-- results:start -->", "<!-- results:end -->"
DUMMIES = ("dummy_mean", "dummy_most_frequent", "dummy_stratified")


def _f(v: Any, nd: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v) if v is not None and str(v) != "nan" else "–"
    return "–" if not np.isfinite(x) else f"{x:.{nd}f}"


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    """Markdown table at 3 decimals, the precision of the report's .tex tables."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_f(v, nd) if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def read(tables: Path, name: str) -> pd.DataFrame | None:
    p = tables / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def current_fingerprint(cfg: dict[str, Any]) -> str | None:
    """The data fingerprint make train would record now, or None if there is no data yet."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_eval

    try:
        return train_eval.Study(cfg).fingerprint
    except (FileNotFoundError, ValueError):
        return None


def run_info(cfg: dict[str, Any], exp: str) -> dict[str, Any]:
    from latte.evaluate import latest_run_dir

    d = latest_run_dir(cfg, exp)
    if d is None or not (d / "run.json").exists():
        return {}
    return json.loads((d / "run.json").read_text())


def emb_note(info: dict[str, Any]) -> str:
    e = info.get("embedding") or {}
    if not e.get("model"):
        return "embedding models skipped"
    return f"embeddings: {e['model']}" + (" (fallback)" if e.get("fell_back") else "") + \
        (" (RANDOM weights, test only)" if e.get("pretrained") is False else "")


def build(cfg: dict[str, Any]) -> str:
    from latte.data import load_dataset
    from latte.feedback import score_error_text

    tables = cfg_path(cfg, "tables_dir")
    ra, rb = cfg["labelling"]["raters"][:2]
    parts: list[str] = []
    fp = current_fingerprint(cfg)

    try:  # data summary straight from the current data, so it can never be stale
        data = load_dataset(cfg).table
    except (FileNotFoundError, ValueError):
        data = pd.DataFrame()
    if not data.empty:
        parts.append(f"**Data (now):** {len(data)} usable cups with features, {int(data['labelled'].sum())} labelled by "
                     f"both raters, {data['session_id'].nunique()} sessions, {data['pourer'].nunique()} pourers. "
                     "Per-cell counts: `results/tables/T1_dataset.csv` (make figures).")

    ag = read(tables, "agreement")
    if ag is not None:
        newest = max((q.stat().st_mtime for q in cfg_path(cfg, "labels_dir").glob("rater_*.csv")), default=0)
        if newest > (tables / "agreement.csv").stat().st_mtime:
            parts.append("**Inter-rater agreement (T2):** out of date, since the labels changed after it was made (make agree).")
        else:
            cols = [c for c in ("dimension", "n", "kappa_w", "kappa_lo", "kappa_hi", "spearman", "mad") if c in ag]
            parts.append("**Inter-rater agreement (T2, make agree):** quadratic-weighted κ with 95% CI, Spearman ρ, "
                         "mean absolute difference.\n\n" + md_table(ag[cols]))

    for exp, name, title in (("E2", "T3_task_a", "Task A, mean Overall score (T3"), ("E3", "T4_task_b", "Task B, diagnosis (T4")):
        t, info = read(tables, name), run_info(cfg, exp)
        if t is None:
            continue
        if fp is None or info.get("fingerprint") != fp:
            parts.append(f"**{title}):** out of date, from data that has since changed (make train).")
            continue
        head = f"**{title}, run {info.get('timestamp', '?')}, {info.get('n_cups', '?')} cups in {info.get('n_sessions', '?')} sessions; {emb_note(info)}):** leave-one-session-out. "
        if exp == "E2":
            cols = [c for c in ("model", "n", "mae", "mae_fold_std", f"mae_vs_{ra}", f"mae_vs_{rb}", "spearman", "r2") if c in t]
            parts.append(head + f"`mae` is against the mean label; `mae_vs_{ra}`/`mae_vs_{rb}` are against one rater, "
                         "the columns comparable with the rater-vs-rater row.\n\n" + md_table(t[cols]))
        else:
            cols = [c for c in ("target", "model", "n", "chance", "bal_acc", "bal_acc_ci_lo", "bal_acc_ci_hi", "macro_f1",
                                "above_chance") if c in t]
            parts.append(head + "Balanced accuracy with a 95% session-bootstrap CI.\n\n" + md_table(t[cols]))

    leak = read(tables, "leakage_check")
    runs_dir = cfg_path(cfg, "runs_dir")
    lc = sorted(runs_dir.glob("*_leakcheck")) if runs_dir.exists() else []
    lc_info = json.loads((lc[-1] / "run.json").read_text()) if lc and (lc[-1] / "run.json").exists() else {}
    if leak is not None and fp is not None and lc_info.get("fingerprint") == fp:
        ok = (leak["verdict"] == "PASS").all()
        parts.append(f"**Leakage check (make leakcheck, run {lc_info.get('timestamp', '?')}):** "
                     + ("every model at chance on labels shuffled within sessions (PASS)." if ok else "some models flagged CHECK.")
                     + " Details: `results/tables/leakage_check.csv`.")
    elif leak is not None or lc:
        parts.append("**Leakage check:** not run on the current data (make leakcheck).")

    final = cfg_path(cfg, "models_dir") / "final.json"
    if final.exists():
        fj = json.loads(final.read_text())
        if fp is not None and fj.get("fingerprint") == fp:
            milk = (fj.get("milk") or {}).get("name") or f"withheld ({fj.get('milk_withheld_reason', '')})"
            parts.append(f"**Demo models (make final):** score model `{fj['score']['name']}`, shown as ± "
                         f"{score_error_text(float(fj['score']['error'])):.1f} (out-of-fold error of the rounded score "
                         f"{_f(fj['score']['error'])}); fixes from `ridge_hand`; milk diagnosis: {milk}.")
        else:
            parts.append("**Demo models:** out of date (make final).")

    if not parts:
        return "_No results yet: run `make features`, `make label`, `make agree`, `make train` and `make figures`, then `make readme`._"
    figs = sorted(p.stem.split("_")[0] for p in cfg_path(cfg, "figures_dir").glob("F*.png"))
    if figs:
        parts.append(f"Figures in [`results/figures/`](results/figures/): {', '.join(figs)}.")
    return "\n\n".join(parts)


def update(readme: Path, block: str) -> bool:
    text = readme.read_text()
    if START not in text or END not in text:
        raise ValueError(f"{readme} has no {START} ... {END} block")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{block}\n{END}{tail}"
    changed = new != text
    if changed:
        readme.write_text(new)
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--readme", help="default: README.md in the project root")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    readme = Path(args.readme) if args.readme else Path(cfg["_root"]) / "README.md"
    try:
        changed = update(readme, build(cfg))
    except (ValueError, FileNotFoundError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"{'Updated' if changed else 'No change to'} the Results section of {readme.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
