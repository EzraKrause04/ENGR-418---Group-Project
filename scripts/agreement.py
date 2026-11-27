"""Inter-rater agreement per rubric dimension (section 7.3, experiment E0).

    python scripts/agreement.py                       # all cups both raters labelled
    python scripts/agreement.py --subset calibration  # the first 30 cups only

Per dimension: quadratic-weighted Cohen's kappa (with 95% bootstrap CI),
Spearman rho, and mean absolute difference between raters. Cups either rater
marked unusable, or marked usable=n in the cup log, are excluded.

Writes results/tables/agreement[_calibration].{csv,tex} and lists large
disagreements in results/qa/disagreements[_calibration].csv for review.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from latte.evaluate import agreement_table
from latte.io import DEFAULT_CONFIG, cfg_path, cup_sort_key, load_config, read_cup_log, write_table
from latte.labels import available_cups, latest_labels, labels_path, read_labels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--subset", choices=["all", "calibration"], default="all")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    lab = cfg["labelling"]
    dims: list[str] = lab["dimensions"]
    if len(lab["raters"]) != 2:
        ap.error("agreement needs exactly two raters in the config")
    ra, rb = lab["raters"]
    a = latest_labels(read_labels(labels_path(cfg, ra), dims))
    b = latest_labels(read_labels(labels_path(cfg, rb), dims))

    # Cups in scope: ingested, not log-unusable, not in an excluded session.
    cups = available_cups({**cfg, "labelling": {**lab, "skip_log_unusable": True}})
    excluded_sessions = set(cfg["design"].get("exclude_sessions") or [])
    cups = sorted((c for c in cups if c.split("-")[0] not in excluded_sessions), key=cup_sort_key)
    if args.subset == "calibration":
        cups = cups[: int(lab["calibration_n"])]

    both = [c for c in cups if c in a.index and c in b.index]
    unusable = [c for c in both if a.at[c, "unusable"] == "y" or b.at[c, "unusable"] == "y"]
    keep = [c for c in both if c not in unusable]
    print(
        f"Cups in scope: {len(cups)} | labelled by both: {len(both)} | "
        f"unusable (either rater): {len(unusable)} | used: {len(keep)}"
    )
    if args.subset == "calibration" and len(both) < len(cups):
        print(f"WARNING: calibration set not complete ({len(both)}/{len(cups)} labelled by both raters).")
    if len(keep) < 2:
        suffix = "" if args.subset == "all" else "_calibration"
        for ext in (".csv", ".tex"):  # an old table must not pass for a new one
            (cfg_path(cfg, "tables_dir") / f"agreement{suffix}{ext}").unlink(missing_ok=True)
        print("Not enough doubly-labelled usable cups to compute agreement yet.")
        return 1

    seed = int(cfg["seed"])
    table = agreement_table(a.loc[keep], b.loc[keep], dims, n_boot=int(lab["bootstrap_n"]), seed=seed)
    suffix = "" if args.subset == "all" else "_calibration"
    caption = "Inter-rater agreement per rubric dimension" + (" (calibration round)" if suffix else "")
    csv_path, tex_path = write_table(
        table, cfg_path(cfg, "tables_dir") / f"agreement{suffix}", caption, f"tab:agreement{suffix}"
    )
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nwrote {csv_path} and {tex_path}")

    flag = int(lab["disagreement_flag"])
    dis = disagreements(a.loc[keep], b.loc[keep], dims, flag, (ra, rb))
    qa_path = cfg_path(cfg, "qa_dir") / f"disagreements{suffix}.csv"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    dis.to_csv(qa_path, index=False)
    print(f"wrote {qa_path} ({len(dis)} cups with |{ra}-{rb}| >= {flag} on some dimension)")

    k = float(table.loc[table["dimension"] == "overall", "kappa_w"].iloc[0]) if "overall" in dims else float("nan")
    if k == k:  # not NaN
        if k >= lab["kappa_target"]:
            verdict = f"meets the target (>= {lab['kappa_target']})"
        elif k < lab["kappa_revise_below"]:
            verdict = (
                f"is below {lab['kappa_revise_below']}: review the disagreements, revise the anchors "
                "in docs/rubric.md (log it in the changelog), then relabel"
            )
        else:
            verdict = f"is below the {lab['kappa_target']} target but above the {lab['kappa_revise_below']} revise line"
        print(f"\nOverall weighted kappa = {k:.2f}: {verdict}.")
    return 0


def disagreements(a: pd.DataFrame, b: pd.DataFrame, dims: list[str], flag: int, names: tuple[str, str]) -> pd.DataFrame:
    """Cups where the raters differ by >= ``flag`` points on any dimension."""
    ra, rb = names
    rows = []
    for c in a.index:
        diffs = {d: abs(float(a.at[c, d]) - float(b.at[c, d])) for d in dims}
        if max(diffs.values()) >= flag:
            row: dict = {"cup_id": c, "max_diff": max(diffs.values())}
            for d in dims:
                row[f"{d}_{ra}"], row[f"{d}_{rb}"] = a.at[c, d], b.at[c, d]
            row[f"comment_{ra}"], row[f"comment_{rb}"] = a.at[c, "comment"], b.at[c, "comment"]
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["max_diff", "cup_id"], ascending=[False, True]) if len(out) else out


if __name__ == "__main__":
    sys.exit(main())
