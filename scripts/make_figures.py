"""Figures F1-F8 and table T1 for the report (section 13). T2-T5 come from make agree / make train.

    make figures

Every figure is a 300 dpi PNG, one IEEE column (3.5 in) wide, in results/figures/:

    F1_pipeline.png           raw photo -> masked crop -> milk mask -> feature overlay, for one thin, good and thick cup
    F2_examples.png           cups whose mean Overall score is closest to 1, 3 and 5
    F3_agreement.png          inter-rater weighted kappa per rubric dimension, with 95% CI (make agree, else E0)
    F4_feature_validity.png   each handcrafted feature vs its linked rubric dimension, with Spearman rho (E1)
    F5_task_a.png             predicted vs actual Overall for the best E2 model: y = x line and the
                              human-disagreement band (+/- the rater-A vs rater-B MAE)
    F6_confusion.png          summed out-of-fold confusion matrices for B1 (milk) and B2 (drop-in), with
                              the model fixed in feedback.milk_model
    F7_importance.png         permutation importance of the best handcrafted models (E4)
    F8_features_by_milk.png   box plots of each feature by planned milk condition

    results/tables/T1_dataset.{csv,tex}   cups per milk x drop-in cell: logged, usable, labelled by both

A figure whose inputs don't exist yet is skipped with the command to run, and
any old copy is deleted. Captions and all report text are the students'.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latte import plots
from latte.data import load_dataset
from latte.evaluate import latest_run_dir
from latte.features import MODEL_FEATURES, RUBRIC_LINK, extract_features, feature_params
from latte.feedback import feedback_params
from latte.geometry import circles_path, geometry_params, read_circles, resolve_circle
from latte.io import DEFAULT_CONFIG, cfg_path, cup_sort_key, load_config, load_rgb, read_cup_log, write_table
from latte.overlay import CYAN, MAGENTA, YELLOW, draw_axis, draw_outline, draw_point, zoom_view
from latte.preprocess import prepare_cfg
from latte.segment import lab, segment, segment_params

FIGS = {
    "F1": "F1_pipeline.png", "F2": "F2_examples.png", "F3": "F3_agreement.png", "F4": "F4_feature_validity.png",
    "F5": "F5_task_a.png", "F6": "F6_confusion.png", "F7": "F7_importance.png", "F8": "F8_features_by_milk.png",
}
DUMMIES = ("dummy_mean", "dummy_most_frequent", "dummy_stratified")


class Skip(Exception):
    """A figure can't be made yet; the message says what to run."""


def need(data: pd.DataFrame) -> None:
    if data.empty:
        raise Skip("no features yet (make features)")


# ---------------------------------------------------------------------------
# Photo figures
# ---------------------------------------------------------------------------


def _cup_panels(cfg: dict[str, Any], cup: str, saved: pd.DataFrame, size: int = 300) -> dict[str, np.ndarray]:
    rgb = load_rgb(cfg_path(cfg, "raw") / f"{cup}.jpg")
    c = resolve_circle(cup, rgb, saved, geometry_params(cfg), run_auto=False).circle
    prep = prepare_cfg(rgb, c, cfg)
    M = segment(prep.crop, prep.mask, segment_params(cfg), int(cfg["seed"]))
    f = extract_features(lab(prep.crop)[:, :, 0], M, prep.mask, feature_params(cfg))
    photo, _ = zoom_view(rgb, c, size, margin=1.4)
    mask = np.zeros_like(prep.crop)
    mask[prep.mask] = (150, 98, 58)
    mask[M] = (245, 245, 245)
    ov = prep.crop.copy()
    n = ov.shape[0]
    draw_outline(ov, M, MAGENTA, 3)
    if M.any():
        ys, xs = np.nonzero(M)
        draw_axis(ov, n, float(xs.mean()), float(ys.mean()), f["symmetry_angle"], n * 0.47, YELLOW, 3)
        draw_point(ov, n, float(xs.mean()), float(ys.mean()), CYAN, 10)
    return {"photo": photo, "crop": prep.crop, "mask": mask, "overlay": ov}


def _has_photo(cfg: dict[str, Any], cup: str) -> bool:
    return (cfg_path(cfg, "raw") / f"{cup}.jpg").exists()


def f1_pipeline(cfg: dict[str, Any], data: pd.DataFrame, out: Path) -> Path:
    need(data)
    milk = cfg["design"]["milk_conditions"]
    saved = read_circles(circles_path(cfg))
    picks = []
    for m in milk:
        d = data[(data["milk_condition"] == m) & data.index.map(lambda c: _has_photo(cfg, c))]
        if d.empty:
            raise Skip(f"no usable photo with milk_condition={m} yet (make ingest, make features)")
        checked = d[d.get("circle_checked", pd.Series("n", index=d.index)) == "y"]
        pool = checked if len(checked) else d
        if pool["labelled"].any():  # the cup closest to that condition's median Overall score
            lab_pool = pool[pool["labelled"]]
            med = lab_pool["overall"].median()
            pick = (lab_pool["overall"] - med).abs().sort_values(kind="stable").index[0]
        else:
            pick = sorted(pool.index, key=cup_sort_key)[0]
        picks.append((m, pick))
    plots.setup()
    fig, axes = plots.figure(2.75, ncols=4, nrows=len(picks), wspace=0.01, hspace=0.01)
    titles = ["Photo", "Masked crop", "Milk mask", "Features"]
    for r, (m, cup) in enumerate(picks):
        p = _cup_panels(cfg, cup, saved)
        for c, key in enumerate(["photo", "crop", "mask", "overlay"]):
            ax = axes[r, c]
            ax.imshow(p[key])
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], pad=2)
        axes[r, 0].set_ylabel(m, rotation=0, ha="right", va="center", labelpad=4, color=plots.INK)
    return plots.save(fig, out / FIGS["F1"])


def f2_examples(cfg: dict[str, Any], data: pd.DataFrame, out: Path) -> Path:
    need(data)
    lab_ = data[data["labelled"] & data.index.map(lambda c: _has_photo(cfg, c))]
    if len(lab_) < 3:
        raise Skip("needs at least 3 cups labelled by both raters, with photos (make label)")
    saved = read_circles(circles_path(cfg))
    picks, used = [], set()
    for target in (1.0, 3.0, 5.0):
        d = (lab_["overall"] - target).abs().drop(index=list(used), errors="ignore").sort_values(kind="stable")
        picks.append((target, d.index[0]))
        used.add(d.index[0])
    plots.setup()
    fig, axes = plots.figure(1.35, ncols=3, wspace=0.01)
    for ax, (target, cup) in zip(axes, picks):
        ax.imshow(_cup_panels(cfg, cup, saved)["crop"])
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Overall {lab_.at[cup, 'overall']:.1f}", pad=2)
    return plots.save(fig, out / FIGS["F2"])


# ---------------------------------------------------------------------------
# Result figures
# ---------------------------------------------------------------------------


def f3_agreement(cfg: dict[str, Any], out: Path) -> Path:
    """E0 (agreement on exactly the Task A cups, tied to a fresh make train run), else make agree's table."""
    tables = cfg_path(cfg, "tables_dir")
    t, source = None, ""
    try:
        fresh_run(cfg, "E0")
        if (tables / "E0_agreement.csv").exists():
            t, source = pd.read_csv(tables / "E0_agreement.csv"), "E0 (make train)"
    except Skip:
        pass
    if t is None and (tables / "agreement.csv").exists():
        newest_labels = max((p.stat().st_mtime for p in cfg_path(cfg, "labels_dir").glob("rater_*.csv")), default=0)
        if newest_labels > (tables / "agreement.csv").stat().st_mtime:
            raise Skip("the agreement table is older than the latest labels (make agree)")
        t, source = pd.read_csv(tables / "agreement.csv"), "make agree"
    if t is None:
        raise Skip("no agreement table yet (make agree, or make train for E0)")
    print(f"    F3 uses the agreement table from {source}")
    t = t.iloc[::-1]
    target = float(cfg["labelling"]["kappa_target"])
    plots.setup()
    fig, ax = plots.figure(1.8)
    y = np.arange(len(t))
    lo = (t["kappa_w"] - t["kappa_lo"]).clip(lower=0).fillna(0)
    hi = (t["kappa_hi"] - t["kappa_w"]).clip(lower=0).fillna(0)
    ax.errorbar(t["kappa_w"], y, xerr=[lo, hi], fmt="o", color=plots.SERIES[0], ms=3.5, elinewidth=1, capsize=1.5)
    ax.axvline(target, color=plots.INK2, lw=0.7, ls="--")
    ax.text(target, len(t) - 0.4, f" target {target:g}", color=plots.INK2, fontsize=6.5, va="bottom")
    ax.set_yticks(y, t["dimension"])
    low = t["kappa_lo"].min(skipna=True)
    ax.set_xlim(min(-0.05, (low if np.isfinite(low) else 0) - 0.05), 1.0)
    ax.set_xlabel(f"quadratic-weighted Cohen's kappa (n = {int(t['n'].max())} cups)")
    ax.grid(axis="y", visible=False)
    return plots.save(fig, out / FIGS["F3"])


def f4_validity(cfg: dict[str, Any], data: pd.DataFrame, out: Path) -> Path:
    from latte.evaluate import spearman

    need(data)
    lab_ = data[data["labelled"]]
    if len(lab_) < 5:
        raise Skip("needs at least 5 cups labelled by both raters (make label)")
    feats = list(RUBRIC_LINK)
    plots.setup()
    rows = -(-len(feats) // 2)
    fig, axes = plots.figure(1.25 * rows, ncols=2, nrows=rows, hspace=0.05, wspace=0.08)
    rng = np.random.default_rng(0)
    for ax, f in zip(axes.flat, feats):
        d = lab_[[f, RUBRIC_LINK[f]]].dropna()
        ax.scatter(d[f], d[RUBRIC_LINK[f]] + rng.uniform(-0.12, 0.12, len(d)), s=4, color=plots.SERIES[0], alpha=0.7, linewidths=0)
        rho = spearman(d[f].to_numpy(), d[RUBRIC_LINK[f]].to_numpy())
        ax.set_title(f + (f"\nrho = {rho:.2f}, n = {len(d)}" if np.isfinite(rho) else ""), pad=2, fontsize=6.5)
        ax.set_ylabel(f"{RUBRIC_LINK[f]} score", labelpad=1)
        ax.set_yticks([1, 2, 3, 4, 5])
    for ax in list(axes.flat)[len(feats):]:
        ax.axis("off")
    return plots.save(fig, out / FIGS["F4"])


_FINGERPRINT: dict[str, str] = {}


def current_fingerprint(cfg: dict[str, Any]) -> str:
    """The data fingerprint make train would record now (same Study as train_eval/fit_final)."""
    if "fp" not in _FINGERPRINT:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import train_eval

        try:
            _FINGERPRINT["fp"] = train_eval.Study(cfg).fingerprint
        except (FileNotFoundError, ValueError) as err:
            raise Skip(f"can't check the run against the current data ({err})") from err
    return _FINGERPRINT["fp"]


def fresh_run(cfg: dict[str, Any], exp: str) -> Path:
    """The latest run folder of exp, only if it was made from the current data (else Skip)."""
    d = latest_run_dir(cfg, exp)
    if d is None or not (d / "metrics.json").exists():
        raise Skip(f"no {exp} run yet (make train)")
    run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
    if run.get("fingerprint") != current_fingerprint(cfg):
        raise Skip(f"the latest {exp} run is from older data (rerun make train)")
    return d


def _e2(cfg: dict[str, Any]) -> tuple[Path, pd.DataFrame]:
    d = fresh_run(cfg, "E2")
    m = json.loads((d / "metrics.json").read_text())
    if "table" not in m:
        raise Skip(f"the latest E2 run was skipped ({m.get('note', '')})")
    return d, pd.DataFrame(m["table"])


def f5_task_a(cfg: dict[str, Any], out: Path) -> Path:
    """Predicted vs mean label. The band is like for like: a single rater's typical distance from the
    mean label. With two raters, |A - mean| = |A - B| / 2, so the band is +/- MAE(A, B) / 2."""
    d, t = _e2(cfg)
    ra, rb = cfg["labelling"]["raters"][:2]
    models = t[~t["model"].isin(DUMMIES) & ~t["model"].str.startswith("human") & t["mae"].notna()]
    if models.empty:
        raise Skip("no non-dummy E2 model results")
    best = str(models.sort_values("mae")["model"].iloc[0])
    row = models.set_index("model").loc[best]
    human = t[t["model"].str.startswith("human")]
    ab = float(human[f"mae_vs_{ra}"].iloc[0]) if len(human) else np.nan
    oof = pd.read_csv(d / "oof_task_a.csv")
    y, p = oof["y"].to_numpy(float), oof[best].to_numpy(float)
    lo = min(0.8, float(np.nanmin(p)) - 0.2)
    hi = max(5.2, float(np.nanmax(p)) + 0.2)
    plots.setup()
    fig, ax = plots.figure(3.2)
    xs = np.linspace(lo, hi, 50)
    if np.isfinite(ab):
        ax.fill_between(xs, xs - ab / 2, xs + ab / 2, color=plots.GRID, alpha=0.9, lw=0,
                        label=f"± a rater's MAE vs the mean label ({ab / 2:.2f})")
    ax.plot(xs, xs, color=plots.INK2, lw=0.8, label="y = x")
    jitter = np.random.default_rng(0).uniform(-0.06, 0.06, len(y))  # labels sit on 0.5 steps
    ax.scatter(y + jitter, p, s=7, facecolors="none", edgecolors=plots.SERIES[0], linewidths=0.7, zorder=3,
               label=f"{best}, out-of-fold (x jittered)")
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"mean Overall score (raters {ra} and {rb})")
    ax.set_ylabel("predicted Overall score")
    title = (f"n = {len(oof)}. MAE vs mean {row['mae']:.2f} (a rater: {ab / 2:.2f}); "
             f"vs {ra} {row[f'mae_vs_{ra}']:.2f}, vs {rb} {row[f'mae_vs_{rb}']:.2f} ({ra} vs {rb}: {ab:.2f})")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    ax.set_title(title, fontsize=6, loc="left", wrap=True)
    return plots.save(fig, out / FIGS["F5"])


def f6_confusion(cfg: dict[str, Any], out: Path) -> Path:
    d = fresh_run(cfg, "E3")
    model = str(feedback_params(cfg)["milk_model"])
    mats = []
    for target, title in (("milk_condition", "B1 milk condition"), ("dropin", "B2 drop-in")):
        p = d / f"confusion_{target}_{model}.csv"
        if p.exists():
            cm = pd.read_csv(p, index_col=0)
            if len(cm) >= 2:  # a 1x1 matrix (one class present) says nothing
                mats.append((title, cm))
    if not mats:
        raise Skip(f"no confusion matrix for {model} with at least two classes in the latest E3 run "
                   "(make train; a task with one class present has nothing to show)")
    plots.setup()
    fig, axes = plots.figure(1.75, ncols=len(mats), wspace=0.08)
    axes = np.atleast_1d(axes)
    for k, (ax, (title, cm)) in enumerate(zip(axes, mats)):
        rows = cm.to_numpy(float)
        norm = rows / np.maximum(rows.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap=plots.ramp_cmap(), vmin=0, vmax=1)
        for i in range(rows.shape[0]):
            for j in range(rows.shape[1]):
                ax.text(j, i, f"{int(rows[i, j])}", ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.55 else plots.INK)
        ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(cm.index)), cm.index)
        ax.set_xlabel("predicted", labelpad=1)
        if k == 0:
            ax.set_ylabel("planned", labelpad=1)
        ax.set_title(f"{title}\n({model})", pad=3)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    return plots.save(fig, out / FIGS["F6"])


def f7_importance(cfg: dict[str, Any], out: Path) -> Path:
    fresh_run(cfg, "E4")
    p = cfg_path(cfg, "tables_dir") / "E4_importance.csv"
    if not p.exists():
        raise Skip("no E4 table yet (make train)")
    t = pd.read_csv(p)
    tasks = [x for x in ("task_a", "milk_condition", "dropin") if x in set(t["task"])]
    names = {"task_a": "Task A: Overall score (MAE increase)", "milk_condition": "B1 milk (balanced-accuracy drop)",
             "dropin": "B2 drop-in (balanced-accuracy drop)"}
    plots.setup()
    fig, axes = plots.figure(0.45 + 0.17 * len(MODEL_FEATURES) * len(tasks), nrows=len(tasks), hspace=0.06)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        d = t[t["task"] == task].sort_values("importance_mean")
        ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"].fillna(0), color=plots.SERIES[0],
                height=0.6, error_kw={"elinewidth": 0.6, "ecolor": plots.INK2})
        ax.axvline(0, color=plots.INK2, lw=0.6)
        ax.set_title(f"{names[task]}\n{d['model'].iloc[0]}", pad=2, fontsize=6.5)
        ax.grid(axis="y", visible=False)
    return plots.save(fig, out / FIGS["F7"])


def f8_by_milk(cfg: dict[str, Any], data: pd.DataFrame, out: Path) -> Path:
    need(data)
    milk = cfg["design"]["milk_conditions"]
    if data.empty or data["milk_condition"].nunique() < 2:
        raise Skip("needs cups from at least two milk conditions (make features)")
    plots.setup()
    rows = -(-len(MODEL_FEATURES) // 2)
    fig, axes = plots.figure(1.05 * rows, ncols=2, nrows=rows, hspace=0.04, wspace=0.06)
    for k, (ax, f) in enumerate(zip(axes.flat, MODEL_FEATURES)):
        plots.boxes(ax, [data.loc[data["milk_condition"] == m, f].dropna().to_numpy() for m in milk], milk, seed=k)
        ax.set_title(f, pad=2)
    for ax in list(axes.flat)[len(MODEL_FEATURES):]:
        ax.axis("off")
    return plots.save(fig, out / FIGS["F8"])


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------


def t1_dataset(cfg: dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
    log = read_cup_log(cfg_path(cfg, "cup_log"))
    raw = cfg_path(cfg, "raw")
    skip = set(cfg["design"].get("exclude_sessions") or [])
    log = log[log["cup_id"].map(lambda c: (raw / f"{c}.jpg").exists()) & ~log["session_id"].isin(skip)]
    # photographed = ingested photos outside excluded sessions; usable = after usable=n / rater-unusable exclusions
    rows = []
    for m in cfg["design"]["milk_conditions"]:
        for d in cfg["design"]["dropin"]:
            sel = data[(data["milk_condition"] == m) & (data["dropin"] == d)]
            rows.append({"milk_condition": m, "dropin": d,
                         "photographed": int(((log["milk_condition"] == m) & (log["dropin"] == d)).sum()),
                         "usable": len(sel), "labelled_by_both": int(sel["labelled"].sum()),
                         "sessions": int(sel["session_id"].nunique()), "pourers": int(sel["pourer"].nunique())})
    t = pd.DataFrame(rows)
    total = {"milk_condition": "total", "dropin": "", "photographed": int(t["photographed"].sum()),
             "usable": int(t["usable"].sum()), "labelled_by_both": int(t["labelled_by_both"].sum()),
             "sessions": int(data["session_id"].nunique()), "pourers": int(data["pourer"].nunique())}
    return pd.concat([t, pd.DataFrame([total])], ignore_index=True)


# ---------------------------------------------------------------------------


def run(cfg: dict[str, Any]) -> int:
    out = cfg_path(cfg, "figures_dir")
    out.mkdir(parents=True, exist_ok=True)
    try:
        data = load_dataset(cfg).table
    except FileNotFoundError:
        data = pd.DataFrame()
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    jobs = {
        "F1": lambda: f1_pipeline(cfg, data, out),
        "F2": lambda: f2_examples(cfg, data, out),
        "F3": lambda: f3_agreement(cfg, out),
        "F4": lambda: f4_validity(cfg, data, out),
        "F5": lambda: f5_task_a(cfg, out),
        "F6": lambda: f6_confusion(cfg, out),
        "F7": lambda: f7_importance(cfg, out),
        "F8": lambda: f8_by_milk(cfg, data, out),
    }
    made = 0
    for key, job in jobs.items():
        try:
            p = job()
            made += 1
            print(f"{key}: {p.relative_to(cfg['_root']) if p.is_relative_to(cfg['_root']) else p}")
        except Skip as why:
            (out / FIGS[key]).unlink(missing_ok=True)
            print(f"{key}: skipped, {why}")
    tables = cfg_path(cfg, "tables_dir")
    if not data.empty:
        write_table(t1_dataset(cfg, data), tables / "T1_dataset",
                    "Dataset: photographed, usable and doubly-labelled cups per milk condition and drop-in cell",
                    "tab:t1", "{:.0f}")
        print("T1: results/tables/T1_dataset.csv and .tex")
    else:
        for ext in (".csv", ".tex"):
            (tables / f"T1_dataset{ext}").unlink(missing_ok=True)
        print("T1: skipped, no features yet (make features)")
    print(f"{made}/{len(jobs)} figures written to {out.relative_to(cfg['_root']) if out.is_relative_to(cfg['_root']) else out}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args(argv)
    return run(load_config(args.config))


if __name__ == "__main__":
    sys.exit(main())
