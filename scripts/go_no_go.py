"""Go/no-go check for Fri Nov 27, with at least 30 usable cups (section 16).

    make gonogo

It reads features/features.csv, data/cup_log.csv and both raters' labels, and
applies the exclusions in ``latte.data``. It writes:

    results/figures/gonogo_by_milk.png     box plots of each image feature, plus measured foam gain
                                           as a manipulation check, by milk_condition
    results/figures/gonogo_by_dropin.png   the same by drop-in timing
    results/figures/gonogo_overall.png     distribution of the mean Overall score (cups labelled by
                                           both raters); skipped while there are none
    results/tables/gonogo_classifier.*     leave-one-session-out balanced accuracy of logistic
                                           regression for B1 (milk_condition) and B2 (dropin), with a
                                           within-session label-permutation p-value
    results/tables/gonogo_features.*       per-feature separation: Kruskal-Wallis (milk) and
                                           Mann-Whitney (drop-in), Holm-adjusted over the image features

Classifier: median imputation -> standardisation -> LogisticRegression (C = 1),
all fitted inside each training fold. The out-of-fold predictions from all
sessions are pooled and scored with balanced accuracy. Chance = 1 / (number of
classes present). Null distribution: the same procedure with labels shuffled
*within* each session, ``gonogo.n_permutations`` times (seeded).
p = (1 + #null >= observed) / (1 + n).

Effect sizes: epsilon^2 = H / ((n^2 - 1) / (n + 1)) for Kruskal-Wallis.
Rank-biserial r = 2 U / (n1 n2) - 1 for Mann-Whitney, with U for the first
drop-in level, so r > 0 means the first level (on_time) tends to be higher.

Foam gain is measured, not read from the photo. It is shown and tested as a
manipulation check (did the milk conditions really differ?), but it is kept out
of the Holm family and out of "features that separate the conditions".

The printed hints apply section 16's decision rules. The team makes the decision.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kruskal, mannwhitneyu  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from latte.data import load_dataset  # noqa: E402
from latte.features import MODEL_FEATURES  # noqa: E402
from latte.io import DEFAULT_CONFIG, cfg_path, load_config, write_table  # noqa: E402

GONOGO_DEFAULTS: dict[str, Any] = {
    "min_cups": 30,
    "n_permutations": 200,
    "alpha": 0.05,
    "narrow_max_score": 2.0,
    "narrow_frac": 0.9,
}
MANIPULATION = "foam_gain_pct"

# Reference categorical palette, first three slots (validated all-pairs, light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------
# Leave-one-session-out classifier with a permutation null
# ---------------------------------------------------------------------------


def classifier() -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def loso_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions, each session predicted by a model fitted on all the others."""
    pred = np.empty(len(y), dtype=object)
    for g in np.unique(groups):
        test = groups == g
        train = ~test
        if len(np.unique(y[train])) < 2:
            raise ValueError(f"the training data without session {g} contains only one class")
        pred[test] = classifier().fit(X[train], y[train]).predict(X[test])
    return pred


def within_group_shuffle(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = y[rng.permutation(idx)]
    return out


def loso_task(df: pd.DataFrame, target: str, classes: list[str], n_perm: int, seed: int) -> dict[str, Any]:
    """Balanced accuracy, permutation p-value and pooled confusion matrix for one Task B target."""
    d = df[df[target].isin(classes)]
    present = [c for c in classes if (d[target] == c).any()]
    notes = [f"absent: {', '.join(c for c in classes if c not in present)}"] if len(present) < len(classes) else []
    usable = [f for f in MODEL_FEATURES if d[f].notna().any()]
    if len(usable) < len(MODEL_FEATURES):
        notes.append("all-missing features dropped: " + ", ".join(f for f in MODEL_FEATURES if f not in usable))
    row: dict[str, Any] = {"task": target, "n": len(d), "sessions": d["session_id"].nunique(),
                           "classes": len(present), "chance": 1 / len(present) if present else np.nan}
    empty = {"bal_acc": np.nan, "null_95th": np.nan, "p_perm": np.nan}
    if row["sessions"] < 2 or len(present) < 2 or not usable:
        return row | empty | {"note": "; ".join([*notes, "needs at least 2 sessions, 2 classes and 1 feature"])}
    X = d[usable].to_numpy(float)
    y = d[target].to_numpy(object)
    g = d["session_id"].to_numpy(object)
    try:
        pred = loso_predict(X, y, g)
    except ValueError as err:
        return row | empty | {"note": "; ".join([*notes, str(err)])}
    obs = balanced_accuracy_score(y, pred)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        ys = within_group_shuffle(y, g, rng)
        try:
            null.append(balanced_accuracy_score(ys, loso_predict(X, ys, g)))
        except ValueError:
            continue  # a shuffle that leaves a training fold with one class: skip it
    null_a = np.asarray(null)
    row |= {"bal_acc": obs, "null_95th": float(np.quantile(null_a, 0.95)) if len(null_a) else np.nan,
            "p_perm": (1 + int((null_a >= obs).sum())) / (1 + len(null_a)), "note": "; ".join(notes)}
    row["_confusion"] = pd.DataFrame(confusion_matrix(y, pred, labels=present), index=present, columns=present)
    return row


# ---------------------------------------------------------------------------
# Per-feature separation tests
# ---------------------------------------------------------------------------


def holm(p: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values (NaN stays NaN)."""
    ok = p.dropna().sort_values()
    m = len(ok)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * ok.to_numpy()))
    return pd.Series(adj, index=ok.index).reindex(p.index)


def separation_row(df: pd.DataFrame, f: str, milk: list[str], dropin: list[str]) -> dict[str, Any]:
    r: dict[str, Any] = {"feature": f, "source": "measured" if f == MANIPULATION else "photo"}
    groups = [x for x in (df.loc[df["milk_condition"] == c, f].dropna().to_numpy() for c in milk) if len(x)]
    n = sum(len(x) for x in groups)
    if len(groups) >= 2 and n > len(groups) and np.ptp(np.concatenate(groups)) > 0:
        h, p = kruskal(*groups)
        r |= {"milk_H": h, "milk_p": p, "milk_eps2": h / ((n**2 - 1) / (n + 1))}
    else:
        r |= {"milk_H": np.nan, "milk_p": np.nan, "milk_eps2": np.nan}
    a = df.loc[df["dropin"] == dropin[0], f].dropna().to_numpy()
    b = df.loc[df["dropin"] == dropin[1], f].dropna().to_numpy()
    if len(a) and len(b) and np.ptp(np.concatenate([a, b])) > 0:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        r |= {"dropin_p": p, "dropin_r": 2 * u / (len(a) * len(b)) - 1}
    else:
        r |= {"dropin_p": np.nan, "dropin_r": np.nan}
    return r


def separation(df: pd.DataFrame, milk: list[str], dropin: list[str]) -> pd.DataFrame:
    """One row per image feature (Holm-adjusted as a family), then the foam-gain manipulation check (unadjusted)."""
    img = pd.DataFrame([separation_row(df, f, milk, dropin) for f in MODEL_FEATURES])
    img["milk_p_holm"] = holm(img["milk_p"])
    img["dropin_p_holm"] = holm(img["dropin_p"])
    man = pd.DataFrame([separation_row(df, MANIPULATION, milk, dropin)])
    cols = ["feature", "source", "milk_H", "milk_p", "milk_p_holm", "milk_eps2", "dropin_p", "dropin_p_holm", "dropin_r"]
    return pd.concat([img, man], ignore_index=True)[cols]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7, length=2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def box_grid(df: pd.DataFrame, by: str, levels: list[str], sep: pd.DataFrame, p_col: str, title: str, path) -> None:
    feats = [*MODEL_FEATURES, MANIPULATION]
    cols = 4
    rows = -(-len(feats) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.85 * rows + 0.5), facecolor=SURFACE)
    rng = np.random.default_rng(0)
    sep = sep.set_index("feature")
    for ax, f in zip(axes.flat, feats):
        _style(ax)
        data = [df.loc[df[by] == lv, f].dropna().to_numpy() for lv in levels]
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data], widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.2}, whiskerprops={"color": INK2, "linewidth": 0.8},
                        capprops={"color": INK2, "linewidth": 0.8}, boxprops={"linewidth": 1.0})
        for patch, colr in zip(bp["boxes"], SERIES):
            patch.set_facecolor(colr + "33")
            patch.set_edgecolor(colr)
        for i, (d, colr) in enumerate(zip(data, SERIES), 1):
            ax.scatter(i + rng.uniform(-0.14, 0.14, len(d)), d, s=9, color=colr, alpha=0.85, linewidths=0, zorder=3)
        ax.set_xticks(range(1, len(levels) + 1), [f"{lv}\n(n={len(d)})" for lv, d in zip(levels, data)], color=INK2)
        if f == MANIPULATION:
            p = sep.at[f, p_col.replace("_holm", "")]
            sub = f"measured, not from photo; p = {p:.3g}" if np.isfinite(p) else "measured, not from photo"
        else:
            p = sep.at[f, p_col]
            sub = f"p = {p:.3g} (Holm)" if np.isfinite(p) else "p = n/a"
        ax.set_title(f"{f}\n{sub}", fontsize=7.5, color=INK, pad=3)
    for ax in list(axes.flat)[len(feats):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def overall_hist(scores: pd.Series, narrow_max: float, path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.3), facecolor=SURFACE)
    _style(ax)
    bins = np.arange(0.75, 5.5, 0.5)
    ax.hist(scores, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.axvline(narrow_max + 0.25, color=INK2, linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(1, 5.5, 0.5))
    ax.set_xlabel("mean Overall score (raters A and B)", fontsize=7.5, color=INK2)
    ax.set_ylabel("cups", fontsize=7.5, color=INK2)
    ax.set_title(f"n = {len(scores)}; {(scores <= narrow_max).mean():.0%} at or below {narrow_max:g}",
                 fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------


def run(cfg: dict[str, Any]) -> int:
    g = {**GONOGO_DEFAULTS, **(cfg.get("gonogo") or {})}
    milk, dropin = list(cfg["design"]["milk_conditions"]), list(cfg["design"]["dropin"])
    try:
        ds = load_dataset(cfg)
    except (FileNotFoundError, ValueError) as err:
        print(err, file=sys.stderr)
        return 1
    df = ds.table
    print(f"Go/no-go data: {ds.summary()}.")
    for reason, cups in ds.excluded.items():
        if cups:
            print(f"  excluded, {reason}: " + ", ".join(cups[:12]) + (" ..." if len(cups) > 12 else ""))
    if len(df) < int(g["min_cups"]):
        print(f"  WARNING: {len(df)} usable cups, fewer than the {g['min_cups']} the brief asks for. Treat every number below as provisional.")
    if "circle_checked" in df and (unchecked := int((df["circle_checked"] != "y").sum())):
        print(f"  NOTE: {unchecked} cup{'s' * (unchecked != 1)} use a circle nobody has checked yet (make circles), "
              "so their features are provisional.")
    if df.empty:
        print("Nothing to analyse yet.")
        return 0

    tables, figs = cfg_path(cfg, "tables_dir"), cfg_path(cfg, "figures_dir")
    figs.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # SimpleImputer: "Skipping features without observed values"
        res = [loso_task(df, "milk_condition", milk, int(g["n_permutations"]), seed),
               loso_task(df, "dropin", dropin, int(g["n_permutations"]), seed + 1)]
    confusions = {r["task"]: r.pop("_confusion", None) for r in res}
    clf = pd.DataFrame(res)[["task", "n", "sessions", "classes", "bal_acc", "chance", "null_95th", "p_perm", "note"]]
    write_table(clf, tables / "gonogo_classifier",
                "Go/no-go: leave-one-session-out balanced accuracy of logistic regression on handcrafted features",
                "tab:gonogo-clf", "{:.3f}")
    sep = separation(df, milk, dropin)
    write_table(sep, tables / "gonogo_features",
                f"Go/no-go: per-feature separation by milk condition (Kruskal-Wallis, epsilon squared) and drop-in "
                f"(Mann-Whitney, rank-biserial r > 0: {dropin[0]} higher). Holm over the photo features; "
                f"{MANIPULATION} is a measured manipulation check, unadjusted.",
                "tab:gonogo-feat", "{:.3g}")

    box_grid(df, "milk_condition", milk, sep, "milk_p_holm", "Features by planned milk condition", figs / "gonogo_by_milk.png")
    box_grid(df, "dropin", dropin, sep, "dropin_p_holm", "Features by planned drop-in timing", figs / "gonogo_by_dropin.png")
    scores = df.loc[df["labelled"], "overall"].astype(float)
    hist_path = figs / "gonogo_overall.png"
    if len(scores):
        overall_hist(scores, float(g["narrow_max_score"]), hist_path)
    elif hist_path.exists():
        hist_path.unlink()  # don't leave a stale figure from an earlier run

    print("\nTask B, leave-one-session-out logistic regression (balanced accuracy):")
    for r in clf.itertuples(index=False):
        if np.isfinite(r.bal_acc):
            print(f"  {r.task:15s} {r.bal_acc:.3f}  chance {r.chance:.3f}  null 95th pct {r.null_95th:.3f}  "
                  f"p = {r.p_perm:.3f}  (n = {r.n}, {r.sessions} sessions)" + (f"  [{r.note}]" if r.note else ""))
            cm = confusions[r.task]
            if cm is not None:
                print("    confusion (rows = planned, columns = predicted):")
                print("    " + cm.to_string().replace("\n", "\n    "))
        else:
            print(f"  {r.task:15s} not computed: {r.note}")
    alpha = float(g["alpha"])
    photo = sep[sep["source"] == "photo"]
    sig_milk = photo.loc[photo["milk_p_holm"] < alpha, "feature"].tolist()
    sig_drop = photo.loc[photo["dropin_p_holm"] < alpha, "feature"].tolist()
    print(f"\nPhoto features that separate milk conditions (Holm p < {alpha}): {', '.join(sig_milk) or 'none'}")
    print(f"Photo features that separate drop-in timing (Holm p < {alpha}): {', '.join(sig_drop) or 'none'}")

    print("\nDecision hints (section 16; the team decides):")
    b1 = clf.set_index("task").loc["milk_condition"]
    if np.isfinite(b1["bal_acc"]):
        above = b1["p_perm"] < alpha and b1["bal_acc"] > b1["chance"]
        status = (f"B1 milk condition: {b1['bal_acc']:.2f} vs chance {b1['chance']:.2f} "
                  f"(+{b1['bal_acc'] - b1['chance']:.2f}), permutation p = {b1['p_perm']:.3f}; "
                  f"{len(sig_milk)} photo feature{'s' * (len(sig_milk) != 1)} "
                  f"{'separates' if len(sig_milk) == 1 else 'separate'} the conditions.")
        print(f"  - {status}")
        if above and sig_milk:
            print("    -> Both conditions met: diagnosis signal looks present. Continue as planned.")
        elif above or sig_milk:
            print("    -> Mixed evidence: only one of the two conditions holds. Discuss it; keep collecting, and look again "
                  "once there is more data.")
        else:
            print("    -> No diagnosis signal yet. Keep collecting (Task A still needs the data), focus on Task A and the "
                  "contribution-based fixes, and report the diagnosis result honestly as a negative finding. "
                  "Do not switch projects.")
    else:
        print(f"  - B1 milk condition: not computed ({b1['note']}).")
    if len(scores):
        frac = float((scores <= float(g["narrow_max_score"])).mean())
        narrow = frac >= float(g["narrow_frac"])
        print(f"  - Overall scores: {len(scores)} doubly-labelled cups, {frac:.0%} at or below {g['narrow_max_score']:g} -> "
              + ("NARROW quality range: pour more good/on_time cups, or have a skilled friend pour 15-20 cups on the same "
                 "rig (written permission; log them with a distinct pourer ID)."
                 if narrow else "range looks wide enough."))
    else:
        print("  - Overall scores: no cups labelled by both raters yet (make label, both raters).")
    man = sep[sep["source"] == "measured"].iloc[0]
    if df[MANIPULATION].notna().any():
        med = df.groupby("milk_condition")[MANIPULATION].median().reindex(milk)
        p_txt = f", Kruskal-Wallis p = {man['milk_p']:.3g}" if np.isfinite(man["milk_p"]) else ""
        print("  - Manipulation check, median measured foam gain %: "
              + ", ".join(f"{k} {v:.0f}" for k, v in med.items() if np.isfinite(v)) + p_txt)
    root = cfg["_root"]
    rel = lambda p: p.relative_to(root) if p.is_relative_to(root) else p  # noqa: E731
    n_figs = 2 + bool(len(scores))
    print(f"\nWrote {rel(tables / 'gonogo_classifier.csv')}, {rel(tables / 'gonogo_features.csv')} (+ .tex) and "
          f"{n_figs} figures in {rel(figs)}/.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args(argv)
    return run(load_config(args.config))


if __name__ == "__main__":
    sys.exit(main())
