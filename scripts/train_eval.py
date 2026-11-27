"""Train and evaluate experiments E0-E5 with leave-one-session-out CV (section 10).

    make train                           # all experiments
    python scripts/train_eval.py --exp E2,E3
    make leakcheck                       # the same pipelines on labels shuffled within sessions

Data comes from ``latte.data.load_dataset`` (features + cup log + mean labels,
with the exclusions in section 10.1) and ``latte.embed.load_embeddings``. If
there are no complete, up-to-date embeddings, the embedding models are skipped
and the reason is printed.

Protocol (``latte.evaluate.nested_cv``): the outer loop is LeaveOneGroupOut by
session_id, so no session is in both train and test. The inner loop is
GridSearchCV over GroupKFold(cv.inner_folds) on the training sessions only.
Imputation, scaling, PCA and the model are all inside one Pipeline. Results
are reported as pooled out-of-fold metrics, plus mean +/- std over outer folds.
Confidence intervals resample whole *sessions* (cluster bootstrap), because
cups in one session are not independent.

Experiments:

    E0  inter-rater agreement on the Task A cups (the human ceiling)
    E1  Spearman rho of each handcrafted feature with its linked rubric dimension, with a 95%
        session-bootstrap CI (coverage is checked against drop-in, late = 1; n_components has
        no rubric dimension)
    E2  Task A: mean Overall score. MAE, RMSE, Spearman, R^2 against the mean label, plus MAE
        against each rater separately (mae_vs_A, mae_vs_B). The human row puts the rater-A vs
        rater-B MAE in those same mae_vs_ columns, so the comparison is like with like
        (section 10.2): model vs one rater against rater vs rater.
    E3  Task B: B1 milk_condition, B2 dropin (balanced accuracy with a session-bootstrap CI,
        macro-F1, summed confusion matrices), and B3 measured foam_gain_pct (regression)
    E4  permutation importance of the best handcrafted model per task (mean over outer folds), and a
        drop-one-feature-group ablation
    E5  pourer generalisation (train on one pourer, test on the other), only if each pourer has at
        least ``train.e5_min_per_pourer`` cups

Outputs:

    results/runs/<timestamp>_<exp>/      config.yaml, run.json (data and embedding model), metrics.json,
                                         out-of-fold predictions (CSV), confusion matrices
    results/runs/latest.json             for each experiment, the most recent run folder (relative to
                                         results/runs); updated after each experiment, read by fit_final
    results/tables/T3_task_a.*, T4_task_b.*, T4b_foam_gain.*, T5_ablation.*,
    E0_agreement.*, E1_feature_validity.*, E4_importance.*, E5_pourer.*   (CSV + LaTeX)
    results/tables/leakage_check.* and results/runs/<timestamp>_leakcheck/   (make leakcheck only)

A table whose experiment is skipped for lack of data is deleted, so an old
result can't pass for a new one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance

from latte.data import load_dataset
from latte.embed import load_embeddings
from latte.evaluate import (
    CVResult,
    _default_params,
    agreement_table,
    classification_metrics,
    cluster_bootstrap_ci,
    latest_run_dir,  # noqa: F401  (re-exported for callers of this script)
    nested_cv,
    regression_metrics,
    spearman,
    summarise,
    summed_confusion,
)
from latte.features import DESIGN_LINK, FEATURE_GROUPS, MODEL_FEATURES, RUBRIC_LINK
from latte.io import DEFAULT_CONFIG, cfg_path, load_config, write_table
from latte.labels import labels_path, latest_labels, read_labels
from latte.models import FEATURE_SETS, classification_specs, foam_specs, regression_specs

TRAIN_DEFAULTS: dict[str, Any] = {
    "min_cups": 10,
    "min_sessions": 2,
    "e5_min_per_pourer": 40,
    "importance_repeats": 10,
    "leakage_repeats": 10,
    "n_jobs": -1,
}
EXPERIMENTS = ["E0", "E1", "E2", "E3", "E4", "E5"]
TABLES = {"E0": ["E0_agreement"], "E1": ["E1_feature_validity"], "E2": ["T3_task_a"],
          "E3": ["T4_task_b", "T4b_foam_gain"], "E4": ["E4_importance", "T5_ablation"], "E5": ["E5_pourer"]}

os.environ.setdefault("PYTHONWARNINGS", "ignore")  # quiet joblib workers (convergence chatter on tiny folds)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class Study:
    """The analysis table, the embeddings and everything the experiments share."""

    def __init__(self, cfg: dict[str, Any], shuffle_seed: int | None = None):
        self.cfg = cfg
        self.seed = int(cfg["seed"])
        self.t = {**TRAIN_DEFAULTS, **(cfg.get("train") or {})}
        self.inner = int(cfg["cv"]["inner_folds"])
        self.grids = cfg["grids"]
        self.dims: list[str] = cfg["labelling"]["dimensions"]
        self.nboot = int(cfg["labelling"]["bootstrap_n"])
        self.milk = list(cfg["design"]["milk_conditions"])
        self.dropin = list(cfg["design"]["dropin"])
        ds = load_dataset(cfg)
        self.excluded = ds.excluded
        self.data = ds.table
        self.emb_note = ""
        self.emb_cols: list[str] = []
        self.emb_info: dict[str, Any] = {"model": None, "fell_back": None, "pretrained": None}
        self.emb_digest: str | None = None
        try:
            emb = load_embeddings(cfg)
            missing = sorted(set(self.data.index) - set(emb.index))
            if missing:
                raise FileNotFoundError(f"no embedding for {len(missing)} analysed cups, e.g. {', '.join(missing[:3])}")
            self.emb_cols = list(emb.columns)
            self.emb_info = {k: emb.attrs.get(k) for k in ("model", "fell_back", "pretrained")}
            self.emb_digest = (emb.attrs.get("meta") or {}).get("digest")
            if emb.attrs.get("fell_back"):
                self.emb_note = f"embeddings come from the fallback model {emb.attrs.get('model')}"
            if not emb.attrs.get("pretrained", True):
                self.emb_note = "embeddings use RANDOM weights (tests only)"
            self.data = self.data.join(emb, how="left")
        except FileNotFoundError as err:
            self.emb_note = f"embedding models skipped: {err}"
        self.feature_sets = FEATURE_SETS if self.emb_cols else ("hand",)
        self.raters = {r: latest_labels(read_labels(labels_path(cfg, r), self.dims))
                       for r in cfg["labelling"]["raters"]}
        self.fingerprint = self._fingerprint()
        self.shuffled = shuffle_seed is not None
        if self.shuffled:
            rng = np.random.default_rng(shuffle_seed)
            for col in ("overall", "milk_condition", "dropin", "foam_gain_pct"):
                self.data[col] = self._shuffle_within_sessions(self.data[col], rng)

    def _fingerprint(self) -> str:
        """Hash of everything the models are fitted on: cups, features, labels, design factors, embeddings."""
        import hashlib

        cols = ["session_id", "pourer", "milk_condition", "dropin", "foam_gain_pct", *MODEL_FEATURES, *self.dims]
        h = hashlib.sha256(self.data[cols].round(6).to_csv().encode())
        h.update(json.dumps(self.emb_info, sort_keys=True, default=str).encode())
        h.update(str(self.emb_digest).encode())
        return h.hexdigest()[:16]

    def _shuffle_within_sessions(self, s: pd.Series, rng: np.random.Generator) -> pd.Series:
        out = s.copy()
        for _, idx in self.data.groupby("session_id").groups.items():
            ok = [i for i in idx if pd.notna(s.loc[i])]
            out.loc[ok] = s.loc[ok].to_numpy()[rng.permutation(len(ok))]
        return out

    def X(self, rows: pd.Index) -> pd.DataFrame:
        return self.data.loc[rows, MODEL_FEATURES + self.emb_cols]

    def sessions(self, rows: pd.Index) -> pd.Series:
        return self.data.loc[rows, "session_id"]

    def enough(self, rows: pd.Index) -> str:
        n, s = len(rows), self.data.loc[rows, "session_id"].nunique()
        if n < int(self.t["min_cups"]) or s < int(self.t["min_sessions"]):
            return f"needs at least {self.t['min_cups']} cups in {self.t['min_sessions']} sessions (have {n} in {s})"
        return ""

    def task_a_rows(self) -> pd.Index:
        return self.data.index[self.data["labelled"] & self.data["overall"].notna()]

    def emb_caption(self) -> str:
        if not self.emb_cols:
            return " Embedding models skipped (no up-to-date embeddings)."
        extra = " (fallback)" if self.emb_info["fell_back"] else ""
        extra += " (RANDOM weights: test only)" if self.emb_info["pretrained"] is False else ""
        return f" Embeddings: {self.emb_info['model']}{extra}."


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def e0_agreement(st: Study) -> dict[str, Any]:
    rows = st.task_a_rows()
    a, b = (st.raters[r] for r in st.cfg["labelling"]["raters"][:2])
    common = [c for c in rows if c in a.index and c in b.index]
    if len(common) < 2:
        return {"note": "fewer than 2 cups labelled by both raters"}
    return {"table": agreement_table(a.loc[common], b.loc[common], st.dims, st.nboot, st.seed)}


def e1_validity(st: Study) -> dict[str, Any]:
    rows = st.task_a_rows()
    out = []

    def rho(df: pd.DataFrame) -> float:
        return spearman(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())

    for f, dim in RUBRIC_LINK.items():
        d = st.data.loc[rows, [f, dim]].dropna()
        lo, hi = cluster_bootstrap_ci(d, st.sessions(d.index), rho, st.nboot, st.seed)
        out.append({"feature": f, "linked_to": dim, "n": len(d), "sessions": st.sessions(d.index).nunique(),
                    "spearman": rho(d) if len(d) else np.nan, "ci_lo": lo, "ci_hi": hi})
    for f, factor in DESIGN_LINK.items():
        d = st.data[[f, factor]].dropna()
        d = pd.DataFrame({f: d[f], factor: (d[factor] == st.dropin[-1]).astype(float)})  # late = 1
        lo, hi = cluster_bootstrap_ci(d, st.sessions(d.index), rho, st.nboot, st.seed)
        out.append({"feature": f, "linked_to": f"{factor} ({st.dropin[-1]} = 1)", "n": len(d),
                     "sessions": st.sessions(d.index).nunique(), "spearman": rho(d) if len(d) else np.nan,
                     "ci_lo": lo, "ci_hi": hi})
    out.append({"feature": "n_components", "linked_to": "broken pattern (no rubric dimension)", "n": 0,
                "sessions": 0, "spearman": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
    return {"table": pd.DataFrame(out)}


def run_models(st: Study, specs: list[Any], rows: pd.Index, y: pd.Series, classes: list[str] | None = None,
               keep: tuple[str, ...] = (), quiet: bool = False) -> dict[str, CVResult]:
    X, g = st.X(rows), st.sessions(rows)
    res = {}
    for spec in specs:
        if spec.uses_emb and not st.emb_cols:
            continue
        res[spec.name] = nested_cv(spec, X, y.loc[rows], g, st.inner, classes, int(st.t["n_jobs"]), spec.name in keep)
        if not quiet:
            print(f"    {spec.name:22s} done", flush=True)
    return res


def e2_task_a(st: Study) -> dict[str, Any]:
    rows = st.task_a_rows()
    if note := st.enough(rows):
        return {"note": f"Task A {note}"}
    y = st.data["overall"].astype(float)
    specs = regression_specs(MODEL_FEATURES, st.emb_cols, st.grids, st.seed, st.feature_sets)
    res = run_models(st, specs, rows, y, keep=("ridge_hand", "rf_hand"))
    ra, rb = st.cfg["labelling"]["raters"][:2]
    per_rater = {r: st.raters[r]["overall"].astype(float) for r in (ra, rb)}
    table = []
    for name, r in res.items():
        s = summarise(r, y)
        for rr, lab in per_rater.items():
            m = regression_metrics(lab.loc[r.oof.index], r.oof)
            s[f"mae_vs_{rr}"], s[f"spearman_vs_{rr}"] = m["mae"], m["spearman"]
        table.append(s)
    human = regression_metrics(per_rater[ra].loc[rows], per_rater[rb].loc[rows])
    # Like for like: rater vs rater goes only in the "vs one rater" columns, never in the
    # columns measured against the mean label (mae, spearman, ...).
    table.append({"model": f"human: rater {ra} vs rater {rb}", "n": len(rows),
                  f"mae_vs_{ra}": human["mae"], f"mae_vs_{rb}": human["mae"],
                  f"spearman_vs_{ra}": human["spearman"], f"spearman_vs_{rb}": human["spearman"]})
    return {"results": res, "table": pd.DataFrame(table), "y": y.loc[rows]}


def _balacc_ci(st: Study, res: CVResult, y: pd.Series, classes: list[str]) -> tuple[float, float]:
    df = pd.DataFrame({"y": y.loc[res.oof.index], "p": res.oof})
    return cluster_bootstrap_ci(df, st.sessions(df.index),
                                lambda d: classification_metrics(d["y"], d["p"], classes)["bal_acc"], st.nboot, st.seed)


def e3_task_b(st: Study) -> dict[str, Any]:
    rows = st.data.index
    out: dict[str, Any] = {"targets": {}}
    for target, classes in (("milk_condition", st.milk), ("dropin", st.dropin)):
        r_ok = rows[st.data.loc[rows, target].isin(classes)]
        if note := st.enough(r_ok):
            out["targets"][target] = {"note": note}
            continue
        y = st.data[target]
        present = [c for c in classes if (y.loc[r_ok] == c).any()]
        specs = classification_specs(MODEL_FEATURES, st.emb_cols, st.grids, st.seed, st.feature_sets)
        res = run_models(st, specs, r_ok, y, present, keep=("logreg_hand", "rf_hand"))
        table = []
        for r in res.values():
            lo, hi = _balacc_ci(st, r, y, present)
            table.append({"target": target, "chance": 1 / len(present), **summarise(r, y, present),
                          "bal_acc_ci_lo": lo, "bal_acc_ci_hi": hi})
        t = pd.DataFrame(table)
        # True/False when the CI exists; empty (unknown) with fewer than 3 sessions.
        t["above_chance"] = [None if not np.isfinite(lo) else bool(lo > ch) for lo, ch in zip(t["bal_acc_ci_lo"], t["chance"])]
        absent = [c for c in classes if c not in present]
        out["targets"][target] = {"results": res, "table": t, "classes": present, "absent": absent,
                                  "confusion": {n: summed_confusion(r, y, present) for n, r in res.items()},
                                  "y": y.loc[r_ok]}
    foam_rows = rows[st.data.loc[rows, "foam_gain_pct"].notna()]
    if note := st.enough(foam_rows):
        out["foam"] = {"note": note}
    else:
        y = st.data["foam_gain_pct"].astype(float)
        res = run_models(st, foam_specs(MODEL_FEATURES, st.emb_cols, st.grids, st.seed, st.feature_sets), foam_rows, y)
        out["foam"] = {"results": res, "table": pd.DataFrame([summarise(r, y) for r in res.values()]), "y": y.loc[foam_rows]}
    return out


def _best(table: pd.DataFrame, names: list[str], metric: str, lower_is_better: bool) -> str | None:
    t = table[table["model"].isin(names)].dropna(subset=[metric])
    if t.empty:
        return None
    return str(t.sort_values(metric, ascending=lower_is_better)["model"].iloc[0])


def _importance(st: Study, res: CVResult, y: pd.Series, scoring: str) -> pd.DataFrame:
    rows = []
    X = st.X(res.oof.index)
    for k, (est, test) in enumerate(res.fitted):
        Xt, yt = X.iloc[test], y.loc[X.index[test]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pi = permutation_importance(est, Xt[MODEL_FEATURES], yt.to_numpy(), scoring=scoring,
                                        n_repeats=int(st.t["importance_repeats"]), random_state=st.seed + k)
        rows += [{"fold": k, "feature": c, "importance": pi.importances_mean[i]} for i, c in enumerate(MODEL_FEATURES)]
    df = pd.DataFrame(rows)
    return (df.groupby("feature")["importance"].agg(["mean", "std"]).reset_index()
            .rename(columns={"mean": "importance_mean", "std": "importance_std"})
            .sort_values("importance_mean", ascending=False))


def e4_importance_ablation(st: Study, e2: dict[str, Any], e3: dict[str, Any]) -> dict[str, Any]:
    imp, abl, jobs = [], [], []
    if "table" in e2:
        best = _best(e2["table"], ["ridge_hand", "rf_hand"], "mae", True)
        if best:
            jobs.append(("task_a", best, e2["results"][best], e2["y"], "neg_mean_absolute_error", "mae", None))
    for target, d in e3.get("targets", {}).items():
        if "table" in d:
            best = _best(d["table"], ["logreg_hand", "rf_hand"], "bal_acc", False)
            if best:
                jobs.append((target, best, d["results"][best], d["y"], "balanced_accuracy", "bal_acc", d["classes"]))
    for task, best, res, y, scoring, metric, classes in jobs:
        t = _importance(st, res, y, scoring)
        t.insert(0, "model", best)
        t.insert(0, "task", task)
        imp.append(t)
        full = res.pooled(y, classes)[metric]
        for group, feats in FEATURE_GROUPS.items():
            keep = [f for f in MODEL_FEATURES if f not in feats]
            maker = classification_specs if classes else regression_specs
            spec = next(s for s in maker(keep, [], st.grids, st.seed, ("hand",)) if s.name == best)
            r = nested_cv(spec, st.data.loc[y.index, keep], y, st.sessions(y.index), st.inner, classes, int(st.t["n_jobs"]))
            ab = r.pooled(y, classes)[metric]
            abl.append({"task": task, "model": best, "dropped_group": group, "metric": metric,
                        "full": full, "ablated": ab, "delta": ab - full})
        print(f"    E4 {task}: importance and ablation done ({best})", flush=True)
    if not jobs:
        return {"note": "needs E2 and/or E3 results (not enough data)", "importance": pd.DataFrame(), "ablation": pd.DataFrame()}
    return {"importance": pd.concat(imp, ignore_index=True), "ablation": pd.DataFrame(abl)}


def e5_pourer(st: Study) -> dict[str, Any]:
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, GroupKFold

    need = int(st.t["e5_min_per_pourer"])
    pourers = [p for p in st.data["pourer"].unique() if p]
    out = []
    tasks = [("task_a", st.task_a_rows(), st.data["overall"].astype(float), None),
             ("milk_condition", st.data.index[st.data["milk_condition"].isin(st.milk)], st.data["milk_condition"], st.milk),
             ("dropin", st.data.index[st.data["dropin"].isin(st.dropin)], st.data["dropin"], st.dropin)]
    for task, rows, y, classes in tasks:
        counts = st.data.loc[rows, "pourer"].value_counts()
        main = [p for p in pourers if counts.get(p, 0) >= need]
        if len(main) < 2:
            out.append({"task": task, "note": f"needs >= {need} cups for two pourers (have {counts.to_dict()})"})
            continue
        names = (["logreg_hand"] + (["logreg_emb"] if st.emb_cols else [])) if classes else \
            (["ridge_hand"] + (["ridge_emb"] if st.emb_cols else []))
        maker = classification_specs if classes else regression_specs
        for spec in (s for s in maker(MODEL_FEATURES, st.emb_cols, st.grids, st.seed, st.feature_sets) if s.name in names):
            for train_p in main[:2]:
                test_p = next(p for p in main[:2] if p != train_p)
                tr = rows[st.data.loc[rows, "pourer"] == train_p]
                te = rows[st.data.loc[rows, "pourer"] == test_p]
                g = st.sessions(tr)
                est = clone(spec.pipeline)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if g.nunique() >= 2:
                        gs = GridSearchCV(est, spec.grid, cv=GroupKFold(min(st.inner, g.nunique())), error_score=np.nan,
                                          scoring="balanced_accuracy" if classes else "neg_mean_absolute_error",
                                          n_jobs=int(st.t["n_jobs"]))
                        est = gs.fit(st.X(tr), y.loc[tr].to_numpy(), groups=g.to_numpy()).best_estimator_
                    else:
                        est.set_params(**_default_params(spec.grid)).fit(st.X(tr), y.loc[tr].to_numpy())
                pred = est.predict(st.X(te))
                if classes:
                    m = classification_metrics(y.loc[te], pred, [c for c in classes if (y.loc[te] == c).any()])
                else:
                    m = regression_metrics(y.loc[te], pred)
                out.append({"task": task, "model": spec.name, "train_pourer": train_p, "test_pourer": test_p,
                            "n_train": len(tr), "n_test": len(te), **m})
    return {"table": pd.DataFrame(out)}


# ---------------------------------------------------------------------------
# Records and tables
# ---------------------------------------------------------------------------


def _jsonable(o: Any) -> Any:
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return _jsonable(o.to_dict())
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_json(path: Path, obj: Any) -> None:
    """Strict JSON (NaN -> null), written atomically."""
    text = json.dumps(_jsonable(obj), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


class Recorder:
    """Run folders and the merged results/runs/latest.json pointer."""

    def __init__(self, st: Study, stamp: str):
        self.st, self.stamp = st, stamp
        self.runs = cfg_path(st.cfg, "runs_dir")
        self.tables = cfg_path(st.cfg, "tables_dir")

    def run_dir(self, exp: str) -> Path:
        d = self.runs / f"{self.stamp}_{exp}"
        d.mkdir(parents=True, exist_ok=True)
        clean = {k: v for k, v in self.st.cfg.items() if not k.startswith("_")}
        (d / "config.yaml").write_text(yaml.safe_dump(clean, sort_keys=False))
        st = self.st
        write_json(d / "run.json", {
            "experiment": exp, "timestamp": self.stamp, "shuffled_labels": st.shuffled, "fingerprint": st.fingerprint,
            "n_cups": len(st.data), "n_sessions": int(st.data["session_id"].nunique()),
            "n_labelled": int(st.data["labelled"].sum()), "embedding": st.emb_info, "embedding_note": st.emb_note,
            "excluded": {k: len(v) for k, v in st.excluded.items()},
        })
        return d

    def done(self, exp: str, d: Path) -> None:
        """Point latest.json's entry for this experiment at d (merged with the others, atomic)."""
        path = self.runs / "latest.json"
        try:
            rec = json.loads(path.read_text()) if path.exists() else {}
        except json.JSONDecodeError:
            rec = {}
        rec.setdefault("runs", {})[exp] = {"dir": d.name, "timestamp": self.stamp, "embedding": self.st.emb_info}
        write_json(path, rec)

    def drop_tables(self, exp: str) -> None:
        for stem in TABLES[exp]:
            for ext in (".csv", ".tex"):
                (self.tables / f"{stem}{ext}").unlink(missing_ok=True)


def save_oof(d: Path, name: str, results: dict[str, CVResult], y: pd.Series) -> None:
    df = pd.DataFrame({"y": y})
    for m, r in results.items():
        df[m] = r.oof
        if r.proba is not None:
            for c in r.proba.columns:
                df[f"{m}__p_{c}"] = r.proba[c]
    df.index.name = "cup_id"
    df.to_csv(d / f"oof_{name}.csv")


def fmt_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--exp", default=",".join(EXPERIMENTS), help="comma-separated subset of E0,E1,E2,E3,E4,E5")
    ap.add_argument("--shuffle-labels", action="store_true", help="leakage check: shuffle targets within sessions")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    exps = [e.strip().upper() for e in args.exp.split(",") if e.strip()]
    bad = [e for e in exps if e not in EXPERIMENTS]
    if bad:
        ap.error(f"unknown experiment(s) {bad}; choose from {EXPERIMENTS}")
    try:
        if args.shuffle_labels:
            return leakage_check(cfg)
        return run_all(cfg, exps)
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


def run_all(cfg: dict[str, Any], exps: list[str]) -> int:
    st = Study(cfg)
    rec = Recorder(st, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tables = rec.tables
    print(f"Data: {len(st.data)} usable cups in {st.data['session_id'].nunique()} sessions; "
          f"{int(st.data['labelled'].sum())} labelled by both raters.")
    for reason, cups in st.excluded.items():
        if cups:
            print(f"  excluded, {reason}: {len(cups)}")
    if st.emb_note:
        print(f"  NOTE: {st.emb_note}")
    if "circle_checked" in st.data and (u := int((st.data["circle_checked"] != "y").sum())):
        print(f"  NOTE: {u} cups use an unchecked circle (make circles); their features are provisional.")
    emb_cap = st.emb_caption()
    e2: dict[str, Any] = {}
    e3: dict[str, Any] = {}

    if "E0" in exps:
        r = e0_agreement(st)
        d = rec.run_dir("E0")
        if "table" in r:
            write_table(r["table"], tables / "E0_agreement", "E0: inter-rater agreement on the Task A cups", "tab:e0", "{:.2f}")
        else:
            rec.drop_tables("E0")
        write_json(d / "metrics.json", r)
        rec.done("E0", d)
        print(f"E0: {'written' if 'table' in r else r['note']}")
    if "E1" in exps:
        r = e1_validity(st)
        d = rec.run_dir("E1")
        write_table(r["table"], tables / "E1_feature_validity",
                    "E1: Spearman rho between each feature and its linked rubric dimension "
                    "(95% CI from a bootstrap over sessions)", "tab:e1", "{:.2f}")
        write_json(d / "metrics.json", r)
        rec.done("E1", d)
        print("E1: feature validity written")
    if "E2" in exps or "E4" in exps:
        print("E2: Task A (mean Overall score) ...", flush=True)
        e2 = e2_task_a(st)
    if "E2" in exps:
        d = rec.run_dir("E2")
        if "table" in e2:
            ra, rb = cfg["labelling"]["raters"][:2]
            cols = ["model", "n", "mae", "mae_fold_mean", "mae_fold_std", f"mae_vs_{ra}", f"mae_vs_{rb}",
                    "rmse", "rmse_fold_std", "spearman", "spearman_fold_std", f"spearman_vs_{ra}", f"spearman_vs_{rb}",
                    "r2", "r2_fold_std"]
            save_oof(d, "task_a", e2["results"], e2["y"])
            write_json(d / "metrics.json", {"table": e2["table"], "folds": {n: r.folds for n, r in e2["results"].items()}})
            rec.done("E2", d)  # pointer first, then tables: a table is never newer than its run record
            write_table(fmt_table(e2["table"], cols), tables / "T3_task_a",
                        "Task A: leave-one-session-out prediction of the mean Overall score. mae, rmse, spearman and r2 "
                        f"are against the mean label; the _vs_{ra}/_vs_{rb} columns are against one rater and are the "
                        "ones comparable with the human row (rater vs rater)." + emb_cap, "tab:t3", "{:.3f}")
            print(fmt_table(e2["table"], ["model", "n", "mae", "mae_fold_std", f"mae_vs_{ra}", f"mae_vs_{rb}", "spearman", "r2"])
                  .to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="--"))
        else:
            write_json(d / "metrics.json", {"note": e2["note"]})
            rec.done("E2", d)
            rec.drop_tables("E2")
            print(f"  skipped: {e2['note']}")
    if "E3" in exps or "E4" in exps:
        print("E3: Task B (milk condition, drop-in, foam gain) ...", flush=True)
        e3 = e3_task_b(st)
    if "E3" in exps:
        d = rec.run_dir("E3")
        t_all, metrics = [], {}
        for target, r in e3["targets"].items():
            if "table" in r:
                t_all.append(r["table"])
                save_oof(d, target, r["results"], r["y"])
                for n, cm in r["confusion"].items():
                    cm.to_csv(d / f"confusion_{target}_{n}.csv")
                metrics[target] = {"table": r["table"], "classes": r["classes"], "absent_classes": r["absent"],
                                   "folds": {n: x.folds for n, x in r["results"].items()}}
            else:
                metrics[target] = {"note": r["note"]}
                print(f"  {target}: skipped, {r['note']}")
        if "table" in e3["foam"]:
            save_oof(d, "foam_gain_pct", e3["foam"]["results"], e3["foam"]["y"])
            metrics["foam_gain_pct"] = {"table": e3["foam"]["table"],
                                        "folds": {n: x.folds for n, x in e3["foam"]["results"].items()}}
        else:
            metrics["foam_gain_pct"] = {"note": e3["foam"]["note"]}
        write_json(d / "metrics.json", metrics)
        rec.done("E3", d)
        if t_all:
            t = pd.concat(t_all, ignore_index=True)
            write_table(fmt_table(t, ["target", "model", "n", "chance", "bal_acc", "bal_acc_ci_lo", "bal_acc_ci_hi",
                                      "bal_acc_fold_std", "macro_f1", "macro_f1_fold_std", "above_chance"]),
                        tables / "T4_task_b", "Task B: leave-one-session-out diagnosis of milk condition and drop-in "
                        "(95% CI from a bootstrap over sessions; above_chance = CI lower bound > chance, empty if the CI "
                        "needs more sessions)." + emb_cap, "tab:t4", "{:.3f}")
            print(fmt_table(t, ["target", "model", "n", "chance", "bal_acc", "bal_acc_ci_lo", "bal_acc_ci_hi", "macro_f1"])
                  .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        else:
            for ext in (".csv", ".tex"):
                (tables / f"T4_task_b{ext}").unlink(missing_ok=True)
        if "table" in e3["foam"]:
            write_table(fmt_table(e3["foam"]["table"], ["model", "n", "mae", "mae_fold_std", "rmse", "spearman", "r2"]),
                        tables / "T4b_foam_gain", "B3: leave-one-session-out regression of measured foam gain (%)." + emb_cap,
                        "tab:t4b", "{:.3f}")
        else:
            for ext in (".csv", ".tex"):
                (tables / f"T4b_foam_gain{ext}").unlink(missing_ok=True)
    if "E4" in exps:
        print("E4: permutation importance and ablation ...", flush=True)
        r = e4_importance_ablation(st, e2, e3)
        d = rec.run_dir("E4")
        write_json(d / "metrics.json", r)
        rec.done("E4", d)
        if not r["importance"].empty:
            write_table(r["importance"], tables / "E4_importance",
                        "E4: permutation importance of the best handcrafted model (mean over outer folds)", "tab:e4", "{:.3f}")
            write_table(r["ablation"], tables / "T5_ablation", "Drop-one-feature-group ablation (pooled out-of-fold metric)",
                        "tab:t5", "{:.3f}")
        else:
            rec.drop_tables("E4")
            print(f"  skipped: {r['note']}")
    if "E5" in exps:
        print("E5: pourer generalisation ...", flush=True)
        r = e5_pourer(st)
        d = rec.run_dir("E5")
        write_json(d / "metrics.json", r)
        rec.done("E5", d)
        write_table(r["table"], tables / "E5_pourer", "E5: train on one pourer, test on the other." + emb_cap, "tab:e5", "{:.3f}")
        for row in r["table"].to_dict(orient="records"):
            if isinstance(row.get("note"), str) and row["note"]:
                print(f"  {row['task']}: skipped, {row['note']}")
    rel = tables.relative_to(cfg["_root"]) if tables.is_relative_to(cfg["_root"]) else tables
    print(f"Done. Tables in {rel}/, run records in results/runs/{rec.stamp}_*/ (latest.json updated).")
    return 0


def leakage_check(cfg: dict[str, Any]) -> int:
    """Re-run E2/E3's main models on targets shuffled within each session; they should land at chance."""
    t = {**TRAIN_DEFAULTS, **(cfg.get("train") or {})}
    rows = []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    st = None
    for rep in range(int(t["leakage_repeats"])):
        st = Study(cfg, shuffle_seed=int(cfg["seed"]) + 1000 + rep)
        print(f"Shuffle {rep + 1}/{t['leakage_repeats']} ...", flush=True)
        ra = st.task_a_rows()
        if not st.enough(ra):
            y = st.data["overall"].astype(float)
            specs = [s for s in regression_specs(MODEL_FEATURES, st.emb_cols, st.grids, st.seed, st.feature_sets)
                     if s.estimator in ("ridge", "rf")]
            for n, r in run_models(st, specs, ra, y, quiet=True).items():
                s = summarise(r, y)
                rows.append({"rep": rep, "task": "task_a", "model": n, **r.pooled(y),
                             "spearman_within_session": s["spearman_fold_mean"]})
        for target, classes in (("milk_condition", st.milk), ("dropin", st.dropin)):
            rb = st.data.index[st.data[target].isin(classes)]
            if st.enough(rb):
                continue
            y = st.data[target]
            present = [c for c in classes if (y.loc[rb] == c).any()]
            specs = [s for s in classification_specs(MODEL_FEATURES, st.emb_cols, st.grids, st.seed, st.feature_sets)
                     if s.estimator == "logreg" or s.name == "rf_hand"]
            for n, r in run_models(st, specs, rb, y, present, quiet=True).items():
                rows.append({"rep": rep, "task": target, "model": n, "chance": 1 / len(present), **r.pooled(y, present)})
    if not rows:
        for ext in (".csv", ".tex"):
            (cfg_path(cfg, "tables_dir") / f"leakage_check{ext}").unlink(missing_ok=True)
        if st is not None:
            write_json(Recorder(st, stamp).run_dir("leakcheck") / "metrics.json", {"note": "not enough data", "passed": None})
        print("Not enough data for a leakage check yet (any earlier leakage_check table was removed).")
        return 0
    df = pd.DataFrame(rows)
    # Verdict per model: the mean over shuffles must not sit clearly above chance. "Clearly" =
    # more than max(floor, 2 standard errors of that mean), because one shuffle of ~100 cups
    # varies by about +/-0.08 on its own.
    g = df.groupby(["task", "model"])
    agg = g.mean(numeric_only=True).drop(columns="rep").reset_index()
    se = (g.std(ddof=1, numeric_only=True).drop(columns="rep") / np.sqrt(g.size().to_numpy()[:, None])).reset_index()
    verdict = []
    for r, e in zip(agg.to_dict(orient="records"), se.to_dict(orient="records")):
        def s(k: str) -> float:
            v = e.get(k)
            return 0.0 if v is None or not np.isfinite(v) else float(v)
        if r["task"] == "task_a":
            # Within-session rank correlation: the shuffle keeps each session's mean score, so a
            # pooled metric would also reward (legitimately) predicting which sessions scored higher.
            ok = r["spearman_within_session"] <= max(0.10, 2 * s("spearman_within_session"))
        else:
            ok = r["bal_acc"] - r["chance"] <= max(0.05, 2 * s("bal_acc"))
        verdict.append("PASS" if ok else "CHECK")
    agg["repeats"] = g.size().to_numpy()
    agg["verdict"] = verdict
    write_table(agg, cfg_path(cfg, "tables_dir") / "leakage_check",
                "Leakage check: models trained on labels shuffled within sessions (mean over repeats; Task A judged by "
                "the within-session Spearman rho)." + st.emb_caption(), "tab:leak", "{:.3f}")
    rec = Recorder(st, stamp)
    d = rec.run_dir("leakcheck")
    df.to_csv(d / "per_repeat.csv", index=False)
    write_json(d / "metrics.json", {"summary": agg, "passed": all(v == "PASS" for v in verdict)})
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="--"))
    if all(v == "PASS" for v in verdict):
        print("Leakage check PASSED: with shuffled labels every model is at chance level.")
        return 0
    print("Leakage check: some models are clearly above chance on shuffled labels. Investigate before trusting results.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
