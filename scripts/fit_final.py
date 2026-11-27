"""Fit the final demo models on all usable data -> models/ (section 11).

    make final        # after make train

These models are for the demo only. Every number in the report comes from
cross-validation (``make train``). The script reads the most recent E2 and E3
runs (``results/runs/latest.json``) and:

0. **Freshness.** E2 and E3 must have run on exactly the current data. The
   data fingerprint in their run.json (cups, features, labels, design factors,
   embeddings) must match a fresh one, otherwise it stops: "rerun make train".
   On any failure, an old bundle is deleted, so the demo can't serve stale models.
1. **Score model.** It takes ``feedback.score_model`` (default ``best``: the
   non-dummy E2 model with the lowest pooled out-of-fold MAE). X in
   "score +/- X" is the out-of-fold MAE of that model's *rounded* score, i.e.
   of the number the demo actually shows.
2. **Fixes model.** It takes ``ridge_hand``, the handcrafted Ridge whose
   weights give the per-feature contributions.
3. **Milk model.** It takes ``feedback.milk_model``, fixed in advance (default
   ``logreg_hand``). It is not "the best of six": choosing the winner and then
   testing it would pass pure noise too often. It is used only if E3 marked it
   ``above_chance`` (session-bootstrap CI lower bound > chance). Each predicted
   class gets its own threshold: the smallest t >= ``feedback.milk_prob_threshold``
   at which that class's out-of-fold predictions with probability >= t are at
   least ``feedback.milk_min_precision`` correct (``latte.feedback.tune_thresholds``).
   A class with no such t is never shown. If B1 isn't above chance, or no class
   qualifies, the milk diagnosis is withheld and the reason recorded.

Each model is refitted on all usable cups. Its hyperparameters are retuned with
GridSearchCV over GroupKFold(sessions), the same grid and protocol as in CV.
The bundle goes to models/latte_final.joblib (gitignored) and a readable
summary to models/final.json.
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

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval  # noqa: E402  (shares the data assembly with make train)

from latte.evaluate import _default_params, latest_run_dir  # noqa: E402
from latte.feedback import BUNDLE_NAME, feedback_params, round_score, score_error_text, tune_thresholds  # noqa: E402
from latte.features import MODEL_FEATURES  # noqa: E402
from latte.io import DEFAULT_CONFIG, cfg_path, load_config  # noqa: E402
from latte.models import classification_specs, regression_specs  # noqa: E402

DUMMIES = ("dummy_mean", "dummy_most_frequent", "dummy_stratified")


def read_metrics(cfg: dict[str, Any], exp: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    d = latest_run_dir(cfg, exp)
    if d is None or not (d / "metrics.json").exists():
        raise FileNotFoundError(f"no {exp} run found in results/runs/latest.json: run make train first")
    run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
    return d, json.loads((d / "metrics.json").read_text()), run


def refit(spec: Any, X: pd.DataFrame, y: np.ndarray, groups: pd.Series, inner: int, n_jobs: int, scoring: str) -> tuple[Any, dict]:
    """Retune on all data with GridSearchCV(GroupKFold over sessions), then refit; middle grid values if < 2 sessions."""
    est = clone(spec.pipeline)
    g = groups.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec.grid and len(np.unique(g)) >= 2:
            gs = GridSearchCV(est, spec.grid, cv=GroupKFold(min(inner, len(np.unique(g)))), scoring=scoring,
                              n_jobs=n_jobs, error_score=np.nan, refit=True).fit(X, y, groups=g)
            return gs.best_estimator_, dict(gs.best_params_)
        params = _default_params(spec.grid) if spec.grid else {}
        return est.set_params(**params).fit(X, y), params


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    try:
        return run(cfg)
    except BaseException as err:  # any failure, even Ctrl-C: never leave an out-of-date bundle for the demo
        out = cfg_path(cfg, "models_dir")
        removed = [f.name for f in (out / BUNDLE_NAME, out / "final.json") if f.exists()]
        for name in removed:
            (out / name).unlink()
        if removed:
            print(f"Removed the old {', '.join(removed)} so the demo can't use out-of-date models.", file=sys.stderr)
        if isinstance(err, (FileNotFoundError, ValueError)):
            print(f"ERROR: {err}", file=sys.stderr)
            return 1
        raise


def check_fresh(st: Any, run: dict[str, Any], exp: str) -> None:
    fp = run.get("fingerprint")
    if fp != st.fingerprint:
        then = f"{run.get('n_cups', '?')} cups / {run.get('n_sessions', '?')} sessions / {run.get('n_labelled', '?')} labelled"
        now = f"{len(st.data)} / {st.data['session_id'].nunique()} / {int(st.data['labelled'].sum())}"
        raise ValueError(f"the data changed since {exp} ran ({then}; now {now}, or features, labels or embeddings "
                         f"differ{'; ' + st.emb_note if st.emb_note else ''}): rerun make train")


def run(cfg: dict[str, Any]) -> int:
    fp = feedback_params(cfg)
    e2_dir, e2, e2_run = read_metrics(cfg, "E2")
    if "table" not in e2:
        raise ValueError(f"the latest E2 run has no results ({e2.get('note', 'skipped')}): collect and label more cups")
    st = train_eval.Study(cfg)
    check_fresh(st, e2_run, "E2")
    t2 = pd.DataFrame(e2["table"])
    t2 = t2[~t2["model"].isin(DUMMIES) & ~t2["model"].str.startswith("human") & t2["mae"].notna()]
    if fp["score_model"] == "best":
        if t2.empty:
            raise ValueError("no usable E2 model to fit")
        score_name = str(t2.sort_values("mae")["model"].iloc[0])
    else:
        score_name = str(fp["score_model"])
        if score_name not in set(t2["model"]):
            raise ValueError(f"feedback.score_model {score_name!r} is not among the E2 results {sorted(t2['model'])}")
    if "ridge_hand" not in set(t2["model"]):
        raise ValueError("E2 has no ridge_hand result, which the fixes need")
    oof = pd.read_csv(e2_dir / "oof_task_a.csv").set_index("cup_id")
    step = float(fp["score_round"])
    score_err = float(np.mean(np.abs(oof[score_name].map(lambda v: round_score(v, step)) - oof["y"])))
    score_mae = float(t2.set_index("model").at[score_name, "mae"])

    inner, n_jobs, seed = int(cfg["cv"]["inner_folds"]), int(st.t["n_jobs"]), st.seed
    rows_a = st.task_a_rows()
    y_a = st.data.loc[rows_a, "overall"].astype(float).to_numpy()
    g_a = st.sessions(rows_a)
    regs = {s.name: s for s in regression_specs(MODEL_FEATURES, st.emb_cols, st.grids, seed, st.feature_sets)}
    print(f"Fitting the score model {score_name} on {len(rows_a)} labelled cups ({g_a.nunique()} sessions) ...", flush=True)
    score_pipe, score_params = refit(regs[score_name], st.X(rows_a), y_a, g_a, inner, n_jobs, "neg_mean_absolute_error")
    if score_name == "ridge_hand":
        fix_pipe, fix_params = score_pipe, score_params
    else:
        print("Fitting ridge_hand for the fixes ...", flush=True)
        fix_pipe, fix_params = refit(regs["ridge_hand"], st.X(rows_a), y_a, g_a, inner, n_jobs, "neg_mean_absolute_error")

    milk, reason, milk_info = None, "", {}
    milk_name = str(fp["milk_model"])
    try:
        e3_dir, e3, e3_run = read_metrics(cfg, "E3")
        check_fresh(st, e3_run, "E3")
    except FileNotFoundError as err:
        e3, reason = {}, str(err)
    b1 = e3.get("milk_condition", {})
    if "table" not in b1:
        reason = reason or f"E3 has no milk-condition results ({b1.get('note', 'not run')})"
    else:
        t3 = pd.DataFrame(b1["table"]).set_index("model")
        if milk_name not in t3.index:
            raise ValueError(f"feedback.milk_model {milk_name!r} is not among the E3 results {sorted(t3.index)}")
        row = t3.loc[milk_name]
        raw = row.get("above_chance")  # True / False from JSON, or None/NaN when the CI was undefined
        above = None if raw is None or (isinstance(raw, float) and np.isnan(raw)) else bool(raw)
        milk_info = {"model": milk_name, "bal_acc": row["bal_acc"], "ci": [row.get("bal_acc_ci_lo"), row.get("bal_acc_ci_hi")],
                     "chance": row["chance"], "above_chance": above, "rule": "feedback.milk_model, fixed in advance"}
        if above is not True:
            reason = (f"B1 ({milk_name}, balanced accuracy {row['bal_acc']:.2f}) is not shown to be above chance "
                      + ("(the CI needs at least 3 sessions)" if above is None
                         else f"(CI lower bound {row['bal_acc_ci_lo']:.2f} <= chance {row['chance']:.2f})"))
        else:
            oofb = pd.read_csv(e3_dir / "oof_milk_condition.csv", dtype={"y": str}).set_index("cup_id")
            pcols = [c for c in oofb.columns if c.startswith(f"{milk_name}__p_")]
            proba = oofb[pcols].rename(columns=lambda c: c.split("__p_", 1)[1])
            th = tune_thresholds(oofb["y"], proba, float(fp["milk_prob_threshold"]), float(fp["milk_min_precision"]),
                                 int(fp["milk_min_predictions"]))
            milk_info |= {"threshold_search": th}
            if not th["any"]:
                reason = (f"no class reaches out-of-fold precision >= {fp['milk_min_precision']} at a threshold "
                          f">= {fp['milk_prob_threshold']} for {milk_name}")
            else:
                rows_b = st.data.index[st.data["milk_condition"].isin(st.milk)]
                clfs = {s.name: s for s in classification_specs(MODEL_FEATURES, st.emb_cols, st.grids, seed, st.feature_sets)}
                print(f"Fitting the milk model {milk_name} on {len(rows_b)} cups ...", flush=True)
                pipe, params = refit(clfs[milk_name], st.X(rows_b), st.data.loc[rows_b, "milk_condition"].to_numpy(),
                                     st.sessions(rows_b), inner, n_jobs, "balanced_accuracy")
                milk = {"name": milk_name, "pipeline": pipe, "params": params, "thresholds": th["thresholds"],
                        "per_class": th["per_class"], "classes": list(pipe.classes_)}

    emb_needed = any("emb" in n for n in [score_name] + ([milk["name"]] if milk else []))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bundle = {
        "created": stamp,
        "fingerprint": st.fingerprint,
        "score": {"name": score_name, "pipeline": score_pipe, "params": score_params, "error": score_err,
                  "oof_mae_unrounded": score_mae, "run": e2_dir.name},
        "fixes": {"name": "ridge_hand", "pipeline": fix_pipe, "params": fix_params},
        "milk": milk,
        "milk_withheld_reason": reason if milk is None else "",
        "milk_selection": milk_info,
        "features": MODEL_FEATURES,
        "emb_cols": st.emb_cols if emb_needed else [],
        "embedding": st.emb_info if emb_needed else None,
        "feedback_params": fp,
        "pipeline_config": {k: cfg.get(k) for k in ("geometry", "preprocess", "segment", "features", "embed", "seed")},
        "trained_on": {"task_a_cups": len(rows_a), "sessions": int(g_a.nunique())},
    }
    out = cfg_path(cfg, "models_dir")
    out.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out, suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(bundle, tmp)
        os.chmod(tmp, 0o644)
        os.replace(tmp, out / BUNDLE_NAME)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    summary = {k: v for k, v in bundle.items() if k not in ("score", "fixes", "milk")}
    summary["score"] = {k: v for k, v in bundle["score"].items() if k != "pipeline"}
    summary["fixes"] = {k: v for k, v in bundle["fixes"].items() if k != "pipeline"}
    summary["milk"] = None if milk is None else {k: v for k, v in milk.items() if k != "pipeline"}
    train_eval.write_json(out / "final.json", summary)

    print(f"Score: {score_name}, shown as score ± {score_error_text(score_err):.1f} "
          f"(out-of-fold error of the rounded score in E2: {score_err:.3f}).")
    print("Fixes: ridge_hand contributions (w_i * z_i) with the tips in src/latte/tips.yaml.")
    if milk:
        shown = ", ".join(f"{k} at p >= {t:.2f}" for k, t in milk["thresholds"].items() if t is not None)
        never = [k for k, t in milk["thresholds"].items() if t is None]
        print(f"Milk diagnosis: {milk_name}; shown for {shown}" + (f"; never shown for {', '.join(never)}" if never else "") + ".")
    else:
        print(f"Milk diagnosis: WITHHELD. {reason}")
    if emb_needed:
        print(f"The demo needs embeddings from {st.emb_info.get('model')} at inference time.")
    root = cfg["_root"]
    rel = lambda q: q.relative_to(root) if q.is_relative_to(root) else q  # noqa: E731
    print(f"Wrote {rel(out / BUNDLE_NAME)} and {rel(out / 'final.json')}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
