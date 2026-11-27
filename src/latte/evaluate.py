"""Metrics, cross-validation and inter-rater agreement.

* Agreement (section 7.3): weighted kappa, Spearman rho, bootstrap CIs.
* Nested cross-validation (section 10.1): the outer loop is LeaveOneGroupOut
  by session_id; the inner loop is a GridSearchCV over GroupKFold(inner_folds)
  on the training sessions only. Out-of-fold (OOF) predictions are pooled for
  the headline metrics, and per-fold metrics give mean +/- std.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning, UndefinedMetricWarning
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

SCORE_LEVELS = [1, 2, 3, 4, 5]


def weighted_kappa(a: np.ndarray, b: np.ndarray, levels: list[int] = SCORE_LEVELS) -> float:
    """Quadratic-weighted Cohen's kappa.

    kappa_w = 1 - sum_ij w_ij O_ij / sum_ij w_ij E_ij, with w_ij = (i - j)^2 / (k - 1)^2,
    O the observed and E the chance-expected (outer product of marginals)
    confusion matrices over the fixed score levels 1..5. NaN if undefined
    (e.g. both raters gave a single identical score to every cup).
    """
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(a) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = cohen_kappa_score(a, b, labels=levels, weights="quadratic")
    return float(k)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation rho (NaN if either input is constant)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def bootstrap_ci(
    a: np.ndarray, b: np.ndarray, stat, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat(a, b)``, resampling cups (pairs) with replacement."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    vals = np.array([stat(a[i], b[i]) for i in idx], float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def agreement_table(
    a: pd.DataFrame, b: pd.DataFrame, dims: list[str], n_boot: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """Per-dimension agreement between two raters on their common cups.

    Columns: n, weighted kappa (+95% bootstrap CI), Spearman rho, mean absolute
    difference |A - B|, exact agreement and within-one agreement (fractions).
    ``a`` and ``b`` are indexed by cup_id; cups missing a score are dropped per dimension.
    """
    common = a.index.intersection(b.index)
    rows = []
    for d in dims:
        x, y = a.loc[common, d].astype(float), b.loc[common, d].astype(float)
        ok = x.notna() & y.notna()
        x, y = x[ok].to_numpy(), y[ok].to_numpy()
        lo, hi = bootstrap_ci(x, y, weighted_kappa, n_boot, seed)
        diff = np.abs(x - y)
        rows.append(
            {
                "dimension": d,
                "n": int(len(x)),
                "kappa_w": weighted_kappa(x, y),
                "kappa_lo": lo,
                "kappa_hi": hi,
                "spearman": spearman(x, y),
                "mad": float(diff.mean()) if len(x) else float("nan"),
                "exact": float((diff == 0).mean()) if len(x) else float("nan"),
                "within1": float((diff <= 1).mean()) if len(x) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics for the experiments
# ---------------------------------------------------------------------------


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """MAE = mean |y - p|; RMSE = sqrt(mean (y - p)^2); Spearman rho(y, p); R^2 = 1 - SS_res / SS_tot."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return {"mae": np.nan, "rmse": np.nan, "spearman": np.nan, "r2": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        r2 = r2_score(y, p) if len(y) >= 2 and np.ptp(y) > 0 else np.nan
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(root_mean_squared_error(y, p)),
            "spearman": spearman(y, p), "r2": float(r2)}


def classification_metrics(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Balanced accuracy = mean per-class recall over classes present in y; macro-F1 over ``classes``."""
    y, p = np.asarray(y, object), np.asarray(p, object)
    if len(y) == 0:
        return {"bal_acc": np.nan, "macro_f1": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"bal_acc": float(balanced_accuracy_score(y, p)),
                "macro_f1": float(f1_score(y, p, labels=classes, average="macro", zero_division=0))}


# ---------------------------------------------------------------------------
# Nested leave-one-session-out cross-validation
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Everything one model produced under nested CV."""

    name: str
    task: str
    oof: pd.Series  # out-of-fold prediction per cup_id
    proba: pd.DataFrame | None  # classification: OOF class probabilities (columns = classes)
    folds: pd.DataFrame  # one row per outer fold: session, n, metrics, chosen params
    fitted: list[tuple[Any, np.ndarray]] = field(default_factory=list)  # (fitted pipeline, test positions)

    def pooled(self, y: pd.Series, classes: list[str] | None = None) -> dict[str, float]:
        yy = y.loc[self.oof.index]
        return classification_metrics(yy, self.oof, classes or sorted(yy.unique())) if self.task == "clf" \
            else regression_metrics(yy, self.oof)


def _default_params(grid: dict[str, list[Any]]) -> dict[str, Any]:
    """Middle grid value of each parameter: used when there are too few sessions to tune."""
    return {k: v[len(v) // 2] for k, v in grid.items()}


def nested_cv(spec: Any, X: pd.DataFrame, y: pd.Series, groups: pd.Series, inner_folds: int = 3,
              classes: list[str] | None = None, n_jobs: int = 1, keep_fitted: bool = False) -> CVResult:
    """Outer LeaveOneGroupOut(groups); inner GridSearchCV(GroupKFold) on the training sessions only.

    With fewer than 2 training sessions the grid can't be tuned without
    mixing sessions, so the middle grid value is used. A classification
    training fold with a single class predicts that class.
    """
    clf = spec.task == "clf"
    scoring = "balanced_accuracy" if clf else "neg_mean_absolute_error"
    Xv, yv, gv = X, y.to_numpy(), groups.to_numpy()
    oof = pd.Series(index=X.index, dtype=object if clf else float)
    proba = pd.DataFrame(np.nan, index=X.index, columns=classes) if clf and classes else None
    rows, fitted = [], []
    for train, test in LeaveOneGroupOut().split(Xv, yv, gv):
        est = clone(spec.pipeline)
        n_tr_groups = len(np.unique(gv[train]))
        params: dict[str, Any] = {}
        with warnings.catch_warnings():
            for w in (ConvergenceWarning, FitFailedWarning, UserWarning, RuntimeWarning, UndefinedMetricWarning):
                warnings.simplefilter("ignore", w)
            if clf and len(np.unique(yv[train])) < 2:
                const = yv[train][0]
                pred = np.array([const] * len(test), dtype=object)
                prob = pd.DataFrame(0.0, index=X.index[test], columns=classes) if proba is not None else None
                if prob is not None:
                    prob[const] = 1.0
                params = {"note": "single class in training"}
                est = None
            else:
                if spec.grid and n_tr_groups >= 2:
                    gs = GridSearchCV(est, spec.grid, cv=GroupKFold(n_splits=min(inner_folds, n_tr_groups)),
                                      scoring=scoring, n_jobs=n_jobs, error_score=np.nan, refit=True)
                    gs.fit(X.iloc[train], yv[train], groups=gv[train])
                    est, params = gs.best_estimator_, dict(gs.best_params_)
                else:
                    if spec.grid:
                        params = _default_params(spec.grid)
                        est.set_params(**params)
                    est.fit(X.iloc[train], yv[train])
                pred = est.predict(X.iloc[test])
                prob = None
                if proba is not None:
                    pr = est.predict_proba(X.iloc[test])
                    prob = pd.DataFrame(0.0, index=X.index[test], columns=classes)
                    for j, c in enumerate(est.classes_):
                        prob[c] = pr[:, j]
        oof.iloc[test] = pred
        if prob is not None:
            proba.loc[prob.index] = prob.to_numpy()
        if clf:  # per fold, score only the classes present in this session (absent ones would count as F1 = 0)
            present = set(yv[test])
            m = classification_metrics(yv[test], pred, [c for c in classes if c in present] if classes else sorted(present))
        else:
            m = regression_metrics(yv[test], pred)
        rows.append({"session": gv[test][0], "n": len(test), **m, "params": params})
        if keep_fitted and est is not None:
            fitted.append((est, test))
    return CVResult(spec.name, spec.task, oof, proba, pd.DataFrame(rows), fitted)


def summarise(res: CVResult, y: pd.Series, classes: list[str] | None = None) -> dict[str, Any]:
    """Pooled OOF metrics plus mean and std over outer folds."""
    out: dict[str, Any] = {"model": res.name, "n": int(len(res.oof)), "folds": int(len(res.folds))}
    pooled = res.pooled(y, classes)
    for k, v in pooled.items():
        out[k] = v
        col = res.folds[k] if k in res.folds else pd.Series(dtype=float)
        out[f"{k}_fold_mean"] = float(col.mean()) if col.notna().any() else np.nan
        out[f"{k}_fold_std"] = float(col.std(ddof=1)) if col.notna().sum() > 1 else np.nan
    return out


def summed_confusion(res: CVResult, y: pd.Series, classes: list[str]) -> pd.DataFrame:
    """Confusion matrix of the pooled OOF predictions = the per-fold matrices summed (rows = true)."""
    cm = confusion_matrix(y.loc[res.oof.index].to_numpy(object), res.oof.to_numpy(object), labels=classes)
    return pd.DataFrame(cm, index=pd.Index(classes, name="true"), columns=pd.Index(classes, name="predicted"))


def cluster_bootstrap_ci(values: pd.DataFrame, groups: pd.Series, stat, n_boot: int, seed: int,
                         alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat(df)`` resampling whole sessions (clusters) with replacement.

    Cups in one session share milk, light and pourer, so resampling cups one
    by one would make the interval too narrow. NaN if there are fewer than 3 sessions.
    """
    g = groups.loc[values.index]
    ids = g.unique()
    if len(ids) < 3:
        return float("nan"), float("nan")
    by = {k: values.index[g.to_numpy() == k] for k in ids}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        v = stat(values.loc[np.concatenate([by[k] for k in pick])])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def latest_run_dir(cfg: dict[str, Any], exp: str) -> Path | None:
    """The most recent run folder of an experiment, from results/runs/latest.json (written by make train)."""
    from latte.io import cfg_path

    path = cfg_path(cfg, "runs_dir") / "latest.json"
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get("runs", {}).get(exp)
    return cfg_path(cfg, "runs_dir") / entry["dir"] if entry else None
