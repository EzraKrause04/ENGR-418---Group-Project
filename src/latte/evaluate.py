"""Metrics, cross-validation and inter-rater agreement.

Agreement metrics (section 7.3) are implemented now; the CV harness for
experiments E1-E5 is added in build step 8.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

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
