"""sklearn pipelines and hyperparameter grids for experiments E2-E5 (section 10.2).

Every model is a ``Pipeline(prep, model)``. ``prep`` is a ColumnTransformer
that picks the feature set, so imputation, scaling and PCA are all fitted
inside each training fold and nothing leaks across folds (section 10.1).

Feature sets:

* ``hand``: the handcrafted features (``latte.features.MODEL_FEATURES``),
  median-imputed then standardised;
* ``emb``: the cached embedding columns, standardised then PCA (n_components is
  in the grid, and is clipped to what the training fold can support);
* ``hand+emb``: both, side by side.

Model names are ``<estimator>_<feature set>``, e.g. ``ridge_hand``,
``logreg_emb``, ``rf_hand+emb``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_SETS = ("hand", "emb", "hand+emb")


class ClippedPCA(TransformerMixin, BaseEstimator):
    """PCA whose n_components is clipped to min(n_components, n_samples - 1, n_features) at fit time.

    With few cups per training fold, a grid value such as 20 components can
    exceed what the fold supports; clipping keeps the grid valid instead of erroring.
    """

    def __init__(self, n_components: int = 10, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "ClippedPCA":
        X = np.asarray(X, float)
        k = int(max(1, min(self.n_components, X.shape[0] - 1, X.shape[1])))
        self.pca_ = PCA(n_components=k, random_state=self.random_state).fit(X)
        self.n_components_ = k
        return self

    def transform(self, X: Any) -> np.ndarray:
        return self.pca_.transform(np.asarray(X, float))


def prep(feature_set: str, hand_cols: list[str], emb_cols: list[str], seed: int = 42) -> ColumnTransformer:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}")
    parts = []
    if "hand" in feature_set:
        parts.append(("hand", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                                        ("scale", StandardScaler())]), list(hand_cols)))
    if "emb" in feature_set:
        parts.append(("emb", Pipeline([("scale", StandardScaler()), ("pca", ClippedPCA(10, seed))]), list(emb_cols)))
    return ColumnTransformer(parts, remainder="drop")


@dataclass
class Spec:
    """One model of an experiment: a pipeline plus its hyperparameter grid."""

    name: str
    task: str  # "reg" or "clf"
    feature_set: str
    estimator: str  # "dummy_mean", "ridge", "rf", "dummy_most_frequent", "dummy_stratified", "logreg"
    pipeline: Pipeline
    grid: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def uses_emb(self) -> bool:
        return "emb" in self.feature_set


def _pca_grid(fs: str, grids: dict[str, Any]) -> dict[str, list[Any]]:
    return {"prep__emb__pca__n_components": list(grids["pca_components"])} if "emb" in fs else {}


def _rf_grid(grids: dict[str, Any]) -> dict[str, list[Any]]:
    return {"model__n_estimators": list(grids["rf_n_estimators"]), "model__max_depth": list(grids["rf_max_depth"])}


def regression_specs(hand_cols: list[str], emb_cols: list[str], grids: dict[str, Any], seed: int = 42,
                     feature_sets: tuple[str, ...] = FEATURE_SETS) -> list[Spec]:
    """E2 (Task A): DummyRegressor(mean); Ridge on each feature set; RandomForest on handcrafted features."""
    specs = [Spec("dummy_mean", "reg", "hand", "dummy_mean",
                  Pipeline([("prep", prep("hand", hand_cols, emb_cols, seed)), ("model", DummyRegressor(strategy="mean"))]))]
    for fs in feature_sets:
        specs.append(Spec(f"ridge_{fs}", "reg", fs, "ridge",
                          Pipeline([("prep", prep(fs, hand_cols, emb_cols, seed)), ("model", Ridge())]),
                          {"model__alpha": list(grids["ridge_alpha"]), **_pca_grid(fs, grids)}))
    if "hand" in feature_sets:
        specs.append(Spec("rf_hand", "reg", "hand", "rf",
                          Pipeline([("prep", prep("hand", hand_cols, emb_cols, seed)),
                                    ("model", RandomForestRegressor(random_state=seed))]), _rf_grid(grids)))
    return specs


def classification_specs(hand_cols: list[str], emb_cols: list[str], grids: dict[str, Any], seed: int = 42,
                         feature_sets: tuple[str, ...] = FEATURE_SETS) -> list[Spec]:
    """E3 (Task B): two dummies; multinomial LogisticRegression and RandomForest on each feature set."""
    p0 = prep("hand", hand_cols, emb_cols, seed)
    specs = [
        Spec("dummy_most_frequent", "clf", "hand", "dummy_most_frequent",
             Pipeline([("prep", p0), ("model", DummyClassifier(strategy="most_frequent"))])),
        Spec("dummy_stratified", "clf", "hand", "dummy_stratified",
             Pipeline([("prep", prep("hand", hand_cols, emb_cols, seed)),
                       ("model", DummyClassifier(strategy="stratified", random_state=seed))])),
    ]
    for fs in feature_sets:
        specs.append(Spec(f"logreg_{fs}", "clf", fs, "logreg",
                          Pipeline([("prep", prep(fs, hand_cols, emb_cols, seed)),
                                    ("model", LogisticRegression(max_iter=5000))]),
                          {"model__C": list(grids["logreg_C"]), **_pca_grid(fs, grids)}))
        specs.append(Spec(f"rf_{fs}", "clf", fs, "rf",
                          Pipeline([("prep", prep(fs, hand_cols, emb_cols, seed)),
                                    ("model", RandomForestClassifier(random_state=seed))]),
                          {**_rf_grid(grids), **_pca_grid(fs, grids)}))
    return specs


def foam_specs(hand_cols: list[str], emb_cols: list[str], grids: dict[str, Any], seed: int = 42,
               feature_sets: tuple[str, ...] = FEATURE_SETS) -> list[Spec]:
    """B3 (optional): regression of measured foam_gain_pct, with a mean dummy and Ridge per feature set."""
    return [s for s in regression_specs(hand_cols, emb_cols, grids, seed, feature_sets) if s.estimator != "rf"]
