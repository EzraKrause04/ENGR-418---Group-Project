"""Train/eval harness on fake studies (build step 8): nested LOSO CV, E0-E5, run records, leakage check."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import run_script
from fake import fake_study, small_grids
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

from latte.evaluate import classification_metrics, nested_cv, regression_metrics
from latte.io import cfg_path, load_config
from latte.models import ClippedPCA, Spec, classification_specs, regression_specs


def setup(project, **kw):
    small_grids(project, **{k: v for k, v in kw.items() if k in ("e5_min_per_pourer", "min_cups")})
    cfg = load_config(project / "configs" / "default.yaml")
    truth = fake_study(cfg, **{k: v for k, v in kw.items() if k not in ("e5_min_per_pourer", "min_cups")})
    return cfg, truth


def table(cfg, name: str) -> pd.DataFrame:
    return pd.read_csv(cfg_path(cfg, "tables_dir") / f"{name}.csv")


# --- the CV machinery --------------------------------------------------------


class SpyRegressor(RegressorMixin, BaseEstimator):
    """Records which sessions it was trained on (the session is encoded in feature 0)."""

    seen: list[set] = []

    def fit(self, X, y):
        SpyRegressor.seen.append(set(np.asarray(X)[:, 0].round().astype(int)))
        self.m_ = float(np.mean(y))
        return self

    def predict(self, X):
        sessions = set(np.asarray(X)[:, 0].round().astype(int))
        assert not any(sessions & s for s in SpyRegressor.seen[-1:]), "test session was in the training data"
        return np.full(len(X), self.m_)


def test_outer_folds_never_train_on_the_test_session():
    rng = np.random.default_rng(0)
    groups = pd.Series(np.repeat([f"S{i:02d}" for i in range(5)], 8))
    X = pd.DataFrame({"session": np.repeat(np.arange(5), 8).astype(float), "x": rng.normal(size=40)})
    y = pd.Series(rng.normal(size=40))
    SpyRegressor.seen = []
    spec = Spec("spy", "reg", "hand", "spy", Pipeline([("model", SpyRegressor())]))
    res = nested_cv(spec, X, y, groups)
    assert len(res.folds) == 5 and res.oof.notna().all()
    for s, fold in zip(SpyRegressor.seen, res.folds["session"]):
        assert int(fold[1:]) not in s and len(s) == 4


def test_clipped_pca_handles_tiny_folds():
    X = np.random.default_rng(0).normal(size=(6, 30))
    p = ClippedPCA(20).fit(X)
    assert p.n_components_ == 5 and p.transform(X).shape == (6, 5)


def test_metrics_formulas():
    m = regression_metrics(np.array([1, 2, 3, 4.0]), np.array([1, 2, 3, 5.0]))
    assert m["mae"] == pytest.approx(0.25) and m["rmse"] == pytest.approx(0.5) and m["spearman"] == pytest.approx(1.0)
    c = classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert c["bal_acc"] == pytest.approx(0.75)


def test_specs_cover_the_brief():
    grids = {"ridge_alpha": [1.0], "pca_components": [5], "logreg_C": [1.0], "rf_n_estimators": [10], "rf_max_depth": [3]}
    reg = {s.name for s in regression_specs(["a"], ["e1"], grids)}
    assert reg == {"dummy_mean", "ridge_hand", "ridge_emb", "ridge_hand+emb", "rf_hand"}
    clf = {s.name for s in classification_specs(["a"], ["e1"], grids)}
    assert {"dummy_most_frequent", "dummy_stratified", "logreg_hand", "rf_hand", "logreg_emb", "rf_hand+emb"} <= clf


# --- the script ----------------------------------------------------------------


def test_full_run_finds_signal_and_writes_everything(project):
    cfg, truth = setup(project, n_sessions=6, per_session=12, e5_min_per_pourer=30)
    r = run_script("scripts/train_eval.py", project)
    assert r.returncode == 0, r.stderr
    t3 = table(cfg, "T3_task_a").set_index("model")
    assert t3.at["ridge_hand", "mae"] < 0.7 * t3.at["dummy_mean", "mae"]
    assert {"mae_vs_A", "mae_vs_B"} <= set(t3.columns) and "human: rater A vs rater B" in t3.index
    assert "ridge_emb" in t3.index  # embeddings were found and used
    t4 = table(cfg, "T4_task_b").set_index(["target", "model"])
    assert t4.loc[("milk_condition", "logreg_hand"), "bal_acc"] > 0.7
    assert t4.loc[("milk_condition", "dummy_most_frequent"), "bal_acc"] == pytest.approx(1 / 3)
    e1 = table(cfg, "E1_feature_validity").set_index("feature")
    assert e1.at["contrast_dL", "spearman"] > 0.3 and e1.at["contrast_dL", "ci_lo"] <= e1.at["contrast_dL", "spearman"]
    t5 = table(cfg, "T5_ablation")
    assert set(t5["dropped_group"]) == {"contrast", "edge", "symmetry_centring", "coverage", "surface"}
    imp = table(cfg, "E4_importance")
    a = imp[imp["task"] == "task_a"].set_index("feature")["importance_mean"]
    assert a.idxmax() in ("contrast_dL", "symmetry_iou")
    e5 = table(cfg, "E5_pourer")
    assert {"A", "B"} <= set(e5["train_pourer"].dropna())
    assert (cfg_path(cfg, "tables_dir") / "E0_agreement.tex").exists()
    te = load_te()
    e2dir, e3dir = te.latest_run_dir(cfg, "E2"), te.latest_run_dir(cfg, "E3")
    latest = json.loads((cfg_path(cfg, "runs_dir") / "latest.json").read_text())
    assert not latest["runs"]["E2"]["dir"].startswith("/")  # relative: portable across laptops
    oof = pd.read_csv(e2dir / "oof_task_a.csv").set_index("cup_id")
    assert {"y", "ridge_hand", "dummy_mean"} <= set(oof.columns) and len(oof) == 72
    b1 = pd.read_csv(e3dir / "oof_milk_condition.csv")
    probs = b1[[c for c in b1.columns if c.startswith("logreg_hand__p_")]]
    assert probs.shape[1] == 3 and np.allclose(probs.sum(axis=1), 1)
    assert (e2dir / "config.yaml").exists() and json.loads((e2dir / "run.json").read_text())["embedding"]["model"] == "fake_model"
    # Fair human comparison: the rater-vs-rater MAE sits in the per-rater columns, never in "mae".
    human = t3.loc["human: rater A vs rater B"]
    assert np.isnan(human["mae"]) and human["mae_vs_A"] == human["mae_vs_B"] > 0
    assert r"fake\_model" in (cfg_path(cfg, "tables_dir") / "T3_task_a.tex").read_text()
    assert {"bal_acc_ci_lo", "bal_acc_ci_hi", "above_chance"} <= set(t4.columns)
    assert bool(t4.loc[("milk_condition", "logreg_hand"), "above_chance"])
    assert not bool(t4.loc[("milk_condition", "dummy_most_frequent"), "above_chance"])
    for f in cfg_path(cfg, "runs_dir").rglob("*.json"):  # strict JSON everywhere (no NaN tokens)
        json.loads(f.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    e1 = table(cfg, "E1_feature_validity").set_index("feature")
    assert "n_components" in e1.index and e1.at["contrast_dL", "sessions"] == 6


def load_te():
    import importlib.util

    from latte.io import REPO_ROOT

    spec = importlib.util.spec_from_file_location("te", REPO_ROOT / "scripts" / "train_eval.py")
    te = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(te)
    return te


def test_partial_rerun_keeps_other_pointers_and_drops_stale_tables(project):
    cfg, _ = setup(project, n_sessions=4, per_session=12)
    assert run_script("scripts/train_eval.py", project, "--exp", "E1,E2").returncode == 0
    te = load_te()
    e2_first = te.latest_run_dir(cfg, "E2")
    assert run_script("scripts/train_eval.py", project, "--exp", "E1").returncode == 0
    assert te.latest_run_dir(cfg, "E2") == e2_first  # a partial run doesn't lose E2
    # Labels vanish -> E2 is skipped -> the old T3 must not survive.
    for f in cfg_path(cfg, "labels_dir").glob("rater_*.csv"):
        f.unlink()
    r = run_script("scripts/train_eval.py", project, "--exp", "E2")
    assert r.returncode == 0 and "skipped" in r.stdout
    assert not (cfg_path(cfg, "tables_dir") / "T3_task_a.csv").exists()


def test_leakage_check_passes_on_shuffled_labels(project):
    cfg, _ = setup(project, n_sessions=5, per_session=12)
    r = run_script("scripts/train_eval.py", project, "--shuffle-labels")
    assert r.returncode == 0, r.stdout + r.stderr
    leak = table(cfg, "leakage_check")
    assert (leak["verdict"] == "PASS").all()
    assert "Leakage check PASSED" in r.stdout


def test_no_signal_gives_chance_results(project):
    cfg, _ = setup(project, n_sessions=5, per_session=12, signal=False)
    r = run_script("scripts/train_eval.py", project, "--exp", "E2,E3")
    assert r.returncode == 0, r.stderr
    t3 = table(cfg, "T3_task_a").set_index("model")
    assert t3.at["ridge_hand", "r2"] < 0.15
    t4 = table(cfg, "T4_task_b").set_index(["target", "model"])
    assert t4.loc[("milk_condition", "logreg_hand"), "bal_acc"] < 0.5


def test_partial_labels_and_missing_embeddings(project):
    cfg, _ = setup(project, n_sessions=4, per_session=12, labelled_frac=0.5)
    for f in cfg_path(cfg, "features_dir").glob("emb_*"):
        f.unlink()
    r = run_script("scripts/train_eval.py", project, "--exp", "E2,E3")
    assert r.returncode == 0, r.stderr
    assert "embedding models skipped" in r.stdout
    t3 = table(cfg, "T3_task_a")
    assert t3["n"].iloc[0] == 24 and not t3["model"].str.contains("emb").any()
    assert table(cfg, "T4_task_b")["n"].iloc[0] == 48  # Task B uses unlabelled cups too


def test_too_little_data_is_skipped_with_a_note(project):
    cfg, _ = setup(project, n_sessions=1, per_session=6)
    r = run_script("scripts/train_eval.py", project, "--exp", "E2,E3")
    assert r.returncode == 0, r.stderr
    assert "needs at least" in r.stdout


def test_unknown_experiment_is_rejected(project):
    setup(project, n_sessions=2, per_session=6)
    r = run_script("scripts/train_eval.py", project, "--exp", "E9")
    assert r.returncode == 2 and "unknown experiment" in r.stderr


def test_leakage_check_has_teeth(project, monkeypatch, capsys):
    """If the 'shuffle' doesn't actually break the labels, the check must flag it."""
    cfg, _ = setup(project, n_sessions=5, per_session=12)
    te = load_te()
    monkeypatch.setattr(te.Study, "_shuffle_within_sessions", lambda self, s, rng: s)  # a broken shuffle
    cfg["train"]["leakage_repeats"] = 3
    assert te.leakage_check(cfg) == 2
    assert "CHECK" in capsys.readouterr().out


def test_leakage_check_not_fooled_by_sessions_of_different_quality(project):
    """Shuffling within sessions keeps each session's mean score; judging within sessions must still PASS."""
    cfg, _ = setup(project, n_sessions=8, per_session=12, session_effect=1.0, seed=2)
    r = run_script("scripts/train_eval.py", project, "--shuffle-labels")
    assert r.returncode == 0, r.stdout + r.stderr
    leak = table(cfg, "leakage_check")
    assert "spearman_within_session" in leak.columns and (leak["verdict"] == "PASS").all()


def test_above_chance_is_unknown_with_too_few_sessions(project):
    cfg, _ = setup(project, n_sessions=2, per_session=12)
    assert run_script("scripts/train_eval.py", project, "--exp", "E3").returncode == 0
    t4 = table(cfg, "T4_task_b")
    assert t4["bal_acc_ci_lo"].isna().all() and t4["above_chance"].isna().all()


def test_skipped_leakage_check_removes_old_table(project):
    cfg, _ = setup(project, n_sessions=1, per_session=6)
    stale = cfg_path(cfg, "tables_dir") / "leakage_check.csv"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("old")
    r = run_script("scripts/train_eval.py", project, "--shuffle-labels")
    assert r.returncode == 0 and "Not enough data" in r.stdout and not stale.exists()
