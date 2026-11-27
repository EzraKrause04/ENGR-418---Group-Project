"""Feedback engine and final fit (build step 9)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import run_script
from fake import fake_study, small_grids
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from latte.features import MODEL_FEATURES
from latte.feedback import (
    MILK_UNCLEAR,
    NO_ISSUES,
    load_bundle,
    load_tips,
    make_feedback,
    ridge_contributions,
    round_score,
    top_fixes,
    NO_TIP,
    UNMEASURABLE,
    tune_thresholds,
)
from latte.io import cfg_path, load_config


def test_round_score():
    assert round_score(3.24) == 3.0 and round_score(3.26) == 3.5 and round_score(7) == 5.0 and round_score(-1) == 1.0


def test_tips_file_matches_the_features_and_milk_classes(cfg):
    tips = load_tips()
    assert set(tips["fixes"]) <= set(MODEL_FEATURES)
    for f, dirs in tips["fixes"].items():
        assert set(dirs) <= {"low", "high"}, f
        for d in dirs.values():
            assert d["cause"] and d["tip"]
    assert set(tips["milk"]) == set(cfg["design"]["milk_conditions"])


def ridge_on(df: pd.DataFrame, y: np.ndarray) -> Pipeline:
    prep = ColumnTransformer([("hand", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                               MODEL_FEATURES)])
    return Pipeline([("prep", prep), ("model", Ridge(alpha=1.0))]).fit(df, y)


@pytest.fixture
def ridge():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    y = 3 + 0.8 * df["contrast_dL"] - 0.6 * df["centre_offset"] + rng.normal(0, 0.1, 60)
    return ridge_on(df, y.to_numpy()), df


def test_contributions_add_up_to_the_prediction(ridge):
    pipe, df = ridge
    row = df.iloc[[3]]
    c, z = ridge_contributions(pipe, row)
    assert pipe.predict(row)[0] == pytest.approx(pipe.named_steps["model"].intercept_ + c.sum())
    assert c["contrast_dL"] == pytest.approx(pipe.named_steps["model"].coef_[0] * z["contrast_dL"])


def test_top_fixes_pick_most_negative_with_a_reviewed_tip():
    contrib = pd.Series({"contrast_dL": -0.5, "centre_offset": -0.3, "coverage": -0.2, "symmetry_iou": 0.4})
    z = pd.Series({"contrast_dL": -1.2, "centre_offset": 1.5, "coverage": 0.8, "symmetry_iou": 1.0})
    tips = load_tips()
    fixes, skipped = top_fixes(contrib, z, tips, 2, -0.1)
    assert [f.feature for f in fixes] == ["contrast_dL", "centre_offset"]
    assert fixes[0].direction == "low" and fixes[1].direction == "high"
    # coverage is HIGH here, but only "low coverage" has a reviewed tip: it is skipped, not improvised.
    fixes, skipped = top_fixes(contrib, z, tips, 3, -0.1)
    assert [f.feature for f in fixes] == ["contrast_dL", "centre_offset"] and skipped == ["coverage (high, -0.20)"]
    assert top_fixes(pd.Series({"contrast_dL": -0.05}), pd.Series({"contrast_dL": -1.0}), tips, 2, -0.1)[0] == []


def test_tune_thresholds_per_class():
    y = pd.Series(["a"] * 10 + ["b"] * 10)
    proba = pd.DataFrame({"a": np.r_[np.linspace(0.9, 0.55, 10), np.linspace(0.45, 0.1, 10)]})
    proba["b"] = 1 - proba["a"]
    proba.index = y.index
    good = tune_thresholds(y, proba, 0.6, 0.7)
    assert good["thresholds"] == {"a": 0.6, "b": 0.6} and good["per_class"]["a"]["precision"] == 1.0
    flipped = tune_thresholds(y.map({"a": "b", "b": "a"}), proba, 0.6, 0.7)
    assert flipped["thresholds"] == {"a": None, "b": None} and not flipped["any"]  # always wrong -> withhold
    # Class b is unreliable at every confidence, class a is fine: only a may ever be shown.
    mixed = y.copy()
    mixed.iloc[12:20] = "a"
    m = tune_thresholds(mixed, proba, 0.6, 0.7)
    assert m["thresholds"]["a"] == 0.6 and m["thresholds"]["b"] is None
    # Errors only at low confidence: the threshold rises above 0.6.
    lo = y.copy()
    lo.iloc[[6, 7, 12, 13]] = lo.iloc[[6, 7, 12, 13]].map({"a": "b", "b": "a"})
    r = tune_thresholds(lo, proba, 0.6, 0.95, min_n=3)
    assert r["thresholds"]["a"] > 0.6 and r["thresholds"]["b"] > 0.6


def bundle_for(pipe, df, milk_t=0.6):
    from sklearn.linear_model import LogisticRegression

    lab = np.where(df["contrast_dL"] > 0, "thin", "thick")
    milk = Pipeline([("prep", pipe.named_steps["prep"]), ("model", LogisticRegression())]).fit(df, lab)
    return {"score": {"pipeline": pipe, "error": 0.62}, "fixes": {"pipeline": pipe},
            "milk": {"pipeline": milk, "thresholds": {"thin": milk_t, "thick": milk_t}}, "emb_cols": []}


def test_make_feedback_states(ridge):
    pipe, df = ridge
    bundle = bundle_for(pipe, df)
    bad = df.iloc[0].to_dict() | {"contrast_dL": -3.0, "centre_offset": 3.0}
    fb = make_feedback(bad, None, bundle)
    assert fb.score_text.endswith("± 0.7") and fb.score <= 2.0  # 0.62 is rounded UP for display
    assert len(fb.fixes) == 2 and all(t.startswith("Likely cause:") for t in fb.fixes_text)
    assert fb.milk_status == "shown" and fb.milk_text.startswith("Likely: milk too thick (p = ")
    good = df.iloc[0].to_dict() | {"contrast_dL": 3.0, "centre_offset": -3.0}
    assert make_feedback(good, None, bundle).fixes_text == [NO_ISSUES]
    bundle["milk"]["thresholds"] = {"thin": None, "thick": None}  # neither class reliable -> never shown
    assert make_feedback(bad, None, bundle).milk_status == "unclear"
    bundle["milk"], bundle["milk_withheld_reason"] = None, "B1 not above chance"
    w = make_feedback(bad, None, bundle)
    assert w.milk_status == "withheld" and w.milk_text == "" and "chance" in w.milk_reason
    with pytest.raises(ValueError, match="embedding"):
        make_feedback(bad, None, bundle | {"emb_cols": ["emb_000"]})
    with pytest.raises(ValueError, match="missing"):
        make_feedback({"contrast_dL": 1.0}, None, bundle)


def test_fixes_never_contradict_the_milk_diagnosis(ridge):
    pipe, df = ridge
    bundle = bundle_for(pipe, df)
    bad = df.iloc[0].to_dict() | {"contrast_dL": -3.0, "centre_offset": 3.0}
    fb = make_feedback(bad, None, bundle)
    assert fb.milk_class == "thick" and fb.milk_status == "shown"
    contrast_fix = next(f for f in fb.fixes if f.feature == "contrast_dL")
    assert "thin" not in contrast_fix.cause and "aerate a little longer" not in contrast_fix.tip
    bundle["milk"] = None  # no diagnosis shown -> the full wording may mention thin milk
    fb2 = make_feedback(bad, None, bundle | {"milk_withheld_reason": "x"})
    assert "thin" in next(f for f in fb2.fixes if f.feature == "contrast_dL").cause


def test_unmeasured_features_are_never_advised_on(ridge):
    pipe, df = ridge
    bundle = bundle_for(pipe, df)
    no_milk = {f: np.nan for f in MODEL_FEATURES}
    fb = make_feedback(no_milk, None, bundle)
    assert not fb.measurable and fb.score_text == UNMEASURABLE and fb.milk_status == "withheld" and not fb.fixes
    partial = df.iloc[0].to_dict() | {"contrast_dL": 2.0, "centre_offset": np.nan}
    fb = make_feedback(partial, None, bundle)
    assert "centre_offset" in fb.unmeasured and all(f.feature != "centre_offset" for f in fb.fixes)


def test_no_tip_message_and_no_repeated_tips():
    tips = load_tips()
    fixes, skipped = top_fixes(pd.Series({"coverage": -0.8}), pd.Series({"coverage": 2.0}), tips, 2, -0.1)
    assert fixes == [] and skipped
    contrib = pd.Series({"contrast_dL": -0.5, "contrast_michelson": -0.4, "coverage": -0.3})
    z = pd.Series({"contrast_dL": -1.0, "contrast_michelson": -1.0, "coverage": -1.0})
    fixes, _ = top_fixes(contrib, z, tips, 2, -0.1)
    assert [f.feature for f in fixes] == ["contrast_dL", "coverage"]  # the identical michelson tip is not repeated


def test_no_tip_text_in_engine(ridge):
    pipe, df = ridge
    bundle = bundle_for(pipe, df)
    tips = {"fixes": {}, "milk": {}}
    bad = df.iloc[0].to_dict() | {"contrast_dL": -3.0}
    assert make_feedback(bad, None, bundle, tips=tips).fixes_text == [NO_TIP]


# --- fit_final end to end -----------------------------------------------------


def prepare(project, signal: bool = True, **kw):
    small_grids(project)
    cfg = load_config(project / "configs" / "default.yaml")
    fake_study(cfg, n_sessions=6, per_session=12, signal=signal, **kw)
    r = run_script("scripts/train_eval.py", project, "--exp", "E2,E3")
    assert r.returncode == 0, r.stderr
    return cfg


def test_fit_final_with_signal(project):
    cfg = prepare(project)
    r = run_script("scripts/fit_final.py", project)
    assert r.returncode == 0, r.stderr
    b = load_bundle(cfg)
    t3 = pd.read_csv(cfg_path(cfg, "tables_dir") / "T3_task_a.csv").set_index("model")
    assert b["score"]["oof_mae_unrounded"] == pytest.approx(t3.at[b["score"]["name"], "mae"])
    assert b["score"]["error"] > 0
    assert b["score"]["name"] == t3.drop(index=[i for i in t3.index if i.startswith(("dummy", "human"))])["mae"].idxmin()
    assert b["fixes"]["name"] == "ridge_hand" and b["milk"] is not None and b["milk"]["name"] == "logreg_hand"
    shown = {k: t for k, t in b["milk"]["thresholds"].items() if t is not None}
    assert shown and all(0.6 <= t < 1 for t in shown.values())
    assert all(b["milk"]["per_class"][k]["precision"] >= 0.7 for k in shown)
    summary = json.loads((cfg_path(cfg, "models_dir") / "final.json").read_text(),
                         parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert summary["milk"]["name"] == b["milk"]["name"] and "pipeline" not in summary["score"]
    assert summary["fingerprint"] == b["fingerprint"]
    feats = pd.read_csv(cfg_path(cfg, "features_dir") / "features.csv").set_index("cup_id")
    from latte.embed import load_embeddings

    emb = load_embeddings(cfg, verify=False)
    fb = make_feedback(feats.iloc[0].to_dict(), emb.iloc[0].to_numpy() if b["emb_cols"] else None, b)
    assert 1 <= fb.score <= 5 and fb.milk_status in ("shown", "unclear")


def test_fit_final_withholds_milk_without_signal(project):
    cfg = prepare(project, signal=False)
    r = run_script("scripts/fit_final.py", project)
    assert r.returncode == 0, r.stderr
    b = load_bundle(cfg)
    assert b["milk"] is None and "chance" in b["milk_withheld_reason"]
    assert "WITHHELD" in r.stdout


def test_fit_final_respects_score_model_and_needs_train(project):
    small_grids(project)
    r = run_script("scripts/fit_final.py", project)
    assert r.returncode == 1 and "make train" in r.stderr
    cfg = prepare(project)
    p = project / "configs" / "default.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["feedback"]["score_model"] = "ridge_hand"
    p.write_text(yaml.safe_dump(raw))
    assert run_script("scripts/fit_final.py", project).returncode == 0
    assert load_bundle(cfg)["score"]["name"] == "ridge_hand"
    raw["feedback"]["score_model"] = "nope"
    p.write_text(yaml.safe_dump(raw))
    r = run_script("scripts/fit_final.py", project)
    assert r.returncode == 1 and "score_model" in r.stderr


def test_fit_final_refuses_stale_runs_and_removes_old_bundle(project):
    cfg = prepare(project)
    assert run_script("scripts/fit_final.py", project).returncode == 0
    assert (cfg_path(cfg, "models_dir") / "latte_final.joblib").exists()
    # New data arrives (a different study) but make train is not rerun.
    fake_study(cfg, n_sessions=7, per_session=12, signal=False, seed=5)
    r = run_script("scripts/fit_final.py", project)
    assert r.returncode == 1 and "rerun make train" in r.stderr and "Removed the old" in r.stderr
    assert not (cfg_path(cfg, "models_dir") / "latte_final.joblib").exists()
