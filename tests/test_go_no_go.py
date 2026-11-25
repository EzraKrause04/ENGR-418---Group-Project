"""Go/no-go script and the analysis-table loader on fake studies (build step 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import run_script

from latte.data import load_dataset
from latte.features import FEATURES
from latte.io import cfg_path, empty_cup_log, write_cup_log
from latte.labels import append_label, labels_path


def fake_study(cfg, signal: bool, n_sessions: int = 4, per_session: int = 12, n_labelled: int = 30,
               score_shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """A cup log, features.csv and both raters' labels for n_sessions x per_session cups.

    With ``signal``, contrast_dL tracks milk condition and coverage tracks drop-in;
    otherwise every feature is noise.
    """
    rng = np.random.default_rng(seed)
    milk, drop = cfg["design"]["milk_conditions"], cfg["design"]["dropin"]
    cells = [(m, d) for m in milk for d in drop]
    log, feats = empty_cup_log(), []
    for s in range(1, n_sessions + 1):
        order = [cells[i % 6] for i in rng.permutation(per_session)]
        for k, (m, d) in enumerate(order, 1):
            cid = f"S{s:02d}-C{k:02d}"
            log.loc[len(log)] = [cid, f"S{s:02d}", "2026-11-25", "A", m, d, str(10 + 20 * milk.index(m)), "", "20",
                                 f"{cid}.jpg", "y", ""]
            f = {c: rng.normal() for c in FEATURES}
            f["n_components"] = float(rng.integers(1, 3))
            if signal:
                f["contrast_dL"] = 30 - 10 * milk.index(m) + rng.normal(0, 2)
                f["coverage"] = 0.35 - 0.15 * drop.index(d) + rng.normal(0, 0.03)
            feats.append({"cup_id": cid, **f})
    write_cup_log(log, cfg_path(cfg, "cup_log"))
    fdir = cfg_path(cfg, "features_dir")
    fdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feats).to_csv(fdir / "features.csv", index=False)
    dims = cfg["labelling"]["dimensions"]
    for r in cfg["labelling"]["raters"]:
        for cid in log["cup_id"][:n_labelled]:
            sc = int(np.clip(round(rng.normal(3 + score_shift, 1)), 1, 5))
            append_label(labels_path(cfg, r), {"cup_id": cid, **{d: sc for d in dims}, "unusable": "n",
                                               "comment": "", "timestamp": "2026-11-26T10:00:00-08:00"}, dims)
    return log


def fast(project, **gonogo) -> None:
    p = project / "configs" / "default.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["gonogo"] |= {"n_permutations": 49, **gonogo}
    p.write_text(yaml.safe_dump(raw))


def read(cfg, name: str) -> pd.DataFrame:
    return pd.read_csv(cfg_path(cfg, "tables_dir") / f"{name}.csv")


def test_signal_is_found(project, cfg):
    fake_study(cfg, signal=True)
    fast(project)
    r = run_script("scripts/go_no_go.py", project)
    assert r.returncode == 0, r.stderr
    clf = read(cfg, "gonogo_classifier").set_index("task")
    assert clf.at["milk_condition", "bal_acc"] > 0.6 and clf.at["milk_condition", "p_perm"] <= 0.05
    assert clf.at["dropin", "bal_acc"] > 0.8 and clf.at["milk_condition", "sessions"] == 4
    sep = read(cfg, "gonogo_features").set_index("feature")
    assert sep.at["contrast_dL", "milk_p_holm"] < 0.001 and sep.at["coverage", "dropin_p_holm"] < 0.001
    assert sep.at["foam_gain_pct", "milk_p"] < 0.001 and sep.at["foam_gain_pct", "source"] == "measured"
    assert np.isnan(sep.at["foam_gain_pct", "milk_p_holm"])  # kept out of the Holm family
    assert sep.at["coverage", "dropin_r"] > 0.8  # on_time has higher coverage -> r > 0
    assert "diagnosis signal looks present" in r.stdout
    figs = cfg_path(cfg, "figures_dir")
    for f in ("gonogo_by_milk.png", "gonogo_by_dropin.png", "gonogo_overall.png"):
        assert (figs / f).stat().st_size > 10_000
    assert (cfg_path(cfg, "tables_dir") / "gonogo_classifier.tex").exists()


def test_no_signal_is_reported_honestly(project, cfg):
    fake_study(cfg, signal=False)
    fast(project)
    r = run_script("scripts/go_no_go.py", project)
    assert r.returncode == 0, r.stderr
    clf = read(cfg, "gonogo_classifier").set_index("task")
    assert clf.at["milk_condition", "bal_acc"] < 0.55 and clf.at["milk_condition", "p_perm"] > 0.05
    assert "No diagnosis signal yet" in r.stdout
    # Measured foam gain still differs by condition, but it is not a photo feature.
    assert "Photo features that separate milk conditions (Holm p < 0.05): none" in r.stdout
    assert "Manipulation check" in r.stdout


def test_narrow_quality_range_is_flagged(project, cfg):
    fake_study(cfg, signal=True, score_shift=-2.5)
    fast(project)
    r = run_script("scripts/go_no_go.py", project)
    assert "NARROW quality range" in r.stdout


def test_too_few_sessions_and_cups(project, cfg):
    fake_study(cfg, signal=True, n_sessions=1, per_session=12, n_labelled=0)
    fast(project)
    r = run_script("scripts/go_no_go.py", project)
    assert r.returncode == 0, r.stderr
    assert "fewer than the 30" in r.stdout and "needs at least 2 sessions" in r.stdout
    assert "no cups labelled by both raters" in r.stdout
    assert not (cfg_path(cfg, "figures_dir") / "gonogo_overall.png").exists()


def test_missing_features_file_is_an_error(project):
    r = run_script("scripts/go_no_go.py", project)
    assert r.returncode == 1 and "make features" in r.stderr


def test_loader_exclusions(project, cfg):
    log = fake_study(cfg, signal=True, n_sessions=2, per_session=6, n_labelled=12)
    log.loc[log["cup_id"] == "S01-C02", "usable"] = "n"
    write_cup_log(log, cfg_path(cfg, "cup_log"))
    dims = cfg["labelling"]["dimensions"]
    append_label(labels_path(cfg, "B"), {"cup_id": "S01-C03", **{d: "" for d in dims}, "unusable": "y",
                                         "comment": "glare", "timestamp": "2026-11-26T11:00:00-08:00"}, dims)
    p = project / "configs" / "default.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["design"]["exclude_sessions"] = ["S02"]
    p.write_text(yaml.safe_dump(raw))
    from latte.io import load_config

    ds = load_dataset(load_config(p))
    assert ds.excluded["usable=n in cup log"] == ["S01-C02"]
    assert ds.excluded["unusable by a rater"] == ["S01-C03"]
    assert len(ds.excluded["excluded session"]) == 6
    assert list(ds.table.index) == ["S01-C01", "S01-C04", "S01-C05", "S01-C06"]
    assert ds.table["labelled"].all() and ds.table["foam_gain_pct"].dtype == float


def test_holm_matches_hand_computation():
    import importlib.util

    from latte.io import REPO_ROOT

    spec = importlib.util.spec_from_file_location("gng", REPO_ROOT / "scripts" / "go_no_go.py")
    gng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gng)
    p = pd.Series([0.01, 0.04, np.nan, 0.03])
    adj = gng.holm(p)
    assert adj[0] == pytest.approx(0.03) and adj[3] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[2])


def test_unusable_from_one_rater_is_excluded_before_the_other_labels(project, cfg):
    fake_study(cfg, signal=True, n_sessions=2, per_session=6, n_labelled=0)
    dims = cfg["labelling"]["dimensions"]
    append_label(labels_path(cfg, "A"), {"cup_id": "S01-C01", **{d: "" for d in dims}, "unusable": "y",
                                         "comment": "glare", "timestamp": "2026-11-26T11:00:00-08:00"}, dims)
    ds = load_dataset(cfg)
    assert ds.excluded["unusable by a rater"] == ["S01-C01"] and "S01-C01" not in ds.table.index


def test_absent_class_sets_chance_from_present_classes(project, cfg):
    log = fake_study(cfg, signal=True)
    keep = log[log["milk_condition"] != "thick"]
    write_cup_log(keep, cfg_path(cfg, "cup_log"))
    f = pd.read_csv(cfg_path(cfg, "features_dir") / "features.csv")
    f[f["cup_id"].isin(keep["cup_id"])].to_csv(cfg_path(cfg, "features_dir") / "features.csv", index=False)
    fast(project)
    r = run_script("scripts/go_no_go.py", project)
    assert r.returncode == 0, r.stderr
    clf = read(cfg, "gonogo_classifier").set_index("task")
    assert clf.at["milk_condition", "chance"] == pytest.approx(0.5) and clf.at["milk_condition", "classes"] == 2
    assert "absent: thick" in clf.at["milk_condition", "note"]


def test_all_missing_feature_is_dropped_and_named(project, cfg):
    fake_study(cfg, signal=True)
    fp = cfg_path(cfg, "features_dir") / "features.csv"
    f = pd.read_csv(fp)
    f["bubble_density"] = np.nan
    f.to_csv(fp, index=False)
    fast(project)
    r = run_script("scripts/go_no_go.py", project)
    assert r.returncode == 0, r.stderr
    assert "all-missing features dropped: bubble_density" in r.stdout
    assert r.stderr.count("Skipping features") == 0


def test_broken_cup_log_is_reported_not_a_traceback(project, cfg):
    log = fake_study(cfg, signal=True, n_sessions=2, per_session=6)
    write_cup_log(pd.concat([log, log.iloc[[0]]], ignore_index=True), cfg_path(cfg, "cup_log"))
    r = run_script("scripts/go_no_go.py", project)
    assert r.returncode == 1 and "duplicate cup_id" in r.stderr and "Traceback" not in r.stderr
