"""Agreement metrics, and agreement.py end-to-end on fake labels (build step 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import run_script, tiny_jpeg

from latte.evaluate import agreement_table, spearman, weighted_kappa
from latte.io import cfg_path, write_cup_log
from latte.labels import append_label, labels_path
from latte.schedule import plan_session

DIMS = ["contrast", "definition", "symmetry", "centring", "surface", "overall"]


def test_kappa_perfect_shifted_and_random():
    rng = np.random.default_rng(0)
    a = rng.integers(1, 6, 200)
    assert weighted_kappa(a, a) == 1.0
    shifted = np.clip(a + 1, 1, 5)
    assert 0.5 < weighted_kappa(a, shifted) < 1.0
    assert abs(weighted_kappa(a, rng.integers(1, 6, 200))) < 0.2
    assert np.isnan(weighted_kappa(np.full(10, 3), np.full(10, 3)))
    assert np.isnan(spearman(np.full(10, 3), a[:10]))


def test_agreement_table_columns_and_values():
    idx = pd.Index([f"S01-C{i:02d}" for i in range(1, 21)], name="cup_id")
    a = pd.DataFrame({d: np.tile([1, 2, 3, 4, 5], 4) for d in DIMS}, index=idx)
    b = a.copy()
    b["overall"] = np.clip(a["overall"] + 1, 1, 5)
    t = agreement_table(a, b, DIMS, n_boot=200, seed=1).set_index("dimension")
    assert list(t.columns) == ["n", "kappa_w", "kappa_lo", "kappa_hi", "spearman", "mad", "exact", "within1"]
    assert t.at["contrast", "kappa_w"] == 1.0 and t.at["contrast", "mad"] == 0.0
    assert t.at["overall", "mad"] == 0.8 and t.at["overall", "within1"] == 1.0
    assert t.at["overall", "kappa_lo"] <= t.at["overall", "kappa_w"] <= t.at["overall", "kappa_hi"]


def test_agreement_script_on_fake_labels(project, cfg):
    log = pd.concat(
        [plan_session(pd.DataFrame(columns=["cup_id"]), cfg, f"S0{s}", "A", 1, "2026-11-25") for s in range(1, 8)],
        ignore_index=True,
    )
    write_cup_log(log, cfg_path(cfg, "cup_log"))
    rng = np.random.default_rng(7)
    for i, cup in enumerate(log["cup_id"]):
        tiny_jpeg(cfg_path(cfg, "raw") / f"{cup}.jpg")
        truth = rng.integers(1, 6)
        for rater in ("A", "B"):
            scores = {d: int(np.clip(truth + rng.integers(-1, 2), 1, 5)) for d in DIMS}
            unusable = "y" if (rater == "B" and i == 4) else "n"
            append_label(
                labels_path(cfg, rater),
                {"cup_id": cup, **scores, "unusable": unusable, "comment": "", "timestamp": "2026-11-26T09:00:00-08:00"},
                DIMS,
            )

    r = run_script("scripts/agreement.py", project)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "used: 41" in r.stdout and "Overall weighted kappa" in r.stdout
    table = pd.read_csv(cfg_path(cfg, "tables_dir") / "agreement.csv")
    assert list(table["dimension"]) == DIMS and (table["n"] == 41).all()
    assert table["kappa_w"].between(-1, 1).all()
    tex = (cfg_path(cfg, "tables_dir") / "agreement.tex").read_text()
    assert r"\begin{tabular}" in tex and r"kappa\_w" in tex
    assert (cfg_path(cfg, "qa_dir") / "disagreements.csv").exists()

    r = run_script("scripts/agreement.py", project, "--subset", "calibration")
    assert r.returncode == 0, r.stdout + r.stderr
    cal = pd.read_csv(cfg_path(cfg, "tables_dir") / "agreement_calibration.csv")
    assert (cal["n"] == 29).all()  # first 30 cups minus the one rater B marked unusable


def test_agreement_script_with_no_labels(project):
    r = run_script("scripts/agreement.py", project)
    assert r.returncode == 1 and "Not enough" in r.stdout
