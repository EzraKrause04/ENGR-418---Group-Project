"""QA sheets script and the circle-check tool's logic, on synthetic photos (build step 4)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import run_script
from PIL import Image
from synth import cup_photo

from latte.geometry import append_circle, circles_path, read_circles
from latte.io import REPO_ROOT, cfg_path, empty_cup_log, write_cup_log


def write_photo(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, quality=92)


@pytest.fixture
def photos(cfg) -> dict[str, object]:
    """Three cups in data/raw: a clean one, an off-centre one, and a photo with no cup."""
    raw = cfg_path(cfg, "raw")
    truth = {}
    for cup, kw in {"S00-C01": {}, "S00-C02": {"offset": (0.1, 0.05), "r_frac": 0.13, "seed": 1}}.items():
        img, t = cup_photo(**kw)
        write_photo(raw / f"{cup}.jpg", img)
        truth[cup] = t
    rng = np.random.default_rng(0)
    write_photo(raw / "S00-C03.jpg", np.clip(rng.normal(140, 12, (768, 1024, 3)), 0, 255).astype(np.uint8))
    return truth


def test_qa_sheets_end_to_end(project, cfg, photos):
    append_circle(circles_path(cfg), {"cup_id": "S00-C02", "cx": 600, "cy": 420, "r": 100, "source": "manual",
                                      "checked_by": "A", "img_w": 1024, "img_h": 768, "points": "", "timestamp": "t"})
    r = run_script("scripts/qa_sheets.py", project)
    assert r.returncode == 0, r.stderr
    qa = cfg_path(cfg, "qa_dir")
    assert (qa / "circles_01.jpg").exists() and (qa / "crops_01.jpg").exists()
    st = pd.read_csv(qa / "circles.csv", dtype=str, keep_default_na=False).set_index("cup_id")
    assert list(st.index) == ["S00-C01", "S00-C02", "S00-C03"]
    t = photos["S00-C01"]
    assert float(st.at["S00-C01", "r"]) == pytest.approx(t.r, rel=0.015)
    assert st.at["S00-C01", "source"] == "auto" and st.at["S00-C01", "checked"] == "n"
    assert st.at["S00-C02", "source"] == "manual" and float(st.at["S00-C02", "cx"]) == 600
    assert st.at["S00-C03", "low_confidence"] == "y"
    assert "low confidence: S00-C03" in r.stdout and "make circles" in r.stdout


def test_qa_sheets_skip_log_unusable_and_remove_stale_pages(project, cfg, photos):
    log = empty_cup_log()
    log.loc[0] = ["S00-C03", "S00", "2026-11-24", "A", "good", "on_time", "", "", "", "S00-C03.jpg", "n", "no heart"]
    write_cup_log(log, cfg_path(cfg, "cup_log"))
    qa = cfg_path(cfg, "qa_dir")
    qa.mkdir(parents=True)
    (qa / "circles_09.jpg").write_bytes(b"stale")
    r = run_script("scripts/qa_sheets.py", project, "--per-sheet", "1")
    assert r.returncode == 0, r.stderr
    st = pd.read_csv(qa / "circles.csv", dtype=str, keep_default_na=False)
    assert "S00-C03" not in set(st["cup_id"])
    assert sorted(p.name for p in qa.glob("circles_*.jpg")) == ["circles_01.jpg", "circles_02.jpg"]


def test_qa_sheets_dev_mode_writes_to_its_own_folder(project, cfg):
    img, _ = cup_photo()
    write_photo(cfg_path(cfg, "dev_external") / "train" / "heart_001.jpg", img)
    r = run_script("scripts/qa_sheets.py", project, "--dev")
    assert r.returncode == 0, r.stderr
    st = pd.read_csv(cfg_path(cfg, "qa_dir") / "dev" / "circles.csv", dtype=str, keep_default_na=False)
    assert list(st["cup_id"]) == ["train/heart_001"]
    assert not (cfg_path(cfg, "qa_dir") / "circles.csv").exists()


def test_qa_sheets_with_no_photos(project):
    r = run_script("scripts/qa_sheets.py", project)
    assert r.returncode == 0 and "make ingest" in r.stdout


# --- circle-check tool logic (the browser side was checked by hand) ----------


@pytest.fixture(scope="module")
def annotate():
    spec = importlib.util.spec_from_file_location("annotate_circle", REPO_ROOT / "app" / "annotate_circle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_review_order_save_and_resume(annotate, cfg, photos):
    cs = annotate.CircleSession(cfg)
    order = cs.order("Not checked yet")
    assert order[0] == "S00-C03"  # low confidence first
    assert cs.order("Low confidence only") == ["S00-C03"]
    assert cs.order("All photos") == ["S00-C01", "S00-C02", "S00-C03"]

    cs.save("S00-C01", cs.auto["S00-C01"], "accepted", "A", [])
    t = photos["S00-C02"]
    pts = [(t.cx + t.r * np.cos(a), t.cy + t.r * np.sin(a)) for a in (0.3, 2.2, 4.4)]
    fit, err = annotate.fitted(pts)
    assert err == "" and fit is not None and fit.r == pytest.approx(t.r, abs=0.01)
    cs.save("S00-C02", fit, "manual", "B", pts)

    assert cs.order("Not checked yet") == ["S00-C03"]
    saved = read_circles(circles_path(cfg))
    assert saved.loc["S00-C01", "source"] == "accepted" and saved.loc["S00-C01", "checked_by"] == "A"
    assert saved.loc["S00-C02", "source"] == "manual" and saved.loc["S00-C02", "points"].count(";") == 2
    # A fresh session (e.g. the next day) sees the same state.
    assert annotate.CircleSession(cfg).order("Not checked yet") == ["S00-C03"]


def test_fitted_reports_bad_clicks(annotate):
    assert annotate.fitted([(0, 0), (1, 1)]) == (None, "")
    fit, err = annotate.fitted([(0, 0), (1, 1), (2, 2)])
    assert fit is None and "collinear" in err
