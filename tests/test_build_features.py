"""build_features.py end to end on synthetic photos (build step 5)."""

from __future__ import annotations

import math

import pandas as pd
from conftest import run_script, write_photo
from synth import cup_photo

from latte.features import FEATURES
from latte.io import cfg_path


def test_build_features_end_to_end(project, cfg, photos):
    r = run_script("scripts/build_features.py", project)
    assert r.returncode == 0, r.stderr
    path = cfg_path(cfg, "features_dir") / "features.csv"
    df = pd.read_csv(path).set_index("cup_id")
    assert list(df.index) == ["S00-C01", "S00-C02", "S00-C03"]
    assert list(df.columns[: len(FEATURES)]) == FEATURES
    good = df.loc["S00-C01"]
    assert good["symmetry_iou"] > 0.9 and good["centre_offset"] < 0.05 and good["contrast_dL"] > 20
    assert good["circle_source"] == "auto" and good["circle_checked"] == "n"
    assert math.isnan(df.loc["S00-C03", "contrast_dL"])  # no cup, no milk: missing, not made up
    qa = cfg_path(cfg, "qa_dir")
    assert (qa / "segment_01.jpg").exists() and (qa / "seg_compare_01.jpg").exists()
    assert "provisional" in r.stdout and "Otsu vs k-means" in r.stdout


def test_otsu_method_from_config_gives_similar_features(project, cfg, photos):
    import yaml

    p = project / "configs" / "default.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["segment"]["method"] = "otsu"
    p.write_text(yaml.safe_dump(raw))
    assert run_script("scripts/build_features.py", project).returncode == 0
    df = pd.read_csv(cfg_path(cfg, "features_dir") / "features.csv").set_index("cup_id")
    assert df.loc["S00-C01", "symmetry_iou"] > 0.9


def test_dev_mode_writes_overlays_but_no_features(project, cfg):
    img, _ = cup_photo()
    write_photo(cfg_path(cfg, "dev_external") / "valid" / "h1.jpg", img)
    r = run_script("scripts/build_features.py", project, "--dev")
    assert r.returncode == 0, r.stderr
    assert (cfg_path(cfg, "qa_dir") / "dev" / "segment_01.jpg").exists()
    assert not (cfg_path(cfg, "features_dir") / "features.csv").exists()


def test_bad_method_is_an_error(project):
    import yaml

    p = project / "configs" / "default.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["segment"]["method"] = "watershed"
    p.write_text(yaml.safe_dump(raw))
    tiny = cup_photo(w=200, h=150)[0]
    write_photo(project / "data" / "raw" / "S00-C01.jpg", tiny)
    r = run_script("scripts/build_features.py", project)
    assert r.returncode == 1 and "segment.method" in r.stderr
