"""make repro's helpers (build step 13). The full clone + make all run is exercised by hand (it takes ~1 h)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import time

import pandas as pd
import pytest

from latte.io import REPO_ROOT


@pytest.fixture(scope="module")
def rc():
    spec = importlib.util.spec_from_file_location("repro_check", REPO_ROOT / "scripts" / "repro_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_variables_are_not_passed_to_the_clone(rc, monkeypatch):
    monkeypatch.setenv("MAKEFLAGS", " ARGS=--keep")
    monkeypatch.setenv("MAKELEVEL", "1")
    monkeypatch.setenv("VIRTUAL_ENV", "/x")
    env = rc.clean_env()
    assert not {"MAKEFLAGS", "MAKELEVEL", "VIRTUAL_ENV"} & set(env) and "PATH" in env


def test_inputs_policy(rc, tmp_path):
    src, clone = tmp_path / "src", tmp_path / "clone"
    for rel in ("data/raw/S01-C01.jpg", "data/cup_log.csv", "labels/rater_A.csv", "features/emb_x.npy",
                "data/raw_full/IMG_1.jpg"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text("x")
    clone.mkdir()
    copied, missing = rc.gather_inputs(src, clone, from_data=False, with_cache=False, allow_uncommitted=False)
    assert copied == ["data/raw/S01-C01.jpg"]
    assert sorted(missing) == ["data/cup_log.csv", "labels/rater_A.csv"]  # must be committed
    assert not (clone / "features" / "emb_x.npy").exists() and not (clone / "data" / "raw_full").exists()
    copied, missing = rc.gather_inputs(src, clone, from_data=True, with_cache=True, allow_uncommitted=False)
    assert "features/emb_x.npy" in copied and "data/cup_log.csv" in copied and not missing
    assert not any("raw_full" in c for c in copied)


def test_compare_tables(rc, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    pd.DataFrame({"m": ["x", "y"], "v": [1.0, float("nan")], "ok": [True, False]}).to_csv(a / "t.csv", index=False)
    pd.DataFrame({"m": ["x", "y"], "v": [1.0 + 1e-9, float("nan")], "ok": [True, False]}).to_csv(b / "t.csv", index=False)
    assert rc.compare_tables(a, b, 1e-5, 1e-6) == ([], ["t"])
    pd.DataFrame({"m": ["x", "y"], "v": [1.1, 2.0], "ok": [True, True]}).to_csv(b / "t.csv", index=False)
    pd.DataFrame({"z": [1]}).to_csv(b / "only_ref.csv", index=False)
    diffs, _ = rc.compare_tables(a, b, 1e-5, 1e-6)
    assert any("only in the reference" in d for d in diffs) and any("column v" in d for d in diffs)
    assert any("column ok" in d for d in diffs)
    assert rc.compare_tables(a, tmp_path / "missing", 1e-5, 1e-6) == ([], [])


def test_timeout_kills_the_whole_process_group(rc, tmp_path):
    marker = tmp_path / "alive"
    script = f"sh -c 'sleep 30; touch {marker}' & sleep 30"
    with pytest.raises(subprocess.TimeoutExpired):
        rc.sh(["sh", "-c", script], cwd=tmp_path, log=tmp_path / "log", timeout=1)
    time.sleep(1.5)
    out = subprocess.run(["pgrep", "-f", str(marker)], capture_output=True, text=True).stdout
    assert out.strip() == ""  # the background child died with make's group


def test_stopped_at(rc, tmp_path):
    log = tmp_path / "make_all.log"
    log.write_text("...\nmake: *** [leakcheck] Error 2\n")
    assert rc.stopped_at(log) == "leakcheck"
