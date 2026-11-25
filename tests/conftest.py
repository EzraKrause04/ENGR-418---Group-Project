from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from synth import cup_photo

from latte.io import DEFAULT_CONFIG, REPO_ROOT, cfg_path, load_config


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A throwaway project root holding a copy of the real config and rubric."""
    (tmp_path / "configs").mkdir()
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg["labelling"]["bootstrap_n"] = 200  # keep tests fast
    (tmp_path / "configs" / "default.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "docs").mkdir()
    shutil.copy(REPO_ROOT / "docs" / "rubric.md", tmp_path / "docs" / "rubric.md")
    return tmp_path


@pytest.fixture
def cfg(project: Path) -> dict:
    return load_config(project / "configs" / "default.yaml")


def run_script(script: str, project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a repo script against the throwaway project, as a user would."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--config", str(project / "configs" / "default.yaml"), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def tiny_jpeg(path: Path, size: tuple[int, int] = (16, 16), color=(120, 80, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


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
