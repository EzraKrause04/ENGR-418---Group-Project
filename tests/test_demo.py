"""Inference pipeline behind the demo, and the demo UI's rendering (build step 10)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import yaml
from conftest import run_script
from fake import fake_study, small_grids
from PIL import Image
from synth import cup_photo

from latte.feedback import load_bundle
from latte.inference import Embedder, analyse, as_ingested
from latte.io import REPO_ROOT, load_config


def demo_module():
    spec = importlib.util.spec_from_file_location("demo", REPO_ROOT / "app" / "demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A fake study trained and fitted with handcrafted-only models (no embeddings needed at inference)."""
    import shutil

    root = tmp_path_factory.mktemp("demo")
    (root / "configs").mkdir()
    (root / "docs").mkdir()
    shutil.copy(REPO_ROOT / "configs" / "default.yaml", root / "configs" / "default.yaml")
    shutil.copy(REPO_ROOT / "docs" / "rubric.md", root / "docs" / "rubric.md")
    small_grids(root)
    p = root / "configs" / "default.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["feedback"]["score_model"] = "ridge_hand"
    p.write_text(yaml.safe_dump(raw))
    cfg = load_config(p)
    fake_study(cfg, n_sessions=6, per_session=12)
    assert run_script("scripts/train_eval.py", root, "--exp", "E2,E3").returncode == 0
    assert run_script("scripts/fit_final.py", root).returncode == 0
    return cfg, load_bundle(cfg)


def test_as_ingested_matches_ingest(tmp_path):
    from latte.io import strip_and_resize

    img, _ = cup_photo(w=2048, h=1536)
    src = tmp_path / "big.jpg"
    Image.fromarray(img).save(src, quality=95)
    strip_and_resize(src, tmp_path / "ing.jpg", 1024)
    with Image.open(src) as im:
        a = as_ingested(im)
    with Image.open(tmp_path / "ing.jpg") as im:
        b = np.asarray(im.convert("RGB"))
    assert a.shape == b.shape == (768, 1024, 3)
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 2.0


def test_as_ingested_applies_exif_rotation_and_handles_odd_modes():
    img, _ = cup_photo(w=400, h=300)
    im = Image.fromarray(img)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    import io

    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    buf.seek(0)
    assert as_ingested(Image.open(buf)).shape == (1024, 768, 3)
    assert as_ingested(Image.fromarray(img).convert("RGBA")).shape == (768, 1024, 3)
    assert as_ingested(Image.fromarray(img).convert("L")).shape == (768, 1024, 3)


def test_analyse_with_a_trained_bundle(trained):
    cfg, bundle = trained
    img, _ = cup_photo(heart_offset=(0.3, 0.1))
    an = analyse(Image.fromarray(img), cfg, bundle, Embedder(cfg, bundle))
    assert an.circle_ok and an.feedback is not None and an.feedback.measurable
    assert an.photo_overlay.ndim == 3 and an.crop_overlay.shape == (512, 512, 3)
    assert an.features["centre_offset"] > 0.2
    assert "±" in an.feedback.score_text and an.feedback.fixes_text
    score, fixes, milk, detail = demo_module().render(an)
    assert "/ 5" in score and "typical error ±" in score and "Rim confidence" in detail
    assert fixes.startswith("<h3") and "Top fixes" in fixes


def test_analyse_without_a_bundle_shows_overlays_only(cfg):
    img, _ = cup_photo()
    an = analyse(Image.fromarray(img), cfg, None)
    assert an.feedback is None and np.isfinite(an.features["symmetry_iou"])
    score, fixes, milk, _ = demo_module().render(an)
    assert "make final" in score and milk == "" and fixes == ""


def test_no_rim_means_no_score_fixes_or_diagnosis(trained):
    cfg, bundle = trained
    rng = np.random.default_rng(0)
    noise = np.clip(rng.normal(140, 10, (768, 1024, 3)), 0, 255).astype(np.uint8)
    for img in (noise, cup_photo(r_frac=0.06, heart_offset=(0.2, 0))[0], np.full((600, 800, 3), 250, np.uint8)):
        an = analyse(Image.fromarray(img), cfg, bundle)
        assert not an.circle_ok and an.status == "no_rim" and an.feedback is None
        score, fixes, milk, _ = demo_module().render(an)
        assert "No score" in score and "⚠️" in score and fixes == "" and milk == ""


def test_sixteen_bit_png_is_rescaled_not_whitened(tmp_path):
    img, _ = cup_photo()
    gray = np.asarray(Image.fromarray(img).convert("L"), dtype=np.uint16) * 257
    path = tmp_path / "deep.png"
    Image.fromarray(gray).save(path)
    out = as_ingested(str(path))
    assert 100 < out.mean() < 200  # not clipped to white


def test_ingested_jpeg_is_not_recompressed(tmp_path):
    from latte.io import load_rgb, strip_and_resize

    src = tmp_path / "phone.jpg"
    Image.fromarray(cup_photo(w=2000, h=1500)[0]).save(src, quality=95)
    ing = tmp_path / "S01-C01.jpg"
    strip_and_resize(src, ing, 1024)
    assert np.array_equal(as_ingested(str(ing)), load_rgb(ing))


def test_feedback_failure_keeps_overlays(trained):
    cfg, bundle = trained
    broken = dict(bundle) | {"score": {"pipeline": None, "error": 0.4}}
    an = analyse(Image.fromarray(cup_photo()[0]), cfg, broken)
    assert an.status == "error" and an.crop_overlay is not None and any("No score" in n for n in an.notes)
    score, *_ = demo_module().render(an)
    assert "No score" in score


def test_withheld_milk_is_left_out_entirely(trained):
    cfg, bundle = trained
    b = dict(bundle) | {"milk": None, "milk_withheld_reason": "B1 not above chance"}
    an = analyse(Image.fromarray(cup_photo()[0]), cfg, b)
    _, _, milk, detail = demo_module().render(an)
    assert milk == "" and "not above chance" in detail


def test_demo_ui_builds(trained):
    cfg, bundle = trained
    mod = demo_module()
    assert mod.build_ui(cfg, bundle, []) is not None
    paths, label = mod.examples(cfg, None)
    assert paths == [] and "train" in label  # the fake study has no photos; label warns these are training cups
