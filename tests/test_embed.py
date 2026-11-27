"""Embeddings: rotation averaging, fallback, cache (build step 7). Random weights, so nothing is downloaded."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from conftest import run_script
from synth import heart_crop

import latte.embed as E
from latte.embed import (
    Embedder,
    WeightsUnavailable,
    cache_paths,
    embed_params,
    is_download_error,
    load_embedder,
    load_embeddings,
    rotation_average,
    run,
)
from latte.geometry import append_circle, circles_path
from latte.io import cfg_path, load_config

TINY = {"model": "resnet18", "pretrained": False, "threads": 2}


def use_tiny(project, **extra) -> dict:
    p = project / "configs" / "default.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["embed"] |= {**TINY, **extra}
    p.write_text(yaml.safe_dump(raw))
    return load_config(p)


@pytest.fixture(scope="module")
def tiny() -> Embedder:
    return load_embedder(embed_params({"embed": TINY}))


def download_failure(*args, **kwargs):
    from huggingface_hub.errors import LocalEntryNotFoundError

    raise LocalEntryNotFoundError("offline: cannot reach huggingface.co")


def test_timm_model_shapes(tiny):
    assert tiny.name == "resnet18" and not tiny.fell_back
    assert tiny.size == (224, 224) and tiny.dim == 512
    crop = heart_crop()[0]
    assert tiny([crop, crop]).shape == (2, 512)


def test_rotation_average_is_invariant_to_90_degree_turns(tiny):
    crop = heart_crop(angle_deg=20, offset=(0.2, -0.1))[0]
    rots = [0, 90, 180, 270]
    a = rotation_average(tiny, crop, rots)
    for k in (1, 2, 3):
        assert np.allclose(a, rotation_average(tiny, np.rot90(crop, k).copy(), rots), rtol=1e-4, atol=1e-5)
    # A single orientation is not invariant, which is why we average.
    assert not np.allclose(tiny([crop])[0], tiny([np.rot90(crop).copy()])[0], rtol=1e-3)


def test_rotations_must_be_multiples_of_90(tiny):
    with pytest.raises(ValueError, match="multiples of 90"):
        rotation_average(tiny, heart_crop(size=64)[0], [0, 45])


def test_unknown_model_name_is_an_error_not_a_fallback():
    with pytest.raises(ValueError, match="not a timm model name"):
        load_embedder(embed_params({"embed": {"model": "vit_small_patch14_dinov2.lvd142", "pretrained": False}}))


def test_download_failure_falls_back_to_torchvision_resnet50(monkeypatch):
    monkeypatch.setattr(E, "_timm_embedder", download_failure)
    emb = load_embedder(embed_params({"embed": {"pretrained": False}}))
    assert emb.fell_back and emb.name == "resnet50.tv_in1k" and emb.dim == 2048
    assert "LocalEntryNotFoundError" in emb.error
    assert emb([heart_crop()[0]]).shape == (1, 2048)


def test_bug_in_model_creation_is_not_hidden_by_fallback(monkeypatch):
    def bug(*a, **k):
        raise KeyError("a programming error")

    monkeypatch.setattr(E, "_timm_embedder", bug)
    with pytest.raises(KeyError):
        load_embedder(embed_params({"embed": {"pretrained": False}}))


def test_both_models_unavailable_gives_one_clear_error(monkeypatch):
    import urllib.error

    def no_net(*a, **k):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(E, "_timm_embedder", download_failure)
    monkeypatch.setattr(E, "_torchvision_resnet50", no_net)
    with pytest.raises(WeightsUnavailable, match="--fetch"):
        load_embedder(embed_params({"embed": {"pretrained": False}}))
    assert is_download_error(urllib.error.URLError("x")) and not is_download_error(KeyError("x"))


def test_cache_reuse_updates_and_staleness(project, cfg, photos, capsys):
    cfg = use_tiny(project)
    assert run(cfg) == 0
    assert "3 computed, 0 reused" in capsys.readouterr().out
    npy, ids, meta = cache_paths(cfg, "resnet18", pretrained=False)
    assert npy.name == "emb_resnet18_random.npy"  # random weights never take the real model's name
    first = np.load(npy)
    assert first.shape == (3, 512) and np.isfinite(first).all()
    info = json.loads(meta.read_text())
    assert info["dim"] == 512 and info["rotations"] == [0, 90, 180, 270] and not info["pretrained"]

    assert run(cfg) == 0
    out = capsys.readouterr().out
    assert "up to date" in out and "wasn't loaded" in out  # nothing to compute: no network needed
    assert np.array_equal(np.load(npy), first)

    append_circle(circles_path(cfg), {"cup_id": "S00-C01", "cx": 515, "cy": 380, "r": 120, "source": "manual",
                                      "checked_by": "A", "img_w": 1024, "img_h": 768, "points": "", "timestamp": "t"})
    (cfg_path(cfg, "raw") / "S00-C03.jpg").unlink()
    assert run(cfg) == 0
    out = capsys.readouterr().out
    assert "1 computed, 1 reused, 1 dropped" in out and "1 cup use a circle nobody has checked" in out
    df = load_embeddings(cfg)
    assert list(df.index) == ["S00-C01", "S00-C02"] and df.shape[1] == 512 and df.attrs["model"] == "resnet18"
    assert not np.allclose(df.loc["S00-C01"].to_numpy(), first[0])
    assert np.allclose(df.loc["S00-C02"].to_numpy(), first[1])

    # A new photo makes the cache stale: load_embeddings refuses rather than silently missing a cup.
    import shutil

    shutil.copy(cfg_path(cfg, "raw") / "S00-C02.jpg", cfg_path(cfg, "raw") / "S00-C09.jpg")
    with pytest.raises(FileNotFoundError, match="S00-C09"):
        load_embeddings(cfg)


def test_seed_is_part_of_the_key_for_random_weights(project, cfg, photos, capsys):
    cfg = use_tiny(project)
    run(cfg)
    capsys.readouterr()
    cfg["seed"] = 7
    run(cfg)
    assert "3 computed, 0 reused" in capsys.readouterr().out


def test_corrupt_cache_is_rebuilt_not_a_crash(project, cfg, photos, capsys):
    cfg = use_tiny(project)
    run(cfg)
    npy, ids, meta = cache_paths(cfg, "resnet18", pretrained=False)
    npy.write_bytes(npy.read_bytes()[:100])
    with pytest.raises(FileNotFoundError, match="unreadable"):
        load_embeddings(cfg)
    capsys.readouterr()
    assert run(cfg) == 0
    out = capsys.readouterr().out
    assert "rebuilding" in out and "3 computed" in out
    # ids out of step with the .npy (an interrupted write) is caught by the digest.
    ids.write_text(ids.read_text().replace("S00-C01", "S00-C07"))
    assert load_cache_problem(cfg) .startswith("files don't match")


def load_cache_problem(cfg) -> str:
    return E.load_cache(cfg, "resnet18", False)[3]


def test_fallback_run_keeps_main_cache_and_is_loud(project, cfg, photos, capsys, monkeypatch):
    cfg = use_tiny(project)
    run(cfg)  # main model cache, complete
    monkeypatch.setattr(E, "_timm_embedder", download_failure)
    assert run(cfg) == 0  # the cache is current, so the (failing) download is never attempted
    assert "up to date" in capsys.readouterr().out
    assert run(cfg, force=True) == 0  # force a recompute: now the download fails and it falls back
    out = capsys.readouterr().out
    assert out.count("could not be downloaded") == 2  # at the start and repeated at the end
    assert cache_paths(cfg, "resnet50.tv_in1k", False)[0].exists()
    # The complete main-model cache still wins.
    assert load_embeddings(cfg).attrs["model"] == "resnet18"
    # Only when the main cache is unusable does the fallback get used, with a warning.
    cache_paths(cfg, "resnet18", False)[2].unlink()
    df = load_embeddings(cfg)
    assert df.attrs["model"] == "resnet50.tv_in1k" and df.attrs["fell_back"] and df.shape[1] == 2048
    assert "fallback" in capsys.readouterr().err


def test_cli_smoke_and_no_photos(project, cfg, photos):
    use_tiny(project)
    r = run_script("src/latte/embed.py", project)
    assert r.returncode == 0, r.stderr
    assert "RANDOM weights" in r.stdout and "tests only" in r.stdout


def test_embed_with_no_photos(project, capsys):
    assert run(use_tiny(project)) == 0
    assert "make ingest" in capsys.readouterr().out


def test_load_embeddings_without_cache(cfg):
    with pytest.raises(FileNotFoundError, match="make embed"):
        load_embeddings(cfg)


def test_load_embeddings_refuses_out_of_date_rows(project, cfg, photos):
    cfg = use_tiny(project)
    run(cfg)
    assert load_embeddings(cfg).shape[0] == 3
    # A circle is fixed after make embed: the cached row for that cup no longer matches.
    append_circle(circles_path(cfg), {"cup_id": "S00-C02", "cx": 600, "cy": 420, "r": 101, "source": "manual",
                                      "checked_by": "A", "img_w": 1024, "img_h": 768, "points": "", "timestamp": "t"})
    with pytest.raises(FileNotFoundError, match="out of date.*S00-C02"):
        load_embeddings(cfg)
    assert load_embeddings(cfg, verify=False).shape[0] == 3
