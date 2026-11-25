"""Segmentation and handcrafted features on synthetic hearts (section 8.5, build step 5)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from synth import cup_photo, heart_crop

from latte.features import (
    FEATURE_GROUPS,
    FEATURES,
    MODEL_FEATURES,
    RUBRIC_LINK,
    boundary_roughness,
    bubble_density,
    centre_offset,
    contrast_dL,
    contrast_michelson,
    coverage,
    edge_sharpness,
    extract_features,
    n_components,
    reflect,
    symmetry,
    texture_energy,
)
from latte.geometry import detect_circle
from latte.preprocess import disk_mask, prepare
from latte.segment import clean, lab, segment


def feats(**kw) -> dict[str, float]:
    rgb, D, _ = heart_crop(**kw)
    M = segment(rgb, D, method="otsu")
    return extract_features(lab(rgb)[:, :, 0], M, D)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / (a | b).sum())


# --- the five assertions required by the brief (section 8.5) -----------------


def test_centred_heart_is_symmetric_and_centred():
    f = feats()
    assert f["symmetry_iou"] > 0.9
    assert f["centre_offset"] < 0.05


def test_shifting_the_heart_increases_centre_offset():
    base = feats()["centre_offset"]
    assert feats(offset=(0.15, 0.0))["centre_offset"] > base + 0.1
    assert feats(offset=(0.3, 0.1))["centre_offset"] > feats(offset=(0.15, 0.0))["centre_offset"]


def test_gaussian_blur_reduces_edge_sharpness():
    sharp = feats()["edge_sharpness"]
    assert feats(blur=2)["edge_sharpness"] < sharp
    assert feats(blur=5)["edge_sharpness"] < feats(blur=2)["edge_sharpness"]


def test_darkening_the_milk_reduces_contrast_dL():
    assert feats(milk=(180, 160, 135))["contrast_dL"] < feats()["contrast_dL"] - 10


def test_small_dark_dots_increase_bubble_density():
    clean_cup = feats()["bubble_density"]
    assert feats(dots=40)["bubble_density"] > clean_cup + 1.0
    assert feats(dots=120)["bubble_density"] > feats(dots=40)["bubble_density"]


# --- segmentation -----------------------------------------------------------


@pytest.mark.parametrize("method", ["otsu", "kmeans"])
@pytest.mark.parametrize("kw", [{}, {"blur": 3}, {"noise": 6}, {"milk": (190, 170, 145)}, {"angle_deg": 25}])
def test_both_methods_recover_the_heart(method, kw):
    rgb, D, truth = heart_crop(**kw)
    assert iou(segment(rgb, D, method=method), truth) > 0.95


def test_kmeans_is_deterministic_for_a_seed():
    rgb, D, _ = heart_crop(noise=8)
    assert (segment(rgb, D, method="kmeans", seed=42) == segment(rgb, D, method="kmeans", seed=42)).all()


def test_clean_drops_specks_and_stays_inside_the_disk():
    D = disk_mask(200)
    m = np.zeros_like(D)
    m[60:140, 60:140] = True  # a big blob
    m[20:23, 100:103] = True  # a 9 px speck (< 0.5% of the disk)
    m[0:5, 0:5] = True  # outside the disk
    out = clean(m, D, 3, 0.005)
    assert out[100, 100] and not out[21, 101] and not out[:5, :5].any()
    assert n_components(out) == 1


def test_unknown_method_is_rejected():
    rgb, D, _ = heart_crop(size=64)
    with pytest.raises(ValueError, match="segment.method"):
        segment(rgb, D, method="watershed")


def test_plain_crema_gives_little_or_no_milk():
    rgb, D, _ = heart_crop(heart_frac=0.0001)
    M = segment(rgb, D, method="otsu")
    assert M.sum() < 0.01 * D.sum()


# --- individual features ---------------------------------------------------


def test_contrast_values_match_the_paint():
    D = disk_mask(100)
    L = np.where(D, 40.0, 0.0)
    M = np.zeros_like(D)
    M[30:70, 30:70] = True
    L[M] = 80.0
    assert contrast_dL(L, M, D) == pytest.approx(40.0)
    assert contrast_michelson(L, M, D) == pytest.approx(40 / 120)


def test_features_are_nan_without_milk():
    D = disk_mask(64)
    L = np.where(D, 40.0, 0.0)
    M = np.zeros_like(D)
    for v in (contrast_dL(L, M, D), edge_sharpness(L, M, D), boundary_roughness(M), centre_offset(M, D),
              texture_energy(L, M), symmetry(M)[0]):
        assert math.isnan(v)
    assert coverage(M, D) == 0 and n_components(M) == 0


def test_edge_sharpness_ignores_contrast():
    # The same edge profile at lower contrast keeps about the same sharpness (gradient / contrast).
    a = feats()["edge_sharpness"]
    b = feats(milk=(200, 180, 150))["edge_sharpness"]
    assert b == pytest.approx(a, rel=0.15)


def test_boundary_roughness_disk_vs_heart_vs_ragged():
    m = disk_mask(300)
    assert boundary_roughness(m) == pytest.approx(1.0, abs=0.08)
    rgb, D, truth = heart_crop()
    heart = boundary_roughness(truth)
    assert heart > 1.1
    rng = np.random.default_rng(0)
    ragged = truth.copy()
    ys, xs = np.nonzero(cv2.morphologyEx(truth.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))
    for i in rng.choice(len(xs), 400, replace=False):
        cv2.circle(ragged.view(np.uint8), (int(xs[i]), int(ys[i])), 4, int(rng.integers(0, 2)), -1)
    assert boundary_roughness(ragged) > heart


def test_symmetry_angle_follows_rotation():
    rgb, D, truth = heart_crop(angle_deg=30)
    s_iou, ang = symmetry(truth)
    assert s_iou > 0.95 and ang == pytest.approx(120, abs=2)
    assert symmetry(heart_crop()[2])[1] == pytest.approx(90, abs=2)


def test_lopsided_shape_has_lower_symmetry():
    _, _, heart = heart_crop()
    lop = heart.copy()
    lop[:, : lop.shape[1] // 2] &= np.roll(heart, 25, axis=0)[:, : lop.shape[1] // 2]  # squash the left half
    assert symmetry(lop)[0] < symmetry(heart)[0] - 0.05


def test_reflect_twice_is_identity():
    _, _, m = heart_crop(size=128, angle_deg=17)
    back = reflect(reflect(m, (60.3, 61.7), 33), (60.3, 61.7), 33)
    assert iou(back, m) > 0.97


def test_coverage_matches_area_ratio():
    rgb, D, truth = heart_crop()
    assert coverage(truth, D) == pytest.approx(truth.sum() / D.sum())
    big = heart_crop(heart_frac=0.8)[2]
    assert coverage(big, D) > coverage(truth, D)


def test_n_components_counts_broken_patterns():
    D = disk_mask(200)
    M = np.zeros_like(D)
    M[50:90, 50:90] = True
    M[110:150, 110:150] = True
    assert n_components(M) == 2


def test_bubbles_outside_the_disk_or_on_the_edge_are_ignored():
    rgb, D, truth = heart_crop()
    L = lab(rgb)[:, :, 0]
    base = bubble_density(L, truth, D)
    L2 = L.copy()
    L2[~D] = 0  # already black: nothing to find out there
    assert bubble_density(L2, truth, D) == pytest.approx(base)


def test_texture_energy_rises_with_fine_noise():
    assert feats(noise=6)["texture_energy"] > 3 * feats(noise=1.5)["texture_energy"]


# --- the whole chain ---------------------------------------------------------


def test_photo_to_features_end_to_end():
    img, _ = cup_photo(r_frac=0.2)
    prep = prepare(img, detect_circle(img))
    M = segment(prep.crop, prep.mask)
    f = extract_features(lab(prep.crop)[:, :, 0], M, prep.mask)
    assert list(f) == FEATURES
    assert all(np.isfinite(v) for v in f.values())
    assert f["symmetry_iou"] > 0.9 and f["centre_offset"] < 0.05 and f["contrast_dL"] > 20


def test_feature_bookkeeping_is_consistent():
    grouped = [f for g in FEATURE_GROUPS.values() for f in g]
    assert sorted(grouped) == sorted(MODEL_FEATURES)
    assert set(RUBRIC_LINK) <= set(MODEL_FEATURES)
    assert "symmetry_angle" in FEATURES and "symmetry_angle" not in MODEL_FEATURES


def test_photo_noise_alone_is_not_counted_as_bubbles():
    # Noise in the 1024 px photo is enlarged ~2x by the 512 px crop to blob-sized speckles;
    # the noise-adaptive LoG threshold must not count it, but must still count real dots.
    def density(noise: float, dots: int) -> float:
        img, t = cup_photo(noise=noise, seed=3)
        rng = np.random.default_rng(3)
        for _ in range(dots):
            a, rr = rng.uniform(0, 2 * np.pi), t.r * np.sqrt(rng.uniform(0.0, 0.8))
            cv2.circle(img, (int(t.cx + rr * np.cos(a)), int(t.cy + rr * np.sin(a))), 1, (92, 62, 40), -1, cv2.LINE_AA)
        prep = prepare(img, detect_circle(img))
        return bubble_density(lab(prep.crop)[:, :, 0], segment(prep.crop, prep.mask), prep.mask)

    assert density(10, 0) < 0.5
    assert density(10, 60) > density(10, 0) + 1.0
