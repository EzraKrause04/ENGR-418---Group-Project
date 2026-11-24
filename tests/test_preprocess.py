"""Crop, mask and resize around the liquid circle (build step 4)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from synth import cup_photo

from latte.geometry import Circle, detect_circle
from latte.preprocess import disk_mask, prepare, prepare_cfg


def test_shapes_mask_and_black_outside():
    img, t = cup_photo()
    out = prepare(img, Circle(t.cx, t.cy, t.r))
    assert out.crop.shape == (512, 512, 3) and out.crop.dtype == np.uint8
    assert out.mask.shape == (512, 512) and out.mask.dtype == bool
    assert out.mask.sum() == pytest.approx(np.pi * 256**2, rel=0.01)
    assert not out.crop[~out.mask].any()
    assert out.scale == pytest.approx(512 / (2 * 0.96 * t.r))


def test_circle_centre_maps_to_crop_centre():
    img = np.zeros((600, 800, 3), np.uint8)
    cv2.circle(img, (300, 250), 3, (255, 0, 0), -1)
    out = prepare(img, Circle(300, 250, 100), size=200, radius_shrink=0.0)
    ys, xs = np.nonzero(out.crop[:, :, 0] > 128)
    assert xs.mean() == pytest.approx(99.5, abs=0.6) and ys.mean() == pytest.approx(99.5, abs=0.6)


def test_radius_shrink_drops_a_bright_ring_at_the_rim():
    img = np.full((800, 800, 3), 60, np.uint8)
    cv2.circle(img, (400, 400), 200, (255, 255, 255), 3)  # a bright rim exactly on the circle
    bright = lambda o: (o.crop[o.mask].max(axis=1) > 200).sum()  # noqa: E731
    assert bright(prepare(img, Circle(400, 400, 200), radius_shrink=0.0)) > 50
    assert bright(prepare(img, Circle(400, 400, 200), radius_shrink=0.04)) == 0


def test_downsampling_large_cups_and_padding_past_the_edge():
    img, t = cup_photo(w=2000, h=1500, r_frac=0.4, offset=(0.15, 0.0))
    out = prepare(img, Circle(t.cx, t.cy, t.r))
    assert out.scale < 1
    # The liquid fills the disk: its mean colour is crema/milk, not table or black.
    assert out.crop[out.mask].mean() > 100
    edge = prepare(img, Circle(5, 5, 300))  # mostly off the photo: padded with black, no error
    assert edge.crop.shape == (512, 512, 3)


def test_prepare_cfg_uses_config_values():
    img, t = cup_photo()
    out = prepare_cfg(img, detect_circle(img), {"preprocess": {"size": 256, "radius_shrink": 0.1}})
    assert out.crop.shape == (256, 256, 3)
    assert out.scale == pytest.approx(256 / (2 * 0.9 * out.circle.r))


def test_disk_mask_is_symmetric():
    m = disk_mask(64)
    assert (m == m[::-1]).all() and (m == m[:, ::-1]).all() and (m == m.T).all()
