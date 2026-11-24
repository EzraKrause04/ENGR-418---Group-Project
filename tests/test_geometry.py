"""Liquid-circle detection, manual fitting and the saved-circle store (build step 4)."""

from __future__ import annotations

import numpy as np
import pytest
from synth import CUP, WALL, cup_photo

from latte.geometry import (
    Circle,
    append_circle,
    detect_circle,
    edge_support,
    fit_circle,
    format_points,
    geometry_params,
    parse_points,
    read_circles,
    resolve_circle,
)


def errors(c: Circle, t) -> tuple[float, float]:
    """(centre error, |radius error|) as fractions of the true radius."""
    return np.hypot(c.cx - t.cx, c.cy - t.cy) / t.r, abs(c.r - t.r) / t.r


# --- fitting ---------------------------------------------------------------


def test_fit_circle_is_exact_for_three_points():
    pts = [(10 + 5 * np.cos(a), -3 + 5 * np.sin(a)) for a in (0.1, 2.0, 4.0)]
    assert fit_circle(pts) == pytest.approx((10, -3, 5), abs=1e-9)


def test_fit_circle_least_squares_with_noise():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 2 * np.pi, 200)
    pts = np.column_stack([300 + 120 * np.cos(a), 200 + 120 * np.sin(a)]) + rng.normal(0, 0.5, (200, 2))
    cx, cy, r = fit_circle(pts)
    assert (cx, cy, r) == pytest.approx((300, 200, 120), abs=0.2)


@pytest.mark.parametrize("pts", [[(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)], [(0, 0), (5, 0), (10, 1e-9)]])
def test_fit_circle_rejects_too_few_or_collinear_points(pts):
    with pytest.raises(ValueError):
        fit_circle(pts)


def test_points_round_trip():
    pts = [(1.25, 2.0), (300.0, 4.5)]
    assert parse_points(format_points(pts)) == [(1.2, 2.0), (300.0, 4.5)]
    assert parse_points("") == []


# --- detection on synthetic photos ------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"r_frac": 0.12},  # a cup that fills little of the frame (phone at 30 cm, 1x lens)
        {"r_frac": 0.35},
        {"offset": (0.18, -0.08)},
        {"heart_offset": (0.35, 0.2)},
        {"noise": 8.0, "seed": 3},
        {"w": 768, "h": 1024},
    ],
    ids=["centred", "small", "large", "off-centre", "heart-off-centre", "noisy", "portrait"],
)
def test_detects_liquid_edge_not_rim(kw):
    img, t = cup_photo(**kw)
    c = detect_circle(img)
    dc, dr = errors(c, t)
    # The lip is at 1.10 r and the outer rim at 1.28 r, so a 1.5% radius error rules them out.
    assert dc < 0.01 and dr < 0.015, (c, t)
    assert c.confidence > 0.9
    assert c.source == "auto" and not c.checked


def test_heart_touching_the_edge_still_found_but_less_confident():
    img, t = cup_photo(heart_frac=2.0, heart_offset=(0.0, 0.1))
    c = detect_circle(img)
    dc, dr = errors(c, t)
    assert dc < 0.01 and dr < 0.015
    assert c.confidence < detect_circle(cup_photo()[0]).confidence


def test_cup_cut_by_image_edge_is_found_with_lower_confidence():
    img, t = cup_photo(offset=(0.55, 0.0))
    c = detect_circle(img)
    dc, dr = errors(c, t)
    assert dc < 0.01 and dr < 0.015
    assert 0.5 < c.confidence < 0.9


def test_no_cup_gives_zero_confidence():
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(128, 10, (480, 640, 3)), 0, 255).astype(np.uint8)
    c = detect_circle(img)
    assert c.confidence < geometry_params(None)["low_confidence"]


def test_bright_inside_polarity_for_dark_cups():
    img, t = cup_photo()
    dark = img.copy()
    # Repaint the cup (rim, lip and inner wall) near-black: now the crema is the lighter side of the edge.
    is_cup = (np.abs(img.astype(int) - CUP).sum(axis=2) < 60) | (np.abs(img.astype(int) - WALL).sum(axis=2) < 60)
    dark[is_cup] = (25, 22, 20)
    p = geometry_params({"geometry": {"edge_polarity": "bright_inside"}})
    dc, dr = errors(detect_circle(dark, p), t)
    assert dc < 0.02 and dr < 0.03


def test_edge_support_is_low_off_the_edge():
    img, t = cup_photo()
    on = edge_support(img, Circle(t.cx, t.cy, t.r))
    off = edge_support(img, Circle(t.cx, t.cy, 0.8 * t.r))
    assert on > 0.9 and off < 0.2


def test_invalid_polarity_is_rejected():
    img, _ = cup_photo(w=200, h=150)
    with pytest.raises(ValueError, match="edge_polarity"):
        detect_circle(img, {"edge_polarity": "sideways"})


# --- saved circles ----------------------------------------------------------


def test_saved_circle_overrides_auto_and_last_row_wins(tmp_path):
    img, t = cup_photo()
    path = tmp_path / "circles.csv"
    base = {"cup_id": "S01-C01", "checked_by": "A", "img_w": 1024, "img_h": 768, "points": "", "timestamp": "t"}
    append_circle(path, {**base, "cx": 1, "cy": 2, "r": 3, "source": "manual"})
    append_circle(path, {**base, "cx": t.cx + 4, "cy": t.cy, "r": t.r, "source": "manual", "points": "1.0 2.0;3.0 4.0;5.0 7.0"})
    saved = read_circles(path)
    assert len(saved) == 1 and saved.loc["S01-C01", "cx"] == pytest.approx(t.cx + 4, abs=0.01)

    res = resolve_circle("S01-C01", img, saved)
    assert res.circle.source == "manual" and res.circle.checked
    assert res.circle.cx == pytest.approx(t.cx + 4, abs=0.01)
    assert res.auto is not None and errors(res.auto, t)[0] < 0.01
    assert 0 <= res.circle.confidence <= 1

    other = resolve_circle("S01-C02", img, saved)  # no saved row: auto
    assert other.circle.source == "auto" and other.note == ""


def test_saved_circle_for_a_different_image_size_is_ignored(tmp_path):
    img, _ = cup_photo()
    path = tmp_path / "circles.csv"
    append_circle(path, {"cup_id": "S01-C01", "cx": 5, "cy": 5, "r": 50, "source": "accepted",
                         "checked_by": "B", "img_w": 4032, "img_h": 3024, "points": "", "timestamp": "t"})
    res = resolve_circle("S01-C01", img, read_circles(path))
    assert res.circle.source == "auto" and "4032x3024" in res.note


def test_append_rejects_unchecked_sources(tmp_path):
    with pytest.raises(ValueError):
        append_circle(tmp_path / "c.csv", {"cup_id": "x", "cx": 1, "cy": 1, "r": 1, "source": "auto"})


def test_read_circles_missing_file_is_empty(tmp_path):
    assert read_circles(tmp_path / "nope.csv").empty
