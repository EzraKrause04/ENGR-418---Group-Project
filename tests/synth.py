"""Synthetic overhead latte photos for tests (sections 8.1 and 8.5).

A photo is a table, a white cup seen from above (outer rim, lip, a slightly
shaded inner wall), the brown crema disk inside it, and a white heart on the
crema. The true liquid circle is returned with the image, so detection can be
checked against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

TABLE = (150, 118, 88)
CUP = (238, 236, 230)
WALL = (205, 203, 198)
CREMA = (150, 98, 58)
MILK = (240, 232, 218)


@dataclass(frozen=True)
class Truth:
    cx: float
    cy: float
    r: float


def heart_polygon(cx: float, cy: float, size: float, angle_deg: float = 0.0, n: int = 400) -> np.ndarray:
    """Points of the classic parametric heart, mirror-symmetric about its axis.

    x = 16 sin^3 t, y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t, scaled so the
    heart is ``size`` px wide, centred on its bounding box, point down, then
    rotated by ``angle_deg``.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    x, y = x - (x.max() + x.min()) / 2, y - (y.max() + y.min()) / 2
    k = size / (x.max() - x.min())
    a = np.deg2rad(angle_deg)
    xr = (x * np.cos(a) - y * np.sin(a)) * k + cx
    yr = (x * np.sin(a) + y * np.cos(a)) * k + cy
    return np.stack([xr, yr], axis=1)


def fill_polygon(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int]) -> None:
    """Anti-aliased polygon fill at 1/16 px precision."""
    cv2.fillPoly(img, [np.round(pts * 16).astype(np.int32)], color, lineType=cv2.LINE_AA, shift=4)


def fill_disk(img: np.ndarray, cx: float, cy: float, r: float, color: tuple[int, int, int]) -> None:
    cv2.circle(img, (round(cx * 16), round(cy * 16)), round(r * 16), color, -1, lineType=cv2.LINE_AA, shift=4)


def cup_photo(
    w: int = 1024,
    h: int = 768,
    r_frac: float = 0.16,
    offset: tuple[float, float] = (0.0, 0.0),
    heart_frac: float = 1.1,
    heart_offset: tuple[float, float] = (0.0, 0.0),
    milk: tuple[int, int, int] = MILK,
    noise: float = 3.0,
    blur: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, Truth]:
    """An RGB uint8 overhead cup photo and its true liquid circle.

    ``r_frac`` is the liquid radius as a fraction of min(w, h). ``offset`` moves
    the cup from the image centre, as a fraction of min(w, h). ``heart_frac`` is
    the heart width as a fraction of the liquid radius, and ``heart_offset``
    moves it from the liquid centre, as a fraction of the liquid radius.
    """
    rng = np.random.default_rng(seed)
    m = min(w, h)
    r = r_frac * m
    cx, cy = w / 2 + offset[0] * m, h / 2 + offset[1] * m
    img = np.empty((h, w, 3), np.uint8)
    img[:] = TABLE
    fill_disk(img, cx + 0.04 * r, cy + 0.06 * r, 1.32 * r, (120, 95, 72))  # soft shadow of the cup
    fill_disk(img, cx, cy, 1.28 * r, CUP)  # outer rim
    fill_disk(img, cx, cy, 1.10 * r, WALL)  # inside of the lip, in slight shade
    fill_disk(img, cx, cy, r, CREMA)
    hx, hy = cx + heart_offset[0] * r, cy + heart_offset[1] * r
    fill_polygon(img, heart_polygon(hx, hy, heart_frac * r), milk)
    out = img.astype(np.float32)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    out += rng.normal(0, noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), Truth(cx, cy, r)
