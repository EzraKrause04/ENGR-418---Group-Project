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


def polygon_centroid(pts: np.ndarray) -> tuple[float, float]:
    """Area centroid of a simple polygon (shoelace formula)."""
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2
    return float(((x + x1) * cross).sum() / (6 * a)), float(((y + y1) * cross).sum() / (6 * a))


def heart_crop(
    size: int = 512,
    heart_frac: float = 0.55,
    offset: tuple[float, float] = (0.0, 0.0),
    angle_deg: float = 0.0,
    milk: tuple[int, int, int] = MILK,
    crema: tuple[int, int, int] = CREMA,
    blur: float = 0.0,
    noise: float = 1.5,
    dots: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A preprocessed-style crop (section 8.5): a crema disk with a white, symmetric heart, black outside.

    The heart is ``heart_frac * size`` wide. Its area centroid sits at the disk
    centre plus ``offset * R`` (R = size / 2), and it is rotated by
    ``angle_deg``. ``blur`` is a Gaussian sigma in px, and ``dots`` adds that
    many small dark dots (radius 2-3 px). Returns ``(rgb, D, M_true)``.
    """
    from latte.preprocess import disk_mask

    rng = np.random.default_rng(seed)
    D = disk_mask(size)
    c = (size - 1) / 2
    R = size / 2
    pts = heart_polygon(0, 0, heart_frac * size, angle_deg)
    gx, gy = polygon_centroid(pts)
    pts = pts - [gx, gy] + [c + offset[0] * R, c + offset[1] * R]
    img = np.zeros((size, size, 3), np.uint8)
    img[D] = crema
    fill_polygon(img, pts, milk)
    truth = np.zeros((size, size), np.uint8)
    cv2.fillPoly(truth, [np.round(pts * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    inner = cv2.erode(D.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(bool)
    ys, xs = np.nonzero(inner)
    for i in rng.choice(len(xs), size=dots, replace=False) if dots else []:
        cv2.circle(img, (int(xs[i]), int(ys[i])), int(rng.integers(2, 4)), (92, 62, 40), -1, cv2.LINE_AA)
    out = img.astype(np.float32)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    out += rng.normal(0, noise, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[~D] = 0
    return out, D, truth.astype(bool) & D


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
    moves its area centroid from the liquid centre, as a fraction of the liquid
    radius.
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
    # Centre the heart by its area centroid, like heart_crop, so "centred" means centre_offset = 0.
    pts = heart_polygon(0, 0, heart_frac * r)
    gx, gy = polygon_centroid(pts)
    fill_polygon(img, pts - [gx, gy] + [cx + heart_offset[0] * r, cy + heart_offset[1] * r], milk)
    out = img.astype(np.float32)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    out += rng.normal(0, noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), Truth(cx, cy, r)
