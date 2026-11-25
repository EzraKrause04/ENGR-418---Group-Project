"""Crop, mask and resize the liquid surface to a fixed square (section 8.2).

Given the liquid circle (cx, cy, r) in image pixels:

1. Shrink the radius: r' = r * (1 - radius_shrink), with a default of 4%, so
   the rim's edge and its highlight stay out of the analysis.
2. Take the square [cx - r', cx + r'] x [cy - r', cy + r'] and resample it to
   ``size`` x ``size`` (512 by default). Parts of the square outside the photo
   are filled with black.
3. Mask the disk inscribed in that square: output pixel (u, v) is inside when
   (u + 0.5 - size/2)^2 + (v + 0.5 - size/2)^2 <= (size/2)^2, *and* it lies
   inside the photo (a cup cut off by the frame edge loses that part). Pixels
   outside the mask are set to black.

``scale`` = size / (2 r') is the number of output pixels per photo pixel. A
value above 1 means the crop was upsampled, i.e. the cup covered fewer than
``size`` pixels in the photo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from latte.geometry import Circle

PREPROCESS_DEFAULTS: dict[str, Any] = {"size": 512, "radius_shrink": 0.04}


@dataclass(frozen=True)
class Prepared:
    """A cup ready for segmentation and features."""

    crop: np.ndarray  # (size, size, 3) uint8 RGB, black outside the disk
    mask: np.ndarray  # (size, size) bool, True inside the analysis disk and inside the photo
    circle: Circle  # the liquid circle in photo pixels, before shrinking
    scale: float  # output px per photo px


def preprocess_params(cfg: dict[str, Any] | None) -> dict[str, Any]:
    return {**PREPROCESS_DEFAULTS, **((cfg or {}).get("preprocess") or {})}


def disk_mask(size: int) -> np.ndarray:
    """Boolean disk inscribed in a size x size square (pixel centres at index + 0.5)."""
    t = np.arange(size) + 0.5 - size / 2
    return (t[None, :] ** 2 + t[:, None] ** 2) <= (size / 2) ** 2


def crop_square(rgb: np.ndarray, cx: float, cy: float, half: float, size: int) -> np.ndarray:
    """Resample the square of half-width ``half`` around (cx, cy) to size x size.

    Output pixel u samples photo x = cx + (u + 0.5 - size/2) * (2 half / size).
    Downsampling area-averages first, so fine texture such as bubbles doesn't
    alias. Upsampling is bicubic. Outside the photo is black.
    """
    k = 2 * half / size  # photo px per output px
    src = rgb
    if k > 1:
        w, h = rgb.shape[1], rgb.shape[0]
        nw, nh = max(1, round(w / k)), max(1, round(h / k))
        src = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        sx, sy = nw / w, nh / h
        cx, cy = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
        kx, ky = k * sx, k * sy
    else:
        kx = ky = k
    off = 0.5 - size / 2
    m = np.array([[kx, 0, cx + off * kx], [0, ky, cy + off * ky]], dtype=np.float64)
    return cv2.warpAffine(
        src, m, (size, size),
        flags=(cv2.INTER_LINEAR if k > 1 else cv2.INTER_CUBIC) | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )


def prepare(rgb: np.ndarray, circle: Circle, size: int = 512, radius_shrink: float = 0.04) -> Prepared:
    """Crop, mask and resize one photo around its liquid circle (steps 1-3 above)."""
    if circle.r <= 0:
        raise ValueError("circle radius must be positive")
    r_in = circle.r * (1 - radius_shrink)
    crop = crop_square(rgb, circle.cx, circle.cy, r_in, size)
    in_photo = crop_square(np.full(rgb.shape[:2], 255, np.uint8), circle.cx, circle.cy, r_in, size) > 127
    mask = disk_mask(size) & in_photo
    crop[~mask] = 0
    return Prepared(crop, mask, circle, size / (2 * r_in))


def prepare_cfg(rgb: np.ndarray, circle: Circle, cfg: dict[str, Any] | None) -> Prepared:
    p = preprocess_params(cfg)
    return prepare(rgb, circle, int(p["size"]), float(p["radius_shrink"]))
