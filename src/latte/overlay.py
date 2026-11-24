"""Drawing helpers shared by the QA sheets, the circle tool and (later) the demo.

A ``View`` is a zoomed, square window onto a photo. It converts between photo
pixel coordinates and view pixel coordinates, using the same pixel-centre
convention as ``latte.preprocess.crop_square``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from latte.geometry import Circle
from latte.preprocess import crop_square

GREEN = (46, 160, 67)  # checked by a student
AMBER = (245, 166, 35)  # automatic, not yet checked
RED = (220, 50, 47)  # automatic, low confidence
CYAN = (0, 190, 255)  # automatic circle shown for reference
WHITE = (255, 255, 255)  # analysis edge (radius after the 4% shrink)
MAGENTA = (255, 0, 200)  # clicked points


@dataclass(frozen=True)
class View:
    """Window onto a photo: view pixel u shows photo x = (u + 0.5 - w/2) / zoom + cx."""

    cx: float
    cy: float
    zoom: float
    w: int
    h: int

    def to_view(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.cx) * self.zoom + self.w / 2 - 0.5, (y - self.cy) * self.zoom + self.h / 2 - 0.5

    def to_photo(self, u: float, v: float) -> tuple[float, float]:
        return (u + 0.5 - self.w / 2) / self.zoom + self.cx, (v + 0.5 - self.h / 2) / self.zoom + self.cy


def zoom_view(rgb: np.ndarray, circle: Circle, size: int, margin: float = 1.35) -> tuple[np.ndarray, View]:
    """Square view of half-width ``margin * r`` around the circle, resampled to size x size."""
    half = margin * circle.r
    img = crop_square(rgb, circle.cx, circle.cy, half, size)
    return img, View(circle.cx, circle.cy, size / (2 * half), size, size)


def full_view(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, View]:
    """The whole photo, scaled so its longest side is at most ``max_side``."""
    H, W = rgb.shape[:2]
    z = min(1.0, max_side / max(H, W))
    w, h = max(1, round(W * z)), max(1, round(H * z))
    img = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA) if z < 1 else rgb.copy()
    return img, View((W - 1) / 2, (H - 1) / 2, w / W, w, h)


def draw_circle(img: np.ndarray, view: View, c: Circle, color: tuple[int, int, int], width: int = 2, r: float | None = None) -> None:
    """Draw circle c (photo coordinates) onto a view image in place, with sub-pixel precision."""
    u, v = view.to_view(c.cx, c.cy)
    rad = (c.r if r is None else r) * view.zoom
    cv2.circle(img, (round(u * 16), round(v * 16)), max(1, round(rad * 16)), color, width, cv2.LINE_AA, shift=4)


def draw_cross(img: np.ndarray, view: View, x: float, y: float, color: tuple[int, int, int], size: int = 6, width: int = 2) -> None:
    u, v = (round(t) for t in view.to_view(x, y))
    cv2.line(img, (u - size, v), (u + size, v), color, width, cv2.LINE_AA)
    cv2.line(img, (u, v - size), (u, v + size), color, width, cv2.LINE_AA)


def font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without FreeType
        return ImageFont.load_default()


def contact_sheet(
    tiles: list[np.ndarray],
    captions: list[str],
    caption_colors: list[tuple[int, int, int]],
    cols: int,
    header: str = "",
    pad: int = 6,
) -> Image.Image:
    """Grid of equal-size RGB tiles with a caption strip under each and an optional header line."""
    if not tiles:
        raise ValueError("no tiles")
    th, tw = tiles[0].shape[:2]
    cap_h = 22
    rows = -(-len(tiles) // cols)
    head_h = 30 if header else 0
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, head_h + rows * (th + cap_h + pad) + pad), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    if header:
        draw.text((pad + 2, 8), header, fill=(230, 230, 230), font=font(14))
    f = font(14)
    for i, (tile, cap, col) in enumerate(zip(tiles, captions, caption_colors)):
        x = pad + (i % cols) * (tw + pad)
        y = head_h + pad + (i // cols) * (th + cap_h + pad)
        sheet.paste(Image.fromarray(tile), (x, y))
        draw.rectangle([x, y + th, x + tw - 1, y + th + cap_h - 1], fill=col)
        draw.text((x + 5, y + th + 3), cap, fill=(0, 0, 0) if sum(col) > 400 else (255, 255, 255), font=f)
    return sheet
