"""Liquid-surface circle: automatic detection, manual fits and the saved-circle store (section 8.1).

All coordinates are pixel indices in the ingested image (``data/raw``, 1024 px
on the longest side): ``(0, 0)`` is the centre of the top-left pixel.

Detection looks for the edge between the coffee and the cup wall, not the
outer rim:

1. Work on CIELAB lightness L* (0-100), downscaled so the longest side is
   ``work_size`` and Gaussian-blurred (``blur_sigma``).
2. ``cv2.HoughCircles`` proposes centres, with the radius limited to
   ``[radius_frac_min, radius_frac_max] * min(H, W)``. The image centre is
   always added as a candidate, because the rig centres the cup.
3. Around each centre, L* is resampled along ``n_angles`` rays. A ray has an
   edge *hit* at radius rho where the radial derivative dL*/drho is a local
   maximum above tau. Its sign follows ``edge_polarity``; the default
   ``dark_inside`` means crema darker than the cup wall. tau is the larger of
   ``min_edge`` and ``edge_frac`` times the 98th percentile of the image's
   gradient magnitude. The support s(rho) is the fraction of rays with a hit
   within +/-2 px of rho.
4. The liquid edge is the innermost support peak that reaches ``peak_ratio``
   times the strongest one. Moving out from the centre, the first ring that
   turns from dark to light is the liquid edge; the lip and outer rim lie
   further out. The heart's own edges run light to dark going outward, so
   they don't count.
5. On each ray, the hit nearest that radius is kept, and the points are fitted
   with a trimmed least-squares circle. This is repeated at full resolution
   with a narrower search window.
6. ``confidence`` = the fraction of rays with a hit within ``inlier_tol * r``
   of the final circle, i.e. how much of the circumference shows the edge.
   Below ``low_confidence`` the image is flagged for review.

``data/circles.csv`` holds circles a student has checked by eye. Each row is
either ``accepted`` (the automatic circle, frozen as it was when accepted) or
``manual`` (fitted to points clicked in ``app/annotate_circle.py``). The file
is append-only, and the last row per cup wins. A saved circle always overrides
detection.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

from latte.io import cfg_path

SOURCES = ("auto", "accepted", "manual")
CHECKED_SOURCES = ("accepted", "manual")
CIRCLE_COLUMNS = ["cup_id", "cx", "cy", "r", "source", "checked_by", "img_w", "img_h", "points", "timestamp"]

GEOMETRY_DEFAULTS: dict[str, Any] = {
    "radius_frac_min": 0.10,
    "radius_frac_max": 0.50,
    "low_confidence": 0.5,
    "edge_polarity": "dark_inside",
    "work_size": 512,
    "blur_sigma": 2.0,
    "n_angles": 360,
    "min_edge": 1.0,
    "edge_frac": 0.25,
    "min_support": 0.3,
    "peak_ratio": 0.7,
    "inlier_tol": 0.02,
    "hough_candidates": 8,
}


@dataclass(frozen=True)
class Circle:
    """A circle in image pixel coordinates.

    ``confidence`` is the edge support in [0, 1] (step 6 above). ``source`` is
    ``auto``, ``accepted`` or ``manual``.
    """

    cx: float
    cy: float
    r: float
    confidence: float = float("nan")
    source: str = "auto"

    @property
    def checked(self) -> bool:
        return self.source in CHECKED_SOURCES

    def scaled(self, k: float) -> "Circle":
        """The same circle after resizing the image by ``k``, keeping pixel centres aligned."""
        return replace(self, cx=(self.cx + 0.5) * k - 0.5, cy=(self.cy + 0.5) * k - 0.5, r=self.r * k)


def geometry_params(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """``cfg['geometry']`` with defaults filled in for any key it leaves out."""
    return {**GEOMETRY_DEFAULTS, **((cfg or {}).get("geometry") or {})}


# ---------------------------------------------------------------------------
# Circle fitting
# ---------------------------------------------------------------------------


def fit_circle(points: Iterable[Iterable[float]]) -> tuple[float, float, float]:
    """Least-squares circle through points (Kasa's algebraic fit).

    Solves min over (D, E, F) of sum_i (x_i^2 + y_i^2 + D x_i + E y_i + F)^2,
    giving centre (-D/2, -E/2) and r = sqrt(cx^2 + cy^2 - F). The fit is exact
    for three non-collinear points. Raises ``ValueError`` for fewer than three
    points or (nearly) collinear ones.
    """
    p = np.asarray(list(points), dtype=float).reshape(-1, 2)
    if len(p) < 3:
        raise ValueError("need at least 3 points to fit a circle")
    mean = p.mean(axis=0)
    q = p - mean  # centring keeps the normal equations well conditioned
    a = np.column_stack([q[:, 0], q[:, 1], np.ones(len(q))])
    b = -(q**2).sum(axis=1)
    sol, _, rank, sv = np.linalg.lstsq(a, b, rcond=None)
    spread = np.sqrt((q**2).sum(axis=1).mean())
    if rank < 3 or sv[-1] < 1e-9 * sv[0]:
        raise ValueError("points are collinear: click points spread around the edge")
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r2 = cx**2 + cy**2 - sol[2]
    if not np.isfinite(r2) or r2 <= 0 or np.sqrt(r2) > 1e3 * max(spread, 1e-9):
        raise ValueError("points are (nearly) collinear: click points spread around the edge")
    return float(cx + mean[0]), float(cy + mean[1]), float(np.sqrt(r2))


# ---------------------------------------------------------------------------
# Edge evidence along rays
# ---------------------------------------------------------------------------


def lightness(rgb: np.ndarray) -> np.ndarray:
    """CIELAB L* in 0-100 (float32) of an RGB uint8 image."""
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)[:, :, 0]


@dataclass
class EdgeImage:
    """Blurred L* of one image plus its edge threshold tau, at one resolution."""

    L: np.ndarray
    tau: float
    polarity: str

    @classmethod
    def from_rgb(cls, rgb: np.ndarray, p: dict[str, Any]) -> "EdgeImage":
        if p["edge_polarity"] not in ("dark_inside", "bright_inside", "any"):
            raise ValueError(f"geometry.edge_polarity must be dark_inside, bright_inside or any, got {p['edge_polarity']!r}")
        L = cv2.GaussianBlur(lightness(rgb), (0, 0), float(p["blur_sigma"]))
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8.0  # /8: per-pixel derivative
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        mag = np.hypot(gx, gy)
        tau = max(float(p["min_edge"]), float(p["edge_frac"]) * float(np.percentile(mag, 98)))
        return cls(L, tau, p["edge_polarity"])

    def rays(self, cx: float, cy: float, radii: np.ndarray, n_angles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Signed radial edge strength E[angle, radius] and its local-maximum hits.

        Returns ``(angles, E, hits)``. Samples that fall outside the image are
        never hits.
        """
        h, w = self.L.shape
        ang = np.arange(n_angles) * (2 * np.pi / n_angles)
        xs = cx + np.cos(ang)[:, None] * radii[None, :]
        ys = cy + np.sin(ang)[:, None] * radii[None, :]
        valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
        prof = cv2.remap(self.L, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        step = float(radii[1] - radii[0])
        d = np.zeros_like(prof)
        d[:, 1:-1] = (prof[:, 2:] - prof[:, :-2]) / (2 * step)
        ok = np.zeros_like(valid)
        ok[:, 1:-1] = valid[:, 2:] & valid[:, :-2]
        e = {"dark_inside": d, "bright_inside": -d, "any": np.abs(d)}[self.polarity]
        e = np.where(ok, e, -np.inf)
        hits = np.zeros_like(ok)
        hits[:, 1:-1] = (e[:, 1:-1] > self.tau) & (e[:, 1:-1] >= e[:, :-2]) & (e[:, 1:-1] >= e[:, 2:])
        return ang, e, hits

    def support(self, c: Circle, p: dict[str, Any]) -> float:
        """Fraction of rays from c's centre with an edge hit within inlier_tol * r of r."""
        tol = max(1.0, float(p["inlier_tol"]) * c.r)
        radii = np.arange(max(1.0, c.r - tol - 2), c.r + tol + 2.5, 0.5)
        _, _, hits = self.rays(c.cx, c.cy, radii, int(p["n_angles"]))
        near = np.abs(radii - c.r) <= tol
        return float(hits[:, near].any(axis=1).mean())


def _refine(edges: EdgeImage, c: Circle, window: float, p: dict[str, Any]) -> Circle | None:
    """Re-fit c to the hit nearest its radius on each ray (trimmed least squares)."""
    n = int(p["n_angles"])
    half = max(3.0, window * c.r)
    radii = np.arange(max(1.0, c.r - half), c.r + half + 0.5, 0.5)
    ang, e, hits = edges.rays(c.cx, c.cy, radii, n)
    dist = np.where(hits, np.abs(radii[None, :] - c.r), np.inf)
    j = dist.argmin(axis=1)
    has = np.isfinite(dist[np.arange(n), j])
    if has.sum() < max(12, 0.2 * n):
        return None
    rows = np.flatnonzero(has)
    jj = j[rows]
    # Sub-pixel peak: fit a parabola through the hit and its two neighbours.
    e0, e1, e2 = e[rows, jj - 1], e[rows, jj], e[rows, jj + 1]
    den = e0 - 2 * e1 + e2
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.where(np.isfinite(den) & (den < 0), 0.5 * (e0 - e2) / den, 0.0)
    rho = radii[jj] + np.clip(np.nan_to_num(off), -0.5, 0.5) * (radii[1] - radii[0])
    pts = np.column_stack([c.cx + rho * np.cos(ang[rows]), c.cy + rho * np.sin(ang[rows])])
    tol = max(1.0, float(p["inlier_tol"]) * c.r)
    keep = np.ones(len(pts), bool)
    cx, cy, r = c.cx, c.cy, c.r
    for mult in (3.0, 2.0, 1.0, 1.0):
        if keep.sum() < max(12, 0.2 * n):
            return None
        try:
            cx, cy, r = fit_circle(pts[keep])
        except ValueError:
            return None
        resid = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        keep = resid <= mult * tol
    return replace(c, cx=cx, cy=cy, r=r)


def _innermost_peak(radii: np.ndarray, s: np.ndarray, rmin: float, rmax: float, p: dict[str, Any]) -> float | None:
    inside = (radii >= rmin) & (radii <= rmax)
    left = np.r_[-np.inf, s[:-1]]
    right = np.r_[s[1:], -np.inf]
    peaks = np.flatnonzero(inside & (s >= left) & (s >= right) & (s >= float(p["min_support"])))
    if not len(peaks):
        return None
    best = s[peaks].max()
    first = peaks[s[peaks] >= float(p["peak_ratio"]) * best][0]
    # Support is flat across the +/-2 px dilation, so take the middle of the plateau.
    end = first
    while end + 1 < len(s) and s[end + 1] == s[first]:
        end += 1
    return float((radii[first] + radii[end]) / 2)


def _hough_centres(L: np.ndarray, rmin: float, rmax: float, k: int) -> list[tuple[float, float]]:
    img = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    m = min(L.shape)
    found = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, dp=1, minDist=max(4.0, 0.05 * m),
        param1=100, param2=18, minRadius=int(rmin), maxRadius=int(np.ceil(rmax)),
    )
    if found is None:
        return []
    return [(float(x), float(y)) for x, y, _ in found[0][:k]]


def detect_circle(rgb: np.ndarray, params: dict[str, Any] | None = None) -> Circle:
    """Find the liquid-surface circle in an RGB uint8 photo (steps 1-6 in the module docstring).

    Always returns a circle. When nothing convincing is found, it returns a
    centred guess with confidence 0, so the image is flagged rather than dropped.
    """
    p = {**GEOMETRY_DEFAULTS, **(params or {})}
    H, W = rgb.shape[:2]
    k = min(1.0, float(p["work_size"]) / max(H, W))
    small = cv2.resize(rgb, (max(1, round(W * k)), max(1, round(H * k))), interpolation=cv2.INTER_AREA) if k < 1 else rgb
    k = (small.shape[1] / W + small.shape[0] / H) / 2  # the exact scale after rounding
    coarse, full = EdgeImage.from_rgb(small, p), EdgeImage.from_rgb(rgb, p)
    h, w = small.shape[:2]
    rmin, rmax = float(p["radius_frac_min"]) * min(h, w), float(p["radius_frac_max"]) * min(h, w)
    centres = _hough_centres(coarse.L, rmin, rmax, int(p["hough_candidates"])) + [((w - 1) / 2, (h - 1) / 2)]
    radii = np.arange(max(1.0, rmin - 3), rmax + 4, 1.0)
    best: Circle | None = None
    tried: list[tuple[float, float]] = []
    for cx, cy in centres:
        if any(np.hypot(cx - a, cy - b) < 1.5 for a, b in tried):
            continue
        tried.append((cx, cy))
        _, _, hits = coarse.rays(cx, cy, radii, int(p["n_angles"]))
        s = maximum_filter1d(hits.astype(np.uint8), size=5, axis=1).mean(axis=0)
        r0 = _innermost_peak(radii, s, rmin, rmax, p)
        if r0 is None:
            continue
        c = _refine(coarse, Circle(cx, cy, r0), 0.08, p)
        if c is None:
            continue
        c = c.scaled(1 / k)
        for window in (0.05, 0.03):
            c = _refine(full, c, window, p) or c
        if not (0.9 * rmin / k <= c.r <= 1.1 * rmax / k):
            continue
        c = replace(c, confidence=full.support(c, p), source="auto")
        if best is None or c.confidence > best.confidence:
            best = c
    if best is None:
        return Circle((W - 1) / 2, (H - 1) / 2, 0.3 * min(H, W), 0.0, "auto")
    return best


def edge_support(rgb: np.ndarray, c: Circle, params: dict[str, Any] | None = None) -> float:
    """How much of circle c's circumference lies on a liquid edge (same measure as ``confidence``)."""
    p = {**GEOMETRY_DEFAULTS, **(params or {})}
    return EdgeImage.from_rgb(rgb, p).support(c, p)


# ---------------------------------------------------------------------------
# Saved circles: data/circles.csv
# ---------------------------------------------------------------------------


def circles_path(cfg: dict[str, Any]) -> Path:
    return cfg_path(cfg, "circles")


def format_points(points: Iterable[Iterable[float]]) -> str:
    """``[(x, y), ...]`` -> ``"x y;x y;..."`` with one decimal."""
    return ";".join(f"{x:.1f} {y:.1f}" for x, y in points)


def parse_points(s: str) -> list[tuple[float, float]]:
    if not s.strip():
        return []
    return [(float(a), float(b)) for a, b in (pair.split() for pair in s.split(";"))]


def append_circle(path: Path, row: dict[str, Any]) -> None:
    """Append one checked circle and fsync, creating the file with a header if needed."""
    if row.get("source") not in CHECKED_SOURCES:
        raise ValueError(f"saved circles must be one of {CHECKED_SOURCES}, got {row.get('source')!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    out = {c: row.get(c, "") for c in CIRCLE_COLUMNS}
    for c in ("cx", "cy", "r"):
        out[c] = f"{float(out[c]):.2f}"
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CIRCLE_COLUMNS, extrasaction="raise")
        if new:
            w.writeheader()
        w.writerow(out)
        fh.flush()
        os.fsync(fh.fileno())


def empty_circles() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=str) for c in CIRCLE_COLUMNS}).set_index("cup_id")


def read_circles(path: Path) -> pd.DataFrame:
    """Latest saved circle per cup (file order, last row wins), indexed by cup_id."""
    if not path.exists() or path.stat().st_size == 0:
        return empty_circles()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in CIRCLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[CIRCLE_COLUMNS].drop_duplicates("cup_id", keep="last").set_index("cup_id")
    for c in ("cx", "cy", "r"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@dataclass(frozen=True)
class Resolved:
    """The circle a cup's pipeline should use, plus how it was chosen."""

    circle: Circle  # the saved circle if it is valid, else the automatic one
    auto: Circle | None  # automatic detection (None if it wasn't run)
    note: str = ""  # why a saved circle was ignored, if it was


def resolve_circle(
    cup_id: str,
    rgb: np.ndarray,
    saved: pd.DataFrame,
    params: dict[str, Any] | None = None,
    run_auto: bool = True,
) -> Resolved:
    """Saved circle (accepted or manual) if one exists for this image size, else auto-detection.

    The saved circle's ``confidence`` is recomputed as its edge support on this image.
    """
    p = {**GEOMETRY_DEFAULTS, **(params or {})}
    H, W = rgb.shape[:2]
    auto = detect_circle(rgb, p) if run_auto else None
    note = ""
    if cup_id in saved.index:
        row = saved.loc[cup_id]
        size_ok = str(row["img_w"]) == str(W) and str(row["img_h"]) == str(H)
        if size_ok and np.isfinite([row["cx"], row["cy"], row["r"]]).all() and row["source"] in CHECKED_SOURCES:
            c = Circle(float(row["cx"]), float(row["cy"]), float(row["r"]), source=str(row["source"]))
            return Resolved(replace(c, confidence=edge_support(rgb, c, p)), auto)
        note = (
            f"saved circle ignored: it was made on a {row['img_w']}x{row['img_h']} image, this one is {W}x{H}"
            if not size_ok
            else "saved circle ignored: unreadable row"
        )
    if auto is None:
        auto = detect_circle(rgb, p)
    return Resolved(auto, auto, note)
