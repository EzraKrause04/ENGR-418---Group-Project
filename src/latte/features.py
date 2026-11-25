"""Handcrafted image features (section 8.4).

Each feature is a pure function of arrays and returns a float, with its formula
in the docstring. Notation, all on the preprocessed ``size`` x ``size`` crop:

* L: CIELAB lightness L* (0-100, float).
* D: the analysis disk (bool), inscribed in the square crop, minus any part
  that fell outside the photo. Its centre c0 and radius R come from the crop
  frame, not from D: c0 = ((w - 1) / 2, (h - 1) / 2), R = min(h, w) / 2.
  That way a cut-off disk keeps its true centre.
* M: the milk mask from ``latte.segment`` (bool, a subset of D).
* C = D minus M: the crema.
* |X| is a pixel count. Sobel gradients are divided by 8, so they are L* per px.

Features that need milk (or crema) return NaN when M (or C) is empty. Models
impute these inside the CV pipeline.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.feature import blob_log
from skimage.measure import perimeter_crofton

from latte.segment import disk_kernel

FEATURE_DEFAULTS: dict[str, Any] = {
    "symmetry_angle_step_deg": 2,
    "edge_band_px": 2,
    "log_sigma_min": 1,
    "log_sigma_max": 4,
    "log_num_sigma": 4,
    "log_threshold": 3.0,
    "log_noise_k": 6.0,
    "highpass_sigma": 8.0,
    "texture_erode_px": 2,
}

# Column order of features/features.csv.
FEATURES = [
    "contrast_dL",
    "contrast_michelson",
    "edge_sharpness",
    "boundary_roughness",
    "symmetry_iou",
    "symmetry_angle",
    "centre_offset",
    "coverage",
    "n_components",
    "bubble_density",
    "texture_energy",
]
# symmetry_angle is the heart's orientation, which says nothing about quality,
# because the cup can be turned. It is kept for the QA overlays but not given to models.
MODEL_FEATURES = [f for f in FEATURES if f != "symmetry_angle"]
# Groups for the drop-one-group ablation (E4).
FEATURE_GROUPS: dict[str, list[str]] = {
    "contrast": ["contrast_dL", "contrast_michelson"],
    "edge": ["edge_sharpness", "boundary_roughness"],
    "symmetry_centring": ["symmetry_iou", "centre_offset"],
    "coverage": ["coverage", "n_components"],
    "surface": ["bubble_density", "texture_energy"],
}
# Rubric dimension each feature should track (E1 feature validity, section 10.2).
RUBRIC_LINK: dict[str, str] = {
    "contrast_dL": "contrast",
    "contrast_michelson": "contrast",
    "edge_sharpness": "definition",
    "boundary_roughness": "definition",
    "symmetry_iou": "symmetry",
    "centre_offset": "centring",
    "bubble_density": "surface",
    "texture_energy": "surface",
}
# Design factor a feature should track (coverage -> drop-in timing).
DESIGN_LINK: dict[str, str] = {"coverage": "dropin"}

NAN = float("nan")


def feature_params(cfg: dict[str, Any] | None) -> dict[str, Any]:
    return {**FEATURE_DEFAULTS, **((cfg or {}).get("features") or {})}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def centroid(mask: np.ndarray) -> tuple[float, float]:
    """(x, y) mean pixel position of a non-empty mask."""
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def disk_geometry(D: np.ndarray) -> tuple[float, float, float]:
    """(cx, cy, R) of the analysis disk inscribed in D's frame: ((w - 1) / 2, (h - 1) / 2, min(h, w) / 2)."""
    h, w = D.shape
    return (w - 1) / 2, (h - 1) / 2, min(h, w) / 2


def gradient_magnitude(L: np.ndarray) -> np.ndarray:
    """|grad L*| from 3x3 Sobel, in L* per px."""
    gx = cv2.Sobel(L.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(L.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy)


def boundary_band(M: np.ndarray, D: np.ndarray, width: int) -> np.ndarray:
    """Pixels within ``width`` px of M's outline, away from the disk edge.

    band = (dilate(M, w) & ~erode(M, w)) & erode(D, w + 1). Where M touches the
    disk's edge, that part of its outline is the mask boundary, not a pour
    edge, so it is left out.
    """
    k = disk_kernel(width)
    m = M.astype(np.uint8)
    ring = cv2.dilate(m, k).astype(bool) & ~cv2.erode(m, k).astype(bool)
    return ring & cv2.erode(D.astype(np.uint8), disk_kernel(width + 1)).astype(bool)


def largest_component(M: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(M.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(M, dtype=bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def reflect(M: np.ndarray, centre: tuple[float, float], angle_deg: float) -> np.ndarray:
    """M mirrored across the line through ``centre`` at ``angle_deg``.

    Angles are in image coordinates (x right, y down): 0 = horizontal line,
    90 = vertical line. A point p maps to Q (p - c) + c, with
    Q = [[cos 2t, sin 2t], [sin 2t, -cos 2t]].
    """
    t = math.radians(2 * angle_deg)
    q = np.array([[math.cos(t), math.sin(t)], [math.sin(t), -math.cos(t)]])
    c = np.asarray(centre, float)
    a = np.hstack([q, (c - q @ c)[:, None]])
    h, w = M.shape
    out = cv2.warpAffine(M.astype(np.uint8), a, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return out.astype(bool)


def symmetry(M: np.ndarray, step_deg: float = 2) -> tuple[float, float]:
    """(best IoU, best angle): max over t in {0, step, ..., 180 - step} of IoU(M, reflect(M, centroid(M), t))."""
    if not M.any():
        return NAN, NAN
    cx, cy = centroid(M)
    # Work in a window that holds M and any reflection of it (radius = farthest M pixel from the centroid).
    ys, xs = np.nonzero(M)
    rad = int(math.ceil(np.hypot(xs - cx, ys - cy).max())) + 2
    x0, y0 = int(math.floor(cx)) - rad, int(math.floor(cy)) - rad
    size = 2 * rad + 2
    win = np.zeros((size, size), bool)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + size, M.shape[1]), min(y0 + size, M.shape[0])
    win[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = M[sy0:sy1, sx0:sx1]
    c = (cx - x0, cy - y0)
    area = win.sum()
    best, best_t = -1.0, NAN
    for t in np.arange(0, 180, step_deg):
        r = reflect(win, c, float(t))
        inter = np.count_nonzero(win & r)
        iou = inter / (2 * area - inter)
        if iou > best:
            best, best_t = iou, float(t)
    return float(best), best_t


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def contrast_dL(L: np.ndarray, M: np.ndarray, D: np.ndarray) -> float:
    """Milk-crema lightness difference: mean L*(M) - mean L*(C), in L* units."""
    C = D & ~M
    if not M.any() or not C.any():
        return NAN
    return float(L[M].mean() - L[C].mean())


def contrast_michelson(L: np.ndarray, M: np.ndarray, D: np.ndarray) -> float:
    """Michelson contrast: (mean L*(M) - mean L*(C)) / (mean L*(M) + mean L*(C)); unitless, in [-1, 1]."""
    C = D & ~M
    if not M.any() or not C.any():
        return NAN
    lm, lc = float(L[M].mean()), float(L[C].mean())
    return (lm - lc) / (lm + lc) if lm + lc > 0 else NAN


def edge_sharpness(L: np.ndarray, M: np.ndarray, D: np.ndarray, band_px: int = 2) -> float:
    """Edge steepness relative to contrast: mean |grad L*| over the band / contrast_dL.

    The band is every pixel within ``band_px`` of M's outline, excluding the disk
    edge (``boundary_band``). Dividing by contrast_dL stops this from being a
    second contrast measure: the result is in 1/px, about 1 / (edge width). It
    is NaN if the band is empty or contrast_dL < 1 L*.
    """
    band = boundary_band(M, D, band_px)
    dl = contrast_dL(L, M, D)
    if not band.any() or not np.isfinite(dl) or dl < 1:
        return NAN
    return float(gradient_magnitude(L)[band].mean() / dl)


def boundary_roughness(M: np.ndarray) -> float:
    """Isoperimetric ratio of the largest milk component: P^2 / (4 pi A).

    P is the Crofton perimeter estimate (skimage.measure.perimeter_crofton,
    4 directions), which is within 0.5% of 2 pi r on digital disks, and A is the
    pixel count. The value is 1 for a disk and grows for jagged or feathered
    outlines.
    """
    big = largest_component(M)
    a = big.sum()
    if a == 0:
        return NAN
    return float(perimeter_crofton(big, directions=4) ** 2 / (4 * math.pi * a))


def symmetry_iou(M: np.ndarray, step_deg: float = 2) -> float:
    """Mirror symmetry: max over t in [0, 180) (step ``step_deg``) of IoU(M, M reflected about the line through centroid(M) at angle t)."""
    return symmetry(M, step_deg)[0]


def symmetry_angle(M: np.ndarray, step_deg: float = 2) -> float:
    """The angle t (degrees, image coordinates, 90 = vertical) that maximises symmetry_iou."""
    return symmetry(M, step_deg)[1]


def centre_offset(M: np.ndarray, D: np.ndarray) -> float:
    """Distance of the milk from the cup centre: ||centroid(M) - c0|| / R (0 = centred, 1 = at the edge)."""
    if not M.any():
        return NAN
    mx, my = centroid(M)
    cx, cy, r = disk_geometry(D)
    return float(math.hypot(mx - cx, my - cy) / r)


def coverage(M: np.ndarray, D: np.ndarray) -> float:
    """Fraction of the surface that is milk: |M| / |D|."""
    return float(M.sum() / D.sum()) if D.any() else NAN


def n_components(M: np.ndarray) -> float:
    """Number of 8-connected milk components left after the segmentation's small-component filter."""
    n, _ = cv2.connectedComponents(M.astype(np.uint8), connectivity=8)
    return float(n - 1)


def bubble_density(
    L: np.ndarray,
    M: np.ndarray,
    D: np.ndarray,
    sigma_min: float = 1,
    sigma_max: float = 4,
    num_sigma: int = 4,
    threshold: float = 3.0,
    highpass_sigma: float = 8.0,
    noise_k: float = 6.0,
) -> float:
    """Dark blobs (bubbles) per 10^4 px^2 of surface.

    H = G_R(L*) - L*, a high-pass that is positive where a spot is darker than
    its surroundings. R is the region the pixel belongs to (milk M or crema C),
    and G_R is a Gaussian blur (sigma = ``highpass_sigma``) normalised inside R,
    G_R(f) = G(f 1_R) / G(1_R). Blurring each region separately means the
    milk-crema edge and the black outside the disk do not count as dark spots.
    Blobs are local maxima of the scale-normalised LoG, -sigma^2 lap G_sigma * H,
    over sigma in [sigma_min, sigma_max], above tau (skimage.feature.blob_log),
    where tau = max(``threshold``, ``noise_k`` * 1.4826 MAD) and MAD is the
    median absolute deviation of that LoG response at sigma_min inside D.
    Without the noise term, photo noise that the 512 px resampling has enlarged
    to blob size would be counted as bubbles. The MAD ignores the few real
    bubbles. A blob counts if its centre is at least 2 sigma_max inside D and
    more than sigma_max from M's outline: density = count / (|D| / 10^4).
    """
    if not D.any():
        return NAN
    Lf = L.astype(np.float64)
    h = np.zeros_like(Lf)
    for region in (M & D, D & ~M):
        if region.any():
            rf = region.astype(np.float64)
            blur = gaussian_filter(Lf * rf, highpass_sigma) / np.maximum(gaussian_filter(rf, highpass_sigma), 1e-6)
            h[region] = (blur - Lf)[region]
    ok = cv2.erode(D.astype(np.uint8), disk_kernel(int(math.ceil(2 * sigma_max)))).astype(bool)
    resp = -gaussian_laplace(h, sigma_min) * sigma_min**2
    vals = resp[ok] if ok.any() else resp[D]
    noise = 1.4826 * float(np.median(np.abs(vals - np.median(vals))))
    tau = max(threshold, noise_k * noise)
    blobs = blob_log(h, min_sigma=sigma_min, max_sigma=sigma_max, num_sigma=num_sigma, threshold=tau)
    if M.any():
        ok &= ~boundary_band(M, np.ones_like(D), int(math.ceil(sigma_max)))
    count = sum(1 for y, x, _ in blobs if ok[int(y), int(x)])
    return float(count / (D.sum() / 1e4))


def texture_energy(L: np.ndarray, M: np.ndarray, erode_px: int = 2) -> float:
    """Fine texture of the milk: Var(lap L*) over M eroded by ``erode_px``.

    lap is the 4-neighbour Laplacian. The erosion keeps the milk-crema edge
    itself out of the measure.
    """
    inner = cv2.erode(M.astype(np.uint8), disk_kernel(erode_px)).astype(bool) if erode_px > 0 else M
    if inner.sum() < 2:
        return NAN
    lap = cv2.Laplacian(L.astype(np.float32), cv2.CV_32F, ksize=1)
    return float(lap[inner].var())


# ---------------------------------------------------------------------------
# All features for one cup
# ---------------------------------------------------------------------------


def extract_features(L: np.ndarray, M: np.ndarray, D: np.ndarray, params: dict[str, Any] | None = None) -> dict[str, float]:
    """Every feature in ``FEATURES`` for one crop (L = CIELAB L*, M = milk, D = disk)."""
    p = {**FEATURE_DEFAULTS, **(params or {})}
    s_iou, s_ang = symmetry(M, float(p["symmetry_angle_step_deg"]))
    out = {
        "contrast_dL": contrast_dL(L, M, D),
        "contrast_michelson": contrast_michelson(L, M, D),
        "edge_sharpness": edge_sharpness(L, M, D, int(p["edge_band_px"])),
        "boundary_roughness": boundary_roughness(M),
        "symmetry_iou": s_iou,
        "symmetry_angle": s_ang,
        "centre_offset": centre_offset(M, D),
        "coverage": coverage(M, D),
        "n_components": n_components(M),
        "bubble_density": bubble_density(
            L, M, D, float(p["log_sigma_min"]), float(p["log_sigma_max"]), int(p["log_num_sigma"]),
            float(p["log_threshold"]), float(p["highpass_sigma"]), float(p["log_noise_k"]),
        ),
        "texture_energy": texture_energy(L, M, int(p["texture_erode_px"])),
    }
    return {k: out[k] for k in FEATURES}
