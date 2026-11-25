"""Milk vs crema segmentation inside the analysis disk (section 8.3).

Two methods, chosen with ``segment.method`` in the config:

* ``otsu``: Otsu's threshold t on CIELAB L* over the disk pixels. Milk
  M = {L* > t}. Otsu picks the t that maximises the between-class variance
  w0 w1 (mu0 - mu1)^2 of the L* histogram.
* ``kmeans``: k-means with k = 2 on (L*, a*, b*) of the disk pixels (the
  course's clustering topic). Milk is the cluster with the higher mean L*. It
  is fitted on a seeded random sample of ``kmeans_fit_pixels`` pixels, then
  every pixel goes to its nearest centre.

Post-processing (both methods): a morphological opening, then a closing, with a
disk of radius ``morph_disk_radius`` (3 px); restrict to the disk; then remove
connected components smaller than ``min_component_frac`` (0.5%) of the disk
area (8-connectivity).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

SEGMENT_DEFAULTS: dict[str, Any] = {
    "method": "kmeans",
    "morph_disk_radius": 3,
    "min_component_frac": 0.005,
    "kmeans_fit_pixels": 20000,
}
METHODS = ("otsu", "kmeans")


def segment_params(cfg: dict[str, Any] | None) -> dict[str, Any]:
    return {**SEGMENT_DEFAULTS, **((cfg or {}).get("segment") or {})}


def lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB (L* 0-100, a*, b*) as float32, from an RGB uint8 image."""
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def disk_kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def raw_otsu(L: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """M = mask & (L* > t), with t = Otsu's threshold of L* inside the mask."""
    vals = L[mask]
    if vals.size == 0 or np.ptp(vals) == 0:
        return np.zeros_like(mask)
    return mask & (L > threshold_otsu(vals))


def raw_kmeans(lab_img: np.ndarray, mask: np.ndarray, seed: int = 42, fit_pixels: int = 20000) -> np.ndarray:
    """M = pixels nearest the brighter of two k-means centres in (L*, a*, b*)."""
    X = lab_img[mask].astype(np.float64)
    if len(X) < 2 or np.ptp(X[:, 0]) == 0:
        return np.zeros_like(mask)
    rng = np.random.default_rng(seed)
    fit = X[rng.choice(len(X), size=min(fit_pixels, len(X)), replace=False)]
    km = KMeans(n_clusters=2, n_init=4, random_state=seed).fit(fit)
    milk = int(np.argmax(km.cluster_centers_[:, 0]))
    out = np.zeros_like(mask)
    out[mask] = km.predict(X) == milk
    return out


def clean(m: np.ndarray, mask: np.ndarray, disk_radius: int = 3, min_component_frac: float = 0.005) -> np.ndarray:
    """Open, then close (disk of ``disk_radius``), keep inside ``mask``, drop components < frac x mask area."""
    k = disk_kernel(disk_radius)
    u = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k)
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k)
    u &= mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component_frac * mask.sum()
    return keep[labels]


def segment(rgb: np.ndarray, mask: np.ndarray, params: dict[str, Any] | None = None, seed: int = 42, method: str | None = None) -> np.ndarray:
    """Milk mask M (bool, same shape as ``mask``) for a preprocessed crop."""
    p = {**SEGMENT_DEFAULTS, **(params or {})}
    method = method or p["method"]
    img = lab(rgb)
    if method == "otsu":
        m = raw_otsu(img[:, :, 0], mask)
    elif method == "kmeans":
        m = raw_kmeans(img, mask, seed, int(p["kmeans_fit_pixels"]))
    else:
        raise ValueError(f"segment.method must be one of {METHODS}, got {method!r}")
    return clean(m, mask, int(p["morph_disk_radius"]), float(p["min_component_frac"]))
