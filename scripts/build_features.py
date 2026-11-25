"""Segment every cup and compute its handcrafted features (sections 8.3 and 8.4).

    make features                     # data/raw -> features/features.csv + QA overlays
    make features ARGS=--dev          # overlays only, for data/dev_external (never used for training)

For each usable photo, it takes the circle (a saved one from data/circles.csv
if there is one, otherwise the automatic detection), makes the 512 px masked
crop, segments the milk with ``segment.method``, and computes every feature in
``latte.features.FEATURES``. It writes:

    features/features.csv          one row per cup_id: the features, plus circle source/confidence
                                   and crop scale
    results/qa/segment_NN.jpg      crop with the milk outline (magenta), the milk centroid (cyan) and
                                   the symmetry axis (yellow)
    results/qa/seg_compare_NN.jpg  Otsu (cyan) vs k-means (magenta) outlines, with their IoU, for
                                   choosing segment.method on the pilot photos

Features for photos whose circle hasn't been checked by eye yet are
provisional. The script says how many there are.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from latte.features import FEATURES, extract_features, feature_params
from latte.geometry import circles_path, empty_circles, geometry_params, read_circles, resolve_circle
from latte.io import DEFAULT_CONFIG, _atomic_write_csv, cfg_path, cup_sort_key, load_config, load_rgb
from latte.labels import available_cups
from latte.overlay import CYAN, MAGENTA, YELLOW, contact_sheet, draw_axis, draw_outline, draw_point
from latte.preprocess import prepare_cfg
from latte.segment import METHODS, lab, segment, segment_params

TILE = 280


def items(cfg: dict[str, Any], dev: bool) -> list[tuple[str, Path]]:
    if dev:
        root = cfg_path(cfg, "dev_external")
        exts = {e.lower() for e in cfg["ingest"]["extensions"]}
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts) if root.exists() else []
        return [(str(p.relative_to(root).with_suffix("")), p) for p in files]
    raw = cfg_path(cfg, "raw")
    return [(c, raw / f"{c}.jpg") for c in sorted(available_cups(cfg), key=cup_sort_key)]


def run(cfg: dict[str, Any], dev: bool, per_sheet: int, cols: int) -> int:
    gp, sp, fp = geometry_params(cfg), segment_params(cfg), feature_params(cfg)
    seed = int(cfg["seed"])
    method = sp["method"]
    if method not in METHODS:
        print(f"segment.method must be one of {METHODS}, got {method!r}", file=sys.stderr)
        return 1
    other = next(m for m in METHODS if m != method)
    todo = items(cfg, dev)
    if not todo:
        where = cfg_path(cfg, "dev_external" if dev else "raw")
        print(f"No images in {where}. {'Put the licensed dev set there first.' if dev else 'Run make ingest first.'}")
        return 0
    saved = empty_circles() if dev else read_circles(circles_path(cfg))

    rows: list[dict[str, Any]] = []
    seg_tiles, seg_caps, cmp_tiles, cmp_caps = [], [], [], []
    for i, (cid, path) in enumerate(todo, 1):
        rgb = load_rgb(path)
        res = resolve_circle(cid, rgb, saved, gp, run_auto=False)
        prep = prepare_cfg(rgb, res.circle, cfg)
        M = segment(prep.crop, prep.mask, sp, seed)
        M_other = segment(prep.crop, prep.mask, sp, seed, method=other)
        L = lab(prep.crop)[:, :, 0]
        f = extract_features(L, M, prep.mask, fp)
        c = res.circle
        rows.append({"cup_id": cid, **f, "circle_source": c.source, "circle_checked": "y" if c.checked else "n",
                     "circle_confidence": round(c.confidence, 3), "crop_scale": round(prep.scale, 3)})

        size = prep.crop.shape[0]
        tile = cv2.resize(prep.crop, (TILE, TILE), interpolation=cv2.INTER_AREA)
        cmp_tile = tile.copy()
        draw_outline(tile, M, MAGENTA, 2)
        if M.any():
            ys, xs = np.nonzero(M)
            mx, my = float(xs.mean()), float(ys.mean())
            draw_axis(tile, size, mx, my, f["symmetry_angle"], size * 0.45, YELLOW, 1)
            draw_point(tile, size, mx, my, CYAN)
        seg_tiles.append(tile)
        seg_caps.append(f"{cid}  sym {f['symmetry_iou']:.2f}  off {f['centre_offset']:.2f}  cov {f['coverage']:.2f}")
        masks = {method: M, other: M_other}
        draw_outline(cmp_tile, masks["otsu"], CYAN, 2)
        draw_outline(cmp_tile, masks["kmeans"], MAGENTA, 1)
        union = (M | M_other).sum()
        agree = (M & M_other).sum() / union if union else 1.0
        cmp_tiles.append(cmp_tile)
        cmp_caps.append(f"{cid}  IoU {agree:.3f}")
        rows[-1]["_iou_methods"] = agree
        if i % 20 == 0:
            print(f"  {i}/{len(todo)} done", flush=True)

    out_dir = cfg_path(cfg, "qa_dir") / ("dev" if dev else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in [*out_dir.glob("segment_*.jpg"), *out_dir.glob("seg_compare_*.jpg")]:
        old.unlink()
    n_pages = -(-len(todo) // per_sheet)
    grey = [(60, 60, 60)] * per_sheet
    for page in range(n_pages):
        sl = slice(page * per_sheet, (page + 1) * per_sheet)
        tag = f"{page + 1:02d}"
        pre = "DEV " if dev else ""
        contact_sheet(seg_tiles[sl], seg_caps[sl], grey, cols,
                      f"{pre}segmentation ({method}), page {page + 1}/{n_pages}.   magenta: milk outline   "
                      "cyan: milk centroid   yellow: symmetry axis").save(out_dir / f"segment_{tag}.jpg", quality=90)
        contact_sheet(cmp_tiles[sl], cmp_caps[sl], grey, cols,
                      f"{pre}Otsu (cyan) vs k-means (magenta), page {page + 1}/{n_pages}.   "
                      f"IoU = overlap of the two milk masks").save(out_dir / f"seg_compare_{tag}.jpg", quality=90)

    table = pd.DataFrame(rows)
    ious = table.pop("_iou_methods")
    table[FEATURES] = table[FEATURES].astype(float).round(6)
    rel = lambda p: p.relative_to(cfg["_root"]) if p.is_relative_to(cfg["_root"]) else p  # noqa: E731
    print(f"Segmented {len(table)} {'dev' if dev else 'usable'} photo{'s' * (len(table) != 1)} with {method}.")
    print(f"  Otsu vs k-means agreement: median IoU {ious.median():.3f}, lowest {ious.min():.3f} "
          f"({table['cup_id'][ious.idxmin()]})")
    if dev:
        print(f"  dev images: overlays only in {rel(out_dir)}/ (no features are written for them).")
        return 0
    feat_path = cfg_path(cfg, "features_dir") / "features.csv"
    _atomic_write_csv(table[["cup_id", *FEATURES, "circle_source", "circle_checked", "circle_confidence", "crop_scale"]], feat_path)
    nan = table[FEATURES].isna().sum()
    if nan.any():
        print("  missing values (e.g. no milk found): " + ", ".join(f"{k} {v}" for k, v in nan.items() if v))
    unchecked = int((table["circle_checked"] == "n").sum())
    if unchecked:
        print(f"  {unchecked} cup{'s' * (unchecked != 1)} still use an unchecked circle, so their features are "
              "provisional. Run make circles.")
    print(f"  wrote {rel(feat_path)} and {2 * n_pages} QA sheets in {rel(out_dir)}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dev", action="store_true", help="overlays for data/dev_external only; no feature table")
    ap.add_argument("--per-sheet", type=int, default=20)
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args(argv)
    return run(load_config(args.config), args.dev, args.per_sheet, args.cols)


if __name__ == "__main__":
    sys.exit(main())
