"""Contact sheets of every cup's liquid circle and masked crop -> results/qa/ (section 8.1).

    make qa                      # every usable photo in data/raw
    make qa ARGS=--dev           # the external dev images in data/dev_external (output is gitignored)

For each photo it uses the saved circle from data/circles.csv if there is one,
otherwise the automatic detection. It writes:

    results/qa/circles_NN.jpg  zoomed view of each circle. Caption colours:
                               green = checked by a student, amber = automatic and not yet checked,
                               red = automatic with low confidence. The thin white ring is the
                               analysis edge (radius minus 4%). A cyan ring is the automatic
                               circle where a saved one replaced it.
    results/qa/crops_NN.jpg    the 512 x 512 masked crops that the features will see
    results/qa/circles.csv     one row per cup: circle used, source, confidence, crop scale

Acceptance (section 8.1): every usable photo has a circle a student has checked
by eye. Check them with ``make circles``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from latte.geometry import circles_path, empty_circles, geometry_params, read_circles, resolve_circle
from latte.io import DEFAULT_CONFIG, _atomic_write_csv, cfg_path, cup_sort_key, list_images, load_config, load_rgb
from latte.labels import available_cups
from latte.overlay import AMBER, CYAN, GREEN, RED, WHITE, contact_sheet, draw_circle, zoom_view
from latte.preprocess import prepare_cfg, preprocess_params

TILE = 300
CROP_TILE = 220


def dev_images(cfg: dict[str, Any]) -> list[tuple[str, Path]]:
    root = cfg_path(cfg, "dev_external")
    exts = {e.lower() for e in cfg["ingest"]["extensions"]}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts) if root.exists() else []
    return [(str(p.relative_to(root).with_suffix("")), p) for p in files]


def team_images(cfg: dict[str, Any]) -> list[tuple[str, Path]]:
    raw = cfg_path(cfg, "raw")
    return [(c, raw / f"{c}.jpg") for c in sorted(available_cups(cfg), key=cup_sort_key)]


def run(cfg: dict[str, Any], dev: bool, per_sheet: int, cols: int) -> int:
    gp, pp = geometry_params(cfg), preprocess_params(cfg)
    low = float(gp["low_confidence"])
    out_dir = cfg_path(cfg, "qa_dir") / ("dev" if dev else "")
    items = dev_images(cfg) if dev else team_images(cfg)
    if not items:
        where = cfg_path(cfg, "dev_external" if dev else "raw")
        print(f"No images in {where}. {'Put the licensed dev set there first.' if dev else 'Run make ingest first.'}")
        return 0
    saved = empty_circles() if dev else read_circles(circles_path(cfg))  # dev images never get saved circles

    rows: list[dict[str, Any]] = []
    tiles, caps, cap_cols, crops, crop_caps = [], [], [], [], []
    for i, (cid, path) in enumerate(items, 1):
        rgb = load_rgb(path)
        res = resolve_circle(cid, rgb, saved, gp)
        c, auto = res.circle, res.auto
        prep = prepare_cfg(rgb, c, cfg)
        low_conf = (not c.checked) and c.confidence < low
        color = GREEN if c.checked else (RED if low_conf else AMBER)

        view_img, view = zoom_view(rgb, c, TILE)
        if auto is not None and c.checked and np.hypot(auto.cx - c.cx, auto.cy - c.cy) + abs(auto.r - c.r) > 1.0:
            draw_circle(view_img, view, auto, CYAN, 1)
        draw_circle(view_img, view, c, color, 2)
        draw_circle(view_img, view, c, WHITE, 1, r=c.r * (1 - float(pp["radius_shrink"])))
        tiles.append(view_img)
        label = c.source  # auto | accepted | manual
        caps.append(f"{cid}  {label}  {c.confidence:.2f}")
        cap_cols.append(color)
        crops.append(cv2.resize(prep.crop, (CROP_TILE, CROP_TILE), interpolation=cv2.INTER_AREA))
        crop_caps.append(f"{cid}  x{prep.scale:.2f}")

        rows.append({
            "cup_id": cid, "source": c.source, "checked": "y" if c.checked else "n",
            "confidence": round(c.confidence, 3), "low_confidence": "y" if low_conf else "n",
            "cx": round(c.cx, 2), "cy": round(c.cy, 2), "r": round(c.r, 2),
            "auto_cx": round(auto.cx, 2) if auto else "", "auto_cy": round(auto.cy, 2) if auto else "",
            "auto_r": round(auto.r, 2) if auto else "", "auto_confidence": round(auto.confidence, 3) if auto else "",
            "crop_scale": round(prep.scale, 3), "note": res.note,
        })
        if i % 20 == 0:
            print(f"  {i}/{len(items)} done", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in [*out_dir.glob("circles_*.jpg"), *out_dir.glob("crops_*.jpg")]:
        old.unlink()  # stale pages from a previous, longer run
    legend = "green: checked   amber: auto, not checked   red: low confidence   white: analysis edge (r - 4%)   cyan: auto circle"
    n_pages = -(-len(items) // per_sheet)
    written: list[Path] = []
    for page in range(n_pages):
        sl = slice(page * per_sheet, (page + 1) * per_sheet)
        tag = f"{page + 1:02d}"
        head = f"{'DEV ' if dev else ''}circles, page {page + 1}/{n_pages}.   {legend}"
        p1 = out_dir / f"circles_{tag}.jpg"
        contact_sheet(tiles[sl], caps[sl], cap_cols[sl], cols, head).save(p1, quality=90)
        p2 = out_dir / f"crops_{tag}.jpg"
        crop_head = f"{'DEV ' if dev else ''}masked {pp['size']} px crops, page {page + 1}/{n_pages}.   xN = crop px per photo px (above 1 = upsampled)"
        contact_sheet(crops[sl], crop_caps[sl], [(60, 60, 60)] * len(crop_caps[sl]), cols + 1, crop_head).save(p2, quality=90)
        written += [p1, p2]
    status = pd.DataFrame(rows)
    _atomic_write_csv(status, out_dir / "circles.csv")

    n = len(status)
    checked = status["checked"] == "y"
    lowc = status["low_confidence"] == "y"
    print(f"QA on {n} {'dev' if dev else 'usable'} photo{'s' * (n != 1)}:")
    if not dev:
        n_acc, n_man = (status["source"] == "accepted").sum(), (status["source"] == "manual").sum()
        print(f"  checked by eye:  {checked.sum()} ({n_acc} accepted, {n_man} manual)")
        print(f"  not checked yet: {(~checked).sum()}, of which {lowc.sum()} have low confidence (< {low})")
    else:
        print(f"  low confidence (< {low}): {lowc.sum()}")
    if lowc.any():
        print("    low confidence: " + ", ".join(status.loc[lowc, "cup_id"].head(30)) + (" ..." if lowc.sum() > 30 else ""))
    for note in status.loc[status["note"] != "", ["cup_id", "note"]].itertuples(index=False):
        print(f"  {note.cup_id}: {note.note}")
    med = float(status["crop_scale"].median())
    if med > 1.25:
        print(f"  note: median crop scale x{med:.2f}, so the {pp['size']} px crops are upsampled "
              f"(the liquid spans about {pp['size'] / med:.0f} px in the photo).")
    rel = lambda p: p.relative_to(cfg["_root"]) if p.is_relative_to(cfg["_root"]) else p  # noqa: E731
    print(f"  wrote {len(written)} sheets and {rel(out_dir / 'circles.csv')} in {rel(out_dir)}/")
    if not dev and (~checked).any():
        print("  next: make circles   (accept or fix each circle; low-confidence ones come first)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dev", action="store_true", help="run on data/dev_external instead of data/raw")
    ap.add_argument("--per-sheet", type=int, default=20)
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args(argv)
    return run(load_config(args.config), args.dev, args.per_sheet, args.cols)


if __name__ == "__main__":
    sys.exit(main())
