"""Ingest phone photos: data/raw_full -> data/raw (section 6.5).

    python scripts/ingest.py

For every photo named <cup_id>.jpg in data/raw_full/ this strips all metadata
(EXIF incl. GPS, XMP, ICC), applies the EXIF rotation, resizes to 1024 px on
the longest side and writes data/raw/<cup_id>.jpg. It then checks that every
photo has a cup-log row and every cup-log row has a photo, and prints the
counts per design cell.

Exit status is 1 if anything needs fixing (unmatched photos or rows, a bad log,
HEIC files, duplicate names), after ingesting whatever was valid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from latte.io import (
    DEFAULT_CONFIG,
    cfg_path,
    cup_sort_key,
    has_metadata,
    list_images,
    load_config,
    read_cup_log,
    strip_and_resize,
    validate_cup_log,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--force", action="store_true", help="re-process photos even if data/raw is up to date")
    ap.add_argument(
        "--allow-missing-photos",
        action="store_true",
        help="don't fail on planned log rows that have no photo yet (session in progress)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    src_dir, dst_dir = cfg_path(cfg, "raw_full"), cfg_path(cfg, "raw")
    log = read_cup_log(cfg_path(cfg, "cup_log"))
    errors: list[str] = []
    warnings: list[str] = []

    problems = validate_cup_log(log, cfg)
    if problems:
        print("Cup log problems (fix data/cup_log.csv, then re-run):")
        print("\n".join(f"  - {p}" for p in problems))
        return 1

    # --- find source photos -------------------------------------------------
    if not src_dir.exists():
        print(f"No {src_dir} folder. Copy the phone photos there first.")
        return 1
    heic = [p.name for p in src_dir.iterdir() if p.suffix.lower() in (".heic", ".heif")]
    if heic:
        errors.append(
            f"{len(heic)} HEIC file(s) can't be read ({', '.join(heic[:5])}...). Set the iPhone camera to "
            "Settings > Camera > Formats > Most Compatible, or export them as JPEG."
        )
    photos: dict[str, Path] = {}
    for p in list_images(src_dir, cfg["ingest"]["extensions"]):
        if p.stem in photos:
            errors.append(f"two photos for {p.stem}: {photos[p.stem].name} and {p.name}")
            continue
        photos[p.stem] = p

    # --- match photos <-> log rows -----------------------------------------
    logged = set(log["cup_id"])
    no_row = sorted(set(photos) - logged, key=cup_sort_key)
    no_photo = sorted(logged - set(photos), key=cup_sort_key)
    if no_row:
        errors.append(f"{len(no_row)} photo(s) with no cup-log row: {', '.join(no_row)}")
    if no_photo:
        msg = f"{len(no_photo)} cup-log row(s) with no photo: {', '.join(no_photo)}"
        (warnings if args.allow_missing_photos else errors).append(msg)

    # --- ingest ---------------------------------------------------------------
    max_side, quality = int(cfg["ingest"]["max_side"]), int(cfg["ingest"]["jpeg_quality"])
    counts = {"new": 0, "updated": 0, "unchanged": 0}
    for cup_id in sorted(set(photos) & logged, key=cup_sort_key):
        src, dst = photos[cup_id], dst_dir / f"{cup_id}.jpg"
        if dst.exists() and not args.force and dst.stat().st_mtime >= src.stat().st_mtime:
            counts["unchanged"] += 1
            continue
        counts["updated" if dst.exists() else "new"] += 1
        strip_and_resize(src, dst, max_side, quality)
        if has_metadata(dst):  # belt and braces: never commit a photo with metadata
            dst.unlink()
            errors.append(f"{cup_id}: metadata survived stripping; output deleted")

    orphans = sorted({p.stem for p in list_images(dst_dir, [".jpg"])} - logged, key=cup_sort_key)
    if orphans:
        warnings.append(f"{len(orphans)} file(s) in {dst_dir.name}/ have no cup-log row: {', '.join(orphans)}")

    print(
        f"Ingested into {dst_dir}: {counts['new']} new, {counts['updated']} updated, "
        f"{counts['unchanged']} unchanged.\n"
    )
    print(cell_counts(log, dst_dir, cfg).to_string(index=False))
    for w in warnings:
        print(f"\nWARNING: {w}")
    for e in errors:
        print(f"\nERROR: {e}")
    return 1 if errors else 0


def cell_counts(log: pd.DataFrame, raw_dir: Path, cfg: dict) -> pd.DataFrame:
    """Per-cell counts of ingested cups and their usable flags."""
    d = cfg["design"]
    have = {p.stem for p in list_images(raw_dir, [".jpg"])}
    df = log[log["cup_id"].isin(have)]
    rows = []
    for milk in d["milk_conditions"]:
        for drop in d["dropin"]:
            c = df[(df["milk_condition"] == milk) & (df["dropin"] == drop)]
            rows.append(
                {
                    "milk": milk,
                    "dropin": drop,
                    "photos": len(c),
                    "usable_y": int((c["usable"] == "y").sum()),
                    "usable_n": int((c["usable"] == "n").sum()),
                    "unmarked": int((c["usable"] == "").sum()),
                    "target": d["target_per_cell"],
                }
            )
    out = pd.DataFrame(rows)
    total = {"milk": "TOTAL", "dropin": ""} | {k: int(out[k].sum()) for k in ("photos", "usable_y", "usable_n", "unmarked", "target")}
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


if __name__ == "__main__":
    sys.exit(main())
