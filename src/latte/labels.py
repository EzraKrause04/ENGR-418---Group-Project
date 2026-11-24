"""Rater label storage, blind presentation order and rubric anchors (section 7).

Label files are append-only CSVs (``labels/rater_<X>.csv``). Re-labelling a cup
appends a new row; the *last* row for a cup in file order is the one that counts.
That makes autosave crash-safe and keeps a full audit trail.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from latte.io import cfg_path, list_images, read_cup_log


def label_columns(dims: list[str]) -> list[str]:
    return ["cup_id", *dims, "unusable", "comment", "timestamp"]


def labels_path(cfg: dict[str, Any], rater: str) -> Path:
    return cfg_path(cfg, "labels_dir") / f"rater_{rater}.csv"


def now_iso() -> str:
    """Local time with UTC offset, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_label(path: Path, row: dict[str, Any], dims: list[str]) -> None:
    """Append one label row and fsync, creating the file with a header if needed."""
    cols = label_columns(dims)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="raise")
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})
        fh.flush()
        os.fsync(fh.fileno())


def read_labels(path: Path, dims: list[str]) -> pd.DataFrame:
    """Every row of a rater's label file, in file order (scores as floats)."""
    cols = label_columns(dims)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame({c: pd.Series(dtype=str) for c in cols})
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[cols].copy()
    for d in dims:
        df[d] = pd.to_numeric(df[d], errors="coerce")
    return df


def latest_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the last row per cup_id (file order), indexed by cup_id."""
    return df.drop_duplicates("cup_id", keep="last").set_index("cup_id")


# ---------------------------------------------------------------------------
# Blind order
# ---------------------------------------------------------------------------


def order_key(seed: int, rater: str, cup_id: str) -> str:
    """Rater-specific pseudo-random sort key for one image.

    Hashing each cup independently (instead of shuffling the whole list) keeps
    the order stable when new images arrive: they slot in at random positions
    among the unlabelled ones rather than reshuffling or piling up at the end.
    """
    return hashlib.sha256(f"{seed}|{rater}|{cup_id}".encode()).hexdigest()


def blind_order(cup_ids: Iterable[str], rater: str, seed: int) -> list[str]:
    return sorted(cup_ids, key=lambda c: order_key(seed, rater, c))


def available_cups(cfg: dict[str, Any]) -> list[str]:
    """Ingested cups the raters should see (optionally minus usable=n in the log)."""
    cups = [p.stem for p in list_images(cfg_path(cfg, "raw"), [".jpg"])]
    if cfg["labelling"].get("skip_log_unusable", True):
        log = read_cup_log(cfg_path(cfg, "cup_log"))
        bad = set(log.loc[log["usable"] == "n", "cup_id"])
        cups = [c for c in cups if c not in bad]
    return cups


def label_queue(
    cups: Iterable[str],
    labelled: pd.DataFrame,
    rater: str,
    seed: int,
    relabel_before: datetime | None = None,
) -> list[str]:
    """Cups still to label, in this rater's blind order.

    ``labelled`` is the rater's latest-label table. With ``relabel_before``,
    cups whose latest label is older than that moment are queued again (used
    after the rubric anchors are revised).
    """
    done: set[str] = set(labelled.index)
    if relabel_before is not None and len(labelled):
        ts = labelled["timestamp"].map(_parse_ts)
        done = {c for c, t in ts.items() if t is not None and t >= relabel_before}
    return blind_order([c for c in cups if c not in done], rater, seed)


def _parse_ts(s: str) -> datetime | None:
    try:
        t = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.astimezone()


def parse_since(s: str) -> datetime:
    """Parse a ``--relabel-before`` value (date or datetime, local time if no offset)."""
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.astimezone()


# ---------------------------------------------------------------------------
# Final labels
# ---------------------------------------------------------------------------


def mean_labels(cfg: dict[str, Any]) -> pd.DataFrame:
    """Per-cup mean of the two raters' latest scores (section 7.3).

    Only cups labelled by every rater are returned. ``unusable_any`` is True if
    any rater ticked unusable; downstream code excludes those cups.
    """
    dims: list[str] = cfg["labelling"]["dimensions"]
    raters: list[str] = cfg["labelling"]["raters"]
    latest = {r: latest_labels(read_labels(labels_path(cfg, r), dims)) for r in raters}
    common = sorted(set.intersection(*(set(t.index) for t in latest.values())))
    out = pd.DataFrame(index=pd.Index(common, name="cup_id"))
    for d in dims:
        out[d] = np.mean([latest[r].loc[common, d].to_numpy(float) for r in raters], axis=0) if common else []
    out["unusable_any"] = [any(latest[r].at[c, "unusable"] == "y" for r in raters) for c in common]
    return out


# ---------------------------------------------------------------------------
# Rubric anchors (read from docs/rubric.md so the tool never drifts from it)
# ---------------------------------------------------------------------------


def parse_rubric_anchors(md_path: Path, dims: list[str]) -> dict[str, dict[str, str]]:
    """Read the anchor table in ``docs/rubric.md``.

    Expects a markdown table whose header starts ``| Dimension | 1 | 3 | 5 |``.
    Returns ``{dim: {"1": text, "3": text, "5": text}}`` for each dimension row
    found (case-insensitive name match); missing rows are simply absent.
    """
    if not md_path.exists():
        return {}
    anchors: dict[str, dict[str, str]] = {}
    in_table = False
    for line in md_path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.strip().startswith("|") else []
        if not cells:
            in_table = False
            continue
        if cells[0].lower() == "dimension":
            in_table = True
            header = cells[1:]
            continue
        if not in_table or re.fullmatch(r"[:\- ]+", "".join(cells)):
            continue
        name = re.sub(r"[*_`]", "", cells[0]).strip().lower()
        if name in dims:
            anchors[name] = {h: re.sub(r"\*\*(.+?)\*\*", r"\1", v) for h, v in zip(header, cells[1:]) if v}
    return anchors
