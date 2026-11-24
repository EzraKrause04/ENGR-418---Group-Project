"""Config, cup-log, image and table I/O shared by every script.

Nothing in here knows about models; it only moves data on and off disk in the
formats fixed by the project brief (sections 6.4, 6.5 and 10.1).
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load a YAML config and resolve its repository root.

    The config's ``root`` key is interpreted relative to the config file, so
    scripts behave the same whatever the current working directory is.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    cfg_path = cfg_path.resolve()
    with cfg_path.open() as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    cfg["_root"] = (cfg_path.parent / cfg.get("root", "..")).resolve()
    cfg["_config_path"] = cfg_path
    return cfg


def cfg_path(cfg: dict[str, Any], key: str) -> Path:
    """Absolute path for ``cfg['paths'][key]``."""
    return Path(cfg["_root"]) / cfg["paths"][key]


# ---------------------------------------------------------------------------
# Cup log (section 6.4)
# ---------------------------------------------------------------------------

CUP_LOG_COLUMNS: list[str] = [
    "cup_id",
    "session_id",
    "date",
    "pourer",
    "milk_condition",
    "dropin",
    "foam_gain_pct",
    "milk_temp_c",
    "seconds_to_photo",
    "photo_file",
    "usable",
    "notes",
]

CUP_ID_RE = re.compile(r"^(S\d{2})-C(\d{2,3})$")


def cup_sort_key(cup_id: str) -> tuple[str, int, str]:
    """Chronological sort key: session, then cup number (so C100 sorts after C99)."""
    m = CUP_ID_RE.match(cup_id)
    return (m.group(1), int(m.group(2)), "") if m else ("~", 0, cup_id)


def empty_cup_log() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=str) for c in CUP_LOG_COLUMNS})


def read_cup_log(path: Path) -> pd.DataFrame:
    """Read the cup log as strings (blank cells stay ``""``, never NaN)."""
    if not path.exists():
        return empty_cup_log()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in CUP_LOG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[CUP_LOG_COLUMNS].copy()
    for c in CUP_LOG_COLUMNS:
        df[c] = df[c].astype(str).str.strip()
    return df


def write_cup_log(df: pd.DataFrame, path: Path) -> None:
    """Write the cup log atomically, so an interrupted write can't truncate it."""
    _atomic_write_csv(df[CUP_LOG_COLUMNS], path)


def validate_cup_log(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Return human-readable problems with the cup log (empty list = valid)."""
    design = cfg["design"]
    problems: list[str] = []
    dup = df["cup_id"][df["cup_id"].duplicated()].unique()
    if len(dup):
        problems.append(f"duplicate cup_id: {', '.join(dup)}")
    for i, row in df.iterrows():
        where = f"row {int(i) + 2} ({row['cup_id'] or 'no cup_id'})"
        m = CUP_ID_RE.match(row["cup_id"])
        if not m:
            problems.append(f"{where}: cup_id must look like S05-C03")
        elif m.group(1) != row["session_id"]:
            problems.append(f"{where}: session_id {row['session_id']!r} does not match cup_id")
        if row["milk_condition"] not in design["milk_conditions"]:
            problems.append(f"{where}: milk_condition {row['milk_condition']!r} not in {design['milk_conditions']}")
        if row["dropin"] not in design["dropin"]:
            problems.append(f"{where}: dropin {row['dropin']!r} not in {design['dropin']}")
        if row["usable"] not in ("", "y", "n"):
            problems.append(f"{where}: usable must be y, n or blank, got {row['usable']!r}")
        for col in ("foam_gain_pct", "milk_temp_c", "seconds_to_photo"):
            if row[col] and not _is_number(row[col]):
                problems.append(f"{where}: {col} {row[col]!r} is not a number")
    return problems


def _is_number(s: str) -> bool:
    try:
        return math.isfinite(float(s))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def strip_and_resize(src: Path, dst: Path, max_side: int, quality: int = 95) -> tuple[int, int]:
    """Write an EXIF-free, upright copy of ``src`` with longest side ``max_side``.

    The EXIF orientation is applied to the pixels first (so the photo stays
    upright once the tag is gone), then the pixels are copied into a brand-new
    image so no EXIF, GPS, XMP or ICC metadata can survive the save.
    Returns the output (width, height).
    """
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale != 1:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        clean = Image.frombytes("RGB", im.size, im.tobytes())
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, suffix=".jpg")
    os.close(fd)
    try:
        clean.save(tmp, "JPEG", quality=quality, optimize=True)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return clean.size


def has_metadata(path: Path) -> bool:
    """True if the image file carries EXIF, XMP or an ICC profile."""
    with Image.open(path) as im:
        return bool(im.getexif()) or any(k in im.info for k in ("exif", "xmp", "XML:com.adobe.xmp", "icc_profile"))


def list_images(folder: Path, extensions: Iterable[str]) -> list[Path]:
    """Image files directly inside ``folder`` (case-insensitive extension match)."""
    if not folder.exists():
        return []
    exts = {e.lower() for e in extensions}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


# ---------------------------------------------------------------------------
# Results tables (CSV + LaTeX)
# ---------------------------------------------------------------------------

_TEX_SPECIAL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def tex_escape(s: str) -> str:
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in str(s))


def to_latex(df: pd.DataFrame, caption: str, label: str, float_fmt: str = "{:.2f}") -> str:
    """A booktabs table for ``\\input{}`` into the IEEEtran report."""
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "--" if math.isnan(v) else float_fmt.format(v)
        return tex_escape(v)

    cols = "l" + "r" * (len(df.columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        " & ".join(tex_escape(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(fmt(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_table(df: pd.DataFrame, stem: Path, caption: str, label: str, float_fmt: str = "{:.2f}") -> tuple[Path, Path]:
    """Write ``stem.csv`` and ``stem.tex``; returns both paths."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = stem.with_suffix(".csv"), stem.with_suffix(".tex")
    _atomic_write_csv(df, csv_path)
    tex_path.write_text(to_latex(df, caption, label, float_fmt))
    return csv_path, tex_path


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
