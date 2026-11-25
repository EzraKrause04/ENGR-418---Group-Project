"""Assemble the analysis table: features + cup log + mean rater labels (section 10.1).

One row per cup_id. A cup is excluded if any of these holds:

* ``usable = n`` in the cup log (case-insensitive);
* its session is listed in ``design.exclude_sessions``;
* *either* rater's latest label ticks ``unusable``, even if the other rater
  hasn't labelled the cup yet;
* it has no row in features/features.csv (not ingested, or features not built).

Cups that are not yet labelled by both raters are kept by default, with NaN
scores; the go/no-go check needs them for Task B. Pass
``require_labels=True`` to keep only doubly-labelled cups (Task A).

The cup log is validated first (``latte.io.validate_cup_log``). A hand edit
that breaks it, such as a duplicated row, raises ``ValueError`` naming the problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from latte.features import FEATURES
from latte.io import cfg_path, read_cup_log, validate_cup_log
from latte.labels import labels_path, latest_labels, mean_labels, read_labels

LOG_COLUMNS = ["session_id", "date", "pourer", "milk_condition", "dropin", "foam_gain_pct", "milk_temp_c", "seconds_to_photo", "usable"]


@dataclass
class Dataset:
    table: pd.DataFrame  # indexed by cup_id
    excluded: dict[str, list[str]] = field(default_factory=dict)  # reason -> cup_ids

    def summary(self) -> str:
        parts = [f"{len(self.table)} cups kept"]
        parts += [f"{len(v)} excluded ({k})" for k, v in self.excluded.items() if v]
        return ", ".join(parts)


def features_path(cfg: dict[str, Any]) -> Path:
    return cfg_path(cfg, "features_dir") / "features.csv"


def rater_unusable(cfg: dict[str, Any]) -> set[str]:
    """Cups whose latest label from *any* rater has unusable = y."""
    dims: list[str] = cfg["labelling"]["dimensions"]
    bad: set[str] = set()
    for r in cfg["labelling"]["raters"]:
        lab = latest_labels(read_labels(labels_path(cfg, r), dims))
        bad |= set(lab.index[lab["unusable"].astype(str).str.strip().str.lower() == "y"])
    return bad


def load_dataset(cfg: dict[str, Any], require_labels: bool = False) -> Dataset:
    """Join features, cup log and mean labels, applying the exclusions above."""
    fpath = features_path(cfg)
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} not found: run make features first")
    feats = pd.read_csv(fpath, dtype={"cup_id": str}).set_index("cup_id")
    log = read_cup_log(cfg_path(cfg, "cup_log"))
    log["usable"] = log["usable"].str.lower()
    problems = validate_cup_log(log, cfg)
    if problems:
        raise ValueError("data/cup_log.csv has problems; fix them first:\n  " + "\n  ".join(problems))
    log = log.set_index("cup_id")
    dims: list[str] = cfg["labelling"]["dimensions"]

    excluded: dict[str, list[str]] = {}
    raw = cfg_path(cfg, "raw")
    unbuilt = [c for c in log.index if log.at[c, "usable"] != "n" and c not in feats.index and (raw / f"{c}.jpg").exists()]
    excluded["ingested but no features (run make features)"] = sorted(unbuilt)
    no_log = sorted(set(feats.index) - set(log.index))
    excluded["no cup-log row"] = no_log
    t = feats.drop(index=no_log).join(log[LOG_COLUMNS], how="inner")
    bad_log = t.index[t["usable"] == "n"]
    excluded["usable=n in cup log"] = sorted(bad_log)
    t = t.drop(index=bad_log)
    skip = set(cfg["design"].get("exclude_sessions") or [])
    in_skip = t.index[t["session_id"].isin(skip)]
    excluded["excluded session"] = sorted(in_skip)
    t = t.drop(index=in_skip)
    rater_bad = sorted(set(t.index) & rater_unusable(cfg))
    excluded["unusable by a rater"] = rater_bad
    t = t.drop(index=rater_bad)

    t = t.join(mean_labels(cfg)[dims], how="left")
    if require_labels:
        unl = t.index[t[dims].isna().any(axis=1)]
        excluded["not labelled by both raters"] = sorted(unl)
        t = t.drop(index=unl)
    t["labelled"] = t[dims].notna().all(axis=1)
    for c in ("foam_gain_pct", "milk_temp_c", "seconds_to_photo", *FEATURES):
        t[c] = pd.to_numeric(t[c], errors="coerce").astype(float)
    return Dataset(t.sort_index(), excluded)
