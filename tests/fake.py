"""Fake studies for the modelling tests: cup log, features.csv, both raters' labels and an embedding cache.

No photos are needed. ``load_embeddings`` only checks the cache against
photos in data/raw, and there are none here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from latte.embed import IDS_COLUMNS, _digest, cache_paths
from latte.features import FEATURES
from latte.io import cfg_path, empty_cup_log, write_cup_log
from latte.labels import append_label, labels_path


def fake_study(cfg: dict, n_sessions: int = 6, per_session: int = 12, signal: bool = True, emb_dim: int = 16,
               pourers: tuple[str, ...] = ("A", "B"), labelled_frac: float = 1.0, seed: int = 0,
               session_effect: float = 0.0) -> pd.DataFrame:
    """Write a complete fake study into the project behind ``cfg``; return the per-cup truth table.

    With ``signal``:
    * the Overall score rises with contrast_dL and symmetry_iou;
    * contrast_dL also tracks milk condition (thin > good > thick);
    * coverage tracks drop-in;
    * two embedding dimensions carry the Overall score.

    ``session_effect`` adds a per-session quality offset (SD in quality units),
    as when the pourers improve over time.

    Without it, every feature and embedding column is noise.
    """
    rng = np.random.default_rng(seed)
    milk, drop = cfg["design"]["milk_conditions"], cfg["design"]["dropin"]
    cells = [(m, d) for m in milk for d in drop]
    log, feats, truth = empty_cup_log(), [], []
    for s in range(1, n_sessions + 1):
        pourer = pourers[(s - 1) % len(pourers)]
        offset = rng.normal(0, session_effect) if session_effect else 0.0
        for k, i in enumerate(rng.permutation(per_session), 1):
            m, d = cells[i % 6]
            cid = f"S{s:02d}-C{k:02d}"
            log.loc[len(log)] = [cid, f"S{s:02d}", "2026-11-26", pourer, m, d,
                                 str(10 + 20 * milk.index(m) + rng.normal(0, 3)), "", "20", f"{cid}.jpg", "y", ""]
            f = {c: rng.normal() for c in FEATURES}
            f["n_components"] = float(rng.integers(1, 3))
            quality = rng.normal() + offset
            if signal:
                f["contrast_dL"] = 30 - 10 * milk.index(m) + 4 * quality + rng.normal(0, 1)
                f["symmetry_iou"] = 0.8 + 0.08 * quality + rng.normal(0, 0.02)
                f["coverage"] = 0.35 - 0.15 * drop.index(d) + rng.normal(0, 0.03)
            overall = float(np.clip(3 + 1.1 * quality - 0.4 * milk.index(m) * signal, 1, 5)) if signal else rng.uniform(1, 5)
            feats.append({"cup_id": cid, **f})
            truth.append({"cup_id": cid, "quality": quality, "overall": overall})
    write_cup_log(log, cfg_path(cfg, "cup_log"))
    fdir = cfg_path(cfg, "features_dir")
    fdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feats).to_csv(fdir / "features.csv", index=False)
    truth_df = pd.DataFrame(truth).set_index("cup_id")

    dims = cfg["labelling"]["dimensions"]
    n_lab = int(round(labelled_frac * len(truth_df)))
    for r in cfg["labelling"]["raters"]:
        for cid in truth_df.index[:n_lab]:
            base = truth_df.at[cid, "overall"]
            row = {d: int(np.clip(round(base + rng.normal(0, 0.5)), 1, 5)) for d in dims}
            append_label(labels_path(cfg, r), {"cup_id": cid, **row, "unusable": "n", "comment": "",
                                               "timestamp": "2026-11-27T10:00:00-08:00"}, dims)

    emb = rng.normal(size=(len(truth_df), emb_dim)).astype(np.float32)
    if signal:
        emb[:, 0] += 1.5 * truth_df["quality"].to_numpy()
        emb[:, 1] -= 1.0 * truth_df["quality"].to_numpy()
    write_fake_embeddings(cfg, list(truth_df.index), emb)
    return truth_df


def write_fake_embeddings(cfg: dict, cups: list[str], emb: np.ndarray) -> None:
    model = cfg["embed"]["model"]
    npy, ids_p, meta_p = cache_paths(cfg, model, bool(cfg["embed"].get("pretrained", True)))
    ids = pd.DataFrame({"cup_id": cups, "key": [f"k{i}" for i in range(len(cups))],
                        "circle_source": "accepted", "circle_checked": "y"}, columns=IDS_COLUMNS)
    np.save(npy, emb)
    ids.to_csv(ids_p, index=False)
    meta_p.write_text(json.dumps({"model": model, "pretrained": True, "dim": int(emb.shape[1]),
                                  "digest": _digest(emb, ids)}))


def small_grids(project, **train) -> None:
    """Shrink the grids and permutation repeats so modelling tests run in seconds."""
    p = project / "configs" / "default.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["grids"] = {"ridge_alpha": [0.1, 10.0], "pca_components": [2, 4], "logreg_C": [0.1, 1.0],
                    "rf_n_estimators": [30], "rf_max_depth": [3]}
    raw["embed"]["model"] = "fake_model"
    raw["train"] = {**raw.get("train", {}), "importance_repeats": 2, "leakage_repeats": 6, "n_jobs": 1, **train}
    raw["labelling"]["bootstrap_n"] = 100
    p.write_text(yaml.safe_dump(raw))
