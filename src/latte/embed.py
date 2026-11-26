"""Pretrained-network embeddings of every ingested cup, averaged over 4 rotations and cached (section 9).

    make embed                         # python -m latte.embed --config configs/default.yaml
    python -m latte.embed --force      # recompute every cup
    python -m latte.embed --fetch      # only download / check the weights (do this once while online)

For each photo in data/raw, it takes the circle (a saved one from
data/circles.csv, otherwise auto-detection) and makes the 512 px masked crop
(``latte.preprocess``). It then:

1. resizes the crop to the network's input size (area averaging; 518 x 518 for
   DINOv2 ViT-S/14) and normalises it with the network's mean and std;
2. runs the frozen network with its classifier head removed
   (``num_classes=0``), giving one vector per image (384 numbers for ViT-S/14,
   2048 for ResNet-50);
3. averages that vector over the rotations in ``embed.rotations`` (0, 90, 180
   and 270 degrees). A cup has no "up", so rotation is free augmentation. The
   average over the four rotations is unchanged when the crop itself is
   rotated by a multiple of 90 degrees.

Model: ``embed.model``, a timm name (default ``vit_small_patch14_dinov2.lvd142m``).
An unknown name is an error. Only when the weights can't be *downloaded*
(network or hub error) does it fall back to torchvision ResNet-50 with ImageNet
weights (``embed.fallback: resnet50``, as section 9 asks), and it says so loudly.
A fallback writes its own cache (``emb_resnet50.tv_in1k.*``). It never
overwrites the main model's cache.

Cache, in ``features/``, one set per model (with ``_random`` added to the name
when ``embed.pretrained`` is false, which is for tests only):

    emb_<model>.npy        float32 array, one row per cup
    emb_<model>_ids.csv    cup_id, key, circle_source, circle_checked, in .npy row order. The key
                           changes when the photo, its circle, the model or the settings change.
    emb_<model>.json       model, dimension, rotations, input size, versions, and a hash that ties
                           the .npy and the ids file together (written last)

A rerun embeds only cups that are new or whose key changed, and drops cups that
are no longer in data/raw. A cache whose files don't match each other is
ignored and rebuilt. PCA is *not* applied here: it belongs inside each
cross-validation training fold (build step 8). Read the cache with
``load_embeddings(cfg)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from latte.geometry import circles_path, geometry_params, read_circles, resolve_circle  # noqa: E402
from latte.io import DEFAULT_CONFIG, cfg_path, cup_sort_key, list_images, load_config, load_rgb  # noqa: E402
from latte.preprocess import prepare_cfg, preprocess_params  # noqa: E402

EMBED_DEFAULTS: dict[str, Any] = {
    "model": "vit_small_patch14_dinov2.lvd142m",
    "fallback": "resnet50",
    "rotations": [0, 90, 180, 270],
    "pretrained": True,
    "threads": 0,  # 0 = all CPU cores
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FALLBACK_NAME = "resnet50.tv_in1k"
IDS_COLUMNS = ["cup_id", "key", "circle_source", "circle_checked"]


class WeightsUnavailable(RuntimeError):
    """Neither the configured model nor the fallback could be loaded."""


def embed_params(cfg: dict[str, Any] | None) -> dict[str, Any]:
    return {**EMBED_DEFAULTS, **((cfg or {}).get("embed") or {})}


@dataclass
class Embedder:
    """A frozen feature extractor and the preprocessing it expects."""

    name: str  # model actually used
    model: torch.nn.Module
    size: tuple[int, int]  # (H, W) input
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    dim: int
    fell_back: bool = False
    error: str = ""  # why the configured model failed, if it did

    def tensor(self, crops: list[np.ndarray]) -> torch.Tensor:
        """RGB uint8 crops -> normalised float tensor (N, 3, H, W)."""
        h, w = self.size
        arr = np.stack([
            cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA if c.shape[0] > h else cv2.INTER_CUBIC) for c in crops
        ]).astype(np.float32) / 255.0
        arr = (arr - np.asarray(self.mean, np.float32)) / np.asarray(self.std, np.float32)
        return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())

    @torch.inference_mode()
    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        return self.model(self.tensor(crops)).float().cpu().numpy()


def _output_dim(model: torch.nn.Module, size: tuple[int, int]) -> int:
    with torch.inference_mode():
        return int(model(torch.zeros(1, 3, *size)).shape[1])


def _timm_embedder(name: str, pretrained: bool) -> Embedder:
    import timm

    model = timm.create_model(name, pretrained=pretrained, num_classes=0).eval()
    dc = timm.data.resolve_model_data_config(model)
    _, h, w = dc["input_size"]
    return Embedder(name, model, (h, w), tuple(dc["mean"]), tuple(dc["std"]), _output_dim(model, (h, w)))


def _torchvision_resnet50(pretrained: bool) -> Embedder:
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = torch.nn.Identity()
    return Embedder(FALLBACK_NAME, model.eval(), (224, 224), IMAGENET_MEAN, IMAGENET_STD, 2048)


def is_download_error(err: BaseException) -> bool:
    """True for failures to fetch weights (network, hub, missing local file), not for bugs or typos."""
    import urllib.error

    try:
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        hub = (HfHubHTTPError, LocalEntryNotFoundError)
    except ImportError:  # pragma: no cover
        hub = ()
    kinds = (ConnectionError, TimeoutError, urllib.error.URLError, OSError, *hub)
    e: BaseException | None = err
    while e is not None:
        if isinstance(e, kinds):
            return True
        e = e.__cause__ or e.__context__
    return False


def load_embedder(p: dict[str, Any]) -> Embedder:
    """The configured timm model; torchvision ResNet-50 only if its weights can't be downloaded.

    Raises ValueError for an unknown model name, and WeightsUnavailable if the
    fallback can't be loaded either.
    """
    import timm

    name, pretrained = str(p["model"]), bool(p["pretrained"])
    try:
        from timm.models import get_pretrained_cfg

        known = timm.is_model(name) and get_pretrained_cfg(name) is not None
    except RuntimeError:  # a known architecture with a misspelt pretrained tag
        known = False
    if not known:
        raise ValueError(f"embed.model {name!r} is not a timm model name (check for a typo)")
    try:
        return _timm_embedder(name, pretrained)
    except Exception as err:
        if not is_download_error(err) or str(p.get("fallback") or "") != "resnet50":
            raise
        primary = f"{type(err).__name__}: {err}"
    try:
        emb = _torchvision_resnet50(pretrained)
    except Exception as err2:
        raise WeightsUnavailable(
            f"Could not load {name} ({primary.splitlines()[0]}) or the ResNet-50 fallback "
            f"({type(err2).__name__}: {str(err2).splitlines()[0] if str(err2) else ''}).\n"
            "Connect to the internet and run `python -m latte.embed --fetch` once; the weights are then cached."
        ) from err2
    emb.fell_back, emb.error = True, primary
    return emb


def rotation_average(embedder: Embedder, crop: np.ndarray, rotations: list[int]) -> np.ndarray:
    """mean over r in rotations of f(rot_r(crop)); rotations must be multiples of 90 degrees."""
    if any(r % 90 for r in rotations):
        raise ValueError("embed.rotations must be multiples of 90 degrees")
    views = [np.rot90(crop, k=(r // 90) % 4).copy() for r in rotations]
    return embedder(views).mean(axis=0)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_name(model_name: str, pretrained: bool) -> str:
    s = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_name)
    return s if pretrained else f"{s}_random"


def cache_paths(cfg: dict[str, Any], model_name: str, pretrained: bool = True) -> tuple[Path, Path, Path]:
    d = cfg_path(cfg, "features_dir")
    s = cache_name(model_name, pretrained)
    return d / f"emb_{s}.npy", d / f"emb_{s}_ids.csv", d / f"emb_{s}.json"


def cup_key(photo: Path, circle: Any, model_name: str, p: dict[str, Any], pp: dict[str, Any], seed: int) -> str:
    """Hash of everything that determines a cup's embedding (seed only matters for random weights)."""
    h = hashlib.sha256(photo.read_bytes())
    h.update(json.dumps({
        "circle": [round(circle.cx, 2), round(circle.cy, 2), round(circle.r, 2)], "model": model_name,
        "pretrained": bool(p["pretrained"]), "rotations": list(p["rotations"]),
        "size": int(pp["size"]), "shrink": float(pp["radius_shrink"]),
        "seed": None if p["pretrained"] else seed,
    }, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _digest(emb: np.ndarray, ids: pd.DataFrame) -> str:
    h = hashlib.sha256(np.ascontiguousarray(emb, np.float32).tobytes())
    h.update("\n".join(ids["cup_id"] + "|" + ids["key"]).encode())
    return h.hexdigest()[:16]


def _empty_ids() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=str) for c in IDS_COLUMNS})


def load_cache(cfg: dict[str, Any], model_name: str, pretrained: bool = True) -> tuple[np.ndarray | None, pd.DataFrame, dict, str]:
    """(embeddings, ids, meta, problem). A missing or inconsistent cache gives (None, empty, {}, why)."""
    npy, ids_p, meta_p = cache_paths(cfg, model_name, pretrained)
    if not (npy.exists() and ids_p.exists() and meta_p.exists()):
        return None, _empty_ids(), {}, "missing"
    try:
        emb = np.load(npy)
        ids = pd.read_csv(ids_p, dtype=str, keep_default_na=False)
        meta = json.loads(meta_p.read_text())
    except (ValueError, OSError, EOFError, pd.errors.ParserError, json.JSONDecodeError) as err:
        return None, _empty_ids(), {}, f"unreadable ({type(err).__name__})"
    if emb.ndim != 2 or not set(IDS_COLUMNS) <= set(ids.columns) or len(ids) != len(emb) or ids["cup_id"].duplicated().any():
        return None, _empty_ids(), {}, "files don't match each other"
    if meta.get("digest") != _digest(emb, ids):
        return None, _empty_ids(), {}, "files don't match each other (interrupted write?)"
    return emb, ids[IDS_COLUMNS], meta, ""


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def stale_cups(cfg: dict[str, Any], ids: pd.DataFrame, model_name: str, p: dict[str, Any]) -> list[str]:
    """Photos in data/raw whose cached key no longer matches (photo, circle or settings changed since make embed)."""
    pp, gp = preprocess_params(cfg), geometry_params(cfg)
    raw = cfg_path(cfg, "raw")
    saved = read_circles(circles_path(cfg))
    keys = dict(zip(ids["cup_id"], ids["key"]))
    out = []
    for photo in list_images(raw, [".jpg"]):
        cup = photo.stem
        if cup not in keys:
            continue
        circle = resolve_circle(cup, load_rgb(photo), saved, gp, run_auto=False).circle
        if cup_key(photo, circle, model_name, p, pp, int(cfg["seed"])) != keys[cup]:
            out.append(cup)
    return sorted(out, key=cup_sort_key)


def load_embeddings(cfg: dict[str, Any], verify: bool = True) -> pd.DataFrame:
    """Cached embeddings indexed by cup_id (columns emb_000, ...), checked against data/raw.

    It uses the configured model's cache. Only if that cache is missing,
    doesn't cover every photo in data/raw, or (with ``verify``) holds an
    embedding made from an older photo or circle, does it use a complete,
    up-to-date ResNet-50 fallback cache, with a printed warning. The returned
    frame has ``attrs['model']``, ``attrs['fell_back']`` and
    ``attrs['pretrained']`` set. Rows for cups whose photo has since left
    data/raw are kept, and callers join on their own cup list. Raises
    FileNotFoundError (naming what is wrong) when no cache is complete and up to
    date, because stale embeddings must never reach a model.
    """
    p = embed_params(cfg)
    pre = bool(p["pretrained"])
    raw = cfg_path(cfg, "raw")
    cups = {ph.stem for ph in list_images(raw, [".jpg"])}
    tried = []
    for name, fell in ((str(p["model"]), False), (FALLBACK_NAME, True)):
        emb, ids, meta, problem = load_cache(cfg, name, pre)
        if emb is None:
            tried.append(f"{name}: {problem}")
            continue
        missing = sorted(cups - set(ids["cup_id"]))
        if missing:
            tried.append(f"{name}: no embedding for {len(missing)} photo(s), e.g. {', '.join(missing[:3])}")
            continue
        stale = stale_cups(cfg, ids, name, p) if verify else []
        if stale:
            tried.append(f"{name}: {len(stale)} embedding(s) out of date (photo or circle changed), e.g. {', '.join(stale[:3])}")
            continue
        if fell:
            print(f"WARNING: using the ResNet-50 fallback embeddings, because the {p['model']} cache is not "
                  f"usable ({tried[0].split(': ', 1)[1]}). Results will be labelled {FALLBACK_NAME}.", file=sys.stderr)
        cols = [f"emb_{i:03d}" for i in range(emb.shape[1])]
        df = pd.DataFrame(emb, index=pd.Index(ids["cup_id"], name="cup_id"), columns=cols)
        df.attrs.update(model=name, fell_back=fell, pretrained=pre, meta=meta)
        return df
    raise FileNotFoundError("no up-to-date embedding cache; run make embed. (" + "; ".join(tried) + ")")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(cfg: dict[str, Any], force: bool = False, fetch_only: bool = False) -> int:
    p, pp, gp = embed_params(cfg), preprocess_params(cfg), geometry_params(cfg)
    seed = int(cfg["seed"])
    torch.set_num_threads(int(p["threads"]) or (os.cpu_count() or 1))
    torch.manual_seed(seed)
    raw = cfg_path(cfg, "raw")
    cups = sorted((ph.stem for ph in list_images(raw, [".jpg"])), key=cup_sort_key)
    if not cups and not fetch_only:
        print(f"No photos in {raw}. Run make ingest first.")
        return 0
    weights = "pretrained weights" if p["pretrained"] else "RANDOM weights (tests only; not for results)"
    print(f"Loading {p['model']} with {weights}. The first run downloads the weights; "
          "offline, the hub can take about a minute to give up.", flush=True)
    try:
        embedder = load_embedder(p)
    except (ValueError, WeightsUnavailable) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    banner = ""
    if embedder.fell_back:
        banner = (f"WARNING: {p['model']} could not be downloaded ({embedder.error.splitlines()[0]}).\n"
                  f"  Using the fallback {embedder.name}; its cache is separate and the main model's cache is untouched.\n"
                  f"  Run `python -m latte.embed --fetch` while online, then `make embed` again.")
        print(banner, flush=True)
    if fetch_only:
        print(f"OK: {embedder.name} is available ({embedder.dim} dims).")
        return 0 if not embedder.fell_back else 1

    old, old_ids, _, problem = (None, _empty_ids(), {}, "") if force else load_cache(cfg, embedder.name, bool(p["pretrained"]))
    if problem and problem != "missing":
        print(f"  note: the existing cache is {problem}; rebuilding it.")
    prev = {} if old is None else {c: (k, old[i]) for i, (c, k) in enumerate(zip(old_ids["cup_id"], old_ids["key"]))}

    saved = read_circles(circles_path(cfg))
    rows, idx, n_new, notes = [], [], 0, []
    for i, cup in enumerate(cups, 1):
        photo = raw / f"{cup}.jpg"
        rgb = load_rgb(photo)
        res = resolve_circle(cup, rgb, saved, gp, run_auto=False)
        if res.note:
            notes.append(f"{cup}: {res.note}")
        key = cup_key(photo, res.circle, embedder.name, p, pp, seed)
        if cup in prev and prev[cup][0] == key:
            vec = prev[cup][1]
        else:
            vec = rotation_average(embedder, prepare_cfg(rgb, res.circle, cfg).crop, list(p["rotations"]))
            n_new += 1
        rows.append(np.asarray(vec, np.float32))
        idx.append({"cup_id": cup, "key": key, "circle_source": res.circle.source,
                    "circle_checked": "y" if res.circle.checked else "n"})
        if i % 10 == 0:
            print(f"  {i}/{len(cups)} cups ({n_new} embedded, {i - n_new} from cache)", flush=True)

    emb = np.stack(rows)
    ids = pd.DataFrame(idx, columns=IDS_COLUMNS)
    npy, ids_p, meta_p = cache_paths(cfg, embedder.name, bool(p["pretrained"]))
    import timm
    import torchvision

    meta = {
        "model": embedder.name, "requested_model": p["model"], "fell_back": embedder.fell_back,
        "fallback_reason": embedder.error.splitlines()[0] if embedder.error else "",
        "pretrained": bool(p["pretrained"]), "dim": int(emb.shape[1]), "input_size": list(embedder.size),
        "mean": list(embedder.mean), "std": list(embedder.std), "rotations": list(p["rotations"]),
        "n_cups": len(cups), "digest": _digest(emb, ids),
        "versions": {"torch": torch.__version__, "timm": timm.__version__, "torchvision": torchvision.__version__},
    }
    _atomic(npy, lambda t: np.save(t, emb))
    _atomic(ids_p, lambda t: ids.to_csv(t, index=False))
    _atomic(meta_p, lambda t: Path(t).write_text(json.dumps(meta, indent=2) + "\n"))  # last: commits the set

    dropped = sorted(set(prev) - set(cups))
    rel = lambda q: q.relative_to(cfg["_root"]) if q.is_relative_to(cfg["_root"]) else q  # noqa: E731
    print(f"Embeddings for {len(cups)} cups with {embedder.name} ({emb.shape[1]} dims, averaged over "
          f"{len(p['rotations'])} rotations): {n_new} computed, {len(cups) - n_new} reused"
          + (f", {len(dropped)} dropped (no longer in data/raw)" if dropped else "") + ".")
    unchecked = int((ids["circle_checked"] == "n").sum())
    if unchecked:
        print(f"  {unchecked} cup{'s' * (unchecked != 1)} use a circle nobody has checked yet, so those embeddings "
              "are provisional. Run make circles.")
    for n in notes:
        print(f"  {n}")
    print(f"  wrote {rel(npy)}, {rel(ids_p)} and {rel(meta_p)}")
    if not p["pretrained"]:
        print("  NOTE: random weights; these embeddings are for tests only.")
    if banner:
        print(banner)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--force", action="store_true", help="recompute every cup, ignoring the cache")
    ap.add_argument("--fetch", action="store_true", help="only download/check the model weights, then stop")
    args = ap.parse_args(argv)
    return run(load_config(args.config), args.force, args.fetch)


if __name__ == "__main__":
    sys.exit(main())
