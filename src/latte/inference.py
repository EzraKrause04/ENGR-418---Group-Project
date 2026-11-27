"""One photo in, overlays and feedback out: the pipeline behind the demo (section 12).

The steps mirror training exactly, so the models see the same kind of input:

1. ``as_ingested``: apply the EXIF orientation, convert to RGB, resize so the
   longest side is ``ingest.max_side`` (1024, LANCZOS), and round-trip through
   JPEG at ``ingest.jpeg_quality``, as ``make ingest`` does. No metadata survives.
2. Find the liquid circle (``latte.geometry.detect_circle``).
3. Make the 512 px masked crop (``latte.preprocess``).
4. Segment the milk (``latte.segment``) and compute the handcrafted features.
5. Compute the embedding, only when the bundle's models need one, with the same
   network and rotation averaging as ``make embed``.
6. Run ``latte.feedback.make_feedback``.

Without a trained bundle (``make final`` not run yet), steps 1-4 still run and
the overlays and feature values are shown, with no score or advice.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pathlib import Path

from PIL import Image, ImageOps

from latte.features import extract_features, feature_params
from latte.feedback import Feedback, make_feedback
from latte.geometry import Circle, detect_circle, geometry_params
from latte.io import to_8bit_rgb
from latte.overlay import (CYAN, GREEN, MAGENTA, RED, WHITE, YELLOW, draw_axis, draw_circle, draw_outline, draw_point,
                           full_view, zoom_view)
from latte.preprocess import prepare_cfg, preprocess_params
from latte.segment import lab, segment, segment_params


def as_ingested(src: Image.Image | str | Path, max_side: int = 1024, quality: int = 95) -> np.ndarray:
    """An uploaded photo processed like ``make ingest``: upright, 8-bit RGB, longest side max_side, JPEG round trip.

    A file that ``make ingest`` already produced (a JPEG of exactly max_side with
    no orientation tag) is read as is, like build_features does, so it isn't
    JPEG-compressed a second time.
    """
    if isinstance(src, (str, Path)):
        with Image.open(src) as im0:
            if (im0.format == "JPEG" and max(im0.size) == max_side and im0.mode == "RGB"
                    and not im0.getexif().get(0x0112)):
                return np.asarray(im0).copy()
            im = im0.copy()
            im.info.update(im0.info)
            exif = im0.getexif()
        if exif:
            im.getexif().update(exif)
        src = im
    im = to_8bit_rgb(ImageOps.exif_transpose(src))
    w, h = im.size
    s = max_side / max(w, h)
    if s != 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    Image.frombytes("RGB", im.size, im.tobytes()).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as out:
        return np.asarray(out.convert("RGB")).copy()


@dataclass
class Analysis:
    photo: np.ndarray  # the photo as ingested
    circle: Circle
    circle_ok: bool  # confidence >= geometry.low_confidence
    photo_overlay: np.ndarray  # zoomed view of the cup with the detected circle (whole photo if unsure)
    crop_overlay: np.ndarray  # 512 crop with milk outline, centroid and symmetry axis
    features: dict[str, float]
    feedback: Feedback | None  # None unless status == "scored" (or "unmeasurable")
    status: str = "scored"  # scored | no_model | no_rim | unmeasurable | error
    notes: list[str] = field(default_factory=list)


class Embedder:
    """Lazily loads the network that the bundle's embedding models were trained with."""

    def __init__(self, cfg: dict[str, Any], bundle: dict[str, Any] | None):
        self.cfg, self.bundle, self._emb = cfg, bundle, None

    @property
    def needed(self) -> bool:
        return bool(self.bundle and self.bundle.get("emb_cols"))

    def warm(self) -> None:
        """Load the network now (at startup), so a missing download fails before anyone presses Analyse."""
        if self.needed:
            self(np.zeros((64, 64, 3), np.uint8))

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        """Rotation-averaged embedding from the *same* network the models were trained on (no silent fallback)."""
        from latte.embed import FALLBACK_NAME, _torchvision_resnet50, embed_params, load_embedder, rotation_average

        p = embed_params(self.cfg)
        info = (self.bundle or {}).get("embedding") or {}
        if self._emb is None:
            pretrained = bool(info.get("pretrained", p["pretrained"]))
            if not pretrained:  # random weights: the same seeded network as make embed
                import torch

                torch.manual_seed(int(self.cfg.get("seed", 42)))
            if info.get("model") == FALLBACK_NAME:
                self._emb = _torchvision_resnet50(pretrained)
            else:
                self._emb = load_embedder({**p, "model": info.get("model") or p["model"], "pretrained": pretrained,
                                           "fallback": ""})
        return rotation_average(self._emb, crop, list(p["rotations"]))


def analyse(im: Image.Image | str | Path, cfg: dict[str, Any], bundle: dict[str, Any] | None,
            embedder: Embedder | None = None, view_size: int = 640) -> Analysis:
    """Run steps 1-6 on one uploaded photo.

    No score, fixes or diagnosis are given when the rim wasn't found reliably
    (status ``no_rim``): the models were trained only on checked circles, so their
    error bars say nothing about a crop the detector itself doubts. A failure in
    the embedding or feedback step keeps the overlays and says why (``error``).
    """
    ing = cfg.get("ingest") or {}
    rgb = as_ingested(im, int(ing.get("max_side", 1024)), int(ing.get("jpeg_quality", 95)))
    gp = geometry_params(cfg)
    circle = detect_circle(rgb, gp)
    ok = circle.confidence >= float(gp["low_confidence"])
    notes = [] if ok else [f"Couldn't find the cup rim reliably (confidence {circle.confidence:.2f}), so there is no score. "
                           "Photograph straight down with the whole rim in frame."]

    # Zoom in on the cup (a little room around the rim) so it reads well on screen; whole photo if the rim is unsure.
    photo_ov, view = zoom_view(rgb, circle, view_size, margin=1.8) if ok else full_view(rgb, view_size)
    shrink = float(preprocess_params(cfg)["radius_shrink"])
    draw_circle(photo_ov, view, circle, GREEN if ok else RED, 3)
    draw_circle(photo_ov, view, circle, WHITE, 1, r=circle.r * (1 - shrink))

    prep = prepare_cfg(rgb, circle, cfg)
    M = segment(prep.crop, prep.mask, segment_params(cfg), int(cfg.get("seed", 42)))
    L = lab(prep.crop)[:, :, 0]
    feats = extract_features(L, M, prep.mask, feature_params(cfg))

    crop_ov = prep.crop.copy()
    size = crop_ov.shape[0]
    draw_outline(crop_ov, M, MAGENTA, 3)
    if M.any():
        ys, xs = np.nonzero(M)
        mx, my = float(xs.mean()), float(ys.mean())
        draw_axis(crop_ov, size, mx, my, feats["symmetry_angle"], size * 0.47, YELLOW, 2)
        draw_point(crop_ov, size, mx, my, CYAN, 8)
    else:
        notes.append("No milk pattern was found on the surface.")

    fb, status = None, "scored"
    if not ok:
        status = "no_rim"
    elif bundle is None:
        status = "no_model"
    else:
        try:
            emb = embedder(prep.crop) if embedder is not None and embedder.needed else None
            fb = make_feedback(feats, emb, bundle)
            status = "scored" if fb.measurable else "unmeasurable"
        except Exception as err:  # keep the overlays; say why there is no score
            status = "error"
            msg = str(err).splitlines()[0] if str(err) else ""
            notes.append(f"No score: {type(err).__name__}" + (f": {msg}" if msg else ""))
    return Analysis(rgb, circle, ok, photo_ov, crop_ov, feats, fb, status, notes)
