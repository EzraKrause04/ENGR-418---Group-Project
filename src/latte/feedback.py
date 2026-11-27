"""Feedback engine: score +/- error, top fixes from Ridge contributions, milk diagnosis (section 11).

Inputs are one cup's handcrafted features (and its embedding, if the score
model uses one) plus the bundle that ``scripts/fit_final.py`` writes to
``models/latte_final.joblib``. Those final models are for the demo only; every
reported number comes from cross-validation (``make train``).

1. **Score.** The score model's prediction of the mean Overall score, rounded
   to ``feedback.score_round`` (0.5) and clipped to 1-5. It is shown as
   "score +/- X". X is the out-of-fold MAE in E2 of the *rounded* score, so it
   describes the number actually shown, rounded up to 0.1 with a floor of 0.1.
2. **Top fixes.** From the final handcrafted Ridge model, the contribution of
   feature i is c_i = w_i * z_i: w_i is its Ridge weight (score points per
   standard deviation) and z_i its standardised value after the pipeline's
   median imputation. The prediction is intercept + sum_i c_i. The
   ``feedback.n_fixes`` most negative contributions with
   c_i < ``feedback.min_contribution`` are mapped to a tip in ``tips.yaml`` by
   feature and direction (z < 0 is "low", z > 0 is "high"). Rules:
   - a feature or direction without a reviewed tip is skipped, never improvised;
   - a tip already shown is not repeated, since correlated features share tips;
   - a feature that couldn't be measured (NaN, filled in by the pipeline's
     median) is never offered as a fix;
   - a tip that names a milk condition (``implies_milk``) is replaced by its
     ``cause_alt``/``tip_alt``, or dropped, when the shown milk diagnosis names
     a different condition.

   "No major issues found." appears only when no measured contribution is
   below the cut-off. If some are, but none has a reviewed tip, a neutral note
   says so instead.
3. **Milk diagnosis.** If the final B1 classifier's top class has a tuned
   threshold and its probability reaches it: "Likely: milk too thick (p = 0.71)"
   plus its tip. Thresholds are tuned per predicted class
   (``tune_thresholds``), so a class the model is unreliable on is never shown.
   Otherwise: "Milk condition unclear from this photo."
4. **Withheld.** If E3 did not show B1 above chance, or no threshold reached
   the required out-of-fold precision, the bundle has no milk model and the
   diagnosis is left out entirely, with the reason recorded.

This is checked before any of the above: if the milk pattern couldn't be
measured (contrast_dL missing, e.g. no milk found), nothing is scored or
advised, and the result says "Could not measure the milk pattern in this photo."
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from latte.features import MODEL_FEATURES
from latte.io import cfg_path

FEEDBACK_DEFAULTS: dict[str, Any] = {
    "score_round": 0.5,
    "min_contribution": -0.1,
    "n_fixes": 2,
    "milk_prob_threshold": 0.6,
    "milk_min_precision": 0.7,
    "milk_min_predictions": 5,
    "score_model": "best",
    "milk_model": "logreg_hand",
}
TIPS_PATH = Path(__file__).with_name("tips.yaml")
BUNDLE_NAME = "latte_final.joblib"
NO_ISSUES = "No major issues found."
NO_TIP = "Some features lowered the score, but there is no reviewed tip for them yet."
MILK_UNCLEAR = "Milk condition unclear from this photo."
UNMEASURABLE = "Could not measure the milk pattern in this photo."


def feedback_params(cfg: dict[str, Any] | None) -> dict[str, Any]:
    return {**FEEDBACK_DEFAULTS, **((cfg or {}).get("feedback") or {})}


def load_tips(path: Path | None = None) -> dict[str, Any]:
    tips = yaml.safe_load((path or TIPS_PATH).read_text()) or {}
    tips.setdefault("fixes", {})
    tips.setdefault("milk", {})
    return tips


def bundle_path(cfg: dict[str, Any]) -> Path:
    return cfg_path(cfg, "models_dir") / BUNDLE_NAME


def load_bundle(cfg: dict[str, Any]) -> dict[str, Any]:
    path = bundle_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run make train, then make final")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def round_score(s: float, step: float = 0.5) -> float:
    """round(s / step) * step, clipped to the rubric's 1-5 range."""
    return float(np.clip(np.round(s / step) * step, 1.0, 5.0))


def hand_columns(pipe: Any) -> list[str]:
    """The handcrafted columns a fitted pipeline's 'hand' branch uses (in order)."""
    for name, _, cols in pipe.named_steps["prep"].transformers_:
        if name == "hand":
            return list(cols)
    raise ValueError("this pipeline has no handcrafted-feature branch")


def ridge_contributions(pipe: Any, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(c, z) for one row: z_i = standardised, imputed feature i; c_i = w_i * z_i (score points).

    The handcrafted Ridge pipeline's prediction equals intercept + sum_i c_i.
    """
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    cols = hand_columns(pipe)
    z = np.asarray(prep.transform(X), float)[0]
    w = np.asarray(model.coef_, float).ravel()
    if len(w) != len(cols):
        raise ValueError("the fixes model must be a Ridge on handcrafted features only")
    return pd.Series(w * z, index=cols), pd.Series(z, index=cols)


@dataclass
class Fix:
    feature: str
    direction: str  # "low" or "high" relative to the training average
    contribution: float  # score points (negative = costs points)
    z: float
    cause: str
    tip: str


@dataclass
class Feedback:
    score: float  # rounded
    score_raw: float
    score_error: float  # out-of-fold MAE of the rounded score (E2), before display rounding
    score_text: str
    fixes: list[Fix] = field(default_factory=list)
    fixes_text: list[str] = field(default_factory=list)
    skipped_fixes: list[str] = field(default_factory=list)  # big negative contributions with no reviewed tip
    milk_status: str = "withheld"  # "shown", "unclear" or "withheld"
    milk_text: str = ""
    milk_class: str | None = None
    milk_p: float | None = None
    milk_tip: str = ""
    milk_reason: str = ""  # why the diagnosis is withheld, if it is
    contributions: dict[str, float] = field(default_factory=dict)
    measurable: bool = True
    unmeasured: list[str] = field(default_factory=list)  # features that were NaN (imputed, never advised on)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tip_for(entry: dict[str, Any], milk_shown: str | None) -> tuple[str, str] | None:
    """(cause, tip) for a tips.yaml entry, adapted so it can't contradict the shown milk diagnosis."""
    implied = entry.get("implies_milk")
    if milk_shown and implied and implied != milk_shown:
        if entry.get("tip_alt"):
            return str(entry.get("cause_alt", "")), str(entry["tip_alt"])
        return None
    return str(entry.get("cause", "")), str(entry["tip"])


def top_fixes(contrib: pd.Series, z: pd.Series, tips: dict[str, Any], n: int, min_contribution: float,
              unmeasured: set[str] | None = None, milk_shown: str | None = None,
              used_tips: set[str] | None = None) -> tuple[list[Fix], list[str]]:
    """The n most negative measured contributions below ``min_contribution`` that have a reviewed, non-repeated tip.

    ``used_tips`` are tips already on screen (e.g. the milk diagnosis tip), which aren't repeated.
    """
    fixes, skipped, used = [], [], set(used_tips or ())
    for f, c in contrib.sort_values().items():
        if not c < min_contribution:
            break
        direction = "low" if z[f] < 0 else "high"
        if unmeasured and f in unmeasured:
            skipped.append(f"{f} (not measured)")
            continue
        entry = (tips.get("fixes", {}).get(f) or {}).get(direction)
        pair = _tip_for(entry, milk_shown) if entry and entry.get("tip") else None
        if pair is None:
            skipped.append(f"{f} ({direction}, {c:+.2f})")
            continue
        if pair[1] in used:
            continue  # the same advice already given for a correlated feature
        used.add(pair[1])
        fixes.append(Fix(f, direction, float(c), float(z[f]), *pair))
        if len(fixes) == n:
            break
    return fixes, skipped


def feature_frame(features: dict[str, float], emb: np.ndarray | None, emb_cols: list[str]) -> pd.DataFrame:
    missing = [f for f in MODEL_FEATURES if f not in features]
    if missing:
        raise ValueError(f"features missing from the input: {', '.join(missing)}")
    row = {f: float(features[f]) if features[f] is not None and not pd.isna(features[f]) else np.nan for f in MODEL_FEATURES}
    row = {f: v if np.isfinite(v) else np.nan for f, v in row.items()}  # +/-inf counts as unmeasured too
    if emb_cols:
        if emb is None or len(emb) != len(emb_cols):
            raise ValueError(f"this model needs a {len(emb_cols)}-dimensional embedding")
        row |= {c: float(v) for c, v in zip(emb_cols, emb)}
    return pd.DataFrame([row])


def score_error_text(err: float) -> float:
    """Displayed error: rounded *up* to 0.1, never below 0.1 (never implies certainty)."""
    return max(0.1, float(np.ceil(err * 10 - 1e-9) / 10))


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def make_feedback(features: dict[str, float], emb: np.ndarray | None, bundle: dict[str, Any],
                  tips: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> Feedback:
    """Score, fixes and milk diagnosis for one cup (steps 1-5 in the module docstring)."""
    p = {**FEEDBACK_DEFAULTS, **(params or bundle.get("feedback_params") or {})}
    tips = tips if tips is not None else load_tips()
    X = feature_frame(features, emb, bundle.get("emb_cols") or [])
    unmeasured = {f for f in MODEL_FEATURES if not np.isfinite(X.at[0, f])}
    err = float(bundle["score"]["error"])
    if "contrast_dL" in unmeasured:
        return Feedback(score=float("nan"), score_raw=float("nan"), score_error=err, measurable=False,
                        score_text=UNMEASURABLE, fixes_text=[UNMEASURABLE], milk_status="withheld",
                        milk_reason="the milk pattern could not be measured", unmeasured=sorted(unmeasured))

    raw = float(bundle["score"]["pipeline"].predict(X)[0])
    rounded = round_score(raw, float(p["score_round"]))
    fb = Feedback(score=rounded, score_raw=raw, score_error=err,
                  score_text=f"{rounded:.1f} ± {score_error_text(err):.1f}", unmeasured=sorted(unmeasured))

    milk = bundle.get("milk")
    if not milk:
        fb.milk_status, fb.milk_reason = "withheld", str(bundle.get("milk_withheld_reason") or "no milk model")
    else:
        proba = milk["pipeline"].predict_proba(X)[0]
        classes = list(milk["pipeline"].classes_)
        k = int(np.argmax(proba))
        fb.milk_class, fb.milk_p = str(classes[k]), float(proba[k])
        t = (milk.get("thresholds") or {}).get(fb.milk_class)
        if t is not None and fb.milk_p >= float(t):
            info = tips.get("milk", {}).get(fb.milk_class, {})
            fb.milk_status = "shown"
            fb.milk_text = f"Likely: {info.get('label') or 'milk ' + fb.milk_class} (p = {fb.milk_p:.2f})"
            fb.milk_tip = str(info.get("tip") or "")
        else:
            fb.milk_status, fb.milk_text = "unclear", MILK_UNCLEAR

    contrib, z = ridge_contributions(bundle["fixes"]["pipeline"], X)
    fb.contributions = {k: float(v) for k, v in contrib.items()}
    shown = fb.milk_class if fb.milk_status == "shown" else None
    fb.fixes, fb.skipped_fixes = top_fixes(contrib, z, tips, int(p["n_fixes"]), float(p["min_contribution"]),
                                           unmeasured, shown, used_tips={fb.milk_tip} if fb.milk_tip else None)
    if fb.fixes:
        fb.fixes_text = [f"Likely cause: {f.cause.rstrip('.')}. Try: {f.tip.rstrip('.')}." for f in fb.fixes]
    elif any(c < float(p["min_contribution"]) for f, c in contrib.items() if f not in unmeasured):
        fb.fixes_text = [NO_TIP]
    else:
        fb.fixes_text = [NO_ISSUES]
    return fb


# ---------------------------------------------------------------------------
# Threshold tuning (used by fit_final on out-of-fold probabilities)
# ---------------------------------------------------------------------------


def tune_thresholds(y: pd.Series, proba: pd.DataFrame, start: float = 0.6, min_precision: float = 0.7,
                    min_n: int = 5) -> dict[str, Any]:
    """Per predicted class, the smallest threshold t >= start (step 0.01) with out-of-fold precision >= min_precision.

    For class k: precision_k(t) = fraction correct among cups predicted k with
    top probability >= t, over at least ``min_n`` such cups. A class with no
    qualifying t gets None, and the demo never shows it. Returns
    {"thresholds": {k: t or None}, "per_class": {k: {...}}, "any": bool}.
    """
    top = proba.max(axis=1).to_numpy()
    pred = proba.idxmax(axis=1).to_numpy()
    truth = y.loc[proba.index].to_numpy()
    thresholds, info = {}, {}
    for k in proba.columns:
        thresholds[k], info[k] = None, {"precision": None, "n": 0, "share_of_cups": 0.0}
        for t in np.round(np.arange(start, 1.0, 0.01), 2):
            sel = (pred == k) & (top >= t)
            if sel.sum() < min_n:
                break
            prec = float((truth[sel] == k).mean())
            if prec >= min_precision:
                thresholds[k] = float(t)
                info[k] = {"precision": prec, "n": int(sel.sum()), "share_of_cups": float(sel.mean())}
                break
    return {"thresholds": thresholds, "per_class": info, "any": any(v is not None for v in thresholds.values())}
