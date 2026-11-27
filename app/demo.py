"""Latte Coach demo for the video: photo -> overlays, score +/- error, top fixes, milk diagnosis (section 12).

    make demo                        # http://127.0.0.1:7863 (after make train and make final)
    make demo ARGS="--examples a.jpg b.jpg"

Upload a photo, or use the webcam, and press Analyse. The page shows:

* the cup with the detected liquid circle (green; red, and no score, if the rim
  wasn't found reliably);
* the analysed surface with the milk outline (magenta), its centre (cyan) and
  symmetry axis (yellow);
* the score as "3.5 / 5" with its typical error "± 0.5", the top fixes
  ("Likely cause ... Try ..."), and the milk diagnosis. The diagnosis says
  "unclear" when the model isn't confident, and is left out entirely when the
  model didn't beat chance (``make final`` decides; see latte.feedback).

Without ``make final`` the page still shows the overlays and measured features
and says that no trained model exists yet. Example photos must be team-owned.
By default they are the first usable ingested cups, which the final models
were fitted on, and the page says so. Use ``--examples`` for held-out team photos.

Privacy: the server listens on 127.0.0.1 only. Gradio keeps uploads in its
temporary folder, and this app asks it to delete them after 10 minutes. Uploads
are processed like ``make ingest`` (EXIF is applied, then dropped) and never
written into the project.
"""

from __future__ import annotations

import argparse
import html
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import UnidentifiedImageError  # noqa: E402

from latte.feedback import load_bundle, score_error_text  # noqa: E402
from latte.inference import Analysis, Embedder, analyse  # noqa: E402
from latte.io import DEFAULT_CONFIG, cfg_path, cup_sort_key, load_config  # noqa: E402
from latte.labels import available_cups  # noqa: E402

CSS = """
.lc-card { border-radius: 12px; padding: .9rem 1rem; background: var(--block-background-fill); }
.lc-score { font-size: 2.6rem; font-weight: 700; line-height: 1.1; }
.lc-score small { font-size: 1.1rem; font-weight: 400; }
.lc-sub { font-size: 0.95rem; opacity: .75; margin-top: .25rem; }
.lc-fix { margin: .35rem 0; }
.lc-milk-shown { font-weight: 600; }
.lc-muted { opacity: .7; font-size: .9rem; }
.lc-warn { font-weight: 600; }
h3.lc-h { margin: .2rem 0 .4rem; font-size: 1.05rem; }
footer { display: none !important; }
"""

TITLE = "Latte Coach"
SUBTITLE = "Score a latte-art heart from one photo and get the two most useful fixes."
HONESTY = ("Scores are model estimates with their typical error (±). Fixes and the milk diagnosis name "
           "<i>likely</i> causes, never certain ones; the diagnosis is only shown when the model was reliable "
           "in cross-validation.")


def examples(cfg: dict[str, Any], given: list[str] | None, n: int = 4) -> tuple[list[str], str]:
    """(paths, label). Default: the first usable ingested cups, labelled as training cups."""
    if given is not None:
        missing = [g for g in given if not Path(g).exists()]
        if missing:
            print(f"NOTE: --examples files not found, skipped: {', '.join(missing)}")
        return [str(Path(g)) for g in given if Path(g).exists()], "Example cups (team photos)"
    raw = cfg_path(cfg, "raw")
    cups = sorted(available_cups(cfg), key=cup_sort_key)[:n]
    return [str(raw / f"{c}.jpg") for c in cups], "Example cups (team photos, used to train the model)"


def section(title: str, body: str) -> str:
    return f"<h3 class='lc-h'>{html.escape(title)}</h3>{body}" if body else ""


def render(an: Analysis) -> tuple[str, str, str, str]:
    """HTML for: score card, fixes panel, milk panel, details. Empty panels render as nothing."""
    notes = "".join(f"<p class='lc-warn'>⚠️ {html.escape(n)}</p>" for n in an.notes)
    fb = an.feedback
    fixes = milk = ""
    if an.status == "scored" and fb is not None:
        score = (f"<div class='lc-score'>{fb.score:.1f} <small>/ 5</small></div>"
                 f"<div class='lc-sub'>predicted Overall score · typical error ± {score_error_text(fb.score_error):.1f}</div>")
        body = "".join(f"<p class='lc-fix'>{i}. {html.escape(t)}</p>" for i, t in enumerate(fb.fixes_text, 1)) \
            if fb.fixes else f"<p class='lc-fix'>{html.escape(fb.fixes_text[0])}</p>"
        fixes = section("Top fixes", body)
        if fb.milk_status == "shown":
            milk = section("Milk", f"<p class='lc-milk-shown'>{html.escape(fb.milk_text)}</p>"
                           + (f"<p>{html.escape(fb.milk_tip)}</p>" if fb.milk_tip else ""))
        elif fb.milk_status == "unclear":
            milk = section("Milk", f"<p>{html.escape(fb.milk_text)}</p>")
        # withheld: the diagnosis is left out entirely (section 11.4)
    elif an.status == "no_model":
        score = ("<div class='lc-score'>–</div><div class='lc-sub'>No trained model yet: run <code>make train</code> "
                 "then <code>make final</code>. The overlays below are real.</div>")
    elif an.status == "unmeasurable" and fb is not None:
        score = f"<div class='lc-score'>–</div><div class='lc-sub'>{html.escape(fb.score_text)}</div>"
    else:  # no_rim, error
        score = "<div class='lc-score'>–</div><div class='lc-sub'>No score for this photo.</div>"
    detail = f"<p class='lc-muted'>Rim confidence {an.circle.confidence:.2f}.</p>"
    if an.status != "no_rim":  # features from a rejected crop would mean nothing
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td style='text-align:right'>{v:.3f}</td></tr>"
                       for k, v in an.features.items() if np.isfinite(v))
        missing = [k for k, v in an.features.items() if not np.isfinite(v)]
        detail += (f"<table class='lc-muted'>{rows}</table>"
                   + (f"<p class='lc-muted'>Not measured: {html.escape(', '.join(missing))}</p>" if missing else ""))
    if fb is not None and fb.milk_status == "withheld" and fb.milk_reason:
        detail += f"<p class='lc-muted'>Milk diagnosis not shown: {html.escape(fb.milk_reason)}</p>"
    return notes + score, fixes, milk, detail


def build_ui(cfg: dict[str, Any], bundle: dict[str, Any] | None, example_paths: list[str],
             example_label: str = "Example cups (team photos)", embedder: Embedder | None = None) -> gr.Blocks:
    embedder = embedder or Embedder(cfg, bundle)

    def run(path: str | None):
        if not path:
            gr.Warning("Upload a photo or take one with the webcam first.")
            return None, None, "", "", "", ""
        try:
            an = analyse(path, cfg, bundle, embedder)
        except (UnidentifiedImageError, OSError, SyntaxError) as err:  # not an image, or a truncated one
            gr.Warning(f"Couldn't read this photo ({type(err).__name__}). Try a JPEG or PNG.")
            return None, None, "", "", "", ""
        except Exception:  # a bug: say so plainly in the page, full traceback in the console
            traceback.print_exc()
            gr.Warning("Something went wrong analysing this photo; the details are in the terminal running make demo.")
            return None, None, "", "", "", ""
        score, fixes, milk, detail = render(an)
        return an.photo_overlay, an.crop_overlay, score, fixes, milk, detail

    with gr.Blocks(title=TITLE, analytics_enabled=False, delete_cache=(600, 600)) as ui:
        gr.Markdown(f"# ☕ {TITLE}\n{SUBTITLE}")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                photo = gr.Image(type="filepath", sources=["upload", "webcam"], label="Your latte (overhead photo)", height=360)
                go = gr.Button("Analyse", variant="primary", size="lg")
                if example_paths:
                    gr.Examples(example_paths, inputs=photo, label=example_label)
                gr.HTML(f"<p class='lc-muted'>{HONESTY}</p>")
            with gr.Column(scale=2):
                with gr.Row():
                    found = gr.Image(label="Cup found", interactive=False, height=300, buttons=["fullscreen"])
                    surface = gr.Image(label="Pattern analysed", interactive=False, height=300, buttons=["fullscreen"])
                score = gr.HTML(elem_classes=["lc-card"])
                with gr.Row(equal_height=True):
                    fixes = gr.HTML()
                    milk = gr.HTML()
                with gr.Accordion("Details", open=False):
                    details = gr.HTML()
        outs = [found, surface, score, fixes, milk, details]
        go.click(run, [photo], outs)
        photo.upload(run, [photo], outs)
    return ui


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--port", type=int, default=7863)
    ap.add_argument("--examples", nargs="*", help="team-owned example photos (default: the first usable ingested cups)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    try:
        bundle = load_bundle(cfg)
        print(f"Loaded models from {cfg_path(cfg, 'models_dir')} (score: {bundle['score']['name']}, "
              f"milk: {(bundle.get('milk') or {}).get('name', 'withheld')}).")
    except FileNotFoundError as err:
        bundle = None
        print(f"NOTE: {err}. The demo will show overlays and features only.")
    embedder = Embedder(cfg, bundle)
    if embedder.needed:
        try:
            embedder.warm()
        except Exception as err:
            model = (bundle.get("embedding") or {}).get("model")
            print(f"ERROR: the score/milk models need embeddings from {model}, which can't be loaded "
                  f"({type(err).__name__}). Run `python -m latte.embed --fetch` while online.", file=sys.stderr)
            return 1
    paths, label = examples(cfg, args.examples)
    ui = build_ui(cfg, bundle, paths, label, embedder)
    ui.launch(server_name="127.0.0.1", server_port=args.port, css=CSS, theme=gr.themes.Soft(), footer_links=[])
    return 0


if __name__ == "__main__":
    sys.exit(main())
