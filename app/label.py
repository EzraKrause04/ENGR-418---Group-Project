"""Blind labelling tool (section 7.2).

    make label                         # choose Rater A or B in the page
    python app/label.py --rater A      # preselect a rater
    python app/label.py --rater A --relabel-before 2026-11-27T18:00
                                       # re-queue cups labelled before an anchor revision

The page shows only the photo and an anonymized index ("Image 18 of 42"): never
the cup_id, condition, pourer or session. Images arrive as pixels, so not even
the file name reaches the browser. Every submission is appended to
labels/rater_<X>.csv and fsynced; closing the tab loses nothing, and the next
start resumes with the cups this rater hasn't labelled (including newly
ingested ones).

Keyboard: 1-5 scores the highlighted row and moves down, Up/Down change row,
U toggles unusable, Enter saves and shows the next image.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from datetime import datetime
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from latte.io import DEFAULT_CONFIG, cfg_path, load_config  # noqa: E402
from latte.labels import (  # noqa: E402
    append_label,
    available_cups,
    label_queue,
    labels_path,
    latest_labels,
    now_iso,
    parse_rubric_anchors,
    parse_since,
    read_labels,
)

TITLES = {
    "contrast": "Contrast",
    "definition": "Definition",
    "symmetry": "Symmetry",
    "centring": "Centring",
    "surface": "Surface",
    "overall": "Overall",
}

CSS = """
.lc-dim { border-radius: 10px; transition: box-shadow 120ms; }
.lc-active { box-shadow: 0 0 0 2px var(--color-accent) !important; }
#lc-progress { font-size: 1.05rem; font-weight: 600; }
#lc-keys { opacity: 0.75; font-size: 0.85rem; }
"""

KEYS_JS = """
<script>
(() => {
  const DIMS = __DIMS__;
  let active = 0;
  const row = (i) => document.getElementById('dim-' + DIMS[i]);
  const highlight = () => DIMS.forEach((_, i) => row(i)?.classList.toggle('lc-active', i === active));
  const typing = (t) => t && (t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && t.type === 'text'));
  document.addEventListener('click', (e) => {
    DIMS.forEach((d, i) => {
      if (row(i)?.contains(e.target) && e.target.tagName === 'INPUT') {
        active = Math.min(i + 1, DIMS.length - 1);
      }
    });
    setTimeout(highlight, 0);
  });
  document.addEventListener('keydown', (e) => {
    if (typing(e.target)) {
      if (e.key === 'Escape') e.target.blur();
      if (e.key === 'Enter') { active = 0; setTimeout(highlight, 0); }  // Enter in the comment box saves
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (/^[1-5]$/.test(e.key)) {
      // .click() also fires the mouse handler above, which advances `active`;
      // assign the next row explicitly so a keypress moves down exactly one row.
      const next = Math.min(active + 1, DIMS.length - 1);
      row(active)?.querySelector(`input[type="radio"][value="${e.key}"]`)?.click();
      active = next;
    } else if (e.key === 'ArrowDown') {
      active = Math.min(active + 1, DIMS.length - 1);
    } else if (e.key === 'ArrowUp') {
      active = Math.max(active - 1, 0);
    } else if (e.key === 'u' || e.key === 'U') {
      document.querySelector('#lc-unusable input[type="checkbox"]')?.click();
    } else if (e.key === 'Enter') {
      if (e.repeat) { e.preventDefault(); return; }
      document.getElementById('lc-submit')?.click();
      active = 0;
    } else {
      return;
    }
    e.preventDefault();
    highlight();
  });
  setInterval(highlight, 400);  // re-apply after Gradio re-renders
})();
</script>
"""


def build_ui(cfg: dict[str, Any], preset_rater: str | None, relabel_before: datetime | None) -> gr.Blocks:
    dims: list[str] = cfg["labelling"]["dimensions"]
    raters: list[str] = cfg["labelling"]["raters"]
    seed = int(cfg["seed"])
    anchors = parse_rubric_anchors(cfg_path(cfg, "rubric"), dims)

    def info(d: str) -> str | None:
        a = anchors.get(d, {})
        if len(a) == 1:  # a single-text row such as Overall's "holistic judgement" note
            return next(iter(a.values()))
        return " · ".join(f"{k} = {v}" for k, v in a.items()) or None

    def latest(rater: str):
        return latest_labels(read_labels(labels_path(cfg, rater), dims))

    def pixels(cup_id: str) -> Image.Image:
        with Image.open(cfg_path(cfg, "raw") / f"{cup_id}.jpg") as im:
            return im.convert("RGB").copy()

    def view(state: dict[str, Any], status: str = "") -> list[Any]:
        """Values for: token, image, progress, *dims, unusable, comment, status, back."""
        rater = state.get("rater")
        blank_dims = [gr.update(value=None) for _ in dims]
        if not rater:
            return ["", None, "Choose your rater ID and press **Start**.", *blank_dims, False, "", status, gr.update(interactive=False)]
        lab = latest(rater)
        cups = available_cups(cfg)
        queue = label_queue(cups, lab, rater, seed, relabel_before)
        if not state.get("revisit"):
            state["current"] = queue[0] if queue else None
        state["token"] = secrets.token_hex(6)
        back = gr.update(interactive=bool(state.get("history")))
        cur = state["current"]
        if cur is None:
            done_msg = f"**All done, Rater {rater}.** New photos appear here after the next ingest."
            return [state["token"], None, done_msg, *blank_dims, False, "", status, back]
        total = len(cups) if relabel_before is None else len(queue) + state.get("n_saved", 0)
        if state.get("revisit"):
            progress = "Revisiting an earlier image: your saved scores are shown. Save to replace them."
            saved = lab.loc[cur]
            vals = [gr.update(value=int(saved[d]) if saved[d] == saved[d] else None) for d in dims]
            return [state["token"], pixels(cur), progress, *vals, saved["unusable"] == "y", saved["comment"], status, back]
        progress = f"Image {total - len(queue) + 1} of {total}"
        return [state["token"], pixels(cur), progress, *blank_dims, False, "", status, back]

    def start(rater: str | None):
        if rater not in raters:
            state: dict[str, Any] = {}
            return [state, gr.update(), *view(state, "Pick A or B first.")]
        state = {"rater": rater, "current": None, "history": [], "revisit": False, "n_saved": 0}
        mode = f" Relabelling cups scored before {relabel_before:%Y-%m-%d %H:%M}." if relabel_before else ""
        return [state, gr.update(interactive=False), *view(state, f"Labelling as **Rater {rater}**.{mode} Reload the page to switch rater.")]

    def submit(state: dict[str, Any], token: str, *vals: Any):
        scores, unusable, comment = list(vals[: len(dims)]), bool(vals[len(dims)]), vals[len(dims) + 1] or ""
        if not state.get("rater") or state.get("current") is None:
            return [state, *view(state, "Nothing to save.")]
        if token != state.get("token"):
            # A second Enter arrived before the page showed the next image: ignore it,
            # otherwise the previous image's scores would be saved against this one.
            return [state, *([gr.update()] * (len(dims) + 7))]
        missing = [TITLES.get(d, d) for d, s in zip(dims, scores) if s is None]
        if missing and not unusable:
            msg = f"Score every dimension (missing: {', '.join(missing)}), or tick Unusable."
            gr.Warning(msg)
            return [state, *([gr.update()] * (len(dims) + 5)), msg, gr.update()]
        cup = state["current"]
        row = {"cup_id": cup, "unusable": "y" if unusable else "n", "comment": " ".join(comment.split()), "timestamp": now_iso()}
        row |= {d: "" if s is None else int(s) for d, s in zip(dims, scores)}
        append_label(labels_path(cfg, state["rater"]), row, dims)
        state["history"].append(cup)
        state["revisit"] = False
        state["n_saved"] = state.get("n_saved", 0) + 1
        return [state, *view(state, f"Saved ({state['n_saved']} this session).")]

    def back(state: dict[str, Any]):
        if not state.get("history"):
            return [state, *view(state, "No earlier image in this session.")]
        state["current"] = state["history"].pop()
        state["revisit"] = True
        return [state, *view(state, "")]

    head = KEYS_JS.replace("__DIMS__", json.dumps(dims))
    with gr.Blocks(title="Latte Coach: labelling", analytics_enabled=False) as ui:
        state = gr.State({})
        token = gr.Textbox(visible=False)
        with gr.Row(equal_height=True):
            rater = gr.Radio(raters, value=preset_rater, label="Rater", scale=1)
            start_btn = gr.Button("Start", variant="primary", scale=0, min_width=120)
            status = gr.Markdown(scale=3)
        with gr.Row():
            with gr.Column(scale=3):
                progress = gr.Markdown(elem_id="lc-progress")
                image = gr.Image(
                    type="pil",
                    format="png",  # lossless: Gradio's default WebP re-encode can soften bubbles and edges
                    interactive=False,
                    show_label=False,
                    height=640,
                    buttons=["fullscreen"],
                )
            with gr.Column(scale=2):
                radios = [
                    gr.Radio([1, 2, 3, 4, 5], label=TITLES.get(d, d), info=info(d), elem_id=f"dim-{d}", elem_classes=["lc-dim"])
                    for d in dims
                ]
                unusable = gr.Checkbox(label="Unusable (blurry, glare, not a heart)", elem_id="lc-unusable")
                comment = gr.Textbox(label="Comment (optional)", lines=1, max_lines=1, placeholder="Enter saves")
                with gr.Row():
                    back_btn = gr.Button("Previous", interactive=False)
                    submit_btn = gr.Button("Save & next", variant="primary", elem_id="lc-submit")
                gr.Markdown(
                    "Keys: **1-5** score the highlighted row and move down · **Up/Down** change row · "
                    "**U** unusable · **Enter** save. Esc leaves the comment box.",
                    elem_id="lc-keys",
                )

        outs = [token, image, progress, *radios, unusable, comment, status, back_btn]
        start_btn.click(start, [rater], [state, rater, *outs])
        for trigger in (submit_btn.click, comment.submit):
            trigger(submit, [state, token, *radios, unusable, comment], [state, *outs])
        back_btn.click(back, [state], [state, *outs])
        if preset_rater:
            ui.load(start, [rater], [state, rater, *outs])
    ui.lc_head = head  # type: ignore[attr-defined]
    return ui


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--rater", help="preselect a rater (A or B)")
    ap.add_argument("--relabel-before", help="re-queue cups whose latest label is older than this (YYYY-MM-DD[THH:MM])")
    ap.add_argument("--port", type=int, help="default: labelling.port in the config")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    rater = args.rater.upper() if args.rater else None
    if rater and rater not in cfg["labelling"]["raters"]:
        ap.error(f"--rater must be one of {cfg['labelling']['raters']}")
    since = parse_since(args.relabel_before) if args.relabel_before else None
    ui = build_ui(cfg, rater, since)
    ui.launch(
        server_name="127.0.0.1",
        server_port=args.port or int(cfg["labelling"]["port"]),
        css=CSS,
        head=ui.lc_head,  # type: ignore[attr-defined]
        theme=gr.themes.Soft(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
