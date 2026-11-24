"""Check every liquid circle by eye, and fix the wrong ones (section 8.1).

    make circles                     # opens http://127.0.0.1:7861
    python app/annotate_circle.py --who A

For each usable photo, the tool shows a zoomed view of the automatic circle.
Low-confidence photos come first. Then either:

  * **Accept** (key A): the automatic circle is right. It is saved, frozen, as ``accepted``.
  * **Fix**: click 3 or more points on the liquid edge, where the crema meets the cup
    wall, spread around the circle. A circle is fitted through them (least squares).
    Then **Save** (key S) stores it as ``manual``, with the clicked points.

Rows are appended to data/circles.csv, and the last row per cup wins. Saved
circles always override detection in the rest of the pipeline. Other keys:
C clears the points, F switches between the zoomed view and the whole photo,
and Left/Right move between photos.

The tool shows cup IDs, because QA needs them, but never the milk condition,
drop-in, pourer or session details.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from latte.geometry import (  # noqa: E402
    Circle,
    append_circle,
    circles_path,
    detect_circle,
    edge_support,
    fit_circle,
    format_points,
    geometry_params,
    read_circles,
)
from latte.io import DEFAULT_CONFIG, cfg_path, cup_sort_key, load_config, load_rgb  # noqa: E402
from latte.labels import available_cups, now_iso  # noqa: E402
from latte.overlay import AMBER, CYAN, GREEN, MAGENTA, RED, WHITE, draw_circle, draw_cross, full_view, zoom_view  # noqa: E402
from latte.preprocess import preprocess_params  # noqa: E402

VIEW_SIZE = 760
FILTERS = ["Not checked yet", "Low confidence only", "All photos"]

CSS = """
#ac-info { font-size: 0.95rem; }
#ac-keys { opacity: 0.75; font-size: 0.85rem; }
"""

KEYS_JS = """
<script>
(() => {
  const click = (id) => document.getElementById(id)?.click();
  document.addEventListener('keydown', (e) => {
    const t = e.target;
    if (t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT' && t.type === 'text')) return;
    if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
    const k = e.key.toLowerCase();
    const map = {a: 'ac-accept', s: 'ac-save', c: 'ac-clear', arrowleft: 'ac-prev', arrowright: 'ac-next'};
    if (map[k]) { click(map[k]); e.preventDefault(); }
    else if (k === 'f') { document.querySelector('#ac-full input[type="checkbox"]')?.click(); e.preventDefault(); }
  });
})();
</script>
"""


class CircleSession:
    """Everything the UI needs that isn't per-browser-tab state."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.gp = geometry_params(cfg)
        self.shrink = float(preprocess_params(cfg)["radius_shrink"])
        self.path = circles_path(cfg)
        self.raw = cfg_path(cfg, "raw")
        self.cups = sorted(available_cups(cfg), key=cup_sort_key)
        self.auto: dict[str, Circle] = {}
        self.sizes: dict[str, tuple[int, int]] = {}
        for i, cup in enumerate(self.cups, 1):
            rgb = self.photo(cup)
            self.auto[cup] = detect_circle(rgb, self.gp)
            self.sizes[cup] = (rgb.shape[1], rgb.shape[0])
            if i % 20 == 0:
                print(f"  detected {i}/{len(self.cups)}", flush=True)

    def photo(self, cup: str) -> np.ndarray:
        return load_rgb(self.raw / f"{cup}.jpg")

    def saved(self, cup: str, df: Any = None) -> tuple[Circle, dict[str, str]] | None:
        df = read_circles(self.path) if df is None else df
        if cup not in df.index:
            return None
        row = df.loc[cup]
        if (str(row["img_w"]), str(row["img_h"])) != tuple(map(str, self.sizes[cup])):
            return None  # made on a different image size: treated as unchecked, like the pipeline does
        return Circle(float(row["cx"]), float(row["cy"]), float(row["r"]), source=str(row["source"])), dict(row)

    def low(self, cup: str) -> bool:
        return self.auto[cup].confidence < float(self.gp["low_confidence"])

    def order(self, flt: str) -> list[str]:
        if flt == "All photos":
            return list(self.cups)
        df = read_circles(self.path)
        todo = [c for c in self.cups if self.saved(c, df) is None]
        if flt == "Low confidence only":
            return [c for c in todo if self.low(c)]
        return sorted(todo, key=lambda c: (not self.low(c), cup_sort_key(c)))

    def save(self, cup: str, c: Circle, source: str, who: str, points: list[tuple[float, float]]) -> None:
        w, h = self.sizes[cup]
        append_circle(self.path, {
            "cup_id": cup, "cx": c.cx, "cy": c.cy, "r": c.r, "source": source, "checked_by": who,
            "img_w": w, "img_h": h, "points": format_points(points), "timestamp": now_iso(),
        })


def fitted(points: list[tuple[float, float]]) -> tuple[Circle | None, str]:
    if len(points) < 3:
        return None, ""
    try:
        cx, cy, r = fit_circle(points)
    except ValueError as err:
        return None, str(err)
    return Circle(cx, cy, r, source="manual"), ""


def build_ui(cs: CircleSession, preset_who: str | None) -> gr.Blocks:
    students: list[str] = cs.cfg["labelling"]["raters"]

    def render(st: dict[str, Any], status: str = "") -> list[Any]:
        """Values for: image, info, status, save button."""
        order = st.get("order", [])
        if not order:
            msg = "**Nothing to check in this list.** Every usable photo has a checked circle." if cs.cups else "**No photos in data/raw yet.** Run `make ingest` first."
            return [None, msg, status, gr.update(interactive=False)]
        st["i"] = min(max(st.get("i", 0), 0), len(order) - 1)
        cup = order[st["i"]]
        rgb = cs.photo(cup)
        auto = cs.auto[cup]
        saved = cs.saved(cup)
        pts: list[tuple[float, float]] = st.get("points", [])
        fit, fit_err = fitted(pts)
        focus = fit or (saved[0] if saved else auto)
        if st.get("full"):
            img, view = full_view(rgb, VIEW_SIZE * 2)
        else:
            if st.get("view_cup") != cup:  # keep the window still while points are clicked
                st["view_cup"], st["view_circle"] = cup, (saved[0] if saved else auto)
            img, view = zoom_view(rgb, st["view_circle"], VIEW_SIZE)
        st["view"] = (view.cx, view.cy, view.zoom, view.w, view.h)
        width = 2 if img.shape[1] <= 1100 else 3
        if saved:
            draw_circle(img, view, auto, CYAN, 1)
            draw_circle(img, view, saved[0], GREEN, width)
        else:
            draw_circle(img, view, auto, RED if cs.low(cup) else AMBER, width)
        if fit:
            draw_circle(img, view, fit, MAGENTA, width)
        draw_circle(img, view, focus, WHITE, 1, r=focus.r * (1 - cs.shrink))
        for x, y in pts:
            draw_cross(img, view, x, y, MAGENTA)

        lines = [f"### {cup}", f"Photo {st['i'] + 1} of {len(order)} in this list"]
        conf = f"{auto.confidence:.2f}" + (" (**low**: check carefully)" if cs.low(cup) else "")
        lines.append(f"Automatic circle: edge support {conf}")
        if saved:
            row = saved[1]
            what = "accepted automatic circle" if row["source"] == "accepted" else "manual circle"
            lines.append(f"Saved: <span style='color:#2ea043'>**{what}**</span>, checked by {row['checked_by']}, {row['timestamp'][:16].replace('T', ' ')}")
        else:
            lines.append("Saved: none, **not checked yet**")
        if pts:
            lines.append(f"Points clicked: {len(pts)}" + ("" if len(pts) >= 3 else f" (click {3 - len(pts)} more on the liquid edge)"))
        if fit:
            lines.append(f"Fitted circle: r = {fit.r:.1f} px, edge support {edge_support(rgb, fit, cs.gp):.2f}")
        if fit_err:
            lines.append(f"Can't fit: {fit_err}")
        legend = ("Amber/red: automatic · green: saved · cyan: automatic (under a saved one) · "
                  "magenta: your points and fit · white: analysis edge (r minus 4%)")
        lines.append(f"<span style='opacity:.7'>{legend}</span>")
        return [img, "\n\n".join(lines), status, gr.update(interactive=fit is not None)]

    def start(who: str | None, flt: str):
        st: dict[str, Any] = {"who": who, "filter": flt, "order": cs.order(flt), "i": 0, "points": [], "full": False}
        return [st, *render(st)]

    def on_filter(st: dict[str, Any], flt: str):
        st |= {"filter": flt, "order": cs.order(flt), "i": 0, "points": [], "view_cup": None}
        return [st, *render(st)]

    def on_who(st: dict[str, Any], who: str | None):
        st["who"] = who
        return st

    def on_click(st: dict[str, Any], evt: gr.SelectData):
        if not st.get("order") or "view" not in st or evt.index is None:
            return [st, *render(st)]
        cx, cy, zoom, w, h = st["view"]
        u, v = float(evt.index[0]), float(evt.index[1])
        x, y = (u + 0.5 - w / 2) / zoom + cx, (v + 0.5 - h / 2) / zoom + cy
        st.setdefault("points", []).append((round(x, 1), round(y, 1)))
        return [st, *render(st)]

    def move(st: dict[str, Any], step: int):
        st["i"] = st.get("i", 0) + step
        st["points"] = []
        return [st, *render(st)]

    def need_who(st: dict[str, Any]) -> str | None:
        if st.get("who") in students:
            return None
        gr.Warning("Choose who is checking (A or B) first.")
        return "Choose who is checking first."

    def accept(st: dict[str, Any]):
        if not st.get("order"):
            return [st, *render(st)]
        if (msg := need_who(st)) is not None:
            return [st, *render(st, msg)]
        cup = st["order"][st["i"]]
        cs.save(cup, cs.auto[cup], "accepted", st["who"], [])
        return advance(st, f"Accepted the automatic circle for {cup}.")

    def save_manual(st: dict[str, Any]):
        if not st.get("order"):
            return [st, *render(st)]
        if (msg := need_who(st)) is not None:
            return [st, *render(st, msg)]
        fit, err = fitted(st.get("points", []))
        if fit is None:
            return [st, *render(st, err or "Click at least 3 points on the liquid edge first.")]
        cup = st["order"][st["i"]]
        cs.save(cup, fit, "manual", st["who"], st["points"])
        return advance(st, f"Saved a manual circle for {cup} ({len(st['points'])} points).")

    def advance(st: dict[str, Any], msg: str):
        st["points"] = []
        if st["i"] < len(st["order"]) - 1:
            st["i"] += 1
        else:
            msg += " That was the last photo in this list."
        return [st, *render(st, msg)]

    def clear(st: dict[str, Any]):
        st["points"] = []
        return [st, *render(st)]

    def toggle_full(st: dict[str, Any], full: bool):
        st["full"] = bool(full)
        return [st, *render(st)]

    with gr.Blocks(title="Latte Coach: circle check", analytics_enabled=False) as ui:
        st = gr.State({})
        with gr.Row(equal_height=True):
            who = gr.Radio(students, value=preset_who, label="Checked by", scale=1)
            flt = gr.Dropdown(FILTERS, value=FILTERS[0], label="Show", scale=1)
            status = gr.Markdown(scale=2)
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(
                    type="numpy", format="png", interactive=False, show_label=False,
                    height=VIEW_SIZE, buttons=["fullscreen"], elem_id="ac-image",
                )
            with gr.Column(scale=2):
                info = gr.Markdown(elem_id="ac-info")
                accept_btn = gr.Button("Accept automatic circle (A)", variant="primary", elem_id="ac-accept")
                with gr.Row():
                    save_btn = gr.Button("Save clicked circle (S)", interactive=False, elem_id="ac-save")
                    clear_btn = gr.Button("Clear points (C)", elem_id="ac-clear")
                with gr.Row():
                    prev_btn = gr.Button("← Previous", elem_id="ac-prev")
                    next_btn = gr.Button("Next →", elem_id="ac-next")
                full = gr.Checkbox(label="Whole photo (F)", value=False, elem_id="ac-full")
                gr.Markdown(
                    "To fix a circle, click **3 or more points on the liquid edge** (where the crema meets the "
                    "cup wall, not the outer rim), spread around it, then **Save**. If the automatic circle is "
                    "on the wrong object, tick **Whole photo** first.\n\n"
                    "Keys: **A** accept · **S** save · **C** clear · **F** whole photo · **←/→** previous/next",
                    elem_id="ac-keys",
                )
        outs = [image, info, status, save_btn]
        ui.load(start, [who, flt], [st, *outs])
        flt.change(on_filter, [st, flt], [st, *outs])
        who.change(on_who, [st, who], [st])
        image.select(on_click, [st], [st, *outs])
        accept_btn.click(accept, [st], [st, *outs])
        save_btn.click(save_manual, [st], [st, *outs])
        clear_btn.click(clear, [st], [st, *outs])
        prev_btn.click(lambda s: move(s, -1), [st], [st, *outs])
        next_btn.click(lambda s: move(s, +1), [st], [st, *outs])
        full.change(toggle_full, [st, full], [st, *outs])
    return ui


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--who", help="preselect who is checking (A or B)")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    who = args.who.upper() if args.who else None
    if who and who not in cfg["labelling"]["raters"]:
        ap.error(f"--who must be one of {cfg['labelling']['raters']}")
    print("Detecting circles ...", flush=True)
    cs = CircleSession(cfg)
    print(f"{len(cs.cups)} usable photos; saved circles in {cs.path}")
    build_ui(cs, who).launch(server_name="127.0.0.1", server_port=args.port, css=CSS, head=KEYS_JS, theme=gr.themes.Soft())
    return 0


if __name__ == "__main__":
    sys.exit(main())
