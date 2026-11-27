"""Shared figure style for the report (section 13): one IEEE column, 300 dpi, a validated palette.

Colours: the first three slots of the reference categorical palette (blue,
orange, aqua). They are validated as all-pairs distinguishable, including
for colour-vision deficiency, on the light surface. Every category is also
named on an axis or in a legend, so colour is never the only cue. Sequential
data (confusion matrices) uses one blue hue from light to dark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLUMN_IN = 3.5  # IEEE single-column width, inches
DPI = 300
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#ffffff", "#d6e6f8", "#a9cbf0", "#6ea6e3", "#2a78d6", "#1d5aa6", "#123d73"]


def setup() -> None:
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
    })


def figure(height_in: float, ncols: int = 1, nrows: int = 1, hspace: float | None = None, wspace: float | None = None,
           **kw: Any):
    """A figure exactly one column wide. Constrained layout keeps the saved width fixed, so every
    figure prints at the same text size when scaled to \\columnwidth."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(COLUMN_IN, height_in), layout="constrained", **kw)
    engine = fig.get_layout_engine()
    engine.set(w_pad=0.02, h_pad=0.02, **({"hspace": hspace} if hspace is not None else {}),
               **({"wspace": wspace} if wspace is not None else {}))
    return fig, axes


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)  # no tight bbox: the width stays exactly COLUMN_IN
    plt.close(fig)
    return path


def ramp_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("lc_blue", BLUE_RAMP)


def boxes(ax: plt.Axes, groups: list[np.ndarray], labels: list[str], seed: int = 0) -> None:
    """Thin box plots with the individual cups as jittered dots (colour = group, also named on the x-axis)."""
    rng = np.random.default_rng(seed)
    bp = ax.boxplot([g if len(g) else [np.nan] for g in groups], widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.0}, whiskerprops={"color": INK2, "linewidth": 0.6},
                    capprops={"color": INK2, "linewidth": 0.6}, boxprops={"linewidth": 0.8})
    for patch, colr in zip(bp["boxes"], SERIES):
        patch.set_facecolor(colr + "33")
        patch.set_edgecolor(colr)
    for i, (g, colr) in enumerate(zip(groups, SERIES), 1):
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(g)), g, s=4, color=colr, alpha=0.8, linewidths=0, zorder=3)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.grid(axis="x", visible=False)
