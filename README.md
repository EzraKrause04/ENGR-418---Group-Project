# Latte Coach

A phone-photo tool that scores a latte-art heart, names the features that cost it the most points, and estimates the likely milk or pour problem.

ENGR 418 (Applied Machine Learning for Engineers), UBC Okanagan, team project.

> **Status (2026-11-27):** build steps 1–11 done: repo scaffold, data-collection tooling (`make_schedule`, `ingest`), blind labelling tool and agreement script, liquid-circle detection with a manual check tool, crop/mask and QA sheets, milk/crema segmentation and handcrafted features, the go/no-go check, cached pretrained embeddings, the train/evaluate harness (E0–E5, leakage check), the feedback engine with the final demo models, the demo app, and the report figures, tables and LaTeX skeleton.
> This README's results table (step 12) and a clean-clone reproducibility pass (step 13) come next. This README is completed in build step 12.

## Setup

Requires Python 3.11. With [uv](https://docs.astral.sh/uv/) installed, `make setup` uses it, which is fast and fetches Python 3.11 if needed. Without uv it falls back to `python3.11 -m venv`.

```bash
git clone git@github.com:EzraKrause04/ENGR-418---Group-Project.git latte-coach
cd latte-coach
make setup
make test
```

If `python3.11` isn't on your PATH, point the fallback at it: `make setup PY311=/path/to/python3.11`.
The Makefile always calls `.venv/bin/python` directly, so you never need to activate the venv.

## Daily workflow

| When | Command | What it does |
|---|---|---|
| Before pouring | `make schedule SESSION=S05 POURER=A BLOCKS=2` | Randomized cup order (blocks of 6, every cell once per block), appended to `data/cup_log.csv` |
| After pouring | copy photos to `data/raw_full/`, fill in the cup log, then `make ingest` | Strips EXIF/GPS, resizes to 1024 px, writes `data/raw/`, checks photos ↔ log, prints counts per cell |
| After ingest | `make qa`, then `make circles` | Contact sheets of every detected liquid circle (`results/qa/`), then a browser tool (http://127.0.0.1:7861) to accept or fix each one |
| After the pilot (and any rig change) | `make features` | Milk/crema segmentation and handcrafted features → `features/features.csv`; QA overlays `results/qa/segment_*.jpg` and Otsu vs k-means `seg_compare_*.jpg` |
| Next day | `make label` | Blind labelling in the browser (http://127.0.0.1:7860) |
| After calibration | `make agree` | Weighted κ, Spearman ρ and mean absolute difference per rubric dimension |
| Once, while online | `make weights` | Downloads the DINOv2 ViT-S/14 weights (about 90 MB, Hugging Face) so `make embed` works offline |
| After ingest | `make embed` | Pretrained-network embeddings of every cup, averaged over 4 rotations and cached in `features/emb_*` (only new or changed cups are recomputed) |
| Fri Nov 27 (≥ 30 usable cups) | `make gonogo` | Go/no-go check: feature box plots by condition, leave-one-session-out accuracy for milk condition and drop-in, spread of Overall scores, with the brief's decision rules as hints |
| After features, embeddings and labels | `make train` | Experiments E0–E5 with leave-one-session-out cross-validation → `results/tables/T3_task_a.*`, `T4_task_b.*`, `T5_ablation.*`, `E0`/`E1`/`E4`/`E5` tables, and a run record per experiment in `results/runs/` |
| Before trusting results | `make leakcheck` | Re-runs the main models on labels shuffled within sessions; every model must land at chance |
| After `make train` | `make final` | Refits the demo models on all usable cups → `models/latte_final.joblib` (score ± cross-validated error, Ridge-contribution fixes, milk diagnosis only if Task B beat chance); summary in `models/final.json` |
| To show it off | `make demo` | The demo at http://127.0.0.1:7863: upload or take a photo → detected cup, milk outline and symmetry axis, score ± error, top two fixes, milk diagnosis (only when reliable). Example cups: the first ingested photos, or `make demo ARGS="--examples a.jpg b.jpg"` |
| For the report | `make figures` | Figures F1–F8 (300 dpi, one IEEE column) in `results/figures/` and table T1 in `results/tables/`; `report/main.tex` is an IEEEtran skeleton (headings and table/figure hooks only) |

The full pouring and photo protocol is in [docs/protocol.md](docs/protocol.md). The scoring rubric is in [docs/rubric.md](docs/rubric.md).

## Repository layout

```
configs/default.yaml   every path, seed (42) and threshold
data/raw/              ingested photos: EXIF-stripped, 1024 px (committed)
data/raw_full/         original photos (gitignored; team shared drive)
data/cup_log.csv       one row per cup
data/circles.csv       liquid circles checked by eye (append-only; last row per cup wins)
labels/rater_<X>.csv   append-only rater labels (last row per cup wins)
src/latte/             library code (io, schedule, labels, evaluate, ...)
scripts/               command-line steps, one per Makefile target
app/                   Gradio apps: labelling, circle annotation, demo
results/               tables, figures, QA sheets and per-run records
docs/                  protocol, rubric, AI-use log
```

## Data protocol and privacy

- Only photos the team takes. Nothing is scraped from social media or video sites.
- Every photo is EXIF-stripped (including GPS) by `make ingest` before it can be committed. `data/raw_full/` is gitignored.
- Photos from anyone outside the team need that person's written OK first.
- External development images (`data/dev_external/`, gitignored) are used only to develop rim detection and segmentation. They are never used for labelling, training or evaluation.

## Results

_Filled from `results/tables/` in build step 12._

## Key findings

## Limitations

## Attribution

## License
