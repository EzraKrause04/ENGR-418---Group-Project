# Latte Coach

A phone-photo tool that scores a latte-art heart, names the features that cost it the most points, and estimates the likely milk or pour problem.

ENGR 418 (Applied Machine Learning for Engineers), UBC Okanagan, team project.

> **Status (2026-11-25):** build steps 1–6 done: repo scaffold, data-collection tooling (`make_schedule`, `ingest`), blind labelling tool and agreement script, liquid-circle detection with a manual check tool, crop/mask and QA sheets, milk/crema segmentation and handcrafted features, and the go/no-go check.
> Embeddings, models and the demo come next. This README is completed in build step 12.

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
| Fri Nov 27 (≥ 30 usable cups) | `make gonogo` | Go/no-go check: feature box plots by condition, leave-one-session-out accuracy for milk condition and drop-in, spread of Overall scores, with the brief's decision rules as hints |

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
