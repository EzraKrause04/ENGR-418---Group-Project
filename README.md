# Latte Coach

A phone-photo tool that scores a latte-art heart, names the features that cost it the most points, and estimates the likely milk or pour problem.

ENGR 418 (Applied Machine Learning for Engineers), UBC Okanagan, team project.

## Objective

From one overhead phone photo of a latte-art heart:

1. **Score** it on the team's 1–5 rubric (Task A), with its cross-validated error.
2. **Name the two features that cost it the most points**, each with a practical tip.
3. **Estimate the likely cause on the milk side**, whether the milk was too thin, about right or too thick, and whether the pour dropped in late (Task B). The demo shows this only when cross-validation says the model is reliable.

Every number comes from scripts, with leave-one-session-out cross-validation, on cups the team pours in a randomized 3 × 2 design (milk condition × drop-in timing). Both raters label every cup in a blind labelling tool, which hides the cup ID, condition, pourer and session.

## How it works

| Stage | Code | What happens |
|---|---|---|
| Collect | `make schedule`, `make ingest` | Randomized cup order in blocks of 6; photos are EXIF/GPS-stripped and resized to 1024 px |
| Find the cup | `latte.geometry`, `make qa`, `make circles` | The liquid-surface circle is detected automatically, then checked or corrected by hand |
| Preprocess | `latte.preprocess` | Square crop around the circle, radius shrunk 4%, masked and resized to 512 × 512 |
| Segment | `latte.segment` | Milk vs crema: k-means (k = 2) on L\*a\*b\*, or Otsu on L\* |
| Features | `latte.features`, `make features` | Ten handcrafted features (contrast, edge sharpness, boundary roughness, symmetry, centring, coverage, components, bubbles, texture), each with its formula in its docstring |
| Embeddings | `latte.embed`, `make embed` | Frozen DINOv2 ViT-S/14, averaged over 4 rotations; PCA applied only inside CV |
| Label | `make label`, `make agree` | Blind, keyboard-driven labelling by two raters; weighted κ, Spearman ρ |
| Evaluate | `make train`, `make leakcheck` | E0–E5 (below), nested leave-one-session-out CV, shuffled-label leakage check |
| Feedback | `make final`, `latte.feedback` | Score ± cross-validated error; fixes from Ridge contributions w·z; milk diagnosis with per-class thresholds tuned on out-of-fold precision, or withheld |
| Demo | `make demo` | Gradio app (upload or webcam) |
| Report | `make figures`, `report/` | Figures F1–F8, tables T1–T5, an IEEEtran skeleton |

**Experiments:**
- **E0:** human agreement, the ceiling the models are compared against.
- **E1:** whether each feature tracks its rubric dimension.
- **E2:** Task A, score regression: dummy, Ridge on handcrafted features, embeddings (PCA) and both together, and a random forest on handcrafted features.
- **E3:** Task B, milk condition and drop-in: dummies, then logistic regression and random forest on each feature set; also B3, a regression of measured foam gain.
- **E4:** permutation importance and drop-one-group ablation.
- **E5:** generalisation across pourers.

## Setup

Requires Python 3.11. With [uv](https://docs.astral.sh/uv/) installed, `make setup` uses it, which is fast and fetches Python 3.11 if needed. Without uv it falls back to `python3.11 -m venv`.

```bash
git clone https://github.com/EzraKrause04/ENGR-418---Group-Project.git latte-coach   # team: the SSH URL
cd latte-coach
make setup
make test
make weights        # once, while online: the DINOv2 weights (about 90 MB)
```

If `python3.11` isn't on your PATH, point the fallback at it: `make setup PY311=/path/to/python3.11`.
The Makefile always calls `.venv/bin/python` directly, so you never need to activate the venv. `make help` lists the main targets.

## Usage

| When | Command | What it does |
|---|---|---|
| Before pouring | `make schedule SESSION=S05 POURER=A BLOCKS=2` | Randomized cup order (blocks of 6, every cell once per block), appended to `data/cup_log.csv` |
| After pouring | copy photos to `data/raw_full/`, fill in the cup log, then `make ingest` | Strips EXIF/GPS, resizes to 1024 px, writes `data/raw/`, checks photos ↔ log, prints counts per cell |
| After ingest | `make qa`, then `make circles` | Contact sheets of every detected liquid circle (`results/qa/`), then a browser tool (http://127.0.0.1:7861) to accept or fix each one |
| After the circles | `make features`, `make embed` | Segmentation, handcrafted features (`features/features.csv`, overlays in `results/qa/`) and cached embeddings (`features/emb_*`) |
| Next day | `make label` | Blind labelling in the browser (http://127.0.0.1:7860) |
| After calibration | `make agree` | Weighted κ, Spearman ρ and mean absolute difference per rubric dimension |
| Go/no-go (≥ 30 usable cups) | `make gonogo` | Feature box plots by condition, leave-one-session-out accuracy for milk condition and drop-in, spread of Overall scores, with the brief's decision rules as hints |
| After features, embeddings and labels | `make train`, then `make leakcheck` | Experiments E0–E5 → `results/tables/` plus a run record per experiment in `results/runs/`; the leakage check must PASS |
| After `make train` | `make final` | Refits the demo models on all usable cups → `models/latte_final.joblib` (summary in `models/final.json`) |
| To show it off | `make demo` | http://127.0.0.1:7863: upload or take a photo → cup found, milk outline and symmetry axis, score ± error, top two fixes, milk diagnosis (only when reliable). Use held-out photos with `make demo ARGS="--examples a.jpg b.jpg"` |
| For the report | `make figures`, then `make readme` | Figures F1–F8 and table T1; the Results section below is filled from `results/tables/` |
| Everything | `make all` | `test env qa features embed agree gonogo train leakcheck final figures readme`. Steps without enough data yet are skipped with a message. It stops at the first failure, on purpose including a leakage-check CHECK |
| Before submitting | `make repro` | Clones the committed HEAD into a temporary folder, runs `make setup` and `make all` there (recomputing the embeddings), and checks that every table and figure is regenerated and that each table matches this checkout. It fails if the cup log, circles or labels aren't committed. `make repro REPRO=--keep` keeps the clone |

The full pouring and photo protocol is in [docs/protocol.md](docs/protocol.md). The scoring rubric is in [docs/rubric.md](docs/rubric.md), and the feedback tips are in [src/latte/tips.yaml](src/latte/tips.yaml). Interactive apps (`make label`, `make circles`, `make demo`) keep running until Ctrl-C, so start them from a normal terminal.

## Repository layout

```
configs/default.yaml   every path, seed (42) and threshold
data/raw/              ingested photos: EXIF-stripped, 1024 px (committed)
data/raw_full/         original photos (gitignored; team shared drive)
data/cup_log.csv       one row per cup
data/circles.csv       liquid circles checked by eye (append-only; last row per cup wins)
labels/rater_<X>.csv   append-only rater labels (last row per cup wins)
src/latte/             library code: io, geometry, preprocess, segment, features, embed, data,
                       models, evaluate, feedback, inference, plots, overlay, ...
scripts/               command-line steps, one per Makefile target
app/                   Gradio apps: labelling, circle check, demo
features/              features.csv and cached embeddings
models/                final demo models (joblib, gitignored) and final.json
results/               tables (CSV + LaTeX), figures, QA sheets, per-run records
report/                IEEEtran skeleton (headings and hooks only)
docs/                  protocol, rubric, AI-use log
tests/                 pytest suite on synthetic photos and fake studies
```

## Data protocol and privacy

- Only photos the team takes. Nothing is scraped from social media or video sites.
- Every photo is EXIF-stripped (including GPS) by `make ingest` before it can be committed. `data/raw_full/` is gitignored.
- Photos from anyone outside the team need that person's written OK first.
- External development images (`data/dev_external/`, gitignored) are used only to develop rim detection and segmentation. They are never used for labelling, training or evaluation.
- The demo runs locally (127.0.0.1). Gradio's temporary uploads are deleted once they are more than 10 minutes old (checked every 10 minutes), and when the demo stops.

## Results

<!-- results:start -->
_No results yet: run `make features`, `make label`, `make agree`, `make train` and `make figures`, then `make readme`._
<!-- results:end -->

## Key findings

<!-- Students write this section (brief, sections 2 and 15). -->

## Limitations

<!-- Students write this section. The brief (section 7.3) asks it to note that a pourer may recognize their own cups; blind, random-order labelling a day or more later reduces but doesn't remove this. -->

## Attribution

- **DINOv2** (Oquab et al., 2023, Meta AI), ViT-S/14 weights `timm/vit_small_patch14_dinov2.lvd142m`, loaded through [timm](https://github.com/huggingface/pytorch-image-models). Apache-2.0.
- **ResNet-50** ImageNet weights from [torchvision](https://pytorch.org/vision/), used only as a fallback if DINOv2 can't be downloaded.
- Built with [scikit-learn](https://scikit-learn.org/), [scikit-image](https://scikit-image.org/), [OpenCV](https://opencv.org/), [PyTorch](https://pytorch.org/), [Gradio](https://www.gradio.app/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [matplotlib](https://matplotlib.org/). Exact versions are in `requirements.txt` and `results/env.txt`.
- **External development images:** none used so far. If the Roboflow Universe "latte-art-hearts" set is used for pipeline development, credit it here under its licence (CC BY 4.0 requires attribution).
- **Generative AI:** Claude Code built the code, tests, figures and tables, wrote the report skeleton, and wrote this README's descriptive sections (Objective, How it works, Setup, Usage, Repository layout, Data protocol and privacy, Attribution) as well as the generated Results block. Every session is logged in [docs/ai-use-log.md](docs/ai-use-log.md). The report text, this README's Key findings and Limitations, the License choice, the video and the individual AI declarations are the students' own.

## License

<!-- The team chooses: e.g. MIT for the code and CC BY 4.0 (or all rights reserved) for the photos in data/raw/. -->
_To be chosen by the team._
