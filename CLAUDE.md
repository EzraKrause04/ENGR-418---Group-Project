# Latte Coach — ENGR 418 Team Project

A phone-photo tool that scores a latte art heart, names the features that cost it the most points, and estimates the likely milk or pour problem.

This file is the project brief for Claude Code. Read it fully before starting any task. Work through the **Build order** (section 15) in sequence unless the team says otherwise.

---

## 1. Context

- **Course:** ENGR 418 Applied Machine Learning for Engineers, UBC Okanagan (Dr. Ling Bai).
- **Weight and deadline:** worth 50% of the final grade. **Due Sun Dec 13, 23:59, on Canvas.**
- **Team:** two students, referred to as **Rater A** and **Rater B**. They are also the two people pouring the coffee (**Pourer A** and **Pourer B**).
- **Project route:** original development. No public latte-art quality benchmark exists, so we build our own baselines.
- **Course alignment:** prefer methods taught in the course, and use scikit-learn wherever possible. The table shows where each method is used.

  | Course topic | Where we use it |
  |---|---|
  | Feature engineering | Handcrafted image features (section 8.4) |
  | Dimension reduction (PCA) | PCA on pretrained embeddings (section 9) |
  | Linear regression | Ridge regression for quality scores (Task A) |
  | Logistic regression and softmax | Milk-condition and drop-in classifiers (Task B) |
  | Clustering (k-means) | Milk vs crema segmentation (section 8.3) |
  | Neural networks | Frozen pretrained network used as a feature extractor |

## 2. Ground rules (non-negotiable)

1. **Academic integrity.**
   - Claude Code builds code, tooling, tests, figures and results tables.
   - Claude Code does **not** write the prose of the technical report, the README "Key findings" section, the video script, or the individual Generative AI declarations. Those must be the students' own writing.
   - The report template may contain section headings and auto-generated tables or figures, and nothing more.
2. **AI-use log.**
   - Keep `docs/ai-use-log.md` up to date. It is the evidence for each student's individual AI declaration.
   - At the end of every session, or after each completed build-order step, append an entry with:
     - the date and who was at the keyboard (ask if unknown),
     - the task,
     - the key prompts, summarized,
     - the files created or changed,
     - how the output was verified (tests run, manual review),
     - what the students changed by hand.
3. **Git.**
   - Use an SSH remote (`git@github.com:...`).
   - Never ask for, accept, or store a GitHub personal access token. If one appears in chat, tell the user to revoke it.
   - Make small, descriptive commits.
4. **Data provenance.**
   - Use only photos the team takes themselves, plus explicitly licensed datasets.
   - **No scraping** from Reddit, Instagram, YouTube or similar sites. Their terms forbid it, and the repo is public.
   - Any photo poured by someone outside the team needs that person's written OK before it goes in the repo.
5. **Privacy.**
   - Strip all EXIF data, including GPS, from every image before it is committed.
6. **Honesty in the product.**
   - Feedback is phrased as *likely* causes.
   - The demo must never present a tip as certain when the model does not support it (see section 11).

## 3. Deliverables (all due Dec 13)

| Deliverable | Notes |
|---|---|
| Technical report | 4–6 pages, double-column academic format. Sections: Introduction, Related Work, Methods, Results, Conclusion, References. Claude Code sets up the LaTeX skeleton and generates figures and tables. The students write it. |
| GitHub repository | All code and experimental results. Comprehensive README covering objective, setup, usage, data protocol, results and key findings. This is part of the students' portfolio. |
| 3-minute video | The demo app (section 12) is the centrepiece. The students script and record it. |
| Generative AI declaration | One per student, individual. Built from `docs/ai-use-log.md` plus each student's own notes. |

## 4. Tech stack

- **Language:** Python 3.11, with a `venv` and a pinned `requirements.txt`. No GPU needed.
- **Packages:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `opencv-python-headless`, `scikit-image`, `matplotlib`, `pillow`, `pyyaml`, `torch`, `timm`, `gradio`, `pytest`.
- **Config:** all paths, seeds and thresholds live in `configs/default.yaml`. Use a global seed of `42`.
- **Code style:** type hints throughout. Every feature function has a docstring that states its formula.
- **Source of truth:** notebooks are allowed in `notebooks/` for exploration only. All results must come from scripts.
- **Environment record:** save `pip freeze > results/env.txt` with every full run.

## 5. Repository layout

```
latte-coach/
├── CLAUDE.md
├── README.md
├── Makefile
├── requirements.txt
├── configs/default.yaml
├── data/
│   ├── raw_full/          # original phone photos (gitignored; kept in team shared drive)
│   ├── raw/               # ingested: EXIF-stripped, 1024 px longest side (committed)
│   ├── dev_external/      # licensed external images for pipeline development only (gitignored)
│   ├── cup_log.csv        # one row per cup (section 6.4)
│   └── circles.csv        # detected or manually corrected cup circles
├── labels/
│   ├── rater_A.csv
│   └── rater_B.csv
├── docs/
│   ├── rubric.md          # scoring anchors + changelog
│   ├── protocol.md        # pouring and photo protocol
│   └── ai-use-log.md
├── src/latte/
│   ├── io.py
│   ├── geometry.py        # cup circle detection
│   ├── preprocess.py      # crop, mask, resize
│   ├── segment.py         # milk vs crema
│   ├── features.py        # handcrafted features
│   ├── embed.py           # pretrained embeddings
│   ├── models.py          # pipelines and model grids
│   ├── evaluate.py        # CV, metrics, agreement
│   ├── feedback.py        # score, top fixes, diagnosis
│   └── tips.yaml
├── scripts/
│   ├── make_schedule.py
│   ├── ingest.py
│   ├── qa_sheets.py
│   ├── build_features.py
│   ├── agreement.py
│   ├── go_no_go.py
│   ├── train_eval.py
│   ├── fit_final.py
│   └── make_figures.py
├── app/
│   ├── label.py           # Gradio labelling tool
│   ├── annotate_circle.py # manual circle override
│   └── demo.py            # Gradio demo for the video
├── features/              # cached feature tables and embeddings
├── models/                # final fitted models (joblib)
├── results/
│   ├── runs/              # one folder per run: config + metrics.json
│   ├── tables/            # CSV + LaTeX
│   ├── figures/
│   └── qa/                # contact sheets for visual checks
├── report/                # IEEEtran LaTeX skeleton (headings only)
└── tests/
```

**Makefile targets:** `setup`, `ingest`, `qa`, `features`, `embed`, `agree`, `gonogo`, `train`, `final`, `figures`, `demo`, `test`, `all`.

## 6. Data collection protocol

Write this protocol into `docs/protocol.md` as well, so the team can follow it at the machine.

### 6.1 Experimental design

- **Pattern:** hearts only. Use one milk type (whole milk), one cup model, one machine.
- **Factors:** a 3 × 2 design.
  - **Milk condition:** `thin`, `good` or `thick`.
  - **Drop-in timing:** `on_time` (lower the pitcher to the surface when the cup is about ½ full) or `late` (when the cup is about ¾ full).
- **Targets:** 20 cups per cell, so **120 cups minimum** (stretch goal 150).
- **Starting targets for foam gain.** These are calibrated in the pilot (Milestone 1).

  | Condition | Aeration | Foam gain |
  |---|---|---|
  | `thin` | about 1 s | under 10% |
  | `good` | normal | 20–30% |
  | `thick` | long | 45% or more |

- **Foam gain measurement:** mark the milk level in a straight-walled pitcher before and after steaming. Foam gain % = (after − before) / before × 100. Record the measured value. The analysis uses the measured gain as well as the planned condition.

### 6.2 Randomization: `scripts/make_schedule.py`

- **Purpose:** cup order is randomized in blocks of 6 so that each block contains every cell once. This spreads the pourers' improvement over time evenly across conditions.
- **Usage:** `python scripts/make_schedule.py --session S05 --pourer A --blocks 2`
- **Output:**
  - appends planned rows to `data/cup_log.csv`, with measurement fields left blank,
  - prints a readable order to follow at the machine.
- **Cup IDs:** `S05-C01`, `S05-C02`, and so on.

### 6.3 Photo capture rig

- Mount the phone overhead at about 30 cm, on a fixed mount. Use the same table, cup, light source and time of day where possible.
- No flash.
- Lock exposure and focus (AE/AF lock).
- Keep the whole cup rim in frame. No zoom, crop or filters.
- Photograph within 30 s of finishing the pour. Take 2 shots and keep the sharper one.
- Name the file `<cup_id>.jpg`.

### 6.4 `data/cup_log.csv` schema

| Column | Type | Notes |
|---|---|---|
| `cup_id` | str | `S05-C03` |
| `session_id` | str | `S05`; used as the CV group |
| `date` | date | |
| `pourer` | `A`/`B` | |
| `milk_condition` | `thin`/`good`/`thick` | planned |
| `dropin` | `on_time`/`late` | planned |
| `foam_gain_pct` | float | measured |
| `milk_temp_c` | float | optional |
| `seconds_to_photo` | int | |
| `photo_file` | str | |
| `usable` | `y`/`n` | `n` for a blurry photo, glare, or a failed pour that isn't a heart |
| `notes` | str | |

### 6.5 Ingest: `scripts/ingest.py`

- Reads `data/raw_full/`, strips EXIF, resizes to 1024 px on the longest side, and writes to `data/raw/`.
- Validates that every photo has a log row and every log row has a photo.
- Prints the counts per cell.

### 6.6 External development images

The Roboflow Universe "latte-art-hearts" set (115 images) may be used **only** to develop and QA rim detection and segmentation before our own data exists.

- Check the license on the dataset page first. If it is CC BY 4.0, attribute it in the README.
- These images are never used for training, labelling or evaluation.

## 7. Labelling

### 7.1 Rubric (`docs/rubric.md`)

These are draft anchors. **The team edits and approves them before labelling starts.** Score each dimension from 1 to 5, using 2 and 4 for in-between cases.

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Contrast | Pattern barely visible or sunk | Visible but greyish or blotchy | Bright white on even crema |
| Definition | Edges smeared or feathered, no clear outline | Outline mostly clear, some blur | Clean, sharp edges throughout |
| Symmetry | Clearly lopsided | Minor asymmetry | Mirror-symmetric |
| Centring | Touching the rim or far off-centre | Noticeably off-centre | Centred |
| Surface | Visible bubbles, matte or dry foam | A few bubbles or slight dullness | Glossy, no visible bubbles |
| Overall | Holistic judgement, **not** the average of the other five | | |

Each photo also gets an `unusable` checkbox. Record every change to the anchors in a changelog at the bottom of `rubric.md`, with the date and reason.

### 7.2 Labelling tool: `app/label.py` (Gradio)

- **Rater selection:** the user selects Rater A or B.
- **Blind labelling:** images are shown one at a time in a rater-specific random order (seeded by the rater ID). The screen shows only the image and an anonymized index. It never shows the `cup_id`, condition, pourer or session.
- **Inputs:** a 1–5 control for each of the six dimensions, the `unusable` checkbox, and a comment box.
- **Saving:** autosaves every submission to `labels/rater_<X>.csv` (columns: `cup_id`, `dim` scores, `unusable`, `comment`, `timestamp`).
- **Resumable:** the tool skips images already labelled by that rater and picks up new images after each ingest.
- **Keyboard:** number keys plus Enter if feasible.
- **Coverage:** both raters label **every** cup, because the human-agreement ceiling needs both.

### 7.3 Agreement: `scripts/agreement.py`

- **Per dimension, it computes:**
  - quadratic-weighted Cohen's kappa,
  - Spearman ρ,
  - mean absolute difference between raters.
- **Outputs:** `results/tables/agreement.csv` and `agreement.tex`.
- **Calibration round:** both raters label the same first 30 cups.
  - Target weighted κ is 0.6 or higher on Overall.
  - If κ is below 0.4, the team reviews the disagreements, revises the anchors (logged in the changelog), and relabels those 30.
- **Final labels:** the mean of Rater A and Rater B for each dimension.
- **Limitation to record in the README:** a pourer may recognize their own cups. Random-order labelling done a day or more later reduces but doesn't remove this.

## 8. Image pipeline

All processing happens at 512 × 512 after cropping.

### 8.1 Cup detection: `geometry.py`

- **Target circle:** find the **liquid surface** circle, not the outer cup rim.
- **Method:** Hough circles on a blurred grayscale image, with the radius restricted to a plausible range, such as 0.2–0.5 × min(H, W).
- **Output:** `(cx, cy, r, confidence)`.
- **Low confidence:** flag the image for review.
- **Manual override:** `app/annotate_circle.py` lets a student click 3 points on the liquid edge, fits a circle, and saves it to `data/circles.csv`. Saved circles always override auto-detection.
- **QA:** `scripts/qa_sheets.py` writes contact sheets of every detected circle to `results/qa/`.
- **Acceptance:** every usable image has a circle that a student has checked by eye.

### 8.2 Preprocessing: `preprocess.py`

- Crop a square around the circle.
- Shrink the radius by 4% to exclude the rim edge.
- Mask everything outside the circle and resize to 512 × 512.

### 8.3 Segmentation: `segment.py`

Implement two methods and compare them on QA sheets:

- **`otsu`:** Otsu threshold on the CIELAB L channel inside the mask.
- **`kmeans`:** k-means with k = 2 on (L, a, b) pixels inside the mask (course clustering topic). The brighter cluster is milk.

**Post-processing:** morphological open, then close (disk radius 3), then remove components smaller than 0.5% of the circle area. Select the method with `configs/default.yaml`. The default is whichever looks better on the pilot QA sheets.

### 8.4 Handcrafted features: `features.py`

Each feature is a pure function returning a float, with its formula in the docstring. Here M is the milk mask, C is the crema region (the circle minus M), and L is the CIELAB lightness.

| Feature | Definition | Rubric link |
|---|---|---|
| `contrast_dL` | mean L(M) − mean L(C) | Contrast |
| `contrast_michelson` | (L̄_M − L̄_C) / (L̄_M + L̄_C) | Contrast |
| `edge_sharpness` | mean Sobel gradient magnitude of L over a 2-px band around the M boundary, divided by `contrast_dL` (so it isn't a contrast proxy) | Definition |
| `boundary_roughness` | perimeter² / (4π · area) of the largest M component | Definition |
| `symmetry_iou` | max over angles 0–178° (step 2°) of IoU(M, M reflected about a line through M's centroid at that angle); also store `symmetry_angle` | Symmetry |
| `centre_offset` | ‖centroid(M) − circle centre‖ / r | Centring |
| `coverage` | area(M) / circle area | Drop-in timing |
| `n_components` | number of M components left after filtering | Broken pattern |
| `bubble_density` | Laplacian-of-Gaussian blob count (σ = 1–4 px) on high-pass-filtered L inside the circle, per 10⁴ px² | Surface |
| `texture_energy` | variance of the Laplacian of L inside M | Surface |

**Outputs:** `scripts/build_features.py` writes `features/features.csv`, keyed by `cup_id`, and QA overlays (mask outline, centroid, symmetry axis) to `results/qa/`.

### 8.5 Tests: `tests/`

Use synthetic fixtures: a brown disk with a white, symmetric, centred heart drawn on it. The tests must assert:

- the centred heart has `symmetry_iou` above 0.9 and `centre_offset` below 0.05;
- shifting the heart increases `centre_offset`;
- Gaussian blur reduces `edge_sharpness`;
- darkening the milk reduces `contrast_dL`;
- adding small dark dots increases `bubble_density`.

`make test` must pass before any modelling.

## 9. Pretrained embeddings: `embed.py`

- **Model:** `timm` model `vit_small_patch14_dinov2.lvd142m`, pretrained, with `num_classes=0`, run on the 512 crop resized to the model's input size.
- **Fallback:** if the DINOv2 download fails, use `torchvision` ResNet-50 with ImageNet weights.
- **Rotation averaging:** average the embedding over 4 rotations (0°, 90°, 180°, 270°). The cups are round, so rotation is free augmentation.
- **Cache:** save to `features/emb_<model>.npy` along with a `cup_id` index file.
- **PCA:** applied **inside** the CV pipeline (section 10), never before the split.

## 10. Models and evaluation

### 10.1 Protocol: `train_eval.py`

- **Data:** join features, mean labels and `cup_log`. Exclude any cup flagged unusable in the log, or flagged unusable by either rater.
- **Outer CV:** `LeaveOneGroupOut` grouped by `session_id`. No session appears in both train and test.
- **Inner CV:** `GroupKFold(3)` inside `GridSearchCV` for hyperparameters.
- **Pipelines:** everything (scaling, PCA, model) lives inside an sklearn `Pipeline` or `ColumnTransformer`, so nothing leaks across folds.
- **Reporting:** mean ± std across folds, plus metrics pooled over all out-of-fold predictions.
- **Run records:** every run writes `results/runs/<timestamp>_<exp>/` containing the config, `metrics.json` and out-of-fold predictions. Final tables go to `results/tables/` as CSV and `.tex`.

### 10.2 Experiments

- **E0. Human agreement**
  - Output of section 7.3. This sets the ceiling the models are compared against.
- **E1. Feature validity**
  - Spearman ρ between each handcrafted feature and its linked rubric dimension (section 8.4 table), with 95% bootstrap CIs.
  - This tests whether each engineered feature tracks what humans see.
- **E2. Task A: quality score regression**
  - **Target:** mean Overall score.
  - **Models:**
    - `DummyRegressor(mean)`
    - `Ridge` on handcrafted features (alpha grid)
    - `RandomForestRegressor` on handcrafted features (small grid)
    - `Ridge` on PCA of embeddings (n_components ∈ {5, 10, 20})
    - `Ridge` on handcrafted + PCA of embeddings
  - **Metrics:** MAE, RMSE, Spearman ρ, R².
  - **Fair human comparison:**
    - Report each model's MAE against the mean label **and** against each rater separately.
    - The human reference is the MAE between Rater A and Rater B.
    - Compare like with like: model vs one rater ↔ rater vs rater.
- **E3. Task B: diagnosis**
  - **Targets:**
    - B1: `milk_condition` (3 classes)
    - B2: `dropin` (binary)
    - B3 (optional): regression of measured `foam_gain_pct`
  - **Models:**
    - `DummyClassifier` (most_frequent and stratified)
    - multinomial `LogisticRegression` (C grid)
    - `RandomForestClassifier`
    - each on the same feature sets as E2
  - **Metrics:** balanced accuracy, macro-F1, and confusion matrices summed over folds.
- **E4. Importance and ablation**
  - Permutation importance for the best handcrafted model in E2 and E3, averaged over outer folds.
  - Drop-one-feature-group ablation over these groups: contrast, edge, symmetry/centring, coverage, surface.
- **E5. Pourer generalization** (only if each pourer has at least 40 usable cups)
  - Train on one pourer, test on the other, in both directions.
- **E6. Segmentation choice** (optional)
  - Otsu vs k-means, judged by E1 correlations and E2 MAE.

## 11. Feedback engine: `feedback.py`

`scripts/fit_final.py` fits the final models on all usable data and saves them to `models/`. These are used only for the demo. Reported metrics always come from CV.

For a new image, the engine produces:

1. **Score:** the predicted Overall score, rounded to 0.5, shown as "± X". X is the pooled out-of-fold MAE from E2.
2. **Top fixes:**
   - Uses the final handcrafted Ridge model.
   - Each feature's contribution is cᵢ = wᵢ · zᵢ, where zᵢ is the standardized feature value.
   - Take the two most negative contributions with cᵢ < −0.1 score points and map each feature name to a tip in `tips.yaml`.
   - If there are none, show "No major issues found."
3. **Milk diagnosis:**
   - If the final B1 classifier's top probability is at or above the threshold (default 0.6; tuned so out-of-fold precision at that threshold is at least 0.7), show "Likely: milk too thick (p = 0.71)" followed by its tip.
   - Otherwise show "Milk condition unclear from this photo."
4. **Withheld claims:** if E3 shows B1 is no better than chance, the demo omits the milk diagnosis entirely.

**Draft `tips.yaml` content.** The team reviews and edits this before use.

| Feature (problem direction) | Likely cause | Tip |
|---|---|---|
| Low `contrast_dL` | Milk sank: poured from too high for too long, or milk too thin | Bring the spout down to the surface sooner, and aerate a little longer |
| Low `edge_sharpness` | Milk too thick, or foam separated while it sat | Aerate less; swirl and pour right after steaming |
| High `bubble_density` / `texture_energy` | Large bubbles in the milk | Purge the wand, keep the tip just under the surface, and tap and swirl before pouring |
| Low `symmetry_iou` | Pitcher drifted sideways, or cup tilted unevenly | Keep the spout on the cup's centre line and level the cup steadily |
| High `centre_offset` | Design started off-centre | Start the pattern in the centre of the cup |
| Low `coverage` | Dropped in late | Lower the pitcher to the surface when the cup is about half full |
| High `n_components` | Unsteady flow broke the pattern | Keep the flow steady and don't lift the pitcher mid-design |

## 12. Demo app: `app/demo.py` (Gradio)

- **Input:** an uploaded image (and webcam, if easy).
- **Pipeline:** detect the circle, segment, extract features and embeddings, then run the feedback engine.
- **Outputs:**
  - an overlay image (circle, milk outline, symmetry axis),
  - the score ± error,
  - the top two fixes,
  - the milk diagnosis, or its absence.
- **Launch:** `make demo` runs it locally. Make it look clean enough to screen-record for the video.
- **Optional:** deploy to Hugging Face Spaces as a portfolio link, using only team-owned example images.

## 13. Report support (no prose)

`scripts/make_figures.py` writes the following to `results/figures/` as PNG at 300 dpi, sized for a single report column.

| Figure | Content |
|---|---|
| F1 | Pipeline stages (raw → crop → mask → feature overlays) for a `good`, a `thin` and a `thick` cup |
| F2 | Example cups with mean Overall scores near 1, 3 and 5 |
| F3 | Inter-rater agreement for each dimension |
| F4 | Feature vs linked rubric dimension scatter plots (E1) |
| F5 | Predicted vs actual Overall score for the best E2 model, with a y = x line and the human-disagreement band |
| F6 | Confusion matrices for B1 and B2 |
| F7 | Permutation importance (E4) |
| F8 | Feature box plots by milk condition |

**Tables** are written to `results/tables/`:

- **T1:** dataset summary (cups per cell, sessions, pourers, usable count)
- **T2:** agreement
- **T3:** Task A results
- **T4:** Task B results
- **T5:** ablation

**`report/` contents:**

- an IEEEtran conference skeleton with section headings only,
- `\input{}` hooks for the generated tables,
- the figure paths.

The students write all text. Claude Code may format BibTeX entries for sources the students supply, but does not search for or select related work.

## 14. Milestones (today is Mon Nov 23)

| Dates | Milestone |
|---|---|
| Nov 23 | **M0.** Repo scaffold, environment, `make_schedule.py`, `ingest.py`, `label.py`, protocol and rubric drafts |
| Nov 24 | **M1.** Confirm Dr. Bai's approval (office hours Tue 3–5 pm) if not already given. Pilot 12 cups (2 per cell) and calibrate foam-gain targets |
| Nov 24–27 | **M2.** Circle detection, segmentation, features, tests and QA sheets, developed on the pilot photos plus the external dev set |
| Nov 25 – Dec 3 | **Data collection:** about 12 cups a day across both pourers, reaching 120 or more. Ingest and label daily |
| Fri Nov 27 | **M3.** Calibration labelling (first 30 cups, both raters), agreement check, and **go/no-go** (section 16) |
| Nov 28 – Dec 1 | **M4.** Embeddings plus the full train/eval harness running end-to-end on the partial data |
| Thu Dec 3 (evening) | **Data freeze.** No new cups after this point |
| Dec 4–6 | **M5.** Full experiments E0–E5, figures and tables |
| Dec 6–8 | **M6.** Feedback engine and demo app |
| Dec 7–12 | **M7.** Students write the report, README findings, video and individual AI declarations. Claude Code only fixes bugs and regenerates figures |
| Sun Dec 13, 23:59 | **Submit** on Canvas |

## 15. Build order for Claude Code

Each step lists what it must achieve before it counts as done. Append an AI-log entry after each step.

1. **Scaffold**
   - Create the layout from section 5, `requirements.txt`, the Makefile, `configs/default.yaml`, and `.gitignore` (excluding `data/raw_full/`, `data/dev_external/`, `.venv/`, caches and `models/*.joblib`).
   - Create `docs/ai-use-log.md` with an entry template.
   - *Done when:* `make setup` works from a clean clone.
2. **Collection tooling**
   - Build `make_schedule.py`, `ingest.py`, and `docs/protocol.md`.
   - *Done when:* it runs end-to-end on 3 dummy photos.
3. **Labelling tool**
   - Build `app/label.py` and `scripts/agreement.py`.
   - *Done when:* blind display, resuming and autosave all work, and agreement runs on fake labels.
4. **Geometry and preprocessing**
   - Build circle detection, the manual override tool, crop/mask, and QA sheets.
   - *Done when:* students have checked the QA sheets on the pilot and dev images.
5. **Segmentation and features**
   - Build both segmentation methods, all features, and the tests.
   - *Done when:* `make test` passes and the QA overlays look right to the students.
6. **Go/no-go script**
   - Build `scripts/go_no_go.py` (section 16).
7. **Embeddings**
   - Build `embed.py` with caching.
   - *Done when:* embeddings exist for every ingested cup.
8. **Train/eval harness**
   - Build `models.py`, `evaluate.py`, and `train_eval.py` covering E0–E5.
   - *Done when:* it runs on the partial data, and a leakage check (shuffled labels should give chance-level results) passes.
9. **Feedback engine and final fit**
   - Build `fit_final.py`, `feedback.py`, and `tips.yaml`.
10. **Demo**
    - Build `app/demo.py`.
11. **Figures, tables and report skeleton**
    - Build `make_figures.py` and `report/`.
12. **README**
    - Write the objective, setup, usage, data protocol, a results table filled from `results/tables/`, attribution and license.
    - Leave the "Key findings" and "Limitations" sections as headings for the students to write.
13. **Reproducibility pass**
    - *Done when:* `make all` from a clean clone (with `data/raw/` present) regenerates every table and figure.

## 16. Go/no-go check (Fri Nov 27, at least 30 usable cups)

`scripts/go_no_go.py` writes:

- box plots of each feature by `milk_condition` and by `dropin`,
- the leave-one-session-out balanced accuracy of logistic regression for B1 and B2,
- the distribution of mean Overall scores.

The team then decides:

- **Diagnosis signal present** (some features separate the conditions, and B1 balanced accuracy is clearly above 0.33): continue as planned.
- **No diagnosis signal:**
  - keep collecting, because Task A still needs the data,
  - focus on Task A and the contribution-based fixes,
  - report the diagnosis result honestly as a negative finding: under this setup, a single photo does not reveal milk texture,
  - do **not** switch projects at this date.
- **Narrow quality range** (almost all Overall scores ≤ 2): pour more `good`/`on_time` cups, or have a skilled friend pour 15–20 cups **on the same rig** with written permission. Log them with a distinct pourer ID.

---

## 17. Working notes for Claude Code (kept current; not part of the original brief)

- **Location:** the repo lives at `~/dev/latte-coach`. `…/ENGR418/latte-coach` is a symlink to it, because the course folder's path contains `:`, which breaks `source .venv/bin/activate`. Run Python as `.venv/bin/python` or through `make`.
- **Remote:** `origin` = `git@github.com:EzraKrause04/ENGR-418---Group-Project.git` (SSH). Push only when the team asks.
- **Pinned majors are new:** pandas 3, OpenCV 5, Pillow 13, Gradio 6. In Gradio 6, `theme`/`css`/`head` go to `launch()`, not `Blocks()`. `gr.Image` takes `buttons=[...]` and `format=`.
- **Labels:** `labels/rater_<X>.csv` is append-only; the last row per `cup_id` wins (`latte.labels.latest_labels`). Final labels come from `latte.labels.mean_labels`.
- **Circles:** `data/circles.csv` is append-only (last row per cup wins) and holds only circles a student checked: `accepted` (auto circle frozen as accepted) or `manual` (3+ clicked points, least-squares fit). `latte.geometry.resolve_circle` prefers a saved circle when its `img_w`/`img_h` match the photo. Detection = innermost dark-to-light radial edge ring (`geometry.edge_polarity`); `confidence` = edge support in [0, 1]. `radius_frac_min` is 0.10, not the brief's example 0.20, because a cup at 30 cm with a 1x lens is about r = 0.15 x min(H, W).
- **Dev images:** `make qa ARGS=--dev` reads `data/dev_external/` (recursively) and writes to `results/qa/dev/`, which is gitignored.
- **Tests:** synthetic cup photos come from `tests/synth.py` (`cup_photo`, `heart_polygon`); reuse them for the step-5 feature tests.
- **Features:** `latte.features.FEATURES` is the CSV column order. `MODEL_FEATURES` leaves out `symmetry_angle`, which is orientation only. `FEATURE_GROUPS` (for the E4 ablation), `RUBRIC_LINK` (for E1) and `DESIGN_LINK` (coverage → dropin) live there too. `D` is the disk minus any part outside the photo, and the disk centre and R come from the crop frame. `bubble_density` uses a region-wise high-pass (milk and crema blurred separately) and a noise-adaptive LoG threshold, max(`log_threshold`, `log_noise_k` × robust SD), because upsampled photo noise otherwise looks like bubbles. Features without milk are NaN, so models must impute inside the pipeline. Dev images get overlays only, never features.
- **Analysis table:** `latte.data.load_dataset(cfg, require_labels=False)` joins features, cup log and mean labels. It validates the cup log, and it excludes usable=n, `exclude_sessions` and a cup that *either* rater ticked unusable (from each rater's latest labels, even before the other rater gets to it). Step 8 should reuse it.
- **Go/no-go:** `make gonogo`. LOSO logistic regression (impute, scale and model fitted inside each fold), pooled out-of-fold balanced accuracy, within-session permutation p, chance = 1 / (classes present). foam_gain_pct is a manipulation check, kept out of the Holm family and the 'separating features' list. Rank-biserial r > 0 means on_time is higher.
- **Embeddings:** `make embed` (`python -m latte.embed`); `make weights` fetches the weights once. The cache is `features/emb_<model>{.npy,_ids.csv,.json}`. The .json is written last and holds a digest tying the three files together; a mismatched cache is rebuilt. The suffix `_random` marks `embed.pretrained: false`, which is for tests only. An unknown model name is an error. Only a *download* failure falls back to torchvision ResNet-50, which gets its own cache and a loud warning. Step 8 must read embeddings with `latte.embed.load_embeddings(cfg)`: it prefers the configured model, refuses stale caches (photos without an embedding), and sets `df.attrs['model']` / `['fell_back']` so results can name the model actually used. Tests use timm `resnet18` with random weights, so nothing downloads.
- **Train/eval (step 8):** `make train` (`scripts/train_eval.py`, `--exp E2,E3` for a subset) and `make leakcheck` (`--shuffle-labels`). Pipelines and grids are in `latte.models` (feature sets hand / emb / hand+emb; `ClippedPCA` inside the pipeline). Nested CV is `latte.evaluate.nested_cv`: LOSO outer, GridSearchCV(GroupKFold) inner, with pooled OOF metrics plus fold mean and std. CIs use `cluster_bootstrap_ci` over sessions. The human comparison in T3 is like for like: model vs one rater (`mae_vs_A/B`) against rater vs rater in the same columns. The human row's `mae` is NaN. Run records go to `results/runs/<stamp>_<exp>/` (config.yaml, run.json, metrics.json as strict JSON, oof_*.csv, confusion_*.csv). `results/runs/latest.json` is merged per experiment with relative dirs, so step 9 reads it through `train_eval.latest_run_dir(cfg, 'E2'/'E3')`. T4 has `bal_acc_ci_lo/hi` and `above_chance` (CI lower bound > chance); step 9 should use them to decide whether to withhold the milk diagnosis. A skipped experiment deletes its tables. Tests use `tests/fake.py` (fake studies, small grids).
- **Feedback (step 9):** `make final` (`scripts/fit_final.py`) checks that E2 and E3 ran on the current data: the `fingerprint` in run.json from `Study._fingerprint` must match, otherwise it says "rerun make train" and deletes the old bundle. It writes `models/latte_final.joblib` (gitignored) and `models/final.json`. The bundle holds: the score model (`feedback.score_model`, default the best by E2 MAE), with `score.error` = OOF MAE of the *rounded* score, shown rounded up to 0.1; `ridge_hand` for the fixes; and the milk model `feedback.milk_model` (fixed in advance, `logreg_hand`, to avoid winner-selection bias), used only if T4's `above_chance` is True and `tune_thresholds` gives at least one class a threshold ≥ 0.6 with OOF precision ≥ 0.7 (per class; classes without one are never shown). `latte.feedback.make_feedback(features, emb, bundle)` never advises on unmeasured (NaN) features, gives an "unmeasurable" result when contrast_dL is NaN, uses `implies_milk`/`*_alt` tips so fixes never contradict a shown diagnosis, de-duplicates identical tips, and says NO_TIP rather than "No major issues" when a big negative contribution has no reviewed tip. `src/latte/tips.yaml` is a DRAFT that needs team review. Step 10's demo should call `make_feedback`.
- **Course-folder Makefile:** `…/ENGR418/Makefile` (outside the repo) forwards every target to `latte-coach/`, so `make <target>` works from either folder.
- **Demo (step 10):** `make demo` (`app/demo.py`, port 7863) calls `latte.inference.analyse`. That mirrors training: `as_ingested` = ingest's EXIF transpose, 1024 px LANCZOS and a JPEG q95 round trip; then auto circle → crop → segment → features; the embedding (same model as the bundle, no silent fallback) only when the bundle needs it; then `make_feedback`. Without a bundle it shows overlays and features only. Example images must be team-owned (default: the first photos in data/raw).
- **Figures (step 11):** `make figures` (`scripts/make_figures.py`, style in `latte.plots`) writes F1–F8 (300 dpi, 3.5 in) plus T1. A missing input means a skip with the command to run, and the stale PNG is deleted. F6 uses `feedback.milk_model`; F5 uses the best E2 model with the band ± rater-A-vs-B MAE. `report/main.tex` holds headings and `\gentable`/`\genfig` hooks only, with empty captions; `tests/test_figures.py` enforces that it contains no prose. No LaTeX is installed on Rater A's laptop (Overleaf works).
- **README (step 12):** `make readme` (`scripts/update_readme.py`, also in `make all`) rewrites only the block between `<!-- results:start -->` and `<!-- results:end -->`: numbers copied from T1, agreement/E0, T3, T4, the leakage check and models/final.json. Key findings and Limitations hold HTML comments only, for the students; `tests/test_readme.py` enforces this. The License is left for the team to choose.
- **Reproducibility (step 13):** `make repro` (`scripts/repro_check.py`; options go in `REPRO=`, not `ARGS=`) clones HEAD, copies only uncommitted photos (it fails if the cup log, circles or labels aren't committed; the embedding cache is copied only with `--with-cache`), runs `make setup` and `make all` with make variables stripped, kills the whole process group on timeout, checks the 13 tables, 11 figures and records, and compares tables with the reference. `make all` = test env qa features embed agree gonogo train leakcheck final figures readme. agree and final skip (exit 0) when data are short. A leakage CHECK (exit 2) stops `make all` on purpose. Verified 2026-11-28 on a 72-cup fake study with synthetic photos and real DINOv2: clean clone exit 0, everything regenerated, and a rerun gave identical tables. Full runs take about an hour under load.
- **Build order status:** all 13 steps done (step 13 on 2026-11-28). Everything is verified only on synthetic or fake data. Still needed from the team: real data, rubric approval, tips.yaml review, License, Key findings and Limitations, report text, video, AI declarations.
