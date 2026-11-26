# Generative AI use log

Evidence for each student's individual Generative AI declaration. Add an entry
at the end of every session, or after each completed build-order step.
Claude Code drafts the entry; the student at the keyboard checks it and fills in
anything it can't know (especially "Changed by hand").

## Entry template

```markdown
### YYYY-MM-DD: <build step or task>
- **At the keyboard:** Rater A / Rater B
- **Tool:** Claude Code (<model>)
- **Task:**
- **Key prompts (summary):**
- **Files created or changed:**
- **Verification:** tests run, manual review
- **Changed by hand afterwards:**
```

---

## Entries

### 2026-11-23: Build step 1, scaffold
- **At the keyboard:** Rater A
- **Tool:** Claude Code (Claude Opus 5.5)
- **Task:** create the repository layout from the brief (section 5), pinned `requirements.txt`, `pyproject.toml`, `Makefile` (all section-5 targets plus `schedule`, `label`, `circles`, `env`), `configs/default.yaml`, `.gitignore`, and placeholder modules and scripts for later build steps.
- **Key prompts (summary):** the student attached the project brief (now `CLAUDE.md`) and asked Claude Code to start the build order. Claude Code asked who was at the keyboard and for the GitHub remote. The student supplied the team repo URL, and Claude Code added it as an SSH remote, as the brief requires.
- **Files created or changed:** `CLAUDE.md`, `README.md` (stub), `Makefile`, `requirements.txt`, `pyproject.toml`, `configs/default.yaml`, `.gitignore`, `src/latte/*.py` (placeholders for steps 4–9), `scripts/*.py` and `app/*.py` placeholders, `.gitkeep` files.
- **Verification:** `rm -rf .venv && make setup` completed and imported every package. A fresh `git clone` of the commit, followed by `make setup` and `make test`, also passed.
- **Notes:** the repo lives at `~/dev/latte-coach`, with a symlink from the course folder, because the course folder's path contains `:`, which breaks `source .venv/bin/activate`. The versions are pinned to what resolved on 2026-11-23 for Python 3.11.
- **Changed by hand afterwards:** _(student to fill in)_

### 2026-11-23: Build step 2, collection tooling
- **At the keyboard:** as above
- **Tool:** Claude Code (Claude Opus 5.5)
- **Task:** `scripts/make_schedule.py` (blocked randomization, seeded per session, pourer and first cup), `scripts/ingest.py` (EXIF/GPS strip, orientation fix, 1024 px resize, photo ↔ log validation, per-cell counts), and `docs/protocol.md`.
- **Key prompts (summary):** implement section 6 of the brief.
- **Files created or changed:** `src/latte/io.py`, `src/latte/schedule.py`, `scripts/make_schedule.py`, `scripts/ingest.py`, `docs/protocol.md`, `tests/test_schedule.py`, `tests/test_ingest.py`, `tests/conftest.py`.
- **Verification:**
  - `make test` passes.
  - Ran end-to-end on 3 synthetic phone photos carrying GPS EXIF in a scratch project. The GPS was present in the source and absent from the output, and the output was 1024 px on the longest side.
  - A strict ingest flagged the 3 planned-but-unpoured rows; `--allow-missing-photos` accepted them.
- **Changed by hand afterwards:** _(student to fill in)_

### 2026-11-23: Build step 3, labelling tool and agreement
- **At the keyboard:** as above
- **Tool:** Claude Code (Claude Opus 5.5)
- **Task:** `app/label.py` (Gradio, blind, resumable, autosaving, keyboard-driven) and `scripts/agreement.py` (quadratic-weighted κ with bootstrap CI, Spearman ρ, mean absolute difference, calibration subset, disagreement list). Also drafted `docs/rubric.md` from the brief's anchors (pending team approval).
- **Key prompts (summary):** implement section 7 of the brief.
- **Files created or changed:** `src/latte/labels.py`, `src/latte/evaluate.py`, `app/label.py`, `scripts/agreement.py`, `docs/rubric.md`, `tests/test_labels.py`, `tests/test_agreement.py`.
- **Verification:**
  - `make test` passes (21 tests). `agreement.py` runs on fake labels.
  - Claude Code drove the labelling tool in a browser:
    - no cup ID appears in the page, and the image URL is a content hash;
    - autosave appends one row per save, and the newest row wins;
    - Previous plus a re-save corrects an earlier score, and a reload resumes at the right image;
    - a fast double Enter saves only once.
  - Browser testing found two bugs, both now fixed: each number key advanced two rows, and Gradio served the image as lossy WebP (it is now PNG).
- **Changed by hand afterwards:** _(student to fill in)_

### 2026-11-24: Push to GitHub and SSH key set-up
- **At the keyboard:** Rater A
- **Tool:** Claude Code (Claude Opus 5.5)
- **Task:** get `main` onto the team repo over SSH.
- **Key prompts (summary):** the student pasted the repo URL and asked to push. Claude Code diagnosed why the push failed (GitHub rejected the key; the old 2048-bit RSA key was too short to add). It suggested a new Ed25519 key, which the student generated and added to GitHub themselves.
- **Files created or changed:** none in the repo. Four commits were pushed (`75fda71..3dac715`).
- **Verification:** `ssh -T git@github.com` authenticated; `git ls-remote` showed `main` at `3dac715`; `make test` passed before the push.
- **Changed by hand afterwards:** _(student to fill in)_

### 2026-11-24: Build step 4, geometry and preprocessing
- **At the keyboard:** Rater A
- **Tool:** Claude Code (Claude Opus 5.5)
- **Task:**
  - liquid-surface circle detection (`latte.geometry`: Hough centre candidates, the innermost dark-to-light radial edge ring, a trimmed least-squares refit, and confidence = edge support);
  - the manual override and check tool (`app/annotate_circle.py`, `make circles`);
  - crop/mask/resize to 512 × 512 with a 4% radius shrink (`latte.preprocess`);
  - QA contact sheets (`scripts/qa_sheets.py`, `make qa`, `make qa ARGS=--dev`).
- **Key prompts (summary):** "Start step 4", i.e. implement sections 8.1 and 8.2 of the brief.
- **Files created or changed:**
  - `src/latte/geometry.py`, `src/latte/preprocess.py`, `src/latte/overlay.py`, `src/latte/io.py` (`load_rgb`);
  - `scripts/qa_sheets.py`, `app/annotate_circle.py`;
  - `configs/default.yaml` (geometry settings), `Makefile`, `.gitignore` (`results/qa/dev/`);
  - `docs/protocol.md` (circle check after ingest), `README.md`, `CLAUDE.md` (working notes);
  - `tests/synth.py`, `tests/test_geometry.py`, `tests/test_preprocess.py`, `tests/test_qa_and_annotate.py`.
- **Verification:**
  - `make test` passes (56 tests).
  - On synthetic photos, the detector finds the liquid edge, not the lip or the outer rim, to within 1% of r. This holds for small, large, off-centre, noisy and portrait cups, and for cups cut off by the frame edge. A photo with no cup gets confidence 0.
  - Claude Code drove the circle tool in a browser on 8 synthetic photos:
    - three clicks on the edge gave r = 122.9 px (true value 122.9) in the zoomed view, and 99.9 px (true 99.8) in the whole-photo view;
    - the A, S, C, F and arrow keys work; the list puts low-confidence photos first;
    - checked photos leave the "Not checked yet" list;
    - saving without choosing a checker is refused.
  - QA sheets were inspected visually.
  - **Not yet verified on real photos:** there are no pilot or dev images yet. Step 4 is only *done* once the students have checked the QA sheets on the pilot and dev images.
- **Deviation from the brief:** `geometry.radius_frac_min` is 0.10 rather than the brief's example of 0.20. A ~9 cm cup shot from 30 cm with a 1× lens has r ≈ 0.15 × min(H, W).
- **Changed by hand afterwards:** _(student to fill in)_

### 2026-11-24: Sandbox for trying step 4
- **At the keyboard:** Rater A
- **Tool:** Claude Code (Claude Opus 5.5)
- **Task:** "How can we test it now?" Claude Code set up a gitignored sandbox project at `.cache/sandbox/` holding synthetic cup photos. The student tried `make qa` and `make circles` on it (`make … CONFIG=.cache/sandbox/configs/default.yaml`).
- **Files created or changed:** none in the repo. The sandbox lives under the gitignored `.cache/`.
- **Verification:** the student ran `make qa` on the sandbox and accepted a circle in the tool.
- **Changed by hand afterwards:** _(student to fill in)_

### 2026-11-25: Build step 5, segmentation and features
- **At the keyboard:** Rater A
- **Tool:** Claude Code (Claude Opus 5.5)
- **Task:**
  - milk/crema segmentation (`latte.segment`): Otsu on L* and k-means (k = 2) on L\*a\*b\*, each followed by open/close and small-component removal;
  - the ten handcrafted features of section 8.4 (`latte.features`), each a pure function with its formula in the docstring;
  - `scripts/build_features.py` (`make features`), which writes `features/features.csv` plus QA overlays (milk outline, centroid, symmetry axis) and an Otsu vs k-means comparison sheet.
- **Key prompts (summary):** "Start step 5", i.e. implement sections 8.3–8.5 of the brief.
- **Files created or changed:**
  - `src/latte/segment.py`, `src/latte/features.py`, `src/latte/overlay.py` (outline, axis and point drawing), `src/latte/preprocess.py` (the mask now leaves out parts of the disk outside the photo);
  - `scripts/build_features.py`, `configs/default.yaml` (segment and feature settings), `Makefile`, `README.md`, `CLAUDE.md`;
  - `tests/synth.py` (`heart_crop`; hearts centred by area centroid), `tests/conftest.py`, `tests/test_features.py`, `tests/test_build_features.py`.
- **Verification:**
  - `make test` passes (94 tests). That includes the five checks section 8.5 requires: a centred heart has symmetry_iou > 0.9 and centre_offset < 0.05; shifting the heart raises centre_offset; blur lowers edge_sharpness; darker milk lowers contrast_dL; dark dots raise bubble_density.
  - Both segmentation methods recover the synthetic heart with IoU > 0.95.
  - `make features` ran on the 13-photo sandbox, and the overlays and comparison sheets were inspected.
  - Three problems turned up during checking and were fixed:
    1. bubble_density counted the milk/crema edge as bubbles, so the high-pass is now computed within each region;
    2. photo noise that the 512 px resampling enlarges was counted as bubbles, so the LoG threshold now adapts to each photo's noise;
    3. for cups cut off by the frame, the black padding counted as crema, so the mask now leaves out pixels outside the photo.
  - **Not yet verified on real photos.** Step 5 is only *done* once the students have looked at the overlays on the pilot photos and chosen `segment.method` (and checked `log_threshold`).
- **Decisions to review:**
  - boundary_roughness uses the Crofton perimeter, which gives 1.00 for a disk; skimage's default perimeter gives 1.11.
  - `symmetry_angle` is stored but kept out of the model features, because it is only the heart's orientation.
  - texture_energy is computed on the milk eroded by 2 px, so the milk/crema edge itself is left out.
- **Changed by hand afterwards:** _(student to fill in)_

### 2026-11-25: Build step 6, go/no-go script
- **At the keyboard:** Rater A
- **Tool:** Claude Code (Claude Opus 5.5), with a multi-agent review workflow
- **Task:** `scripts/go_no_go.py` (`make gonogo`), implementing section 16. It produces:
  - box plots of every feature by milk condition and by drop-in;
  - leave-one-session-out balanced accuracy of logistic regression for B1 and B2, with a within-session permutation p-value;
  - per-feature Kruskal-Wallis and Mann-Whitney tests with Holm correction;
  - a histogram of the mean Overall score;
  - decision hints that follow section 16's rules.

  Also `src/latte/data.py`, a shared loader that joins features, cup log and labels and applies the exclusions in section 10.1.
- **Key prompts (summary):** "Start step 6".
- **Files created or changed:** `src/latte/data.py`, `scripts/go_no_go.py`, `tests/test_go_no_go.py`, `configs/default.yaml` (the `gonogo` section), `Makefile`, `README.md`, `CLAUDE.md`.
- **Verification:**
  - `make test` passes (105 tests). They run fake studies with and without planted signal, a narrow score range, too few sessions, an absent class, an all-missing feature, a broken cup log, and the exclusion rules.
  - The script ran on a fake 48-cup study: it found the planted signals (B1 0.73 against chance 0.33, p = 0.005), and the figures were inspected.
  - An independent review ran 3 reviewers with different lenses, and 2 skeptics tried to refute each finding. It confirmed 10 findings, all now fixed:
    - measured foam gain was counted as a "photo feature that separates conditions" and inflated the Holm family;
    - a cup one rater ticked unusable stayed in until the other rater labelled it;
    - chance was wrong when a class was absent;
    - the rank-biserial sign ran against the usual convention;
    - the decision hint ignored the "features separate" half of the rule;
    - a duplicated cup-log row crashed the script with an unreadable error;
    - a pandas fill warning appeared when no cup had both raters' labels.

    Unverified low-severity items were fixed as well: warning floods from all-missing features, an empty histogram, no notice about unchecked circles, and a crash in the path print.
  - **Not yet run on real data.** It needs at least 30 usable cups (Fri Nov 27).
- **Changed by hand afterwards:** _(student to fill in)_

### 2026-11-25: Build step 7, embeddings
- **At the keyboard:** Rater A
- **Tool:** Claude Code (Claude Opus 5.5), with a multi-agent review workflow
- **Task:** `src/latte/embed.py` (`make embed`, `make weights`), implementing section 9:
  - frozen timm DINOv2 ViT-S/14 (`num_classes=0`) on the 512 px masked crop, resized to the model's input size;
  - the embedding averaged over 0/90/180/270° rotations;
  - an incremental cache in `features/emb_<model>.*`;
  - a torchvision ResNet-50 fallback used only when the download fails;
  - `load_embeddings(cfg)` for step 8.
- **Key prompts (summary):** "Start step 7".
- **Files created or changed:** `src/latte/embed.py`, `tests/test_embed.py`, `configs/default.yaml` (embed section), `requirements.txt` (torchvision pinned), `Makefile` (`weights`, ARGS for `embed`), `README.md`, `CLAUDE.md`.
- **Verification:**
  - `make test` passes (119 tests). The embedding tests use a small, untrained network (random weights), so nothing is downloaded. They cover:
    - rotation invariance of the averaged embedding;
    - cache reuse, recompute and drop;
    - refusing a stale cache;
    - a corrupted cache;
    - the seed in the cache key;
    - a typo giving an error, not a fallback;
    - a simulated download failure falling back with a loud warning and without overwriting the main cache;
    - both models unavailable giving one clear message.
  - An independent review (2 lenses, 2 skeptics per finding) confirmed 10 problems, all fixed:
    - a fallback silently repointed step 8 to ResNet features;
    - any error, including a typo, triggered the fallback;
    - offline, the script hung for about a minute and then crashed with a raw traceback;
    - a corrupted cache crashed it, and cache files written non-atomically could get out of step;
    - random-weight embeddings were saved under the real model's name;
    - the seed was missing from the cache key;
    - the output dimension was taken from metadata instead of measured;
    - unchecked circles were not reported;
    - one test assertion could never fail.
  - **2026-11-26, after the student approved the download:**
    - `make weights` fetched `timm/vit_small_patch14_dinov2.lvd142m` (`model.safetensors`, 88 MB, from Hugging Face) into the local Hugging Face cache, outside the repo.
    - `make embed` on the 13 synthetic sandbox cups took about 11 s, and a rerun reused all 13 from the cache.
    - Sanity check (cosine similarity): the same heart photographed upright vs. sideways scored 0.999; a dull-milk variant 0.986; a bigger heart 0.886; the photo with no cup 0.677.
    - Still to come: a run on real photos.
- **Changed by hand afterwards:** _(student to fill in)_
