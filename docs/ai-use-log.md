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
