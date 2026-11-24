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
