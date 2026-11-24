# Scoring rubric

> **Status: DRAFT. The team must edit and approve these anchors before labelling starts.**
> Record approval, and every later change, in the changelog at the bottom.

The labelling tool (`app/label.py`) reads the anchor table below and shows the
anchors next to each score, so editing this table changes what raters see.
Keep the table format (`| Dimension | 1 | 3 | 5 |`) intact.

## How to score

- Score each dimension from **1 to 5**. The table anchors 1, 3 and 5; use **2** and **4** for in-between cases.
- Score what is in the photo, not what you think the pour should have looked like.
- **Overall** is a holistic judgement of the heart as a whole. It is **not** the average of the other five.
- Tick **Unusable** instead of scoring when the photo can't be judged: badly blurred, strong glare over the pattern, the rim cut off, or a failed pour that isn't recognisably a heart.

## Anchors

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Contrast | Pattern barely visible or sunk | Visible but greyish or blotchy | Bright white on even crema |
| Definition | Edges smeared or feathered, no clear outline | Outline mostly clear, some blur | Clean, sharp edges throughout |
| Symmetry | Clearly lopsided | Minor asymmetry | Mirror-symmetric |
| Centring | Touching the rim or far off-centre | Noticeably off-centre | Centred |
| Surface | Visible bubbles, matte or dry foam | A few bubbles or slight dullness | Glossy, no visible bubbles |
| Overall | Holistic judgement, not the average of the other five | | |

## Labelling rules

- Label **blind**: don't look at `data/cup_log.csv`, file names or the other rater's label file while labelling.
- Don't discuss individual cups with the other rater until both of you have labelled them.
- Label your own pours **a day or more after** pouring them where possible, to reduce recognising them.
- Use the same screen at a similar brightness for every session.
- **Calibration round:** both raters label the same first 30 cups, then run `make agree` (or
  `python scripts/agreement.py --subset calibration`).
  - Target: weighted κ ≥ 0.6 on Overall.
  - If κ < 0.4: review `results/qa/disagreements_calibration.csv` together, revise the anchors here (log it below), then relabel with
    `python app/label.py --rater X --relabel-before <time of the revision>`.

## Changelog

| Date | Change | Reason | Approved by |
|---|---|---|---|
| 2026-11-23 | Draft anchors copied from the project brief (section 7.1) | Starting point for team review | (pending) |
