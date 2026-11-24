# Pouring and photo protocol

Follow this at the machine. Any deviation goes in the `notes` column of
`data/cup_log.csv` for that cup.

## Fixed setup

| Item | Fixed value |
|---|---|
| Pattern | Heart only |
| Milk | Whole milk, one brand |
| Cup | One cup model |
| Machine | One machine |
| Pitcher | Straight-walled, so the foam-gain marks are meaningful |

## Design: 3 × 2 cells, 20 cups each (120 minimum, stretch goal 150)

| Factor | Levels |
|---|---|
| Milk condition | `thin`, `good`, `thick` |
| Drop-in timing | `on_time`: lower the pitcher to the surface when the cup is about **½ full**<br>`late`: when the cup is about **¾ full** |

Starting foam-gain targets (recalibrate after the pilot and update `configs/default.yaml` → `design.milk_targets`):

| Condition | Aeration | Foam gain |
|---|---|---|
| `thin` | about 1 s | under 10% |
| `good` | normal | 20–30% |
| `thick` | long | 45% or more |

**Foam gain:** mark the milk level in the pitcher before and after steaming.
`foam_gain_pct = (after − before) / before × 100`. Record the *measured* value even when it misses the target.

## Phone and photo rig

- Phone mounted overhead at about **30 cm** on a fixed mount. Same table, cup, light source and time of day where possible.
- iPhone: **Settings → Camera → Formats → Most Compatible** (JPEG, not HEIC). Ingest rejects HEIC.
- **No flash.** No zoom, crop or filters. The whole cup rim must be in frame.
- **Lock exposure and focus**: long-press on the cup until "AE/AF LOCK" shows, before each shot.
- Photograph **within 30 s** of finishing the pour. Take **2 shots** and keep the sharper one.
- Name the kept photo **`<cup_id>.jpg`** (e.g. `S05-C03.jpg`).

## Session checklist

**Before pouring**

1. Pick the next session ID. The pilot is `S00`; real sessions are `S01`, `S02`, … A session is one sitting at the machine.
2. Plan the cup order (blocks of 6, each block covers every cell once):
   ```bash
   make schedule SESSION=S05 POURER=A BLOCKS=2
   ```
   This appends planned rows to `data/cup_log.csv` and prints the order. If the second pourer pours in the same
   session, add their cups with `make schedule SESSION=S05 POURER=B BLOCKS=2 ARGS=--append`.
   `ARGS=--dry-run` prints an order without touching the log.
3. Check the mount, the light and the AE/AF lock.

**For each cup, in the printed order**

1. Mark the milk level, then steam to the cup's milk condition and mark the level again.
2. Pour, dropping in at the cup's timing (½ or ¾ full).
3. Photograph within 30 s: 2 shots, keep the sharper.
4. Fill in the row: `foam_gain_pct`, `seconds_to_photo`, `milk_temp_c` (optional), `usable`, `notes`.
   - `usable = n` for a blurry photo, glare, or a failed pour that isn't a heart. **Photograph and log failed pours anyway.**
   - Keep the planned `milk_condition` and `dropin` even if the pour went wrong. Note what happened instead.

**After the session**

1. Copy the photos to `data/raw_full/` (and to the team shared drive, which holds the originals).
2. Ingest. This strips EXIF/GPS, resizes, checks photos against the log and prints counts per cell:
   ```bash
   make ingest
   ```
   Fix every `ERROR` it prints and re-run. Mid-session, `--allow-missing-photos` tolerates planned rows not poured yet.
3. Commit `data/raw/` and `data/cup_log.csv`. Never commit `data/raw_full/`.
4. Label (each rater, ideally the next day): `make label`.

## Editing `data/cup_log.csv`

Edit in a text editor or a spreadsheet, but keep the header row and save as plain CSV (UTF-8).
Don't let a spreadsheet reformat `cup_id`s or dates. `make ingest` validates the whole log first and names any row it can't read.

| Column | Values |
|---|---|
| `cup_id` | `S05-C03` (filled by `make schedule`) |
| `session_id` | `S05`; the cross-validation group |
| `date` | `YYYY-MM-DD` |
| `pourer` | `A` / `B` (guest pourers get their own ID, e.g. `G1`) |
| `milk_condition` | `thin` / `good` / `thick` (planned) |
| `dropin` | `on_time` / `late` (planned) |
| `foam_gain_pct` | number (measured) |
| `milk_temp_c` | number, optional |
| `seconds_to_photo` | whole seconds |
| `photo_file` | `<cup_id>.jpg` |
| `usable` | `y` / `n` |
| `notes` | free text |

## Data rules

- Only photos the team takes. **No scraping** from Reddit, Instagram, YouTube or similar.
- A cup poured by anyone outside the team needs their **written OK before** the photo enters the repo. Log them under a distinct pourer ID (`--guest`).
- External images (e.g. the Roboflow "latte-art-hearts" set) go only in `data/dev_external/` (gitignored). They're for developing rim detection and segmentation, and are never used for labelling, training or evaluation.
