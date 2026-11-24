"""End-to-end ingest on 3 dummy phone photos carrying EXIF incl. GPS (build step 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import run_script
from PIL import Image

from latte.io import cfg_path, has_metadata, read_cup_log, strip_and_resize, write_cup_log

GPS_IFD = 0x8825
ORIENTATION = 0x0112


def phone_photo(path: Path, size=(4032, 3024), orientation: int = 1) -> None:
    """A fake phone JPEG: left half white, right half brown, with Make, GPS and orientation tags."""
    im = Image.new("RGB", size, (110, 70, 40))
    im.paste((250, 250, 245), (0, 0, size[0] // 2, size[1]))
    exif = Image.Exif()
    exif[0x010F] = "TestPhone"  # Make
    exif[ORIENTATION] = orientation
    exif[GPS_IFD] = {1: "N", 2: (49.0, 56.0, 23.0), 3: "W", 4: (119.0, 23.0, 46.0)}
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", exif=exif, quality=90)


def test_fixture_really_has_gps(tmp_path):
    p = tmp_path / "x.jpg"
    phone_photo(p, size=(64, 48))
    with Image.open(p) as im:
        assert im.getexif().get_ifd(GPS_IFD), "fixture should carry GPS so the strip test means something"
    assert has_metadata(p)


def test_strip_and_resize_removes_metadata_and_applies_rotation(tmp_path):
    src, dst = tmp_path / "in.jpg", tmp_path / "out.jpg"
    phone_photo(src, size=(400, 300), orientation=6)  # 6 = rotate 90 CW to display
    assert strip_and_resize(src, dst, max_side=1024) == (768, 1024)
    assert not has_metadata(dst)
    with Image.open(dst) as im:
        assert im.size == (768, 1024)
        assert not im.getexif()
        top, bottom = im.getpixel((384, 10)), im.getpixel((384, 1013))
    # The white half was on the left before rotating 90 degrees clockwise, so it is now on top.
    assert sum(top) > 600 and sum(bottom) < 300


def test_ingest_end_to_end_on_three_dummy_photos(project, cfg):
    r = run_script("scripts/make_schedule.py", project, "--session", "S01", "--pourer", "A", "--blocks", "1")
    assert r.returncode == 0, r.stderr
    log = read_cup_log(cfg_path(cfg, "cup_log"))
    first3 = list(log["cup_id"][:3])
    raw_full, raw = cfg_path(cfg, "raw_full"), cfg_path(cfg, "raw")
    phone_photo(raw_full / f"{first3[0]}.jpg")
    phone_photo(raw_full / f"{first3[1]}.JPG", orientation=6)
    phone_photo(raw_full / f"{first3[2]}.jpeg", size=(3024, 4032))

    # Three planned rows have no photo yet: strict ingest reports them and exits 1...
    r = run_script("scripts/ingest.py", project)
    assert r.returncode == 1
    assert "3 cup-log row(s) with no photo" in r.stdout
    assert "3 new" in r.stdout  # ...but still ingests what's valid

    for cup in first3:
        out = raw / f"{cup}.jpg"
        assert out.exists() and not has_metadata(out)
        with Image.open(out) as im:
            assert max(im.size) == 1024
    with Image.open(raw / f"{first3[1]}.jpg") as im:
        assert im.size == (768, 1024)  # orientation applied

    # Mid-session flag tolerates planned rows; a re-run is a no-op.
    r = run_script("scripts/ingest.py", project, "--allow-missing-photos")
    assert r.returncode == 0, r.stdout
    assert "0 new, 0 updated, 3 unchanged" in r.stdout
    assert "TOTAL" in r.stdout
    total = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("TOTAL")][0].split()
    assert total[1] == "3"  # photos column

    # Drop the 3 unpoured rows: now everything matches.
    write_cup_log(log[log["cup_id"].isin(first3)], cfg_path(cfg, "cup_log"))
    r = run_script("scripts/ingest.py", project)
    assert r.returncode == 0, r.stdout
    assert "ERROR" not in r.stdout and "WARNING" not in r.stdout


def test_ingest_flags_unlogged_photos_heic_and_bad_log(project, cfg):
    run_script("scripts/make_schedule.py", project, "--session", "S01", "--pourer", "A", "--blocks", "1")
    raw_full = cfg_path(cfg, "raw_full")
    phone_photo(raw_full / "S09-C01.jpg", size=(64, 48))
    (raw_full / "S01-C01.HEIC").write_bytes(b"not really heic")
    r = run_script("scripts/ingest.py", project, "--allow-missing-photos")
    assert r.returncode == 1
    assert "no cup-log row: S09-C01" in r.stdout
    assert "HEIC" in r.stdout
    assert not (cfg_path(cfg, "raw") / "S09-C01.jpg").exists()

    log = read_cup_log(cfg_path(cfg, "cup_log"))
    log.loc[0, "milk_condition"] = "medium"
    log.loc[1, "foam_gain_pct"] = "lots"
    write_cup_log(log, cfg_path(cfg, "cup_log"))
    r = run_script("scripts/ingest.py", project, "--allow-missing-photos")
    assert r.returncode == 1
    assert "'medium' not in" in r.stdout and "'lots' is not a number" in r.stdout


@pytest.mark.parametrize("size", [(500, 400), (4000, 3000)])
def test_resize_always_hits_max_side(tmp_path, size):
    src, dst = tmp_path / "in.jpg", tmp_path / "out.jpg"
    Image.new("RGB", size, (90, 60, 30)).save(src)
    w, h = strip_and_resize(src, dst, max_side=1024)
    assert max(w, h) == 1024 and abs(w / h - size[0] / size[1]) < 0.01
