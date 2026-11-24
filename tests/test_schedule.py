from __future__ import annotations

from conftest import run_script

from latte.io import cfg_path, read_cup_log, validate_cup_log
from latte.schedule import design_cells, plan_session


def test_every_block_contains_every_cell_once(cfg):
    plan = plan_session(read_cup_log(cfg_path(cfg, "cup_log")), cfg, "S05", "A", 4, "2026-11-25")
    cells = set(design_cells(cfg))
    assert len(cells) == 6 and len(plan) == 24
    for b in range(4):
        block = plan.iloc[6 * b : 6 * (b + 1)]
        assert set(zip(block["milk_condition"], block["dropin"])) == cells


def test_plan_is_seeded_and_ids_are_sequential(cfg):
    log = read_cup_log(cfg_path(cfg, "cup_log"))
    p1 = plan_session(log, cfg, "S05", "A", 2, "2026-11-25")
    p2 = plan_session(log, cfg, "S05", "A", 2, "2026-11-25")
    assert p1.equals(p2)
    assert list(p1["cup_id"]) == [f"S05-C{i:02d}" for i in range(1, 13)]
    assert list(p1["photo_file"]) == [f"S05-C{i:02d}.jpg" for i in range(1, 13)]
    assert (p1[["foam_gain_pct", "seconds_to_photo", "usable"]] == "").all().all()
    assert validate_cup_log(p1, cfg) == []
    # a different session gets a different order
    p3 = plan_session(log, cfg, "S06", "A", 2, "2026-11-25")
    assert list(p3["milk_condition"]) != list(p1["milk_condition"]) or list(p3["dropin"]) != list(p1["dropin"])


def test_cli_appends_refuses_duplicates_and_continues_numbering(project, cfg):
    r = run_script("scripts/make_schedule.py", project, "--session", "S05", "--pourer", "A", "--blocks", "2")
    assert r.returncode == 0, r.stderr
    assert "S05-C12" in r.stdout and "block 2" in r.stdout
    log_path = cfg_path(cfg, "cup_log")
    assert len(read_cup_log(log_path)) == 12

    r = run_script("scripts/make_schedule.py", project, "--session", "S05", "--pourer", "B", "--blocks", "1")
    assert r.returncode == 1 and "--append" in r.stderr
    assert len(read_cup_log(log_path)) == 12

    r = run_script("scripts/make_schedule.py", project, "--session", "S05", "--pourer", "B", "--blocks", "1", "--append")
    assert r.returncode == 0, r.stderr
    log = read_cup_log(log_path)
    assert len(log) == 18
    assert list(log["cup_id"].iloc[12:]) == [f"S05-C{i}" for i in range(13, 19)]
    assert set(log["pourer"].iloc[12:]) == {"B"}
    assert validate_cup_log(log, cfg) == []


def test_cli_dry_run_guest_and_validation(project, cfg):
    r = run_script("scripts/make_schedule.py", project, "--session", "S01", "--pourer", "A", "--dry-run")
    assert r.returncode == 0 and "dry run" in r.stdout
    assert not cfg_path(cfg, "cup_log").exists()

    r = run_script("scripts/make_schedule.py", project, "--session", "S01", "--pourer", "G1")
    assert r.returncode != 0 and "--guest" in r.stderr
    r = run_script("scripts/make_schedule.py", project, "--session", "S01", "--pourer", "G1", "--guest", "--blocks", "1")
    assert r.returncode == 0 and "written OK" in r.stdout

    r = run_script("scripts/make_schedule.py", project, "--session", "5", "--pourer", "A")
    assert r.returncode != 0
