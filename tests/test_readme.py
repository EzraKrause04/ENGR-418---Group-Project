"""README: generated Results block and the students' sections (build step 12)."""

from __future__ import annotations

import re

import pytest
from conftest import run_script
from fake import fake_study, small_grids

from latte.io import REPO_ROOT, load_config

README = REPO_ROOT / "README.md"


def section(text: str, title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    assert m, title
    return m.group(1)


def test_repo_readme_leaves_student_sections_empty():
    text = README.read_text()
    for title in ("Key findings", "Limitations"):
        body = re.sub(r"<!--.*?-->", "", section(text, title), flags=re.S)
        assert body.strip() == "", f"README '{title}' must be written by the students (brief section 15)"
    assert "<!-- results:start -->" in text and "<!-- results:end -->" in text
    for title in ("Objective", "Setup", "Usage", "Data protocol and privacy", "Results", "Attribution", "License"):
        section(text, title)


def test_update_readme_fills_only_the_results_block(project, tmp_path):
    small_grids(project)
    cfg = load_config(project / "configs" / "default.yaml")
    readme = tmp_path / "README.md"
    readme.write_text("# X\n\n## Results\n\n<!-- results:start -->\nold\n<!-- results:end -->\n\n## Key findings\n\nmine\n")
    r = run_script("scripts/update_readme.py", project, "--readme", str(readme))
    assert r.returncode == 0, r.stderr
    assert "No results yet" in readme.read_text() and "mine" in readme.read_text()

    fake_study(cfg, n_sessions=4, per_session=12)
    assert run_script("scripts/train_eval.py", project, "--exp", "E2,E3").returncode == 0
    r = run_script("scripts/update_readme.py", project, "--readme", str(readme))
    text = readme.read_text()
    assert "Task A, mean Overall score (T3" in text and "| ridge_hand |" in text and "human: rater A vs rater B" in text
    assert "Task B, diagnosis (T4" in text and "above_chance" in text
    assert text.endswith("## Key findings\n\nmine\n")  # the students' text is untouched
    again = run_script("scripts/update_readme.py", project, "--readme", str(readme))
    assert "No change" in again.stdout and readme.read_text() == text  # idempotent


def test_update_readme_needs_the_markers(project, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# no markers\n")
    r = run_script("scripts/update_readme.py", project, "--readme", str(readme))
    assert r.returncode == 1 and "results:start" in r.stderr


def test_stale_results_are_marked_not_shown(project, tmp_path):
    small_grids(project)
    cfg = load_config(project / "configs" / "default.yaml")
    fake_study(cfg, n_sessions=4, per_session=12)
    assert run_script("scripts/train_eval.py", project, "--exp", "E2,E3").returncode == 0
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n")
    from latte.io import cfg_path, read_cup_log, write_cup_log

    log = read_cup_log(cfg_path(cfg, "cup_log"))
    log.loc[log["cup_id"] == "S01-C01", "usable"] = "n"  # the data change after make train
    write_cup_log(log, cfg_path(cfg, "cup_log"))
    assert run_script("scripts/update_readme.py", project, "--readme", str(readme)).returncode == 0
    text = readme.read_text()
    assert "| ridge_hand |" not in text and text.count("out of date") >= 2 and "47 usable cups" in text
