"""Figures F1-F8, table T1 and the report skeleton (build step 11)."""

from __future__ import annotations

import re

import pandas as pd
import pytest
from conftest import run_script
from fake import fake_study, small_grids
from PIL import Image
from synth import cup_photo

from latte.io import REPO_ROOT, cfg_path, load_config, read_cup_log

FIGS = ["F1_pipeline", "F2_examples", "F3_agreement", "F4_feature_validity", "F5_task_a", "F6_confusion",
        "F7_importance", "F8_features_by_milk"]


@pytest.fixture
def study_with_photos(project):
    small_grids(project)
    cfg = load_config(project / "configs" / "default.yaml")
    truth = fake_study(cfg, n_sessions=4, per_session=12)
    log = read_cup_log(cfg_path(cfg, "cup_log")).set_index("cup_id")
    raw = cfg_path(cfg, "raw")
    raw.mkdir(parents=True, exist_ok=True)
    for i, cup in enumerate(truth.index):
        blur = 2.5 if log.at[cup, "milk_condition"] == "thick" else 0.8
        Image.fromarray(cup_photo(w=640, h=480, blur=blur, seed=i)[0]).save(raw / f"{cup}.jpg", quality=90)
    return cfg


def test_all_figures_and_t1(project, study_with_photos):
    cfg = study_with_photos
    assert run_script("scripts/train_eval.py", project, "--exp", "E0,E1,E2,E3,E4").returncode == 0
    assert run_script("scripts/agreement.py", project).returncode == 0
    r = run_script("scripts/make_figures.py", project)
    assert r.returncode == 0, r.stderr
    figs = cfg_path(cfg, "figures_dir")
    for f in FIGS:
        with Image.open(figs / f"{f}.png") as im:
            dpi = im.info.get("dpi", (0, 0))[0]
            assert round(dpi) == 300, f
            assert abs(im.size[0] - 1050) <= 2, (f, im.size)  # exactly one 3.5 in column at 300 dpi
    t1 = pd.read_csv(cfg_path(cfg, "tables_dir") / "T1_dataset.csv")
    total = t1[t1["milk_condition"] == "total"].iloc[0]
    assert total["photographed"] == 48 and total["usable"] == 48 and total["labelled_by_both"] == 48
    assert total["sessions"] == 4 and len(t1) == 7
    for tex in cfg_path(cfg, "tables_dir").glob("*.tex"):  # generated tables carry no report text
        body = tex.read_text()
        assert "\\caption" not in body and "\\begin{table" not in body, tex.name
        prose = [line for line in body.splitlines() if line and not line.startswith(("%", "\\")) and "&" not in line]
        assert prose == [], (tex.name, prose)  # only comments, commands and table rows

    # The data change after make train: run-based figures refuse rather than mix old and new data.
    log = read_cup_log(cfg_path(cfg, "cup_log"))
    log.loc[log["cup_id"] == "S01-C01", "usable"] = "n"
    from latte.io import write_cup_log

    write_cup_log(log, cfg_path(cfg, "cup_log"))
    r = run_script("scripts/make_figures.py", project)
    assert r.returncode == 0 and r.stdout.count("older data") >= 3
    for f in ("F5_task_a", "F6_confusion", "F7_importance"):
        assert not (cfg_path(cfg, "figures_dir") / f"{f}.png").exists()


def test_missing_inputs_skip_and_remove_stale_figures(project, cfg):
    figs = cfg_path(cfg, "figures_dir")
    figs.mkdir(parents=True)
    (figs / "F5_task_a.png").write_bytes(b"old")
    r = run_script("scripts/make_figures.py", project)
    assert r.returncode == 0, r.stderr
    assert "0/8 figures" in r.stdout and "make train" in r.stdout and "make features" in r.stdout
    assert not (figs / "F5_task_a.png").exists()


def test_report_skeleton_has_headings_and_hooks_only():
    tex = (REPO_ROOT / "report" / "main.tex").read_text()
    sections = re.findall(r"\\section\{([^}]*)\}", tex)
    assert sections == ["Introduction", "Related Work", "Methods", "Results", "Conclusion"]
    body = tex[tex.index("\\begin{document}"):tex.index("\\end{document}")]
    body = "\n".join(line.split("%", 1)[0] for line in body.splitlines())  # drop comments
    # Remove every allowed construct; whatever is left would be prose written by Claude.
    allowed = [r"\\(gentable|genfig)\{[^}]*\}\{make [a-z]+\}", r"\\resizebox\{[^}]*\}\{!\}\{", r"\\(begin|end)\{[^}]*\}(\[[^]]*\])?",
               r"\\(section|label)\{[^}]*\}", r"\\caption\{\}", r"\\title\{TITLE\}",
               r"\\author\{\\IEEEauthorblockN\{AUTHOR A, AUTHOR B\}\\IEEEauthorblockA\{UBC Okanagan, ENGR 418\}\}",
               r"\\(centering|maketitle)", r"\}"]
    rest = body
    for pat in allowed:
        rest = re.sub(pat, "", rest)
    assert rest.split() == [], f"report/main.tex must hold no prose (brief section 2): {rest.split()}"
    preamble = "\n".join(line.split("%", 1)[0] for line in tex[:tex.index("\\begin{document}")].splitlines())
    assert re.findall(r"\\(AtBeginDocument|AtEndDocument)", preamble) == []
    assert set(re.findall(r"\\([A-Za-z]+)", preamble)) <= {
        "documentclass", "usepackage", "newcommand", "gentable", "genfig", "IfFileExists", "input", "includegraphics",
        "columnwidth", "fbox", "parbox", "linewidth", "ttfamily", "small", "detokenize"}
    for f in FIGS:
        assert f"{f}.png" in tex
    for t in ("T1_dataset", "agreement", "T3_task_a", "T4_task_b", "T5_ablation"):
        assert f"{t}.tex" in tex
    assert re.findall(r"\\caption\{([^}]*)\}", tex) == [""] * 13  # 8 figures + 5 tables: captions are the students'
    assert "\\detokenize" in tex  # missing-file boxes can't break the build on underscores
