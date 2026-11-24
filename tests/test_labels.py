from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta

import pandas as pd
from conftest import tiny_jpeg

from latte.io import REPO_ROOT, cfg_path, write_cup_log
from latte.labels import (
    append_label,
    available_cups,
    blind_order,
    label_queue,
    labels_path,
    latest_labels,
    mean_labels,
    parse_rubric_anchors,
    read_labels,
)
from latte.schedule import plan_session

DIMS = ["contrast", "definition", "symmetry", "centring", "surface", "overall"]
CUPS = [f"S01-C{i:02d}" for i in range(1, 31)]


def row(cup, score=3, unusable="n", comment="", ts="2026-11-25T10:00:00-08:00"):
    return {"cup_id": cup, **dict.fromkeys(DIMS, score), "unusable": unusable, "comment": comment, "timestamp": ts}


def test_blind_order_is_rater_specific_random_and_stable():
    a1, a2, b = blind_order(CUPS, "A", 42), blind_order(CUPS, "A", 42), blind_order(CUPS, "B", 42)
    assert a1 == a2 and a1 != b
    assert a1 != sorted(CUPS)  # not chronological, so session/pourer can't be inferred from position
    # New images interleave without disturbing the relative order of existing ones.
    more = blind_order(CUPS + [f"S02-C{i:02d}" for i in range(1, 13)], "A", 42)
    assert [c for c in more if c in set(CUPS)] == a1
    assert more[-12:] != [f"S02-C{i:02d}" for i in range(1, 13)]


def test_append_latest_wins_and_comment_round_trip(tmp_path):
    p = tmp_path / "rater_A.csv"
    append_label(p, row("S01-C01", 2), DIMS)
    append_label(p, row("S01-C02", 4, comment='bubbly, "dull" edge'), DIMS)
    append_label(p, row("S01-C01", 5), DIMS)  # relabel
    df = read_labels(p, DIMS)
    assert len(df) == 3
    lat = latest_labels(df)
    assert lat.at["S01-C01", "overall"] == 5
    assert lat.at["S01-C02", "comment"] == 'bubbly, "dull" edge'


def test_queue_skips_labelled_and_relabel_before_requeues():
    old, new = "2026-11-25T10:00:00-08:00", "2026-11-28T10:00:00-08:00"
    lab = latest_labels(pd.DataFrame([row("S01-C01", ts=old), row("S01-C02", ts=new)]))
    q = label_queue(CUPS, lab, "A", 42)
    assert "S01-C01" not in q and "S01-C02" not in q and len(q) == 28
    cutoff = datetime.fromisoformat("2026-11-27T00:00:00-08:00")
    q2 = label_queue(CUPS, lab, "A", 42, relabel_before=cutoff)
    assert "S01-C01" in q2 and "S01-C02" not in q2
    assert label_queue(CUPS, lab, "A", 42, relabel_before=cutoff - timedelta(days=30)) == q


def test_available_cups_skips_log_unusable(cfg):
    log = plan_session(pd.DataFrame(columns=["cup_id"]), cfg, "S01", "A", 1, "2026-11-25")
    log.loc[2, "usable"] = "n"
    write_cup_log(log, cfg_path(cfg, "cup_log"))
    for c in log["cup_id"]:
        tiny_jpeg(cfg_path(cfg, "raw") / f"{c}.jpg")
    cups = available_cups(cfg)
    assert len(cups) == 5 and log.loc[2, "cup_id"] not in cups


def test_mean_labels_needs_both_raters(cfg):
    append_label(labels_path(cfg, "A"), row("S01-C01", 2), DIMS)
    append_label(labels_path(cfg, "B"), row("S01-C01", 4, unusable="y"), DIMS)
    append_label(labels_path(cfg, "A"), row("S01-C02", 3), DIMS)
    m = mean_labels(cfg)
    assert list(m.index) == ["S01-C01"]
    assert m.at["S01-C01", "overall"] == 3.0 and bool(m.at["S01-C01", "unusable_any"])


def test_rubric_anchors_parse_from_docs():
    anchors = parse_rubric_anchors(REPO_ROOT / "docs" / "rubric.md", DIMS)
    assert set(anchors) == set(DIMS)
    assert set(anchors["contrast"]) == {"1", "3", "5"}
    assert anchors["centring"]["5"] == "Centred"
    assert list(anchors["overall"]) == ["1"] and "Holistic" in anchors["overall"]["1"]


def test_label_app_builds(cfg):
    spec = importlib.util.spec_from_file_location("label_app", REPO_ROOT / "app" / "label.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ui = mod.build_ui(cfg, "A", None)
    assert ui is not None and '"overall"' in ui.lc_head
