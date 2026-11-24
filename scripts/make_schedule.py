"""Plan a randomized pouring session and append it to data/cup_log.csv.

    python scripts/make_schedule.py --session S05 --pourer A --blocks 2

Cups are ordered in blocks of 6; each block holds every
(milk_condition x dropin) cell exactly once. Measurement fields are left blank
for the team to fill in at the machine.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date as _date

import pandas as pd

from latte.io import DEFAULT_CONFIG, cfg_path, load_config, read_cup_log, write_cup_log
from latte.schedule import plan_session


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--session", required=True, help="session ID, e.g. S05 (S00 = pilot)")
    ap.add_argument("--pourer", required=True, help="A or B (other IDs need --guest)")
    ap.add_argument("--blocks", type=int, default=2, help="blocks of 6 cups (default 2)")
    ap.add_argument("--date", default=_date.today().isoformat(), help="YYYY-MM-DD (default today)")
    ap.add_argument("--append", action="store_true", help="add cups to a session that already has rows")
    ap.add_argument("--guest", action="store_true", help="allow a pourer ID outside the team")
    ap.add_argument("--dry-run", action="store_true", help="print the order without writing the log")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    design = cfg["design"]
    session, pourer = args.session.strip().upper(), args.pourer.strip().upper()

    if not re.fullmatch(design["session_pattern"], session):
        ap.error(f"session {session!r} must match {design['session_pattern']} (e.g. S05)")
    if pourer not in design["pourers"]:
        if not args.guest:
            ap.error(f"pourer {pourer!r} is not one of {design['pourers']}; use --guest for a guest pourer")
        if not re.fullmatch(r"[A-Z][A-Z0-9]{0,3}", pourer):
            ap.error("guest pourer IDs are 1-4 letters/digits, e.g. G1")
        print(f"NOTE: guest pourer {pourer}. You need their written OK before their photos go in the repo.\n")
    if args.blocks < 1:
        ap.error("--blocks must be at least 1")
    try:
        _date.fromisoformat(args.date)
    except ValueError:
        ap.error(f"--date {args.date!r} is not YYYY-MM-DD")

    log_path = cfg_path(cfg, "cup_log")
    log = read_cup_log(log_path)
    existing = log[log["session_id"] == session]
    if len(existing) and not args.append:
        who = ", ".join(sorted(existing["pourer"].unique()))
        print(
            f"Session {session} already has {len(existing)} rows (pourer {who}).\n"
            f"Re-run with --append to add more cups to it, or pick a new session ID.",
            file=sys.stderr,
        )
        return 1

    plan = plan_session(log, cfg, session, pourer, args.blocks, args.date)
    print_plan(plan, cfg)

    if args.dry_run:
        print("\n(dry run: cup log not changed)")
        return 0
    write_cup_log(pd.concat([log, plan], ignore_index=True), log_path)
    print(f"\nAppended {len(plan)} planned rows to {log_path}")
    return 0


def print_plan(plan: pd.DataFrame, cfg: dict) -> None:
    d = cfg["design"]
    first = plan.iloc[0]
    print(f"Session {first['session_id']} | pourer {first['pourer']} | {first['date']} | {len(plan)} cups\n")
    head = f"{'#':>3}  {'cup':<9} {'milk':<6} {'aerate / target gain':<26} {'drop-in':<8} lower pitcher when"
    print(head)
    print("-" * len(head))
    block = len(d["milk_conditions"]) * len(d["dropin"])
    for k, row in enumerate(plan.itertuples(index=False)):
        if k and k % block == 0:
            print(f"{'':>3}  -- block {k // block + 1} --")
        t = d["milk_targets"][row.milk_condition]
        aer = f"{t['aeration']} / {t['foam_gain']}"
        print(f"{k + 1:>3}  {row.cup_id:<9} {row.milk_condition:<6} {aer:<26} {row.dropin:<8} {d['dropin_targets'][row.dropin]}")
    print(
        "\nPer cup: mark milk level before/after steaming -> foam_gain_pct = (after - before) / before x 100.\n"
        "Photograph within 30 s (2 shots, keep the sharper), name it <cup_id>.jpg,\n"
        "then fill foam_gain_pct, seconds_to_photo and usable (y/n) in the cup log."
    )


if __name__ == "__main__":
    sys.exit(main())
