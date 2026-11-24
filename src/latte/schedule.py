"""Blocked randomization of cup order (protocol section 6.2).

Each block of 6 cups contains every (milk_condition, dropin) cell exactly once,
in random order, so the pourers' improvement over a session is spread evenly
across conditions instead of piling up in whichever cell happens to come last.
"""

from __future__ import annotations

import itertools
import zlib
from typing import Any

import numpy as np
import pandas as pd

from latte.io import CUP_ID_RE, CUP_LOG_COLUMNS


def design_cells(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """All (milk_condition, dropin) cells of the factorial design."""
    d = cfg["design"]
    return list(itertools.product(d["milk_conditions"], d["dropin"]))


def session_rng(seed: int, session: str, pourer: str, first_cup: int) -> np.random.Generator:
    """Generator seeded from the global seed plus the session, pourer and first cup number.

    Uses CRC32 rather than Python's ``hash()``, which is salted per process.
    """
    key = f"{session}|{pourer}|{first_cup}".encode()
    return np.random.default_rng([seed, zlib.crc32(key)])


def blocked_order(cells: list[tuple[str, str]], n_blocks: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    """Concatenate ``n_blocks`` independent random permutations of ``cells``."""
    order: list[tuple[str, str]] = []
    for _ in range(n_blocks):
        order += [cells[i] for i in rng.permutation(len(cells))]
    return order


def next_cup_number(log: pd.DataFrame, session: str) -> int:
    """1 + the highest cup number already logged for ``session`` (1 if none)."""
    nums = [int(m.group(2)) for cid in log["cup_id"] if (m := CUP_ID_RE.match(cid)) and m.group(1) == session]
    return max(nums, default=0) + 1


def plan_session(
    log: pd.DataFrame,
    cfg: dict[str, Any],
    session: str,
    pourer: str,
    n_blocks: int,
    date: str,
) -> pd.DataFrame:
    """Planned rows for a new run of cups; measurement fields are left blank."""
    first = next_cup_number(log, session)
    rng = session_rng(int(cfg["seed"]), session, pourer, first)
    order = blocked_order(design_cells(cfg), n_blocks, rng)
    rows = []
    for k, (milk, dropin) in enumerate(order):
        cup_id = f"{session}-C{first + k:02d}"
        row = dict.fromkeys(CUP_LOG_COLUMNS, "")
        row.update(
            cup_id=cup_id,
            session_id=session,
            date=date,
            pourer=pourer,
            milk_condition=milk,
            dropin=dropin,
            photo_file=f"{cup_id}.jpg",
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=CUP_LOG_COLUMNS)
