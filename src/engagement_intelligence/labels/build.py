"""Point-in-time eligibility and labels on a weekly assessment grid.

For an assessment at time t (00:00 UTC on the configured weekday):
- eligible : the latest engagement event *recorded by t* occurred in (t - W, t]
- label    : True (lapse) iff no engagement event occurred in (t, t + H]
- censored : t + H + settlement > as_of (label not final in this snapshot) -> dropped
Rows are assigned to split windows by assessment date and purged when their label window
reaches past the start of the next window. Inputs are observable events only; exact
duplicates are removed by event_id first (D-009).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pyarrow as pa

from engagement_intelligence.core.arrow_io import check_table, deduplicate_events
from engagement_intelligence.labels.spec import WEEKDAYS, LabelingConfig, TargetSpec

ASSESSMENT_SCHEMA = pa.schema(
    [
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("assessed_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("cold_start", pa.bool_(), nullable=False),
        pa.field("label", pa.bool_(), nullable=False),  # True = lapse within the horizon
    ]
)

DAY_US = 86_400 * 10**6
_I64_MIN, _I64_MAX = np.iinfo(np.int64).min, np.iinfo(np.int64).max


def _us(value: datetime | date) -> int:
    """Microseconds since epoch (UTC) for an aware datetime or a date (00:00 UTC)."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("datetimes must be timezone-aware")
        naive = value.astimezone(timezone.utc).replace(tzinfo=None)
        return int(np.datetime64(naive, "us").astype(np.int64))
    return int(np.datetime64(value, "D").astype("datetime64[us]").astype(np.int64))


def column_us(table: pa.Table, name: str) -> np.ndarray:
    """A timestamp column as int64 microseconds since epoch (UTC)."""
    return table.column(name).to_numpy().astype("datetime64[us]").astype(np.int64)


def assessment_grid(target: TargetSpec, start: date, end: date) -> np.ndarray:
    """Assessment instants on the configured weekday in [start, end), as int64 microseconds."""
    offset = (WEEKDAYS.index(target.assessment_weekday) - start.weekday()) % 7
    first = np.datetime64(start + timedelta(days=offset), "D")
    days = np.arange(first, np.datetime64(end, "D"), np.timedelta64(7, "D"))
    return days.astype("datetime64[us]").astype(np.int64)


def latest_known(occurred: np.ndarray, recorded: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Latest occurrence time among events recorded at or before each time (int64 min if none)."""
    by_recorded = np.argsort(recorded, kind="stable")
    latest = np.maximum.accumulate(occurred[by_recorded])
    pos = np.searchsorted(recorded[by_recorded], times, side="right")
    return np.where(pos > 0, latest[np.maximum(pos - 1, 0)], _I64_MIN)


def next_occurrence(occurred: np.ndarray, times: np.ndarray) -> np.ndarray:
    """First occurrence strictly after each time (int64 max if none). `occurred` must be sorted."""
    nxt = np.searchsorted(occurred, times, side="right")
    return np.where(nxt < len(occurred), occurred[np.minimum(nxt, len(occurred) - 1)], _I64_MAX)


def _engagement_blocks(target: TargetSpec, events: pa.Table) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Per account: (account_id, occurred ascending, recorded aligned). Expects deduplicated events."""
    accounts = np.array(events.column("account_id").to_pylist())
    occurred = column_us(events, "timestamp")
    recorded = column_us(events, "recorded_at")
    selectors = {(s.event_type, s.actor) for s in target.engagement_events}
    pairs = zip(events.column("event_type").to_pylist(), events.column("actor").to_pylist())
    mask = np.fromiter((p in selectors for p in pairs), dtype=bool, count=events.num_rows)
    idx = np.flatnonzero(mask)
    order = idx[np.lexsort((occurred[idx], accounts[idx]))]
    unique, starts, counts = np.unique(accounts[order], return_index=True, return_counts=True)
    for account, start, k in zip(unique, starts, counts):
        block = order[start:start + k]
        yield str(account), occurred[block], recorded[block]


def engagement_blocks(target: TargetSpec, events: pa.Table) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Per-account engagement event times, after exact deduplication (D-009)."""
    return _engagement_blocks(target, deduplicate_events(events))


def _grid_with_splits(cfg: LabelingConfig, as_of_us: int) -> tuple[np.ndarray, np.ndarray]:
    target, windows = cfg.target, cfg.splits.windows
    horizon = target.horizon_days * DAY_US
    settle = target.label_settlement_days * DAY_US
    grids, names = [], []
    for i, w in enumerate(windows):
        g = assessment_grid(target, w.start, w.end)
        purge_limit = _us(windows[i + 1].start) if i + 1 < len(windows) else _I64_MAX
        g = g[(g + horizon <= purge_limit) & (g + horizon + settle <= as_of_us)]
        grids.append(g)
        names.append(np.full(len(g), w.name, dtype=object))
    return np.concatenate(grids), np.concatenate(names)


def _first_seen(accounts: np.ndarray, timestamps: np.ndarray) -> dict[str, int]:
    order = np.lexsort((timestamps, accounts))
    unique, first = np.unique(accounts[order], return_index=True)
    return dict(zip(unique.tolist(), timestamps[order][first].tolist()))


def build_assessments(cfg: LabelingConfig, events: pa.Table, as_of: datetime) -> pa.Table:
    target, plan = cfg.target, cfg.splits
    events = deduplicate_events(events)
    first_seen = _first_seen(np.array(events.column("account_id").to_pylist()), column_us(events, "timestamp"))
    grid, split_names = _grid_with_splits(cfg, _us(as_of))
    window = target.eligibility_window_days * DAY_US
    horizon = target.horizon_days * DAY_US

    out_accounts, out_times, out_splits, out_labels = [], [], [], []
    for account, occurred, recorded in _engagement_blocks(target, events):
        eligible = latest_known(occurred, recorded, grid) > grid - window
        if not eligible.any():
            continue
        lapse = ~(next_occurrence(occurred, grid) <= grid + horizon)
        out_accounts.append(np.full(int(eligible.sum()), account, dtype=object))
        out_times.append(grid[eligible])
        out_splits.append(split_names[eligible])
        out_labels.append(lapse[eligible])

    if not out_accounts:
        return ASSESSMENT_SCHEMA.empty_table()

    account_col = np.concatenate(out_accounts)
    cold_cut = _us(plan.window(plan.cold_start_after).end)
    table = pa.table(
        {
            "account_id": pa.array(account_col.tolist(), pa.string()),
            "assessed_at": pa.array(np.concatenate(out_times).astype("datetime64[us]"),
                                    pa.timestamp("us", tz="UTC")),
            "split": pa.array(np.concatenate(out_splits).tolist(), pa.string()),
            "cold_start": np.array([first_seen[a] >= cold_cut for a in account_col], dtype=bool),
            "label": np.concatenate(out_labels),
        },
        schema=ASSESSMENT_SCHEMA,
    )
    check_table(table, ASSESSMENT_SCHEMA)
    return table