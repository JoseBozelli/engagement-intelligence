"""Projection of the observable event store onto SignalGuard event records.

SignalGuard consumes Event.to_dict() records with six fields (account_id, timestamp, ref_id,
related_ref_id, event_id, event_type) over its eight-type vocabulary. The reference vocabulary
shares those types, names, and ref-id rules (D-014), so the projection deduplicates exactly
(D-009), optionally keeps events recorded by a cutoff, filters to the shared types, and drops
the envelope fields SignalGuard does not use. Timestamps are ISO strings with a UTC offset.

Usage:
  uv run python -m engagement_intelligence.interop.signalguard \
      --events data/reference/observable/events.parquet \
      --out data/reference/interop/signalguard_events.jsonl [--as-of 2026-07-01T00:00:00+00:00]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from engagement_intelligence.core.arrow_io import check_schema, deduplicate_events
from engagement_intelligence.schemas.events import SIGNALGUARD_COMPATIBLE_TYPES

SIGNALGUARD_FIELDS = ("account_id", "timestamp", "ref_id", "related_ref_id", "event_id", "event_type")


def to_signalguard_records(events: pa.Table, as_of: datetime | None = None) -> list[dict[str, Any]]:
    events = deduplicate_events(events)
    if as_of is not None:
        cutoff = pa.scalar(as_of, type=pa.timestamp("us", tz="UTC"))
        events = events.filter(pc.less_equal(events["recorded_at"], cutoff))
    shared = pa.array(sorted(SIGNALGUARD_COMPATIBLE_TYPES), pa.string())
    events = events.filter(pc.is_in(events["event_type"], value_set=shared)).sort_by(
        [("account_id", "ascending"), ("timestamp", "ascending"), ("event_id", "ascending")])
    records = events.select(list(SIGNALGUARD_FIELDS)).to_pylist()
    for record in records:
        record["timestamp"] = record["timestamp"].isoformat()
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Project observable events onto SignalGuard records.")
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--as-of", type=datetime.fromisoformat, default=None)
    args = parser.parse_args(argv)

    events = pq.read_table(args.events)
    check_schema(events)
    records = to_signalguard_records(events, args.as_of)
    write_jsonl(records, args.out)
    counts = Counter(r["event_type"] for r in records)
    print(f"projected {len(records)} records from {events.num_rows} observable events -> {args.out}")
    for event_type, n in sorted(counts.items()):
        print(f"  {event_type:<20} {n}")


if __name__ == "__main__":
    main()