"""Vocabulary-agnostic Parquet/Arrow boundary for event envelopes.

The raw store may contain exact duplicats (same event_id); deduplication is a downstream concern
keyed on event_id."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TypeVar
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np

from engagement_intelligence.core.envelope import EventEnvelope

E = TypeVar("E", bound=EventEnvelope)

_TS = pa.timestamp("us", tz="UTC")

EVENT_ARROW_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("actor", pa.string(), nullable=False),
        pa.field("timestamp", _TS, nullable=False),
        pa.field("recorded_at", _TS, nullable=False),
        pa.field("ref_id", pa.string(), nullable=True),
        pa.field("related_ref_id", pa.string(), nullable=True),
        pa.field("attributes_json", pa.string(), nullable=False),
        pa.field("schema_version", pa.int16(), nullable=False)
    ]
)

def canonical_json(attributes: Mapping[str, object]) -> str:
    return json.dumps(dict(attributes), sort_keys=True, separators=(",", ":"))

def to_arrow(events: Iterable[EventEnvelope]) -> pa.Table:
    rows = [
        {
            "event_id": e.event_id,
            "tenant_id": e.tenant_id,
            "account_id": e.account_id,
            "event_type": str(e.event_type),
            "actor": str(e.actor),
            "timestamp": e.timestamp,
            "recorded_at": e.recorded_at,
            "ref_id": e.ref_id,
            "related_ref_id": e.related_ref_id,
            "attributes_json": canonical_json(e.attributes),
            "schema_version": e.schema_version,
        }
        for e in events
    ]
    table = pa.Table.from_pylist(rows, schema=EVENT_ARROW_SCHEMA)
    check_schema(table)
    return table

def check_table(table: pa.Table, schema: pa.Schema) -> None:
    """Column presence, types, and non-nullability (pyarrow does not enforce the latter)."""
    for field in schema:
        if field.name not in table.schema.names:
            raise ValueError(f"missing column: {field.name}")
        actual = table.schema.field(field.name).type
        if actual != field.type:
            raise ValueError(f"column {field.name}: expected {field.type}, got {actual}")
        if not field.nullable and table.column(field.name).null_count:
            raise ValueError(f"column {field.name}: nulls in non-nullable column")

def check_schema(table: pa.Table) -> None:
    check_table(table, EVENT_ARROW_SCHEMA)
    if table.num_rows and pc.any(pc.less(table["recorded_at"], table["timestamp"])).as_py():
        raise ValueError("recorded_at must be >= timestamp")

def from_arrow(table: pa.Table, model: type[E] = EventEnvelope) -> list[E]:
    """Boundary validation: materialize every row through `model`."""
    check_schema(table)
    out: list[E] = []
    for row in table.to_pylist():
        attributes = json.loads(row.pop("attributes_json"))
        out.append(model(**row, attributes=attributes))
    return out

def write_parquet(table: pa.Table, path: str | Path, metadata: Mapping[str, str] | None = None) -> None:
    """Write a table, attaching provenance metadata (e.g., a config fingerprint)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if metadata:
        existing = table.schema.metadata or {}
        added = {k.encode(): v.encode() for k, v in metadata.items()}
        table = table.replace_schema_metadata({**existing, **added})
    pq.write_table(table, path)

def read_parquet_metadata(path: str | Path) -> dict[str, str]:
    """User metadata only; library-internal keys are excluded."""
    raw = pq.read_schema(path).metadata or {}
    return {
        k.decode(): v.decode()
        for k, v in raw.items()
        if not k.startswith(b"ARROW:") and k != b"pandas"
    }

def deduplicate_events(table: pa.Table) -> pa.Table:
    """Remove exact duplicates by event_id, keeping the first occurrence in table order (D-009)."""
    if table.num_rows == 0:
        return table
    _, first = np.unique(np.array(table.column("event_id").to_pylist()), return_index=True)
    return table.take(pa.array(np.sort(first)))