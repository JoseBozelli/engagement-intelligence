"""Vocabulary-agnostic Parquet/Arrow boundary for event envelopes.

The raw store may contain exact duplicats (same event_id); deduplication is a downstream concern
keyed on event_id."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TypeVar

import pyarrow as pa

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

def check_schema(table: pa.Table) -> None:
    for field in EVENT_ARROW_SCHEMA:
        if field.name not in table.schema.names:
            raise ValueError(f"missing column: {field.name}")
        actual = table.schema.field(field.name).type
        if actual != field.type:
            raise ValueError(f"column {field.name}: expected {field.type}, got {actual}")
        if not field.nullable and table.column(field.name).null_count:
            raise ValueError(f"column {field.name}: nulls in non-nullable column")

def from_arrow(table: pa.Table, model: type[E] = EventEnvelope) -> list[E]:
    """Boundary validation: materialize every row through `model`."""
    check_schema(table)
    out: list[E] = []
    for row in table.to_pylist():
        attributes = json.loads(row.pop("attributes_json"))
        out.append(model(**row, attributes=attributes))
    return out