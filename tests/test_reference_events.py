from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from engagement_intelligence.core import ids
from engagement_intelligence.core.arrow_io import EVENT_ARROW_SCHEMA, from_arrow, to_arrow, check_schema, deduplicate_events
from engagement_intelligence.schemas.events import (
    REFERENCE_VOCABULARY,
    SIGNALGUARD_COMPATIBLE_TYPES,
    Actor,
    Event,
    EventType,
)

T0 = datetime(2024, 3, 4, 9, 30, tzinfo=timezone.utc)


def make_event(**overrides) -> Event:
    base = dict(
        event_id=ids.make_event_id(1, "acc_000001", 0),
        tenant_id="demo",
        account_id="acc_000001",
        event_type=EventType.MESSAGE_REPLIED,
        actor=Actor.ACCOUNT,
        timestamp=T0,
        recorded_at=T0 + timedelta(hours=2),
        ref_id=ids.make_ref_id(1, "acc_000001", 0),
        attributes={"latency_hours": 5.5, "channel": "email"},
    )
    base.update(overrides)
    return Event(**base)


def _dump(events):
    return [e.model_dump() for e in events]


def test_valid_reference_event():
    e = make_event()
    assert e.event_type is EventType.MESSAGE_REPLIED
    assert e.tenant_id == "demo"


def test_operator_cannot_reply_for_account():
    with pytest.raises(ValidationError, match="actor"):
        make_event(actor=Actor.OPERATOR)


def test_lifecycle_types_require_ref_id():
    with pytest.raises(ValidationError, match="ref_id is required"):
        make_event(ref_id=None)


def test_account_created_forbids_ref_id():
    with pytest.raises(ValidationError, match="ref_id must be None"):
        make_event(event_type=EventType.ACCOUNT_CREATED, actor=Actor.SYSTEM)
    ok = make_event(
        event_type=EventType.ACCOUNT_CREATED, actor=Actor.SYSTEM, ref_id=None, attributes={}
    )
    assert ok.ref_id is None


def test_unknown_event_type_rejected():
    with pytest.raises(ValidationError):
        make_event(event_type="account.closed")  # absent by design: disengagement is silent


def test_related_ref_id_only_on_followup_required():
    ok = make_event(
        event_type=EventType.FOLLOWUP_REQUIRED, actor=Actor.OPERATOR,
        related_ref_id=ids.make_ref_id(1, "acc_000001", 99),
    )
    assert ok.related_ref_id is not None
    with pytest.raises(ValidationError, match="related_ref_id"):
        make_event(
            event_type=EventType.SESSION_COMPLETED,
            related_ref_id=ids.make_ref_id(1, "acc_000001", 99),
        )


def test_reference_vocabulary_is_fully_specified():
    assert REFERENCE_VOCABULARY.event_types == {str(t) for t in EventType}


def test_signalguard_compatible_subset():
    assert SIGNALGUARD_COMPATIBLE_TYPES <= REFERENCE_VOCABULARY.event_types
    assert len(SIGNALGUARD_COMPATIBLE_TYPES) == 8
    assert REFERENCE_VOCABULARY.allows_related_ref_id & SIGNALGUARD_COMPATIBLE_TYPES == {
        "followup.required"
    }


def test_arrow_round_trip_is_lossless():
    events = [
        make_event(),
        make_event(
            event_id=ids.make_event_id(1, "acc_000001", 1),
            event_type=EventType.ACCOUNT_CREATED, actor=Actor.SYSTEM,
            ref_id=None, attributes={},
        ),
    ]
    table = to_arrow(events)
    assert table.schema.equals(EVENT_ARROW_SCHEMA)
    assert _dump(from_arrow(table, model=Event)) == _dump(events)


def test_parquet_round_trip_is_lossless(tmp_path):
    events = [make_event()]
    path = tmp_path / "events.parquet"
    pq.write_table(to_arrow(events), path)
    assert _dump(from_arrow(pq.read_table(path), model=Event)) == _dump(events)


def test_exact_duplicates_are_preserved_in_raw_table():
    e = make_event()
    table = to_arrow([e, e])
    assert table.num_rows == 2
    assert table.column("event_id").to_pylist() == [e.event_id, e.event_id]


def test_attributes_serialization_is_canonical():
    a = to_arrow([make_event(attributes={"b": 1, "a": 2})])
    b = to_arrow([make_event(attributes={"a": 2, "b": 1})])
    assert (
        a.column("attributes_json").to_pylist()
        == b.column("attributes_json").to_pylist()
        == ['{"a":2,"b":1}']
    )

def test_nulls_in_required_columns_are_rejected():
    rows = to_arrow([make_event()]).to_pylist()
    rows[0]["tenant_id"] = None
    table = pa.Table.from_pylist(rows, schema=EVENT_ARROW_SCHEMA)   # pyarrow accepts this silently
    with pytest.raises(ValueError, match="tenant_id"):
        from_arrow(table, model=Event)

def test_table_with_recorded_before_timestamp_rejected():
    rows = to_arrow([make_event()]).to_pylist()
    rows[0]["recorded_at"] = rows[0]["timestamp"] - timedelta(hours=1)
    table = pa.Table.from_pylist(rows, schema=EVENT_ARROW_SCHEMA)
    with pytest.raises(ValueError, match="recorded_at"):
        check_schema(table)

def test_reference_vocabulary_names_are_exact():
    assert {str(t) for t in EventType} == {
        "account.created", "session.booked", "session.completed", "session.cancelled",
        "session.no_show", "followup.required", "followup.completed", "task.required",
        "task.completed", "message.sent", "message.opened", "message.replied",
    }
    assert {str(a) for a in Actor} == {"account", "operator", "system"}

def test_deduplicate_events_keeps_first_occurrence():
    a = make_event()
    b = make_event(event_id=ids.make_event_id(1, "acc_000001", 7))
    deduped = deduplicate_events(to_arrow([a, b, a]))
    assert deduped.column("event_id").to_pylist() == [a.event_id, b.event_id]