from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from engagement_intelligence.core.envelope import EventEnvelope

T0 = datetime(2024, 3, 4, 9, 30, tzinfo=timezone.utc)


def make_envelope(**overrides) -> EventEnvelope:
    base = dict(
        event_id="e1", tenant_id="t1", account_id="u1",
        event_type="thing.opened", actor="user",
        timestamp=T0, recorded_at=T0 + timedelta(minutes=5), ref_id="r1",
    )
    base.update(overrides)
    return EventEnvelope(**base)


def test_valid_envelope_constructs():
    e = make_envelope()
    assert e.schema_version == 1
    assert e.attributes == {}


def test_event_id_has_no_random_default():
    with pytest.raises(ValidationError):
        make_envelope(event_id=None)
    fields = dict(
        tenant_id="t1", account_id="u1", event_type="x", actor="y",
        timestamp=T0, recorded_at=T0,
    )
    with pytest.raises(ValidationError):
        EventEnvelope(**fields)


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError):
        make_envelope(timestamp=datetime(2024, 3, 4, 9, 30))


def test_non_utc_timestamp_normalized_to_utc():
    est = timezone(timedelta(hours=-5))
    e = make_envelope(timestamp=datetime(2024, 3, 4, 4, 30, tzinfo=est))
    assert e.timestamp == T0
    assert e.timestamp.tzinfo == timezone.utc


def test_recorded_before_timestamp_rejected():
    with pytest.raises(ValidationError, match="recorded_at"):
        make_envelope(recorded_at=T0 - timedelta(seconds=1))


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        make_envelope(satisfaction=0.2)  # simulator-only quantities cannot enter the envelope


def test_envelope_is_immutable():
    e = make_envelope()
    with pytest.raises(ValidationError):
        e.actor = "admin"


def test_malformed_identifier_rejected():
    with pytest.raises(ValidationError):
        make_envelope(account_id="u 1")