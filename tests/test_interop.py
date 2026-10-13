from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.compute as pc

from engagement_intelligence.core.arrow_io import deduplicate_events
from engagement_intelligence.core.base_event import validate_correlation_ids
from engagement_intelligence.interop.signalguard import SIGNALGUARD_FIELDS, to_signalguard_records
from engagement_intelligence.schemas.events import SIGNALGUARD_COMPATIBLE_TYPES

AS_OF = datetime(2025, 12, 1, tzinfo=timezone.utc)

# SignalGuard's own ref-id rule sets (signalguard/schemas/event.py), restated as data.
SG_REQUIRES = SIGNALGUARD_COMPATIBLE_TYPES - {"account.created"}
SG_NO_REF = frozenset({"account.created"})
SG_ALLOWS_RELATED = frozenset({"followup.required"})


def test_records_have_exactly_signalguard_fields_and_types(small_pipeline):
    records = to_signalguard_records(small_pipeline.observable)
    assert records
    assert all(tuple(r) == SIGNALGUARD_FIELDS for r in records)
    assert {r["event_type"] for r in records} <= SIGNALGUARD_COMPATIBLE_TYPES
    shared = pa.array(sorted(SIGNALGUARD_COMPATIBLE_TYPES), pa.string())
    deduped = deduplicate_events(small_pipeline.observable)
    assert len(records) == deduped.filter(pc.is_in(deduped["event_type"], value_set=shared)).num_rows
    assert len({r["event_id"] for r in records}) == len(records)


def test_cutoff_keeps_only_events_recorded_by_as_of(small_pipeline):
    events = small_pipeline.observable
    cutoff = pa.scalar(AS_OF, type=pa.timestamp("us", tz="UTC"))
    expected = {r["event_id"] for r in to_signalguard_records(events.filter(pc.less_equal(events["recorded_at"], cutoff)))}
    assert {r["event_id"] for r in to_signalguard_records(events, AS_OF)} == expected


def test_records_satisfy_signalguard_ref_id_rules(small_pipeline):
    for r in to_signalguard_records(small_pipeline.observable):
        validate_correlation_ids(
            account_id=r["account_id"], event_type=r["event_type"], ref_id=r["ref_id"],
            related_ref_id=r["related_ref_id"], requires_ref_id=SG_REQUIRES,
            no_ref_id_types=SG_NO_REF, allows_related_ref_id=SG_ALLOWS_RELATED,
        )