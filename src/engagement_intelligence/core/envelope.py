"""Tenant-scoped, bitemporal event envelope.

Extends BaseCorrelatedEvent (unmodified) with:
- tentant_id:   owning tenant
- event_type:   vocabulary-defined value (checked by an EventVocabulary)
- actor:        vocabulary-defined initiator of the event
- recorded_at:  when the system learned of the event. The inherited `timestamp` is when it occurred.
                Point-in-time computations filter on recorded_at <= t.
- attributes:   small scalar payload
- schema_version

Override: event_id is required (no random default) so datasets are reproducible.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import ConfigDict, Field, field_validator, model_validator

from engagement_intelligence.core import ids
from engagement_intelligence.core.base_event import BaseCorrelatedEvent

AttributeValue = str | int | float | bool

class EventEnvelope(BaseCorrelatedEvent):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    tenant_id: str
    event_type: str
    actor: str
    recorded_at: datetime
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)
    schema_version: int = Field(default=1, ge=1)

    @field_validator("event_id", "tenant_id", "account_id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not ids.is_valid_id(v):
            raise ValueError(f"malformed identifier: {v!r}")
        return v

    @field_validator("ref_id", "related_ref_id")
    @classmethod
    def _optional_id_format(cls, v: str | None) -> str | None:
        if v is not None and not ids.is_valid_id(v):
            raise ValueError(f"malformed identifier: {v!r}")
        return v

    @field_validator("event_type", "actor")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("timestamp", "recorded_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _bitemporal_order(self) -> EventEnvelope:
        if self.recorded_at < self.timestamp:
            raise ValueError("recorded_at must be >= timestamp")
        return self