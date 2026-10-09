"""
Event vocabulary as data.

An Event Vocabulary declares the event types, which actors may emit each type, and the ref-id rule sets
consumed by validate_correlation_ids(). Applications supply their own vocabulary (in code or from config);
envelope and validation code are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from engagement_intelligence.core.base_event import validate_correlation_ids

if TYPE_CHECKING:
    from engagement_intelligence.core.envelope import EventEnvelope
    import pyarrow as pa

def _strs(values) -> frozenset[str]:
    return frozenset(str(v) for v in values)

@dataclass(frozen=True)
class EventVocabulary:
    name: str
    allowed_actors: Mapping[str, frozenset[str]]
    requires_ref_id: frozenset[str]
    no_ref_id_types: frozenset[str]
    allows_related_ref_id: frozenset[str]

    def __post_init__(self) -> None:
        actors = {str(t): _strs(a) for t, a in self.allowed_actors.items()}
        object.__setattr__(self, "allowed_actors", MappingProxyType(actors))
        for f in ("requires_ref_id", "no_ref_id_types", "allows_related_ref_id"):
            object.__setattr__(self, f, _strs(getattr(self, f)))

        types = frozenset(actors)
        if not self.name or not types:
            raise ValueError("vocabulary needs a name and at least one eventy type")
        if any(not a for a in actors.values()):
            raise ValueError("every event type needs at least one allowed actor")
        if self.requires_ref_id & self.no_ref_id_types:
            raise ValueError("requires_ref_id and no_ref_id_types overlap")
        if (self.requires_ref_id | self.no_ref_id_types) != types:
            raise ValueError("every event type must either require or forbid ref_id")
        if not self.allows_related_ref_id <= self.requires_ref_id:
            raise ValueError("allows_related_ref_id must be a subset of requries_ref_id")

    @property
    def event_types(self) -> frozenset[str]:
        return frozenset(self.allowed_actors)

    def _check(self, *, account_id: str, event_type: str, actor: str,
               ref_id: str | None, related_ref_id: str | None) -> None:
        if event_type not in self.allowed_actors:
            raise ValueError(f"unknown event_type {event_type!r} for vocabulary {self.name!r}")
        if actor not in self.allowed_actors[event_type]:
            raise ValueError(f"actor {actor!r} not allowed for event_type {event_type!r}")
        validate_correlation_ids(
            account_id=account_id,
            event_type=event_type,
            ref_id=ref_id,
            related_ref_id=related_ref_id,
            requires_ref_id=self.requires_ref_id,
            no_ref_id_types=self.no_ref_id_types,
            allows_related_ref_id=self.allows_related_ref_id
        )

    def validate(self, event: EventEnvelope) -> None:
        self._check(
            account_id=event.account_id,
            event_type=str(event.event_type),
            actor=str(event.actor),
            ref_id=event.ref_id,
            related_ref_id=event.related_ref_id
        )

    def check_table(self, table: pa.Table) -> None:
        """Validate every row of an event table without constructing objects (bulk boundary)."""
        names = ("account_id", "event_type", "actor", "ref_id", "related_ref_id")
        columns = [table.column(n).to_pylist() for n in names]
        for i, (account_id, event_type, actor, ref_id, related) in enumerate(zip(*columns)):
            try:
                self._check(account_id=account_id, event_type=event_type, actor=actor,
                            ref_id=ref_id, related_ref_id=related)
            except ValueError as exc:
                raise ValueError(f"row {i}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "allowed_actors": {t: sorted(a) for t, a in sorted(self.allowed_actors.items())},
            "requires_ref_id": sorted(self.requires_ref_id),
            "no_ref_id_types": sorted(self.no_ref_id_types),
            "allows_related_ref_id": sorted(self.allows_related_ref_id)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EventVocabulary:
        return cls(
            name=data["name"],
            allowed_actors=data["allowed_actors"],
            requires_ref_id=data["requires_ref_id"],
            no_ref_id_types=data.get("no_ref_id_types", ()),
            allows_related_ref_id=data.get("allows_related_ref_id", ())
        )