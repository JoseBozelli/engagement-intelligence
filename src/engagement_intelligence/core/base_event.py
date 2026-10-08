# Provenance: copied verbatim from SignalGuard (github.com/JoseBozelli/signalguard), module src/signalguard/schemas/
# base_event.py. Do not modify here; extend in evelope.py.
"""
Domain-neutral event correlation base (Zone B / reusable foundation).

Encodes the *shape* of an account-scoped, correlation-ID-linked event without knowing event vocabulary:

- `ref_id`: the ID of the lifecycle instance this event belongs to
- `related_ref_id`: an optional cross-lifecycle reference

which event types requires a ref_id, which must NOT carry one, and which are allowed carry a related_ref_id is
vocabulary-specific knowledge -- that lives in each application's own event module (e.g., SignalGuard's event.py),
which calls `validate_correlation_ids()` from its own validator, passing its own rules sets. This keeps the 
mechanic reusable without this module ever enconding SignalGuard's (or any other application's) specific event types.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

class BaseCorrelatedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    account_id: str
    timestamp: datetime
    ref_id: Optional[str] = None
    related_ref_id: Optional[str] = None
    event_id: str = Field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

def validate_correlation_ids(
        *,
        account_id: str,
        event_type,
        ref_id: Optional[str],
        related_ref_id: Optional[str],
        requires_ref_id: set,
        no_ref_id_types: set,
        allows_related_ref_id: set
) -> None:
    """
    Pure, vocabulary-agnostic validation of the correlation-ID mechanic. Takes the vocabulary-specific rule sets as
    arguments rather than knowing them itself -- directly reusable by any future event vocabulary. Raises ValueError
    on any violation; callers (subclass validators) are expected to let that propagate.
    """
    if not account_id:
        raise ValueError("account_id must be non-empty")

    if event_type in requires_ref_id and ref_id is None:
        raise ValueError(f"ref_id is required for event_type={event_type}")

    if event_type in no_ref_id_types and ref_id is not None:
        raise ValueError(f"ref_id must be None for event_type={event_type}")

    if related_ref_id is not None and event_type not in allows_related_ref_id:
        allowed = ", ".join(str(e) for e in allows_related_ref_id)
        raise ValueError(
            f"related_ref_id is only allowed for event_type in [{allowed}], "
            f"got event_type={event_type}"
        )