"""
Reference event vocabulary for account-level service interactions.

A small, generic vocabulary covering sessions, follow-ups, tasks and messsages. Its eight shared types use
the same names and ref-id rules as SignalGuard's vocabulary, so these events project into SignalGuard by filtering.
Applications may define their own EventVocabulary instead of this one.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import model_validator

from engagement_intelligence.core.envelope import EventEnvelope
from engagement_intelligence.core.vocabulary import EventVocabulary

class EventType(StrEnum):
    ACCOUNT_CREATED = "account.created"
    SESSION_BOOKED = "session.booked"
    SESSION_COMPLETED = "session.completed"
    SESSION_CANCELLED = "session.cancelled"
    SESSION_NO_SHOW = "session.no_show"
    FOLLOWUP_REQUIRED = "followup.required"
    FOLLOWUP_COMPLETED = "followup.completed"
    TASK_REQUIRED = "task.required"
    TASK_COMPLETED = "task.completed"
    MESSAGE_SENT = "message.sent"
    MESSAGE_OPENED = "message.opened"
    MESSAGE_REPLIED = "message.replied"

class Actor(StrEnum):
    ACCOUNT = "account"     # the served party
    OPERATOR = "operator"   # the organization serving the account
    SYSTEM = "system"


_A, _O, _S = Actor.ACCOUNT, Actor.OPERATOR, Actor.SYSTEM

REFERENCE_VOCABULARY = EventVocabulary(
    name="engagement-reference-v1",
    allowed_actors={
        EventType.ACCOUNT_CREATED: {_S},
        EventType.SESSION_BOOKED: {_A, _O},
        EventType.SESSION_COMPLETED: {_A},      # actor=account: the account attended
        EventType.SESSION_CANCELLED: {_A, _O},
        EventType.SESSION_NO_SHOW: {_A},
        EventType.FOLLOWUP_REQUIRED: {_O},
        EventType.FOLLOWUP_COMPLETED: {_A},
        EventType.TASK_REQUIRED: {_O},
        EventType.TASK_COMPLETED: {_A},
        EventType.MESSAGE_SENT: {_O},
        EventType.MESSAGE_OPENED: {_A},
        EventType.MESSAGE_REPLIED: {_A},
    },
    requires_ref_id={t for t in EventType if t is not EventType.ACCOUNT_CREATED},
    no_ref_id_types={EventType.ACCOUNT_CREATED},
    allows_related_ref_id={EventType.FOLLOWUP_REQUIRED},         # links a follow-up to its session
)

# Types shared with SignalGuard's vocabulary (same names, same ref-id rules).
SIGNALGUARD_COMPATIBLE_TYPES: frozenset[str] = frozenset(
    {
        EventType.ACCOUNT_CREATED,
        EventType.SESSION_BOOKED,
        EventType.SESSION_COMPLETED,
        EventType.SESSION_NO_SHOW,
        EventType.TASK_REQUIRED,
        EventType.TASK_COMPLETED,
        EventType.FOLLOWUP_REQUIRED,
        EventType.FOLLOWUP_COMPLETED
    }
)

class Event(EventEnvelope):
    """Envelope typed and validated against the reference vocabulary."""

    event_type: EventType
    actor: Actor

    @model_validator(mode="after")
    def _validate_vocabulary(self) -> Event:
        REFERENCE_VOCABULARY.validate(self)
        return self