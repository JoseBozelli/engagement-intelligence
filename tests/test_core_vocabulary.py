import dataclasses
from datetime import datetime, timezone

import pytest

from engagement_intelligence.core.envelope import EventEnvelope
from engagement_intelligence.core.vocabulary import EventVocabulary

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

TOY = EventVocabulary(
    name="toy",
    allowed_actors={
        "user.joined": {"system"},
        "thing.opened": {"user"},
        "thing.closed": {"user", "admin"},
    },
    requires_ref_id={"thing.opened", "thing.closed"},
    no_ref_id_types={"user.joined"},
    allows_related_ref_id={"thing.closed"},
)


def env(**overrides) -> EventEnvelope:
    base = dict(
        event_id="e1", tenant_id="t1", account_id="u1",
        event_type="thing.opened", actor="user",
        timestamp=T0, recorded_at=T0, ref_id="r1",
    )
    base.update(overrides)
    return EventEnvelope(**base)


def test_conforming_event_passes():
    TOY.validate(env())
    TOY.validate(env(event_type="thing.closed", actor="admin", related_ref_id="r0"))


def test_unknown_event_type_rejected():
    with pytest.raises(ValueError, match="unknown event_type"):
        TOY.validate(env(event_type="thing.deleted"))


def test_disallowed_actor_rejected():
    with pytest.raises(ValueError, match="actor"):
        TOY.validate(env(actor="admin"))


def test_missing_ref_id_rejected_by_shared_mechanic():
    with pytest.raises(ValueError, match="ref_id is required"):
        TOY.validate(env(ref_id=None))


def test_related_ref_id_only_where_allowed():
    with pytest.raises(ValueError, match="related_ref_id"):
        TOY.validate(env(related_ref_id="r0"))


def test_overlapping_rule_sets_rejected():
    with pytest.raises(ValueError, match="overlap"):
        EventVocabulary(
            name="bad", allowed_actors={"a.x": {"u"}},
            requires_ref_id={"a.x"}, no_ref_id_types={"a.x"}, allows_related_ref_id=set(),
        )


def test_unclassified_type_rejected():
    with pytest.raises(ValueError, match="require or forbid"):
        EventVocabulary(
            name="bad", allowed_actors={"a.x": {"u"}, "a.y": {"u"}},
            requires_ref_id={"a.x"}, no_ref_id_types=set(), allows_related_ref_id=set(),
        )


def test_related_must_be_subset_of_requires():
    with pytest.raises(ValueError, match="subset"):
        EventVocabulary(
            name="bad", allowed_actors={"a.x": {"u"}, "a.y": {"u"}},
            requires_ref_id={"a.x"}, no_ref_id_types={"a.y"}, allows_related_ref_id={"a.y"},
        )


def test_dict_round_trip():
    assert EventVocabulary.from_dict(TOY.to_dict()).to_dict() == TOY.to_dict()


def test_vocabulary_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        TOY.name = "other"
    with pytest.raises(TypeError):
        TOY.allowed_actors["x.y"] = frozenset({"u"})