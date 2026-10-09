import numpy as np
import pytest

from engagement_intelligence.core.roles import RoleMap
from engagement_intelligence.schemas.events import REFERENCE_VOCABULARY

ROLES = RoleMap.from_dict({
    "engagement": [{"event_type": "session.completed", "actor": "account"},
                   {"event_type": "message.replied", "actor": "account"}],
    "outreach": [{"event_type": "message.sent", "actor": "operator"}],
})


def test_dict_round_trip_and_stable_fingerprint():
    again = RoleMap.from_dict(ROLES.to_dict())
    assert again.to_dict() == ROLES.to_dict()
    assert again.fingerprint == ROLES.fingerprint


def test_validate_against_vocabulary():
    ROLES.validate_against(REFERENCE_VOCABULARY)
    bad = RoleMap.from_dict({"x": [{"event_type": "message.sent", "actor": "account"}]})
    with pytest.raises(ValueError, match="cannot emit"):
        bad.validate_against(REFERENCE_VOCABULARY)


def test_mask_selects_event_type_and_actor():
    types = np.array(["session.completed", "message.sent", "message.replied", "session.completed"])
    actors = np.array(["account", "operator", "account", "operator"])
    assert ROLES.mask(types, actors, "engagement").tolist() == [True, False, True, False]


def test_empty_role_rejected():
    with pytest.raises(ValueError, match="without selectors"):
        RoleMap.from_dict({"x": []})