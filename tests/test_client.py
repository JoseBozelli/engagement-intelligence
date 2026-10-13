from datetime import datetime, timezone

import pytest

from engagement_intelligence.client import EngagementClient
from engagement_intelligence.scoring.engine import UnknownAccountError
from engagement_intelligence.scoring.schemas import EngagementAssessment

AS_OF = datetime(2025, 12, 1, tzinfo=timezone.utc)


@pytest.fixture
def api(service_client):
    return EngagementClient(service_client)


def _first_scored(api):
    return next(a for a in api.assess(as_of=AS_OF).assessments if a.status == "scored")


def test_model_info_is_typed(api, small_bundle):
    assert [f.name for f in api.model_info().features] == small_bundle[0].feature_names


def test_assess_is_typed_and_read_only_by_default(api):
    first = _first_scored(api)
    assert isinstance(first, EngagementAssessment)
    assert api.risk_history(first.account_id) == []


def test_single_assessment_and_events(api):
    first = _first_scored(api)
    assert api.assessment(first.account_id, as_of=AS_OF).risk_score == pytest.approx(first.risk_score)
    events = api.events(first.account_id, as_of=AS_OF, limit=3)
    assert 0 < len(events) <= 3
    assert all(e.recorded_at <= AS_OF for e in events)


def test_unknown_account_raises(api):
    with pytest.raises(UnknownAccountError):
        api.assessment("acc_999999", as_of=AS_OF)