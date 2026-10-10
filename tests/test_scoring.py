import math
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pytest

from engagement_intelligence.labels.build import column_us, to_us
from engagement_intelligence.scoring.engine import Scorer, UnknownAccountError
from engagement_intelligence.scoring.sources import TableEventSource
from engagement_intelligence.scoring.store import JsonlAssessmentStore

WEEK1 = datetime(2025, 12, 1, tzinfo=timezone.utc)   # first held-out Monday
WEEK2 = datetime(2025, 12, 8, tzinfo=timezone.utc)


@pytest.fixture
def scorer(small_bundle, small_pipeline, tmp_path):
    bundle, _ = small_bundle
    return Scorer(bundle, TableEventSource(small_pipeline.observable), JsonlAssessmentStore(tmp_path / "a.jsonl"))


def _feature_rows_at(features, when):
    return features.filter(pa.array(column_us(features, "assessed_at") == to_us(when)))


def test_eligibility_matches_point_in_time_labels(scorer, small_bundle):
    scored = {a.account_id for a in scorer.assess(WEEK1) if a.status == "scored"}
    assert scored == set(_feature_rows_at(small_bundle[1], WEEK1).column("account_id").to_pylist())


def test_live_scores_equal_bundle_predictions_on_feature_rows(scorer, small_bundle):
    bundle, features = small_bundle
    live = {a.account_id: a.risk_score for a in scorer.assess(WEEK1) if a.status == "scored"}
    rows = _feature_rows_at(features, WEEK1)
    actual = np.array([live[a] for a in rows.column("account_id").to_pylist()])
    np.testing.assert_allclose(actual, bundle.predict_table(rows), rtol=0, atol=1e-12)


def test_risk_change_tracks_band_transitions(scorer):
    first = {a.account_id: a for a in scorer.assess(WEEK1) if a.status == "scored"}
    assert all(a.risk_change.transition == "new" for a in first.values())
    second = [a for a in scorer.assess(WEEK2) if a.status == "scored" and a.account_id in first]
    order = {"low": 0, "medium": 1, "high": 2}
    for a in second:
        prev = first[a.account_id]
        d = order[a.risk_band] - order[prev.risk_band]
        assert a.risk_change.transition == ("unchanged" if d == 0 else "escalated" if d > 0 else "de-escalated")
        assert a.risk_change.score_delta == pytest.approx(a.risk_score - prev.risk_score)
    assert len(scorer.risk_history(second[0].account_id)) == 2


def test_capacity_flag_is_batch_level(scorer, small_bundle):
    scored = [a for a in scorer.assess(WEEK1) if a.status == "scored"]
    k = math.ceil(small_bundle[0].operating_point.review_capacity * len(scored))
    assert sum(a.within_review_capacity for a in scored) == k
    single = scorer.assess(WEEK1, [scored[0].account_id], rank_capacity=False, persist=False)[0]
    assert single.within_review_capacity is None
    assert single.risk_score == pytest.approx(scored[0].risk_score)


def test_drivers_are_exact_linear_contributions(scorer, small_bundle):
    bundle, features = small_bundle
    rows = _feature_rows_at(features, WEEK1)
    X = np.column_stack([rows.column(n).to_numpy() for n in bundle.feature_names])
    raw = bundle.estimator.predict_proba(X)[:, 1]
    total = scorer.explainer.contributions(X).sum(axis=1) + scorer.explainer.intercept
    np.testing.assert_allclose(total, np.log(raw / (1 - raw)), atol=1e-8)
    for a in scorer.assess(WEEK1, persist=False):
        if a.status == "scored":
            c = [d.contribution for d in a.top_drivers]
            assert len(c) <= 3 and all(x > 0 for x in c) and c == sorted(c, reverse=True)


def test_unknown_account_rejected(scorer):
    with pytest.raises(UnknownAccountError):
        scorer.assess(WEEK1, ["acc_999999"], persist=False)


def test_event_source_respects_recording_time(small_pipeline):
    out = TableEventSource(small_pipeline.observable).events(WEEK1)
    assert (column_us(out, "recorded_at") <= to_us(WEEK1)).all()
    assert out.num_rows < small_pipeline.observable.num_rows


def test_store_round_trip(scorer, tmp_path):
    results = scorer.assess(WEEK1)
    account = results[0].account_id
    assert JsonlAssessmentStore(tmp_path / "a.jsonl").history(account) == scorer.risk_history(account)


def test_recent_events_are_point_in_time_and_newest_first(scorer):
    account = next(a.account_id for a in scorer.assess(WEEK1, persist=False) if a.status == "scored")
    events = scorer.recent_events(account, WEEK1, limit=5)
    assert 0 < len(events) <= 5
    assert all(e.recorded_at <= WEEK1 for e in events)
    assert [e.timestamp for e in events] == sorted((e.timestamp for e in events), reverse=True)