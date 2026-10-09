import math
from datetime import timedelta

import numpy as np
import pyarrow as pa
import pytest
from helpers import T0, ev
from pydantic import ValidationError

from engagement_intelligence.core.arrow_io import to_arrow
from engagement_intelligence.features.build import build_features
from engagement_intelligence.features.spec import FeatureSpec
from engagement_intelligence.labels.build import ASSESSMENT_SCHEMA
from engagement_intelligence.simulation.diagnostics import annotate

SPEC = FeatureSpec.model_validate({"name": "test_spec", "features": [
    {"name": "since", "kind": "recency", "role": "engagement", "cap_days": 365, "description": "d"},
    {"name": "n30", "kind": "count", "role": "engagement", "window_days": 30, "description": "d"},
    {"name": "tenure", "kind": "tenure", "description": "d"},
    {"name": "reply30", "kind": "rate", "numerator": "reply", "denominator": "outreach",
     "window_days": 30, "prior_rate": 0.2, "prior_weight": 2, "description": "d"},
    {"name": "open", "kind": "outstanding", "open_role": "request", "close_role": "completion",
     "min_age_days": 7, "description": "d"},
    {"name": "overdue", "kind": "overdue_ratio", "role": "engagement", "lookback_days": 180,
     "cap": 20, "description": "d"},
    {"name": "trend", "kind": "change", "role": "engagement", "short_days": 30, "long_days": 90,
     "description": "d"},
]})


def _features(events, pairs, roles):
    pairs = sorted(pairs)
    n = len(pairs)
    assessments = pa.table({
        "account_id": [a for a, _ in pairs],
        "assessed_at": [T0 + timedelta(days=d) for _, d in pairs],
        "split": ["train"] * n, "cold_start": [False] * n, "label": [False] * n,
    }, schema=ASSESSMENT_SCHEMA)
    table = build_features(SPEC, roles, to_arrow(events), assessments)
    return {(r["account_id"], (r["assessed_at"] - T0).days): r for r in table.to_pylist()}


def test_level_features_on_known_history(roles_cfg):
    events = [ev("a", "account.created", "system", 0)]
    events += [ev("a", "session.completed", "account", d) for d in (0, 10, 20)]
    r = _features(events, [("a", 28)], roles_cfg)[("a", 28)]
    assert r["since"] == pytest.approx(7.5)
    assert r["n30"] == 3
    assert r["tenure"] == pytest.approx(27.5)


def test_late_recorded_events_are_invisible_until_recorded(roles_cfg):
    events = [ev("b", "account.created", "system", 0),
              ev("b", "session.completed", "account", 25, recorded_day=30)]
    rows = _features(events, [("b", 28), ("b", 35)], roles_cfg)
    assert rows[("b", 28)]["since"] == 365 and rows[("b", 28)]["n30"] == 0
    assert rows[("b", 35)]["since"] == pytest.approx(9.5) and rows[("b", 35)]["n30"] == 1


def test_rate_uses_prior_and_observations(roles_cfg):
    events = [ev("c", "account.created", "system", 0),
              ev("d", "account.created", "system", 0),
              ev("d", "message.sent", "operator", 1, ref="m1"),
              ev("d", "message.sent", "operator", 2, ref="m2"),
              ev("d", "message.replied", "account", 3, ref="m1")]
    rows = _features(events, [("c", 7), ("d", 7)], roles_cfg)
    assert rows[("c", 7)]["reply30"] == pytest.approx(0.2)                    # prior only
    assert rows[("d", 7)]["reply30"] == pytest.approx((1 + 0.4) / (2 + 2))    # 1 reply / 2 sent + prior


def test_outstanding_requests_open_until_completion_recorded(roles_cfg):
    events = [ev("e", "account.created", "system", 0),
              ev("e", "followup.required", "operator", 1, ref="q1"),
              ev("e", "followup.completed", "account", 30, ref="q1")]
    rows = _features(events, [("e", 7), ("e", 14), ("e", 35)], roles_cfg)
    assert [rows[("e", d)]["open"] for d in (7, 14, 35)] == [0, 1, 0]   # too young, open, closed


def test_trajectory_features(roles_cfg):
    events = [ev("f", "account.created", "system", 0)]
    events += [ev("f", "session.completed", "account", d) for d in (0, 30, 60)]
    r = _features(events, [("f", 91)], roles_cfg)[("f", 91)]
    assert r["overdue"] == pytest.approx(30.5 * 3 / 180)                     # silence / average interval
    assert r["trend"] == pytest.approx(math.log(((0 + 0.5 * 30 / 90) / 30) / ((2 + 0.5) / 90)))


def test_duplicate_feature_names_rejected():
    f = {"name": "x", "kind": "tenure", "description": "d"}
    with pytest.raises(ValidationError, match="duplicate"):
        FeatureSpec.model_validate({"name": "s", "features": [f, f]})


def test_unknown_role_rejected(roles_cfg):
    spec = FeatureSpec.model_validate({"name": "s", "features": [
        {"name": "x", "kind": "count", "role": "telepathy", "window_days": 30, "description": "d"}]})
    with pytest.raises(ValueError, match="telepathy"):
        spec.validate_roles(roles_cfg)


def test_change_windows_must_be_ordered():
    with pytest.raises(ValidationError, match="short_days"):
        FeatureSpec.model_validate({"name": "s", "features": [
            {"name": "x", "kind": "change", "role": "engagement", "short_days": 90, "long_days": 30,
             "description": "d"}]})


def test_simulated_features_complete_and_consistent_with_label_primitives(
        small_pipeline, labeling_cfg, roles_cfg, features_spec):
    r = small_pipeline
    table = build_features(features_spec, roles_cfg, r.observable, r.assessments)
    assert table.num_rows == r.assessments.num_rows
    assert table.column_names == ["account_id", "assessed_at", *features_spec.names]
    values = np.column_stack([table.column(n).to_numpy() for n in features_spec.names])
    assert np.isfinite(values).all()
    annotated = annotate(labeling_cfg, r.accounts, r.episodes, r.observable, r.assessments)
    np.testing.assert_allclose(table.column("days_since_engagement").to_numpy(),
                               annotated["days_since_engagement"])

def test_change_is_zero_without_recent_activity(roles_cfg):
    events = [ev("g", "account.created", "system", 0), ev("g", "session.completed", "account", 0)]
    r = _features(events, [("g", 200)], roles_cfg)[("g", 200)]
    assert r["trend"] == pytest.approx(0.0)   # no activity in either window -> no change