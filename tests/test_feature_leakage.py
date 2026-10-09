from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from engagement_intelligence.core.arrow_io import EVENT_ARROW_SCHEMA
from engagement_intelligence.features.audit import truncation_audit
from engagement_intelligence.features.build import build_features
from engagement_intelligence.labels.build import ASSESSMENT_SCHEMA

CUTS = [
    datetime(2024, 10, 1, tzinfo=timezone.utc),
    datetime(2025, 3, 3, tzinfo=timezone.utc),
    datetime(2025, 9, 1, tzinfo=timezone.utc),
]


@pytest.fixture(scope="module")
def baseline(small_pipeline, roles_cfg, features_spec):
    return build_features(features_spec, roles_cfg, small_pipeline.observable, small_pipeline.assessments)


def _values(table, names) -> np.ndarray:
    return np.column_stack([table.column(n).to_numpy() for n in names])


@pytest.mark.parametrize("cut", CUTS, ids=[c.date().isoformat() for c in CUTS])
def test_features_use_only_information_recorded_by_assessment_time(
        cut, small_pipeline, roles_cfg, features_spec):
    checked = truncation_audit(features_spec, roles_cfg, small_pipeline.observable,
                               small_pipeline.assessments, cut)
    assert checked > 0


def test_audit_detects_a_leaky_feature(monkeypatch, small_pipeline, roles_cfg, features_spec):
    from engagement_intelligence.features import build

    def leaky_count(f, a):  # ignores recording time: counts every event of the role, including future ones
        return np.full(len(a.times), float(a.masks[f.role].sum()))

    monkeypatch.setitem(build._COMPUTE, "count", leaky_count)
    with pytest.raises(AssertionError, match="recorded after"):
        truncation_audit(features_spec, roles_cfg, small_pipeline.observable,
                         small_pipeline.assessments, CUTS[1])


def test_features_depend_on_recording_time(small_pipeline, roles_cfg, features_spec, baseline):
    events = small_pipeline.observable
    idx = events.schema.get_field_index("recorded_at")
    instant = events.set_column(idx, EVENT_ARROW_SCHEMA.field("recorded_at"), events["timestamp"])
    changed = build_features(features_spec, roles_cfg, instant, small_pipeline.assessments)
    assert not np.array_equal(_values(changed, features_spec.names), _values(baseline, features_spec.names))


def test_labels_are_not_feature_inputs(small_pipeline, roles_cfg, features_spec, baseline):
    a = small_pipeline.assessments
    idx = a.schema.get_field_index("label")
    flipped = a.set_column(idx, ASSESSMENT_SCHEMA.field("label"), pc.invert(a["label"]))
    rebuilt = build_features(features_spec, roles_cfg, small_pipeline.observable, flipped)
    assert np.array_equal(_values(rebuilt, features_spec.names), _values(baseline, features_spec.names))


def test_row_order_and_exact_duplicates_do_not_change_features(
        small_pipeline, roles_cfg, features_spec, baseline):
    events = small_pipeline.observable
    perm = np.random.default_rng(0).permutation(events.num_rows)
    noisy = pa.concat_tables([events.take(pa.array(perm)), events])
    rebuilt = build_features(features_spec, roles_cfg, noisy, small_pipeline.assessments)
    assert np.array_equal(_values(rebuilt, features_spec.names), _values(baseline, features_spec.names))


def test_feature_table_contains_no_truth_label_or_split_columns(baseline, features_spec):
    assert baseline.column_names == ["account_id", "assessed_at", *features_spec.names]
    forbidden = {"label", "split", "cold_start", "archetype"}
    assert not forbidden & set(baseline.column_names)
    assert not any(c.startswith("latent_") for c in baseline.column_names)