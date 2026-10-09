import numpy as np
import pytest

from engagement_intelligence.features.build import build_features
from engagement_intelligence.models.bundle import ModelBundle, build_bundle
from engagement_intelligence.models.data import load_dataset
from engagement_intelligence.models.evaluate_cli import check_first_run
from engagement_intelligence.models.operating import OperatingConfig
from engagement_intelligence.models.train import FittedCandidate, fit_final


@pytest.fixture(scope="module")
def bundle_inputs(small_pipeline, roles_cfg, features_spec, modeling_cfg):
    r = small_pipeline
    features = build_features(features_spec, roles_cfg, r.observable, r.assessments)
    ds = load_dataset(features, r.assessments, features_spec.names)
    train = ds.subset(ds.split == modeling_cfg.splits.train)
    names = modeling_cfg.feature_sets["no_policy"].resolve(features_spec)
    estimator = fit_final("logistic", {"C": 0.01, "max_iter": 2000}, None, train, names,
                          modeling_cfg.log1p_features, modeling_cfg.seed)
    candidate = FittedCandidate("logistic[no_policy]", "logistic", "no_policy", names, {"C": 0.01}, None, [], estimator)
    operating = OperatingConfig(review_capacity=0.1, watch_capacity=0.25, ece_tolerance=0.02, calibration_bins=10,
                                tenure_feature="tenure_days", tenure_buckets_days=(90, 180, 365))
    point = {
        "selected_model": candidate.name,
        "calibration": {"method": "none", "slope": 1.0, "intercept": 0.0},
        "operating_point": {"high_threshold": 0.4, "medium_threshold": 0.15,
                            "review_capacity": 0.1, "watch_capacity": 0.25},
        "operating_config_fingerprint": operating.fingerprint,
        "provenance": {"feature spec fingerprint": features_spec.fingerprint},
        "validation_summary": {},
    }
    return candidate, point, operating, ds


@pytest.fixture
def build(bundle_inputs, features_spec, roles_cfg, labeling_cfg, modeling_cfg):
    def _build(**point_overrides):
        candidate, point, operating, _ = bundle_inputs
        return build_bundle(candidate, {**point, **point_overrides}, features_spec, roles_cfg,
                            labeling_cfg, modeling_cfg, operating)
    return _build


def test_bundle_round_trip_and_deterministic_version(tmp_path, build, bundle_inputs):
    bundle = build()
    assert build().version == bundle.version
    loaded = ModelBundle.load(bundle.save(tmp_path / bundle.version))
    X = bundle_inputs[3].columns(bundle.feature_names)
    np.testing.assert_array_equal(loaded.predict_proba(X), bundle.predict_proba(X))
    assert loaded.manifest["policy_dependent_features_used"] == []
    assert set(loaded.bands(loaded.predict_proba(X)).tolist()) <= {"low", "medium", "high"}


def test_tampered_model_file_is_rejected(tmp_path, build):
    path = build().save(tmp_path / "bundle")
    with open(path / "model.joblib", "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        ModelBundle.load(path)


def test_bundle_rejects_operating_point_for_another_model(build):
    with pytest.raises(ValueError, match="different model"):
        build(selected_model="xgboost[all]")


def test_held_out_evaluation_runs_once(tmp_path):
    record = tmp_path / "test_evaluation.json"
    assert check_first_run(record, None) == []
    record.write_text('{"runs": [{"reason": "first run"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="rerun-reason"):
        check_first_run(record, None)
    assert len(check_first_run(record, "fixed a reporting bug")) == 1