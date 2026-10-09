import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from engagement_intelligence.features.build import build_features
from engagement_intelligence.labels.build import DAY_US
from engagement_intelligence.models.config import CandidateConfig, FeatureSet
from engagement_intelligence.models.data import load_dataset
from engagement_intelligence.models.train import fold_masks, run_experiments


@pytest.fixture(scope="module")
def small_features(small_pipeline, roles_cfg, features_spec):
    return build_features(features_spec, roles_cfg, small_pipeline.observable, small_pipeline.assessments)


@pytest.fixture(scope="module")
def dev_data(small_features, small_pipeline, features_spec, modeling_cfg):
    ds = load_dataset(small_features, small_pipeline.assessments, features_spec.names)
    return ds.subset(np.isin(ds.split, [modeling_cfg.splits.train, modeling_cfg.splits.validation]))


def test_feature_set_resolution(features_spec):
    no_policy = FeatureSet(exclude_policy_dependent=True).resolve(features_spec)
    assert set(no_policy).isdisjoint(features_spec.policy_dependent_names)
    with pytest.raises(ValueError, match="unknown"):
        FeatureSet(exclude=("nope",)).resolve(features_spec)


def test_candidate_configurations_are_the_grid_product():
    configs = CandidateConfig(grid={"a": [1, 2], "b": [3]}, fixed={"c": 0}).configurations()
    assert configs == [{"c": 0, "a": 1, "b": 3}, {"c": 0, "a": 2, "b": 3}]


def test_folds_are_purged_and_disjoint(dev_data, modeling_cfg, labeling_cfg):
    train = dev_data.subset(dev_data.split == modeling_cfg.splits.train)
    horizon_days = labeling_cfg.target.horizon_days
    for fit, score in fold_masks(train, modeling_cfg.cv_folds, horizon_days):
        assert not (fit & score).any()
        assert train.assessed_us[fit].max() + horizon_days * DAY_US <= train.assessed_us[score].min()


def test_misaligned_inputs_rejected(small_features, small_pipeline, features_spec):
    with pytest.raises(ValueError, match="row counts"):
        load_dataset(small_features.slice(1), small_pipeline.assessments, features_spec.names)


def test_held_out_rows_are_rejected(small_features, small_pipeline, features_spec, modeling_cfg, labeling_cfg):
    ds = load_dataset(small_features, small_pipeline.assessments, features_spec.names)  # includes test rows
    with pytest.raises(ValueError, match="development splits"):
        run_experiments(modeling_cfg, features_spec, ds, labeling_cfg.target.horizon_days)


def test_candidates_train_and_beat_chance(dev_data, features_spec, modeling_cfg, labeling_cfg):
    small = modeling_cfg.model_copy(update={
        "feature_sets": {"all": modeling_cfg.feature_sets["all"]},
        "candidates": {k: CandidateConfig(fixed=v.fixed) for k, v in modeling_cfg.candidates.items()},
    })
    candidates = run_experiments(small, features_spec, dev_data, labeling_cfg.target.horizon_days)
    assert [c.name for c in candidates] == ["baseline", "logistic[all]", "xgboost[all]"]
    val = dev_data.subset(dev_data.split == small.splits.validation)
    for c in candidates:
        p = c.predict(val)
        assert np.all((p > 0) & (p < 1))
        assert roc_auc_score(val.y, p) > 0.7
    xgb = candidates[-1]
    assert xgb.estimator.get_params()["n_estimators"] == xgb.cv_results[0]["rounds"]