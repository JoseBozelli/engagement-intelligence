import numpy as np
import pytest
from sklearn.metrics import log_loss

from engagement_intelligence.evaluation.metrics import (
    binary_log_loss,
    capacity_flags,
    clustered_bootstrap,
    expected_calibration_error,
    metric_suite,
)


def test_capacity_flags_take_top_share_within_each_group():
    p = np.array([0.9, 0.1, 0.5, 0.2, 0.8, 0.3, 0.7, 0.6])
    groups = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    flags = capacity_flags(p, groups, 0.25)     # ceil(0.25 * 4) = 1 per group
    assert flags.tolist() == [True, False, False, False, True, False, False, False]


def test_binary_log_loss_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.2
    p = rng.random(500) * 0.98 + 0.01
    assert binary_log_loss(y, p) == pytest.approx(log_loss(y, p))


def test_constant_prediction_at_prevalence_is_calibrated():
    y = np.array([True] * 10 + [False] * 90)
    assert expected_calibration_error(y, np.full(100, 0.1), 10) == pytest.approx(0.0)


def test_perfect_ranking_metrics():
    y = np.array([True] * 10 + [False] * 90)
    p = np.where(y, 0.9, 0.1)
    m = metric_suite(y, p, np.zeros(100), (0.10,), 10)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == pytest.approx(1.0)
    assert m["precision@10%"] == 1.0 and m["lift@10%"] == pytest.approx(10.0)


def test_clustered_bootstrap_resamples_whole_clusters():
    clusters = np.repeat(np.arange(20), 5)

    def whole_clusters_only(idx):
        counts = np.bincount(clusters[idx], minlength=20)
        return float(np.all(counts % 5 == 0))

    assert clustered_bootstrap(clusters, whole_clusters_only, 50, seed=1) == (1.0, 1.0)