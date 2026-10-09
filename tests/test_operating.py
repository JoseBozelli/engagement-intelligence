import numpy as np
import pytest
from pydantic import ValidationError

from engagement_intelligence.evaluation.metrics import expected_calibration_error
from engagement_intelligence.evaluation.slices import bucket_labels, slice_table
from engagement_intelligence.models.operating import (
    Calibration,
    OperatingConfig,
    assign_bands,
    choose_calibration,
    derive_operating_point,
)

CFG = OperatingConfig(review_capacity=0.10, watch_capacity=0.25, ece_tolerance=0.02, calibration_bins=10,
                      tenure_feature="tenure_days", tenure_buckets_days=(90, 180, 365))


def test_operating_point_matches_capacities():
    p = np.random.default_rng(0).random(10_000)
    bands = assign_bands(p, derive_operating_point(p, CFG))
    assert (bands == "high").mean() == pytest.approx(0.10, abs=0.005)
    assert (bands != "low").mean() == pytest.approx(0.25, abs=0.005)


def test_calibrated_scores_are_left_alone():
    rng = np.random.default_rng(1)
    p = rng.random(20_000) * 0.5
    y = rng.random(20_000) < p
    assert choose_calibration(y, p, CFG).method == "none"


def test_miscalibrated_scores_are_recalibrated():
    rng = np.random.default_rng(2)
    true = rng.random(20_000) * 0.3
    y = rng.random(20_000) < true
    p = np.clip(true * 2, 0.0, 0.99)            # overconfident by a factor of two
    cal = choose_calibration(y, p, CFG)
    assert cal.method == "platt"
    assert expected_calibration_error(y, cal.apply(p), 10) < expected_calibration_error(y, p, 10)


def test_identity_calibration():
    p = np.array([0.1, 0.5])
    assert np.array_equal(Calibration("none").apply(p), p)


def test_watch_capacity_must_exceed_review_capacity():
    with pytest.raises(ValidationError, match="watch_capacity"):
        OperatingConfig(**{**CFG.model_dump(), "watch_capacity": 0.05})


def test_slice_table_and_buckets():
    assert bucket_labels(np.array([10, 100, 200, 400]), (90, 180, 365)).tolist() == \
        ["<90", "90-180", "180-365", ">=365"]
    y = np.array([True, False, True, False])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    rows = slice_table(y, p, p >= 0.5, {"group": np.array(["a", "a", "b", "b"])})
    a = next(r for r in rows if r["value"] == "a")
    assert a["n"] == 2 and a["precision"] == 0.5 and a["recall"] == 1.0