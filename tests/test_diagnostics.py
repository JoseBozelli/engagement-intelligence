import numpy as np
import pytest

from engagement_intelligence.labels.build import latest_known, next_occurrence
from engagement_intelligence.simulation.diagnostics import annotate, check_provenance, render_report

I64_MIN, I64_MAX = np.iinfo(np.int64).min, np.iinfo(np.int64).max


def test_latest_known_respects_recording_time():
    occurred = np.array([10, 20, 30])
    recorded = np.array([10, 40, 30])      # the event at 20 is recorded late, at 40
    times = np.array([5, 25, 35, 45])
    assert latest_known(occurred, recorded, times).tolist() == [I64_MIN, 10, 30, 30]


def test_next_occurrence_is_strictly_after():
    occurred = np.array([10, 20, 30])
    assert next_occurrence(occurred, np.array([5, 10, 25, 30])).tolist() == [10, 20, 30, I64_MAX]


def test_provenance_mismatch_rejected(labeling_cfg):
    meta = {"labels.config_fingerprint": labeling_cfg.fingerprint, "labels.source_fingerprint": "abc"}
    with pytest.raises(ValueError, match="different simulation run"):
        check_provenance(meta, "def", labeling_cfg)
    check_provenance(meta, "abc", labeling_cfg)


@pytest.fixture(scope="module")
def annotated(small_pipeline, labeling_cfg):
    r = small_pipeline
    return annotate(labeling_cfg, r.accounts, r.episodes, r.observable, r.assessments)


def test_days_since_engagement_within_eligibility_window(annotated, labeling_cfg):
    d = annotated["days_since_engagement"]
    assert not np.isnan(d).any()
    assert (d >= 0).all() and (d < labeling_cfg.target.eligibility_window_days).all()


def test_labels_track_latent_truth_with_noise(annotated):
    label, latent = annotated["label"], annotated["latent_positive"]
    assert 0 < latent.mean() < 1
    assert label[latent].mean() > label[~latent].mean()     # informative
    assert label[~latent].any() and (~label[latent]).any()  # but noisy, by design (D-001)


def test_report_covers_every_split_and_archetype(small_pipeline, labeling_cfg, annotated):
    report = render_report(small_pipeline, labeling_cfg, annotated)
    for w in labeling_cfg.splits.windows:
        assert f"| {w.name} |" in report
    for name in small_pipeline.sim.archetypes:
        assert f"| {name} |" in report