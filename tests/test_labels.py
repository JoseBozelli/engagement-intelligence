import itertools
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from helpers import T0, ev

from engagement_intelligence.core.arrow_io import to_arrow
from engagement_intelligence.core.config import deep_merge
from engagement_intelligence.labels.build import ASSESSMENT_SCHEMA, build_assessments
from engagement_intelligence.labels.spec import LabelingConfig
from engagement_intelligence.schemas.events import REFERENCE_VOCABULARY, Event
from engagement_intelligence.simulation.events import simulate_world
from engagement_intelligence.simulation.noise import apply_noise
from engagement_intelligence.simulation.population import sample_population, utc_midnight

BASE = {
    "target": {
        "name": "test_target",
        "engagement_role": "engagement",
        "engagement_events": [
            {"event_type": "session.completed", "actor": "account"},
            {"event_type": "message.replied", "actor": "account"},
        ],
        "horizon_days": 60,
        "eligibility_window_days": 90,
        "label_settlement_days": 14,
        "assessment_weekday": "monday",
    },
    "splits": {
        "windows": [
            {"name": "train", "start": "2024-01-01", "end": "2024-07-01"},   # days 0-182
            {"name": "test", "start": "2024-07-01", "end": "2025-01-01"},    # days 182-366
        ],
        "cold_start_after": "train",
    },
}


def _cfg(**overlay) -> LabelingConfig:
    return LabelingConfig.model_validate(deep_merge(BASE, overlay))


def _rows(events, cfg=None, as_of_day=400):
    table = build_assessments(cfg or _cfg(), to_arrow(events), T0 + timedelta(days=as_of_day))
    return {(r["account_id"], (r["assessed_at"] - T0).days): r for r in table.to_pylist()}


# --- point-in-time semantics -------------------------------------------------------

def test_eligibility_uses_only_events_recorded_by_assessment_time():
    events = [ev("a", "account.created", "system", 0),
              ev("a", "session.completed", "account", 3, recorded_day=10)]
    rows = _rows(events)
    assert ("a", 7) not in rows      # occurred day 3, but recorded only on day 10
    assert ("a", 14) in rows


def test_label_counts_engagement_by_occurrence_time():
    events = [ev("b", "account.created", "system", 0),
              ev("b", "session.completed", "account", 0),
              ev("b", "session.completed", "account", 50, recorded_day=70)]
    assert _rows(events)[("b", 7)]["label"] is False   # day 50 is inside (7, 67]


def test_lapse_label_when_no_engagement_in_horizon():
    events = [ev("c", "account.created", "system", 0),
              ev("c", "session.completed", "account", 0),
              ev("c", "session.completed", "account", 100)]
    rows = _rows(events)
    assert rows[("c", 7)]["label"] is True             # (7, 67] is empty
    assert rows[("c", 42)]["label"] is False           # (42, 102] contains day 100


def test_only_configured_engagement_events_count():
    events = [ev("d", "account.created", "system", 0),
              ev("d", "session.completed", "account", 0),
              ev("d", "session.booked", "operator", 20),
              ev("d", "message.opened", "account", 21),
              ev("e", "account.created", "system", 0),
              ev("e", "message.opened", "account", 5)]
    rows = _rows(events)
    assert rows[("d", 7)]["label"] is True
    assert not any(account == "e" for account, _ in rows)


def test_exact_duplicates_do_not_change_results():
    events = [ev("b", "account.created", "system", 0),
              ev("b", "session.completed", "account", 0),
              ev("b", "session.completed", "account", 50)]
    assert _rows(events + events) == _rows(events)


# --- splits ------------------------------------------------------------------------

def test_purge_and_censoring_boundaries():
    events = [ev("g", "account.created", "system", 0)]
    events += [ev("g", "session.completed", "account", d) for d in range(0, 391, 10)]
    days = {"train": [], "test": []}
    for (_, day), r in _rows(events).items():
        days[r["split"]].append(day)
    assert max(days["train"]) == 119   # 119 + 60 <= 182 (next split start); 126 would cross
    assert min(days["test"]) == 182
    assert max(days["test"]) == 322    # 322 + 60 + 14 <= 400 (as_of); 329 would be censored


def test_cold_start_uses_first_observed_event():
    events = [ev("g", "account.created", "system", 0),
              ev("g", "session.completed", "account", 180),
              ev("f", "account.created", "system", 200),
              ev("f", "session.completed", "account", 201)]
    rows = _rows(events)
    assert rows[("f", 203)]["cold_start"] is True
    assert rows[("g", 182)]["cold_start"] is False


@pytest.fixture(scope="module")
def simulated(small_cfg, labeling_cfg):
    accounts = sample_population(small_cfg)
    _, _, clean = simulate_world(small_cfg, accounts)
    observable, _ = apply_noise(small_cfg, accounts, clean)
    as_of = utc_midnight(small_cfg.calendar.end)
    return build_assessments(labeling_cfg, observable, as_of), as_of


def test_simulated_assessments_respect_split_rules(simulated, labeling_cfg):
    table, as_of = simulated
    assert table.schema.equals(ASSESSMENT_SCHEMA)
    windows = labeling_cfg.splits.windows
    assert set(table.column("split").to_pylist()) == {w.name for w in windows}
    horizon = timedelta(days=labeling_cfg.target.horizon_days)
    settle = timedelta(days=labeling_cfg.target.label_settlement_days)
    by_name = {w.name: w for w in windows}
    next_start = {w.name: (windows[i + 1].start if i + 1 < len(windows) else None)
                  for i, w in enumerate(windows)}
    for r in table.to_pylist():
        w = by_name[r["split"]]
        assert w.start <= r["assessed_at"].date() < w.end
        assert r["assessed_at"] + horizon + settle <= as_of
        if next_start[r["split"]] is not None:
            assert (r["assessed_at"] + horizon).date() <= next_start[r["split"]]


# --- specification validation ------------------------------------------------------

def test_overlapping_splits_rejected():
    windows = [{"name": "train", "start": "2024-01-01", "end": "2024-07-01"},
               {"name": "test", "start": "2024-06-01", "end": "2025-01-01"}]
    with pytest.raises(ValidationError, match="overlaps"):
        _cfg(splits={"windows": windows})


def test_unknown_cold_start_split_rejected():
    with pytest.raises(ValidationError, match="holdout"):
        _cfg(splits={"cold_start_after": "holdout"})


def test_target_validated_against_vocabulary():
    cfg = _cfg(target={"engagement_events": [{"event_type": "message.sent", "actor": "account"}]})
    with pytest.raises(ValueError, match="message.sent"):
        cfg.target.validate_against(REFERENCE_VOCABULARY)

def test_engagement_is_resolved_from_roles(roles_cfg, tmp_path):
    import yaml

    from engagement_intelligence.labels.spec import load_labeling_config

    target = {k: v for k, v in BASE["target"].items() if k != "engagement_events"}
    path = tmp_path / "labeling.yaml"
    path.write_text(yaml.safe_dump({"target": target, "splits": BASE["splits"]}), encoding="utf-8")
    cfg = load_labeling_config(path, roles=roles_cfg)
    assert {(s.event_type, s.actor) for s in cfg.target.engagement_events} == roles_cfg.selectors["engagement"]

    path.write_text(yaml.safe_dump(BASE), encoding="utf-8")   # inline events are rejected
    with pytest.raises(ValueError, match="engagement_role"):
        load_labeling_config(path, roles=roles_cfg)