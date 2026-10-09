import copy
from collections import Counter, defaultdict
from datetime import timedelta

import numpy as np
import pytest
from helpers import replace_config
from pydantic import ValidationError

from engagement_intelligence.core.arrow_io import EVENT_ARROW_SCHEMA
from engagement_intelligence.simulation.config import SimConfig
from engagement_intelligence.simulation.events import simulate_world
from engagement_intelligence.simulation.noise import apply_noise
from engagement_intelligence.simulation.population import sample_population, utc_midnight


@pytest.fixture(scope="module")
def noisy_world(small_cfg):
    accounts = sample_population(small_cfg)
    _, _, clean = simulate_world(small_cfg, accounts)
    observable, log = apply_noise(small_cfg, accounts, clean)
    return accounts, clean, observable, log


@pytest.fixture(scope="module")
def observable_rows(noisy_world):
    return noisy_world[2].to_pylist()


def test_unknown_exempt_type_rejected(reference_raw):
    raw = copy.deepcopy(reference_raw)
    raw["noise"]["exempt_types"] = ["account.vanished"]
    with pytest.raises(ValidationError, match="vanished"):
        SimConfig.model_validate(raw)


def test_noise_is_deterministic(noisy_world, small_cfg):
    accounts, clean, observable, log = noisy_world
    observable2, log2 = apply_noise(small_cfg, accounts, clean)
    assert observable2.equals(observable)
    assert log2.equals(log)


def test_observable_uses_event_schema_only(noisy_world):
    assert noisy_world[2].schema.equals(EVENT_ARROW_SCHEMA)


def test_measured_rates_match_config(noisy_world, small_cfg):
    _, clean, _, log = noisy_world
    nz = small_cfg.noise
    exempt = np.isin(np.array(clean.column("event_type").to_pylist()), list(nz.exempt_types))
    dropped = np.array(log.column("dropped").to_pylist())[~exempt]
    delayed = (log.column("delay_hours").to_numpy() > 0)[~exempt]
    duplicated = np.array(log.column("duplicated").to_pylist())[~exempt]
    assert dropped.mean() == pytest.approx(nz.drop_prob, abs=0.006)
    assert delayed.mean() == pytest.approx(nz.delay_prob, abs=0.01)
    assert duplicated.mean() == pytest.approx(nz.duplicate_prob, abs=0.002)


def test_exempt_events_are_untouched(noisy_world, observable_rows):
    accounts = noisy_world[0]
    created = dict(zip(accounts.column("account_id").to_pylist(), accounts.column("created_at").to_pylist()))
    rows = [r for r in observable_rows if r["event_type"] == "account.created"]
    assert len(rows) == accounts.num_rows
    assert all(r["timestamp"] == created[r["account_id"]] == r["recorded_at"] for r in rows)


def test_observable_reconciles_with_noise_log(noisy_world):
    _, _, observable, log = noisy_world
    ids = log.column("event_id").to_pylist()
    observed = log.column("observed").to_pylist()
    duplicated = log.column("duplicated").to_pylist()
    counts = Counter(observable.column("event_id").to_pylist())
    assert set(counts) == {e for e, o in zip(ids, observed) if o}
    assert {e for e, n in counts.items() if n == 2} == {e for e, o, d in zip(ids, observed, duplicated) if o and d}
    assert max(counts.values()) <= 2


def test_recording_delay_and_horizon(noisy_world, observable_rows, small_cfg):
    log = noisy_world[3]
    delay_of = dict(zip(log.column("event_id").to_pylist(), log.column("delay_hours").to_pylist()))
    end = utc_midnight(small_cfg.calendar.end)
    for r in observable_rows:
        assert r["timestamp"] <= r["recorded_at"] < end
        lag_hours = (r["recorded_at"] - r["timestamp"]).total_seconds() / 3600
        assert lag_hours == pytest.approx(delay_of[r["event_id"]], abs=1e-6)


def test_jitter_is_bounded(noisy_world, observable_rows, small_cfg):
    clean = noisy_world[1]
    clean_ts = dict(zip(clean.column("event_id").to_pylist(), clean.column("timestamp").to_pylist()))
    limit = timedelta(minutes=small_cfg.noise.jitter_minutes)
    assert all(abs(r["timestamp"] - clean_ts[r["event_id"]]) <= limit for r in observable_rows)


def test_mechanisms_draw_independently(noisy_world, small_cfg):
    accounts, clean, _, log = noisy_world
    _, log2 = apply_noise(replace_config(small_cfg, "noise.drop_prob", 0.2), accounts, clean)
    for col in ("jitter_seconds", "delay_hours", "duplicated"):
        assert log2.column(col).equals(log.column(col))
    assert not log2.column("dropped").equals(log.column("dropped"))


def test_exact_duplicates_are_identical_rows(observable_rows):
    by_id = defaultdict(list)
    for r in observable_rows:
        by_id[r["event_id"]].append(r)
    pairs = [rows for rows in by_id.values() if len(rows) == 2]
    assert pairs
    assert all(a == b for a, b in pairs)