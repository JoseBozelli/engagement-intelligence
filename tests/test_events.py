import copy
from collections import Counter

import pytest
from pydantic import ValidationError
from datetime import timedelta

from engagement_intelligence.core.ids import make_account_id
from engagement_intelligence.core.arrow_io import from_arrow
from engagement_intelligence.schemas.events import Event
from engagement_intelligence.simulation.config import SimConfig
from engagement_intelligence.simulation.dynamics import simulate_dynamics, DayState, sigmoid
from engagement_intelligence.simulation.events import simulate_world, EventGenerator
from engagement_intelligence.simulation.population import sample_population, utc_midnight, LATENT_PREFIX

SESSION_OUTCOMES = {"session.completed", "session.no_show", "session.cancelled"}


@pytest.fixture(scope="module")
def world(small_cfg):
    accounts = sample_population(small_cfg)
    states, episodes, events = simulate_world(small_cfg, accounts)
    return accounts, states, episodes, events


@pytest.fixture(scope="module")
def rows(world):
    return world[3].to_pylist()


def _archetype_of(accounts) -> dict[str, str]:
    return dict(zip(accounts.column("account_id").to_pylist(), accounts.column("archetype").to_pylist()))


def _is_engagement(cfg, row) -> bool:
    return row["actor"] == "account" and row["event_type"] in cfg.events.messages.nudge_reset_by


# --- configuration ---------------------------------------------------------------

def test_unknown_event_predictor_reference_rejected(reference_raw):
    raw = copy.deepcopy(reference_raw)
    raw["events"]["sessions"]["booking"]["coefficients"]["mood"] = 1.0
    with pytest.raises(ValidationError, match="mood"):
        SimConfig.model_validate(raw)


def test_unknown_nudge_reset_type_rejected(reference_raw):
    raw = copy.deepcopy(reference_raw)
    raw["events"]["messages"]["nudge_reset_by"] = ["session.teleported"]
    with pytest.raises(ValidationError, match="teleported"):
        SimConfig.model_validate(raw)


# --- integrity -------------------------------------------------------------------

def test_world_is_deterministic(world, small_cfg):
    accounts, states, _, events = world
    states2, _, events2 = simulate_world(small_cfg, accounts)
    assert events2.equals(events)
    assert states2.equals(states)


def test_events_pass_full_boundary_validation(world):
    sample = world[3].slice(0, 2000)
    assert len(from_arrow(sample, model=Event)) == sample.num_rows


def test_event_ids_are_unique(world):
    event_ids = world[3].column("event_id").to_pylist()
    assert len(set(event_ids)) == len(event_ids)


def test_each_account_starts_with_one_created_event(rows, world):
    accounts = world[0]
    created = dict(zip(accounts.column("account_id").to_pylist(), accounts.column("created_at").to_pylist()))
    first: dict = {}
    count = Counter()
    for r in rows:
        first.setdefault(r["account_id"], r)
        if r["event_type"] == "account.created":
            count[r["account_id"]] += 1
    assert len(count) == accounts.num_rows and set(count.values()) == {1}
    assert all(first[a]["event_type"] == "account.created" and first[a]["timestamp"] == created[a] for a in created)


def test_events_within_lifetime_and_clean(rows, world, small_cfg):
    accounts = world[0]
    created = dict(zip(accounts.column("account_id").to_pylist(), accounts.column("created_at").to_pylist()))
    end = utc_midnight(small_cfg.calendar.end)
    for r in rows:
        assert created[r["account_id"]] <= r["timestamp"] < end
        assert r["recorded_at"] == r["timestamp"]


def test_session_lifecycles_are_consistent(rows):
    booked = {r["ref_id"]: r["timestamp"] for r in rows if r["event_type"] == "session.booked"}
    outcomes = Counter()
    for r in rows:
        if r["event_type"] in SESSION_OUTCOMES:
            assert booked[r["ref_id"]] <= r["timestamp"]
            outcomes[r["ref_id"]] += 1
    assert outcomes and max(outcomes.values()) == 1


def test_followups_link_to_completed_sessions(rows):
    completed = {r["ref_id"] for r in rows if r["event_type"] == "session.completed"}
    required = {r["ref_id"]: r for r in rows if r["event_type"] == "followup.required"}
    assert required
    assert all(r["related_ref_id"] in completed for r in required.values())
    for r in rows:
        if r["event_type"] == "followup.completed":
            assert required[r["ref_id"]]["timestamp"] <= r["timestamp"]


def test_message_threads_are_consistent(rows):
    sent = {r["ref_id"]: r["timestamp"] for r in rows if r["event_type"] == "message.sent"}
    opened = {r["ref_id"]: r["timestamp"] for r in rows if r["event_type"] == "message.opened"}
    assert opened
    assert all(sent[ref] <= ts for ref, ts in opened.items())
    for r in rows:
        if r["event_type"] == "message.replied":
            assert opened[r["ref_id"]] <= r["timestamp"]


# --- behavior --------------------------------------------------------------------

def test_frequent_archetype_engages_more_than_sparse(rows, world, small_cfg):
    accounts, states, _, _ = world
    archetype_of = _archetype_of(accounts)
    exposure = Counter(archetype_of[a] for a in states.column("account_id").to_pylist())
    engaged = Counter(archetype_of[r["account_id"]] for r in rows if _is_engagement(small_cfg, r))
    rate = {k: engaged[k] / exposure[k] for k in exposure}
    assert rate["stable_high"] > 1.5 * rate["sparse_stable"]


def test_disengaged_days_are_quiet(rows, world, small_cfg):
    states = world[1]
    keys = zip(states.column("account_id").to_pylist(), states.column("date").to_pylist())
    disengaged = dict(zip(keys, states.column("disengaged").to_pylist()))
    counts = Counter(
        disengaged[(r["account_id"], r["timestamp"].date())] for r in rows if _is_engagement(small_cfg, r)
    )
    days = Counter(disengaged.values())
    assert counts[True] / days[True] < 0.25 * counts[False] / days[False]


def test_experience_shocks_feed_back_into_satisfaction_only(world, small_cfg):
    accounts, states, _, _ = world
    baseline, _ = simulate_dynamics(small_cfg, accounts)
    for col in ("account_id", "date", "disrupted", "dip"):
        assert states.column(col).equals(baseline.column(col))
    assert not states.column("satisfaction").equals(baseline.column("satisfaction"))

# ---code fidelity----------------------------------------------------------------------------------------

def test_booking_rate_matches_configured_predictor(small_cfg):
    """With state held fixed, the empirical booking rate must follow the configured predictor."""
    sessions = small_cfg.events.sessions
    inputs = {**{n: 0.5 for n in small_cfg.latents},
              "satisfaction": 0.5, "disrupted": 0.0, "dip": 0.0, "disengaged": 0.0}
    logit = sessions.booking.intercept + sum(
        c * inputs[n] for n, c in sessions.booking.coefficients.items()
    )
    p = sigmoid(logit)
    expected_cycle_days = sessions.lead_days_mean + (1 - p) / p  # lead time + geometric wait

    start = small_cfg.calendar.start
    n_days = (small_cfg.calendar.end - start).days
    fixed = DayState(satisfaction=0.5, disrupted=False, dip=False, propensity=0.0, disengaged=False)
    bookings = 0
    n_accounts = 20
    for i in range(n_accounts):
        account = {
            "account_id": make_account_id(i), "account_index": i, "tenant_id": "demo",
            "created_at": utc_midnight(start),
            **{LATENT_PREFIX + n: 0.5 for n in small_cfg.latents},
        }
        generator = EventGenerator(small_cfg, account)
        for d in range(n_days):
            generator.on_day(d, start + timedelta(days=d), fixed)
        bookings += sum(e["event_type"] == "session.booked" for e in generator.events)

    assert bookings / (n_accounts * n_days) == pytest.approx(1 / expected_cycle_days, rel=0.15)