import copy
from collections import defaultdict

import numpy as np
import pytest
from pydantic import ValidationError

from engagement_intelligence.simulation.config import SimConfig
from engagement_intelligence.simulation.dynamics import AccountDynamics, simulate_dynamics
from engagement_intelligence.simulation.population import sample_population


def _replace(model, path: str, value):
    """Copy of a frozen config with the nested attribute at dotted `path` replaced."""
    head, *rest = path.split(".")
    new = value if not rest else _replace(getattr(model, head), ".".join(rest), value)
    return model.model_copy(update={head: new})


def _stepper(cfg, archetype: str, n_days: int = 200, seed: int = 1) -> AccountDynamics:
    latents = dict(cfg.archetypes[archetype].means)
    return AccountDynamics(cfg, archetype, latents, np.random.default_rng(seed), n_days)


def _archetype_of(accounts) -> dict[str, str]:
    return dict(zip(accounts.column("account_id").to_pylist(), accounts.column("archetype").to_pylist()))


@pytest.fixture(scope="module")
def world(small_cfg):
    accounts = sample_population(small_cfg)
    states, episodes = simulate_dynamics(small_cfg, accounts)
    return accounts, states, episodes


# --- configuration ---------------------------------------------------------------

def test_unknown_predictor_reference_rejected(reference_raw):
    raw = copy.deepcopy(reference_raw)
    raw["dynamics"]["propensity"]["coefficients"]["mood"] = 1.0
    with pytest.raises(ValidationError, match="mood"):
        SimConfig.model_validate(raw)


def test_unknown_anchor_latent_rejected(reference_raw):
    raw = copy.deepcopy(reference_raw)
    raw["dynamics"]["satisfaction"]["anchor_latent"] = "mood"
    with pytest.raises(ValidationError, match="anchor_latent"):
        SimConfig.model_validate(raw)


# --- stepper ----------------------------------------------------------------------

def test_stepper_is_deterministic(small_cfg):
    a, b = _stepper(small_cfg, "stable_high"), _stepper(small_cfg, "stable_high")
    assert [a.step() for _ in range(200)] == [b.step() for _ in range(200)]


def test_satisfaction_follows_negative_drift(small_cfg):
    cfg = _replace(small_cfg, "dynamics.satisfaction.noise_sd", 0.0)
    sim = _stepper(cfg, "gradual_decline")
    values = np.array([sim.step().satisfaction for _ in range(200)])
    assert np.all(np.diff(values) < 0)


def test_shock_shifts_satisfaction(small_cfg):
    a, b = _stepper(small_cfg, "stable_high"), _stepper(small_cfg, "stable_high")
    a.step()
    b.step(satisfaction_shock=-0.5)
    assert b.satisfaction == pytest.approx(a.satisfaction - 0.5)


def test_stepping_past_horizon_raises(small_cfg):
    sim = _stepper(small_cfg, "stable_high", n_days=2)
    sim.step()
    sim.step()
    with pytest.raises(IndexError):
        sim.step()


# --- population-level simulation --------------------------------------------------

def test_one_row_per_account_day(world, small_cfg):
    accounts, states, _ = world
    expected = sum(
        (small_cfg.calendar.end - c.date()).days for c in accounts.column("created_at").to_pylist()
    )
    assert states.num_rows == expected


def test_simulation_is_deterministic(world, small_cfg):
    accounts, states, episodes = world
    states2, episodes2 = simulate_dynamics(small_cfg, accounts)
    assert states2.equals(states)
    assert episodes2.equals(episodes)


def test_episodes_match_daily_flags(world, small_cfg):
    _, states, episodes = world
    flagged = defaultdict(int)
    for a, dg in zip(states.column("account_id").to_pylist(), states.column("disengaged").to_pylist()):
        flagged[a] += dg
    from_episodes = defaultdict(int)
    for a, s, e in zip(
        episodes.column("account_id").to_pylist(),
        episodes.column("start").to_pylist(),
        episodes.column("end").to_pylist(),
    ):
        from_episodes[a] += ((e or small_cfg.calendar.end) - s).days
    assert from_episodes == {a: n for a, n in flagged.items() if n}


def test_disengagement_parameters_do_not_perturb_other_processes(world, small_cfg):
    accounts, states, _ = world
    cfg = _replace(small_cfg, "dynamics.disengagement.hazard.intercept", -3.0)
    altered, _ = simulate_dynamics(cfg, accounts)
    cols = ["account_id", "date", "satisfaction", "disrupted", "dip", "propensity"]
    assert altered.select(cols).equals(states.select(cols))
    assert not altered.column("disengaged").equals(states.column("disengaged"))


def test_low_propensity_archetype_disengages_more(world):
    accounts, _, episodes = world
    archetype_of = _archetype_of(accounts)
    ever = set(episodes.column("account_id").to_pylist())

    def share(name: str) -> float:
        members = [a for a, n in archetype_of.items() if n == name]
        return sum(a in ever for a in members) / len(members)

    assert share("chronic_low") > share("stable_high") + 0.3


def test_disruption_prone_archetype_is_disrupted_more(world):
    accounts, states, _ = world
    archetype_of = _archetype_of(accounts)
    labels = np.array([archetype_of[a] for a in states.column("account_id").to_pylist()])
    disrupted = np.array(states.column("disrupted").to_pylist())
    assert disrupted[labels == "disruption_recovery"].mean() > disrupted[labels == "stable_high"].mean()


def test_dips_only_where_configured(world, small_cfg):
    accounts, states, _ = world
    archetype_of = _archetype_of(accounts)
    labels = np.array([archetype_of[a] for a in states.column("account_id").to_pylist()])
    dip = np.array(states.column("dip").to_pylist())
    for name, archetype in small_cfg.archetypes.items():
        assert bool(dip[labels == name].any()) == (archetype.dynamics.dip_rate_per_year > 0)