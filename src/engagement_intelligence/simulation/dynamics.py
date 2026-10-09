"""Daily latent dynamics per account (simulator truth).

AccountDynamics is a stepper. The event generator drives it one day at a time
and may feed back experience shocks (e.g., an operator cancellation lowering
satisfaction). simulate_dynamics() is a shock-free driver used to validate the
dynamics in isolation.

Randomness comes from the account's own "dynamics" stream, pre-drawn with one
column per process, so changing one process's parameters cannot change another
process's trajectory.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pyarrow as pa

from engagement_intelligence.core.arrow_io import check_table
from engagement_intelligence.core.seeding import rng_for
from engagement_intelligence.simulation.config import LinearPredictor, SimConfig
from engagement_intelligence.simulation.population import LATENT_PREFIX

STATE_SCHEMA = pa.schema(
    [
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("satisfaction", pa.float64(), nullable=False),
        pa.field("disrupted", pa.bool_(), nullable=False),
        pa.field("dip", pa.bool_(), nullable=False),
        pa.field("propensity", pa.float64(), nullable=False),
        pa.field("disengaged", pa.bool_(), nullable=False),
    ]
)

EPISODE_SCHEMA = pa.schema(
    [
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("episode", pa.int32(), nullable=False),
        pa.field("start", pa.date32(), nullable=False),
        pa.field("end", pa.date32(), nullable=True),  # first re-engaged day; null if ongoing at horizon
    ]
)


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


_Bound = tuple[float, tuple[tuple[str, float], ...]]


def _bind(predictor: LinearPredictor, latents: Mapping[str, float]) -> _Bound:
    """Fold static (latent) terms into a constant; keep time-varying terms by name."""
    constant = predictor.intercept + sum(
        c * latents[n] for n, c in predictor.coefficients.items() if n in latents
    )
    dynamic = tuple((n, c) for n, c in predictor.coefficients.items() if n not in latents)
    return constant, dynamic


def _evaluate(bound: _Bound, state: Mapping[str, float]) -> float:
    constant, dynamic = bound
    return constant + sum(c * state[n] for n, c in dynamic)


@dataclass(frozen=True)
class DayState:
    satisfaction: float
    disrupted: bool
    dip: bool
    propensity: float
    disengaged: bool


class AccountDynamics:
    def __init__(
        self,
        cfg: SimConfig,
        archetype: str,
        latents: Mapping[str, float],
        rng: np.random.Generator,
        n_days: int,
    ) -> None:
        dyn = cfg.dynamics
        arch = cfg.archetypes[archetype].dynamics

        self._anchor = latents[dyn.satisfaction.anchor_latent]
        self._reversion = dyn.satisfaction.reversion
        self._noise_sd = dyn.satisfaction.noise_sd
        self._drift_per_day = arch.satisfaction_drift_per_year / 365.25

        self._p_disruption_start = sigmoid(_bind(dyn.disruption.onset, latents)[0])
        self._p_disruption_end = 1.0 / dyn.disruption.mean_duration_days
        self._p_dip_start = arch.dip_rate_per_year / 365.25
        self._p_dip_end = 1.0 / arch.dip_mean_duration_days

        self._propensity = _bind(dyn.propensity, latents)
        self._hazard = _bind(dyn.disengagement.hazard, latents)
        self._recovery = _bind(dyn.disengagement.recovery, latents)

        # Columns: 0 = disruption, 1 = dip, 2 = disengagement.
        self._u = rng.random((n_days, 3))
        self._z = rng.standard_normal(n_days)
        self._day = 0

        self.satisfaction = self._anchor
        self.disrupted = False
        self.dip = False
        self.disengaged = False

    def step(self, satisfaction_shock: float = 0.0) -> DayState:
        d = self._day
        if d >= len(self._z):
            raise IndexError("stepped past the simulated horizon")
        u = self._u[d]

        target = self._anchor + self._drift_per_day * d
        self.satisfaction += (
            self._reversion * (target - self.satisfaction)
            + self._noise_sd * self._z[d]
            + satisfaction_shock
        )

        if self.disrupted:
            self.disrupted = u[0] >= self._p_disruption_end
        else:
            self.disrupted = u[0] < self._p_disruption_start

        if self.dip:
            self.dip = u[1] >= self._p_dip_end
        else:
            self.dip = u[1] < self._p_dip_start

        state = {"satisfaction": self.satisfaction, "disrupted": float(self.disrupted)}
        propensity = _evaluate(self._propensity, state)
        state["propensity"] = propensity

        if self.disengaged:
            self.disengaged = u[2] >= sigmoid(_evaluate(self._recovery, state))
        else:
            self.disengaged = u[2] < sigmoid(_evaluate(self._hazard, state))

        self._day += 1
        return DayState(self.satisfaction, self.disrupted, self.dip, propensity, self.disengaged)


def _run(sim: AccountDynamics, n_days: int) -> dict[str, np.ndarray]:
    out = {
        "satisfaction": np.empty(n_days),
        "disrupted": np.empty(n_days, dtype=bool),
        "dip": np.empty(n_days, dtype=bool),
        "propensity": np.empty(n_days),
        "disengaged": np.empty(n_days, dtype=bool),
    }
    for d in range(n_days):
        s = sim.step()
        out["satisfaction"][d] = s.satisfaction
        out["disrupted"][d] = s.disrupted
        out["dip"][d] = s.dip
        out["propensity"][d] = s.propensity
        out["disengaged"][d] = s.disengaged
    return out


def _runs(flags: np.ndarray) -> list[tuple[int, int | None]]:
    """Maximal True runs as (start, end-exclusive); end is None if the run reaches the horizon."""
    padded = np.concatenate([[False], flags, [False]]).astype(np.int8)
    diff = np.diff(padded)
    starts, ends = np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)
    n = len(flags)
    return [(int(s), None if e == n else int(e)) for s, e in zip(starts, ends)]


def simulate_dynamics(cfg: SimConfig, accounts: pa.Table) -> tuple[pa.Table, pa.Table]:
    """Shock-free daily trajectories and disengagement episodes for every account."""
    state_chunks: list[pa.Table] = []
    episode_rows: list[dict] = []

    for row in accounts.to_pylist():
        created = row["created_at"].date()
        n_days = (cfg.calendar.end - created).days
        latents = {n: row[LATENT_PREFIX + n] for n in cfg.latents}
        rng = rng_for(cfg.seed, "dynamics", row["account_index"])
        sim = AccountDynamics(cfg, row["archetype"], latents, rng, n_days)
        arrays = _run(sim, n_days)

        state_chunks.append(
            pa.table(
                {
                    "account_id": pa.array([row["account_id"]] * n_days, pa.string()),
                    "date": pa.array(np.datetime64(created, "D") + np.arange(n_days), pa.date32()),
                    **arrays,
                },
                schema=STATE_SCHEMA,
            )
        )
        for k, (start, end) in enumerate(_runs(arrays["disengaged"])):
            episode_rows.append(
                {
                    "account_id": row["account_id"],
                    "episode": k,
                    "start": created + timedelta(days=start),
                    "end": None if end is None else created + timedelta(days=end),
                }
            )

    states = pa.concat_tables(state_chunks)
    episodes = pa.Table.from_pylist(episode_rows, schema=EPISODE_SCHEMA)
    check_table(states, STATE_SCHEMA)
    check_table(episodes, EPISODE_SCHEMA)
    return states, episodes