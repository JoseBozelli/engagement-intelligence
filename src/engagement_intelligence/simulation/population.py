"""Account population: archetype, arrival time, and static latent traits.

Output is simulator truth only. Each component draws from its own random stream
keyed by account index, so an account's draws are independent of population size,
generation order, and the parameters of the other components.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pyarrow as pa

from engagement_intelligence.core.arrow_io import check_table
from engagement_intelligence.core.ids import make_account_id
from engagement_intelligence.core.seeding import rng_for
from engagement_intelligence.simulation.config import SimConfig

LATENT_PREFIX = "latent_"


def utc_midnight(d: date) -> datetime:
    return datetime.combine(d, time(), tzinfo=timezone.utc)


def account_schema(cfg: SimConfig) -> pa.Schema:
    return pa.schema(
        [
            pa.field("account_id", pa.string(), nullable=False),
            pa.field("account_index", pa.int32(), nullable=False),
            pa.field("tenant_id", pa.string(), nullable=False),
            pa.field("archetype", pa.string(), nullable=False),
            pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
            *[pa.field(LATENT_PREFIX + n, pa.float64(), nullable=False) for n in cfg.latents],
        ]
    )


def sample_population(cfg: SimConfig) -> pa.Table:
    names = list(cfg.archetypes)
    weights = np.array([cfg.archetypes[n].weight for n in names])
    weights = weights / weights.sum()

    start = utc_midnight(cfg.calendar.start)
    initial_end = start + timedelta(days=cfg.population.initial_window_days)
    arrivals_end = utc_midnight(cfg.population.arrivals_end)

    rows = []
    for i in range(cfg.population.n_accounts):
        archetype = names[rng_for(cfg.seed, "archetype", i).choice(len(names), p=weights)]

        arrival = rng_for(cfg.seed, "arrival", i)
        established = arrival.random() < cfg.population.initial_fraction
        lo, hi = (start, initial_end) if established else (initial_end, arrivals_end)
        offset = int(arrival.integers(0, int((hi - lo).total_seconds())))
        created_at = lo + timedelta(seconds=offset)

        prior = cfg.archetypes[archetype]
        values = rng_for(cfg.seed, "latents", i).normal(
            loc=[prior.means[n] for n in cfg.latents], scale=prior.sd
        )

        row = {
            "account_id": make_account_id(i),
            "account_index": i,
            "tenant_id": cfg.population.tenant_id,
            "archetype": archetype,
            "created_at": created_at,
        }
        row.update({LATENT_PREFIX + n: float(v) for n, v in zip(cfg.latents, values)})
        rows.append(row)

    schema = account_schema(cfg)
    table = pa.Table.from_pylist(rows, schema=schema)
    check_table(table, schema)
    return table