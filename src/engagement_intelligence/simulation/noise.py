"""Observation noise: turns the clean event stream (truth) into the observable store.

Mechanisms (all configured): dropped events, recording delay (lognormal), timestamp
jitter, and exact duplicates (same event_id, identical row; D-009). Exempt types are
never perturbed. Events recorded at or after the horizon are not yet observable.

Every event receives all draws from its account's own "noise" stream, one column per
mechanism, so changing one mechanism's parameters cannot change another's draws.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from engagement_intelligence.core.arrow_io import EVENT_ARROW_SCHEMA, check_schema, check_table
from engagement_intelligence.core.seeding import rng_for
from engagement_intelligence.schemas.events import REFERENCE_VOCABULARY
from engagement_intelligence.simulation.config import SimConfig
from engagement_intelligence.simulation.population import utc_midnight

NOISE_LOG_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("dropped", pa.bool_(), nullable=False),
        pa.field("jitter_seconds", pa.float64(), nullable=False),
        pa.field("delay_hours", pa.float64(), nullable=False),   # 0.0 if not delayed
        pa.field("duplicated", pa.bool_(), nullable=False),
        pa.field("observed", pa.bool_(), nullable=False),         # not dropped and recorded before horizon
    ]
)

_TS_TYPE = pa.timestamp("us", tz="UTC")


def _draws(cfg: SimConfig, accounts: pa.Table, account_ids: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per-event draws from each account's own noise stream (events grouped by account)."""
    nz = cfg.noise
    index_of = dict(zip(accounts.column("account_id").to_pylist(),
                        accounts.column("account_index").to_pylist()))
    n = len(account_ids)
    uniforms, lag_hours, jitter_seconds = np.empty((n, 3)), np.empty(n), np.empty(n)
    unique, starts, counts = np.unique(account_ids, return_index=True, return_counts=True)
    for account, start, k in zip(unique, starts, counts):
        rng = rng_for(cfg.seed, "noise", index_of[account])
        block = slice(start, start + k)
        uniforms[block] = rng.random((k, 3))  # columns: 0 = drop, 1 = delay, 2 = duplicate
        lag_hours[block] = rng.lognormal(np.log(nz.delay_median_hours), nz.delay_sigma, k)
        jitter_seconds[block] = rng.uniform(-1.0, 1.0, k) * nz.jitter_minutes * 60.0
    return uniforms, lag_hours, jitter_seconds


def apply_noise(cfg: SimConfig, accounts: pa.Table, clean: pa.Table) -> tuple[pa.Table, pa.Table]:
    """Return (observable events, noise log). `clean` must be sorted by account_id."""
    nz = cfg.noise
    account_ids = np.array(clean.column("account_id").to_pylist())
    if len(account_ids) and not np.all(account_ids[:-1] <= account_ids[1:]):
        raise ValueError("clean events must be sorted by account_id")

    uniforms, lag_hours, jitter_seconds = _draws(cfg, accounts, account_ids)
    exempt = np.isin(np.array(clean.column("event_type").to_pylist()), list(nz.exempt_types))

    dropped = (uniforms[:, 0] < nz.drop_prob) & ~exempt
    delay_hours = np.where((uniforms[:, 1] < nz.delay_prob) & ~exempt, lag_hours, 0.0)
    duplicated = (uniforms[:, 2] < nz.duplicate_prob) & ~exempt
    jitter_seconds = np.where(exempt, 0.0, jitter_seconds)

    created_of = dict(zip(accounts.column("account_id").to_pylist(),
                          accounts.column("created_at").to_numpy().astype("datetime64[us]")))
    created = np.array([created_of[a] for a in account_ids], dtype="datetime64[us]")
    timestamp = clean.column("timestamp").to_numpy().astype("datetime64[us]")
    observed_ts = np.maximum(timestamp + (jitter_seconds * 1e6).astype("timedelta64[us]"), created)
    recorded = observed_ts + (delay_hours * 3.6e9).astype("timedelta64[us]")

    horizon = np.datetime64(utc_midnight(cfg.calendar.end).replace(tzinfo=None), "us")
    observed = ~dropped & (recorded < horizon)

    log = pa.table(
        {
            "event_id": clean.column("event_id").combine_chunks(),
            "account_id": clean.column("account_id").combine_chunks(),
            "dropped": dropped,
            "jitter_seconds": jitter_seconds,
            "delay_hours": delay_hours,
            "duplicated": duplicated,
            "observed": observed,
        },
        schema=NOISE_LOG_SCHEMA,
    )

    noisy = clean.set_column(
        clean.schema.get_field_index("timestamp"), EVENT_ARROW_SCHEMA.field("timestamp"),
        pa.array(observed_ts, _TS_TYPE),
    )
    noisy = noisy.set_column(
        noisy.schema.get_field_index("recorded_at"), EVENT_ARROW_SCHEMA.field("recorded_at"),
        pa.array(recorded, _TS_TYPE),
    )
    rows = np.sort(np.concatenate([np.flatnonzero(observed), np.flatnonzero(observed & duplicated)]))
    observable = noisy.take(pa.array(rows)).sort_by(
        [("account_id", "ascending"), ("timestamp", "ascending"), ("event_id", "ascending")]
    )

    check_schema(observable)
    REFERENCE_VOCABULARY.check_table(observable)
    check_table(log, NOISE_LOG_SCHEMA)
    return observable, log