"""Point-in-time feature computation from a declarative FeatureSpec.

For each assessment (account, t) only events recorded at or before t are visible; windows
apply to occurrence time. Inputs are observable events and assessment keys only; labels
and truth are never inputs. Exact duplicates are removed by event_id first (D-009).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyarrow as pa

from engagement_intelligence.core.arrow_io import check_table, deduplicate_events
from engagement_intelligence.core.roles import RoleMap
from engagement_intelligence.features.spec import (
    ChangeFeature,
    CountFeature,
    FeatureSpec,
    OutstandingFeature,
    OverdueRatioFeature,
    RateFeature,
    RecencyFeature,
    TenureFeature,
)
from engagement_intelligence.labels.build import DAY_US, column_us

_I64_MIN, _I64_MAX = np.iinfo(np.int64).min, np.iinfo(np.int64).max


def feature_schema(spec: FeatureSpec) -> pa.Schema:
    return pa.schema(
        [
            pa.field("account_id", pa.string(), nullable=False),
            pa.field("assessed_at", pa.timestamp("us", tz="UTC"), nullable=False),
            *[pa.field(name, pa.float64(), nullable=False) for name in spec.names],
        ]
    )


class _AccountView:
    """One account's event history, as visible at each of its assessment times."""

    def __init__(self, occurred, recorded, refs, masks, times) -> None:
        self.occurred, self.recorded, self.refs, self.masks, self.times = occurred, recorded, refs, masks, times
        self.known = recorded[:, None] <= times[None, :]  # events x times

    def visible(self, role: str) -> np.ndarray:
        return self.known & self.masks[role][:, None]

    def count(self, role: str, days: float) -> np.ndarray:
        in_window = self.occurred[:, None] > self.times[None, :] - days * DAY_US
        return (self.visible(role) & in_window).sum(axis=0)

    def last(self, role: str) -> np.ndarray:
        return np.where(self.visible(role), self.occurred[:, None], _I64_MIN).max(axis=0, initial=_I64_MIN)

    def days_since(self, instants: np.ndarray) -> np.ndarray:
        return (self.times - instants) / DAY_US


def _recency(f: RecencyFeature, a: _AccountView) -> np.ndarray:
    last = a.last(f.role)
    return np.where(last == _I64_MIN, f.cap_days, np.minimum(a.days_since(last), f.cap_days))


def _count(f: CountFeature, a: _AccountView) -> np.ndarray:
    return a.count(f.role, f.window_days).astype(float)


def _change(f: ChangeFeature, a: _AccountView) -> np.ndarray:
    # Pseudo-count applies to the long window and is scaled to the short window's length,
    # so equal underlying rates, including zero activity, yield exactly 0.
    short = (a.count(f.role, f.short_days) + f.pseudo_count * f.short_days / f.long_days) / f.short_days
    long = (a.count(f.role, f.long_days) + f.pseudo_count) / f.long_days
    return np.log(short / long)


def _rate(f: RateFeature, a: _AccountView) -> np.ndarray:
    num = a.count(f.numerator, f.window_days)
    den = a.count(f.denominator, f.window_days)
    return (num + f.prior_rate * f.prior_weight) / (den + f.prior_weight)


def _overdue_ratio(f: OverdueRatioFeature, a: _AccountView) -> np.ndarray:
    last = a.last(f.role)
    silence = np.where(last == _I64_MIN, f.lookback_days, a.days_since(last))
    average_interval = f.lookback_days / np.maximum(a.count(f.role, f.lookback_days), 1)
    return np.minimum(silence / average_interval, f.cap)


def _outstanding(f: OutstandingFeature, a: _AccountView) -> np.ndarray:
    closes = a.masks[f.close_role]
    closed_at: dict[str, int] = {}
    for ref, recorded in zip(a.refs[closes], a.recorded[closes]):
        closed_at[ref] = min(closed_at.get(ref, _I64_MAX), int(recorded))
    opens = a.masks[f.open_role]
    close_recorded = np.array([closed_at.get(ref, _I64_MAX) for ref in a.refs[opens]], dtype=np.int64)
    times = a.times[None, :]
    still_open = (
        (a.recorded[opens][:, None] <= times)
        & (a.occurred[opens][:, None] <= times - f.min_age_days * DAY_US)
        & (close_recorded[:, None] > times)
    )
    return still_open.sum(axis=0).astype(float)


def _tenure(f: TenureFeature, a: _AccountView) -> np.ndarray:
    first = np.where(a.known, a.occurred[:, None], _I64_MAX).min(axis=0, initial=_I64_MAX)
    return a.days_since(first)


_COMPUTE: dict[str, Callable] = {
    "recency": _recency,
    "count": _count,
    "change": _change,
    "rate": _rate,
    "overdue_ratio": _overdue_ratio,
    "outstanding": _outstanding,
    "tenure": _tenure,
}


def build_features(spec: FeatureSpec, roles: RoleMap, events: pa.Table, assessments: pa.Table) -> pa.Table:
    """Feature rows aligned one-to-one, in order, with `assessments` (sorted by account_id)."""
    spec.validate_roles(roles)
    events = deduplicate_events(events)
    accounts = np.array(events.column("account_id").to_pylist())
    occurred = column_us(events, "timestamp")
    recorded = column_us(events, "recorded_at")
    refs = np.array(events.column("ref_id").to_pylist(), dtype=object)
    types = np.array(events.column("event_type").to_pylist())
    actors = np.array(events.column("actor").to_pylist())
    masks = {role: roles.mask(types, actors, role) for role in spec.roles_used()}

    order = np.lexsort((occurred, accounts))
    e_unique, e_starts, e_counts = np.unique(accounts[order], return_index=True, return_counts=True)
    blocks = {str(a): order[s:s + k] for a, s, k in zip(e_unique, e_starts, e_counts)}

    assessed = np.array(assessments.column("account_id").to_pylist())
    if len(assessed) and not np.all(assessed[:-1] <= assessed[1:]):
        raise ValueError("assessments must be sorted by account_id")
    times = column_us(assessments, "assessed_at")
    values = np.empty((len(assessed), len(spec.features)))
    a_unique, a_starts, a_counts = np.unique(assessed, return_index=True, return_counts=True)
    for account, start, k in zip(a_unique, a_starts, a_counts):
        idx = blocks.get(str(account))
        if idx is None:
            raise ValueError(f"no events for assessed account {account!r}")
        view = _AccountView(occurred[idx], recorded[idx], refs[idx],
                            {r: m[idx] for r, m in masks.items()}, times[start:start + k])
        for j, f in enumerate(spec.features):
            values[start:start + k, j] = _COMPUTE[f.kind](f, view)

    if np.isnan(values).any():
        raise ValueError("feature computation produced NaN")
    schema = feature_schema(spec)
    table = pa.table(
        {
            "account_id": assessments.column("account_id").combine_chunks(),
            "assessed_at": assessments.column("assessed_at").combine_chunks(),
            **{name: values[:, j] for j, name in enumerate(spec.names)},
        },
        schema=schema,
    )
    check_table(table, schema)
    return table