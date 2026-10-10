"""Batch-first scoring engine.

For an as-of time: eligibility (bundle target spec, point-in-time), features (bundle roles and
feature spec), calibrated risk, bands, batch review capacity, drivers, and risk change against
the stored previous assessment. Vocabulary-agnostic: everything comes from the bundle and the
injected event source and assessment store.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa

from engagement_intelligence.core.arrow_io import deduplicate_events, from_arrow
from engagement_intelligence.core.envelope import EventEnvelope
from engagement_intelligence.core.roles import RoleMap
from engagement_intelligence.features.build import build_features
from engagement_intelligence.features.spec import FeatureSpec
from engagement_intelligence.labels.build import DAY_US, engagement_blocks, latest_known, to_us
from engagement_intelligence.labels.spec import TargetSpec
from engagement_intelligence.models.bundle import ModelBundle
from engagement_intelligence.scoring.explain import make_explainer, top_drivers
from engagement_intelligence.scoring.schemas import EngagementAssessment, compare_bands
from engagement_intelligence.scoring.sources import EventSource
from engagement_intelligence.scoring.store import AssessmentStore

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def from_us(us: int) -> datetime:
    return _EPOCH + timedelta(microseconds=int(us))


class UnknownAccountError(LookupError):
    """Requested accounts have no events recorded by the as-of time."""


class Scorer:
    def __init__(self, bundle: ModelBundle, source: EventSource, store: AssessmentStore, top_k: int = 3) -> None:
        self.bundle, self.source, self.store, self.top_k = bundle, source, store, top_k
        manifest = bundle.manifest
        self.target = TargetSpec.model_validate(manifest["target"])
        self.roles = RoleMap.from_dict(manifest["roles"])
        self.spec = FeatureSpec.model_validate(manifest["feature_spec"])
        self.explainer = make_explainer(bundle)
        self._descriptions = {f.name: f.description for f in self.spec.features}

    def assess(self, as_of: datetime, account_ids: Sequence[str] | None = None, *,
               rank_capacity: bool = True, persist: bool = True) -> list[EngagementAssessment]:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        events = self.source.events(as_of, account_ids)
        accounts = events.column("account_id").to_pylist()
        tenant_of = dict(zip(accounts, events.column("tenant_id").to_pylist()))
        requested = sorted(set(account_ids)) if account_ids is not None else sorted(tenant_of)
        missing = [a for a in requested if a not in tenant_of]
        if missing:
            raise UnknownAccountError(f"no events recorded by {as_of.isoformat()} for: {missing}")

        t = np.array([to_us(as_of)])
        window = self.target.eligibility_window_days * DAY_US
        eligible = [a for a, occurred, recorded in engagement_blocks(self.target, events)
                    if latest_known(occurred, recorded, t)[0] > t[0] - window]

        common = {"assessed_at": as_of, "horizon_days": self.target.horizon_days,
                  "model_version": self.bundle.version}
        out = []
        if eligible:
            keys = pa.table({"account_id": pa.array(eligible, pa.string()),
                             "assessed_at": pa.array([as_of] * len(eligible), pa.timestamp("us", tz="UTC"))})
            features = build_features(self.spec, self.roles, events, keys)
            names = self.bundle.feature_names
            X = np.column_stack([features.column(n).to_numpy() for n in names]).astype(float)
            p = self.bundle.predict_proba(X)
            bands = self.bundle.bands(p)
            capacity = self._capacity(p) if rank_capacity else [None] * len(p)
            contributions = self.explainer.contributions(X)
            previous = self.store.latest_scored_before(eligible, as_of)
            for i, account in enumerate(eligible):
                band, score = str(bands[i]), float(p[i])
                out.append(EngagementAssessment(
                    account_id=account, tenant_id=tenant_of[account], status="scored",
                    risk_score=score, risk_band=band,
                    risk_change=compare_bands(previous.get(account), band, score),
                    within_review_capacity=capacity[i],
                    top_drivers=top_drivers(contributions[i], X[i], names, self._descriptions, self.top_k),
                    **common))
        scored = set(eligible)
        out += [EngagementAssessment(account_id=a, tenant_id=tenant_of[a], status="not_eligible", **common)
                for a in requested if a not in scored]
        out.sort(key=lambda a: a.account_id)
        if persist:
            self.store.append(out)
        return out

    def _capacity(self, p: np.ndarray) -> list[bool]:
        """Top review-capacity share of this batch by score."""
        k = math.ceil(self.bundle.operating_point.review_capacity * len(p))
        flags = np.zeros(len(p), dtype=bool)
        flags[np.argsort(-p, kind="stable")[:k]] = True
        return flags.tolist()

    def recent_events(self, account_id: str, as_of: datetime, limit: int = 20) -> list[EventEnvelope]:
        events = self.source.events(as_of, [account_id])
        if events.num_rows == 0:
            raise UnknownAccountError(f"no events recorded by {as_of.isoformat()} for: {[account_id]}")
        events = deduplicate_events(events).sort_by(
            [("timestamp", "descending"), ("event_id", "ascending")]).slice(0, limit)
        return from_arrow(events, model=EventEnvelope)

    def risk_history(self, account_id: str) -> list[EngagementAssessment]:
        return self.store.history(account_id)