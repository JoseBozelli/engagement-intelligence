"""Per-request observability, composed over the verbatim SignalGuard JSONLTracer."""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Sequence
from typing import Any

from engagement_intelligence.core.tracing import JSONLTracer
from engagement_intelligence.scoring.schemas import BAND_ORDER, EngagementAssessment


class LockedTracer:
    """Thread-safe wrapper: one JSON line per request, never interleaved."""

    def __init__(self, tracer: JSONLTracer) -> None:
        self._tracer = tracer
        self._lock = threading.Lock()

    def log(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._tracer.log(record)

    def read_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._tracer.read_all()


def batch_summary(assessments: Sequence[EngagementAssessment]) -> dict[str, Any]:
    scored = [a for a in assessments if a.status == "scored"]
    n = len(scored)
    bands = Counter(a.risk_band for a in scored)
    return {
        "records": len(assessments),
        "scored": n,
        "not_eligible": len(assessments) - n,
        "band_shares": {band: (bands[band] / n if n else 0.0) for band in BAND_ORDER},
        "within_capacity": sum(bool(a.within_review_capacity) for a in scored),
    }