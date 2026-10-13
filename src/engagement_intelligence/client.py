"""Typed client for the Engagement Intelligence HTTP service.

Used by the demo UI and the reference shape for downstream agent tools
(get_engagement_assessment, get_recent_events, get_risk_history). Defaults to read-only:
assess() does not persist unless asked to; scheduled batch jobs own persistence.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import httpx

from engagement_intelligence.core.envelope import EventEnvelope
from engagement_intelligence.scoring.engine import UnknownAccountError
from engagement_intelligence.scoring.schemas import EngagementAssessment
from engagement_intelligence.service.schemas import AssessResponse, ModelInfo


class EngagementClient:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    @classmethod
    def connect(cls, base_url: str, timeout: float = 60.0) -> EngagementClient:
        return cls(httpx.Client(base_url=base_url, timeout=timeout))

    def close(self) -> None:
        self._http.close()

    def _send(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._http.request(method, path, **kwargs)
        if response.status_code == 404:
            raise UnknownAccountError(response.json().get("detail", path))
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _params(**values: Any) -> dict[str, Any]:
        return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in values.items() if v is not None}

    def health(self) -> dict[str, str]:
        return self._send("GET", "/health")

    def model_info(self) -> ModelInfo:
        return ModelInfo.model_validate(self._send("GET", "/model-info"))

    def assess(self, as_of: datetime | None = None, account_ids: Sequence[str] | None = None,
               persist: bool = False) -> AssessResponse:
        body: dict[str, Any] = {"persist": persist}
        if as_of is not None:
            body["as_of"] = as_of.isoformat()
        if account_ids is not None:
            body["account_ids"] = list(account_ids)
        return AssessResponse.model_validate(self._send("POST", "/assess", json=body))

    def assessment(self, account_id: str, as_of: datetime | None = None) -> EngagementAssessment:
        data = self._send("GET", f"/accounts/{account_id}/assessment", params=self._params(as_of=as_of))
        return EngagementAssessment.model_validate(data)

    def events(self, account_id: str, as_of: datetime | None = None, limit: int = 20) -> list[EventEnvelope]:
        data = self._send("GET", f"/accounts/{account_id}/events", params=self._params(as_of=as_of, limit=limit))
        return [EventEnvelope.model_validate(e) for e in data]

    def risk_history(self, account_id: str) -> list[EngagementAssessment]:
        data = self._send("GET", f"/accounts/{account_id}/risk-history")
        return [EngagementAssessment.model_validate(a) for a in data]