"""Engagement assessment contract, consumed by the API, the demo UI, agents, and downstream systems."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Band = Literal["low", "medium", "high"]
Transition = Literal["new", "unchanged", "escalated", "de-escalated"]
Status = Literal["scored", "not_eligible"]
BAND_ORDER = {"low": 0, "medium": 1, "high": 2}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Driver(_Frozen):
    feature: str
    description: str
    value: float          # raw feature value, in business units
    contribution: float   # log-odds contribution relative to an average training account


class RiskChange(_Frozen):
    transition: Transition
    previous_assessed_at: datetime | None = None
    previous_band: Band | None = None
    score_delta: float | None = None


class EngagementAssessment(_Frozen):
    account_id: str
    tenant_id: str
    assessed_at: datetime
    status: Status
    horizon_days: int = Field(ge=1)
    risk_score: float | None = Field(default=None, ge=0, le=1)
    risk_band: Band | None = None
    risk_change: RiskChange | None = None
    within_review_capacity: bool | None = None
    top_drivers: tuple[Driver, ...] = ()
    model_version: str

    @field_validator("assessed_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("assessed_at must be timezone-aware")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _consistency(self) -> EngagementAssessment:
        scored_fields = (self.risk_score, self.risk_band, self.risk_change)
        if self.status == "scored" and any(f is None for f in scored_fields):
            raise ValueError("scored assessments require risk_score, risk_band, and risk_change")
        if self.status == "not_eligible" and (any(f is not None for f in scored_fields) or self.top_drivers):
            raise ValueError("not_eligible assessments carry no score, band, change, or drivers")
        return self


def compare_bands(previous: EngagementAssessment | None, band: str, score: float) -> RiskChange:
    if previous is None:
        return RiskChange(transition="new")
    delta = BAND_ORDER[band] - BAND_ORDER[previous.risk_band]
    transition = "unchanged" if delta == 0 else ("escalated" if delta > 0 else "de-escalated")
    return RiskChange(transition=transition, previous_assessed_at=previous.assessed_at,
                      previous_band=previous.risk_band, score_delta=score - previous.risk_score)