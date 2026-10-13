"""Request and response models for the HTTP service."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from engagement_intelligence.scoring.schemas import EngagementAssessment


class AssessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: datetime | None = None
    account_ids: list[str] | None = Field(default=None, min_length=1)
    persist: bool = True


class BatchSummary(BaseModel):
    records: int
    scored: int
    not_eligible: int
    band_shares: dict[str, float]
    within_capacity: int


class AssessResponse(BaseModel):
    request_id: str
    model_version: str
    as_of: datetime
    summary: BatchSummary
    assessments: list[EngagementAssessment]


class FeatureInfo(BaseModel):
    name: str
    description: str


class ModelInfo(BaseModel):
    model_version: str
    model: dict[str, Any]
    features: list[FeatureInfo]
    policy_dependent_features_used: list[str]
    target: dict[str, Any]
    calibration: dict[str, Any]
    operating_point: dict[str, Any]
    training: dict[str, Any]
    provenance: dict[str, Any]
    validation_summary: dict[str, Any]