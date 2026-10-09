"""Model development configuration, validated against the feature specification."""

from __future__ import annotations

import itertools
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engagement_intelligence.core.config import fingerprint, load_yaml_layers
from engagement_intelligence.features.spec import FeatureSpec
from engagement_intelligence.models.estimators import KINDS


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DevelopmentSplits(_Frozen):
    train: str
    validation: str


class BaselineConfig(_Frozen):
    feature: str
    params: dict[str, Any] = Field(default_factory=dict)


class FeatureSet(_Frozen):
    exclude: tuple[str, ...] = ()
    exclude_policy_dependent: bool = False

    def resolve(self, spec: FeatureSpec) -> list[str]:
        unknown = set(self.exclude) - set(spec.names)
        if unknown:
            raise ValueError(f"feature set excludes unknown features: {sorted(unknown)}")
        dropped = set(self.exclude)
        if self.exclude_policy_dependent:
            dropped |= set(spec.policy_dependent_names)
        return [n for n in spec.names if n not in dropped]


class Fold(_Frozen):
    score_start: date
    score_end: date

    @model_validator(mode="after")
    def _order(self) -> Fold:
        if self.score_end <= self.score_start:
            raise ValueError("fold score_end must be after score_start")
        return self


class CandidateConfig(_Frozen):
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)

    def configurations(self) -> list[dict[str, Any]]:
        keys = sorted(self.grid)
        return [{**self.fixed, **dict(zip(keys, values))}
                for values in itertools.product(*(self.grid[k] for k in keys))]


class EvaluationConfig(_Frozen):
    capacities: tuple[float, ...] = Field(min_length=1)
    bootstrap_replicates: int = Field(ge=10)
    calibration_bins: int = Field(ge=2)

    @model_validator(mode="after")
    def _capacities(self) -> EvaluationConfig:
        if not all(0 < c < 1 for c in self.capacities):
            raise ValueError("capacities must be in (0, 1)")
        return self


class ModelingConfig(_Frozen):
    seed: int = Field(ge=0)
    splits: DevelopmentSplits
    baseline: BaselineConfig
    feature_sets: dict[str, FeatureSet] = Field(min_length=1)
    log1p_features: tuple[str, ...] = ()
    cv_folds: tuple[Fold, ...] = Field(min_length=1)
    candidates: dict[str, CandidateConfig] = Field(min_length=1)
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def _kinds(self) -> ModelingConfig:
        unknown = set(self.candidates) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown candidate kinds: {sorted(unknown)}")
        return self

    def validate_against(self, spec: FeatureSpec) -> None:
        missing = ({self.baseline.feature} | set(self.log1p_features)) - set(spec.names)
        if missing:
            raise ValueError(f"modeling config references unknown features: {sorted(missing)}")
        for feature_set in self.feature_sets.values():
            feature_set.resolve(spec)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


def load_modeling_config(*paths: str | Path, spec: FeatureSpec | None = None) -> ModelingConfig:
    cfg = ModelingConfig.model_validate(load_yaml_layers(*paths))
    if spec is not None:
        cfg.validate_against(spec)
    return cfg