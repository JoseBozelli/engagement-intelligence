"""Target and split specifications: deployment configuration in business units (D-025).

The target (engagement definition, horizon, eligibility, settlement, cadence) is what a
production scorer needs. The split plan is used only for model development. Event types
and actors are plain strings, validated against whichever EventVocabulary is in use.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engagement_intelligence.core.config import deep_merge, fingerprint, load_yaml_layers
from engagement_intelligence.core.roles import RoleMap
from engagement_intelligence.core.vocabulary import EventVocabulary

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
Weekday = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EventSelector(_Frozen):
    event_type: str
    actor: str


class TargetSpec(_Frozen):
    name: str
    engagement_role: str
    engagement_events: tuple[EventSelector, ...] = Field(min_length=1)
    horizon_days: int = Field(ge=1)
    eligibility_window_days: int = Field(ge=1)
    label_settlement_days: int = Field(ge=0)
    assessment_weekday: Weekday

    def validate_against(self, vocabulary: EventVocabulary) -> None:
        for s in self.engagement_events:
            if s.event_type not in vocabulary.allowed_actors:
                raise ValueError(f"engagement event type {s.event_type!r} not in vocabulary {vocabulary.name!r}")
            if s.actor not in vocabulary.allowed_actors[s.event_type]:
                raise ValueError(f"actor {s.actor!r} cannot emit {s.event_type!r}")


class SplitWindow(_Frozen):
    name: str
    start: date
    end: date

    @model_validator(mode="after")
    def _order(self) -> SplitWindow:
        if self.end <= self.start:
            raise ValueError(f"split {self.name!r}: end must be after start")
        return self


class SplitPlan(_Frozen):
    windows: tuple[SplitWindow, ...] = Field(min_length=1)
    cold_start_after: str

    @model_validator(mode="after")
    def _consistency(self) -> SplitPlan:
        names = [w.name for w in self.windows]
        if len(set(names)) != len(names):
            raise ValueError("split names must be unique")
        for prev, nxt in zip(self.windows, self.windows[1:]):
            if nxt.start < prev.end:
                raise ValueError(f"split {nxt.name!r} overlaps or precedes {prev.name!r}")
        if self.cold_start_after not in names:
            raise ValueError(f"cold_start_after {self.cold_start_after!r} is not a split")
        return self

    def window(self, name: str) -> SplitWindow:
        return next(w for w in self.windows if w.name == name)


class LabelingConfig(_Frozen):
    target: TargetSpec
    splits: SplitPlan

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


def load_labeling_config(*paths: str | Path, roles: RoleMap,
                         vocabulary: EventVocabulary | None = None) -> LabelingConfig:
    """Load labeling config, resolving target.engagement_role into concrete selectors."""
    raw = load_yaml_layers(*paths)
    target = raw.get("target", {})
    if "engagement_events" in target:
        raise ValueError("define engagement through target.engagement_role, not inline events")
    role = target.get("engagement_role")
    if role not in roles.selectors:
        raise ValueError(f"target.engagement_role {role!r} is not a defined role")
    raw = deep_merge(raw, {"target": {"engagement_events": roles.to_dict()[role]}})
    cfg = LabelingConfig.model_validate(raw)
    if vocabulary is not None:
        cfg.target.validate_against(vocabulary)
    return cfg