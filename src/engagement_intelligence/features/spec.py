"""Declarative feature specification (business units; D-025).

Each feature names a kind, the roles it reads, its window parameters, a human-readable
description (used for downstream driver explanations), and whether it depends on the
operator's outreach policy.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engagement_intelligence.core.config import fingerprint, load_yaml_layers
from engagement_intelligence.core.roles import RoleMap

_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_KEY_COLUMNS = {"account_id", "assessed_at"}
_ROLE_FIELDS = ("role", "numerator", "denominator", "open_role", "close_role")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Feature(_Frozen):
    name: str
    description: str = Field(min_length=1)
    policy_dependent: bool = False


class RecencyFeature(_Feature):
    kind: Literal["recency"]
    role: str
    cap_days: float = Field(gt=0)


class CountFeature(_Feature):
    kind: Literal["count"]
    role: str
    window_days: int = Field(ge=1)


class ChangeFeature(_Feature):
    kind: Literal["change"]
    role: str
    short_days: int = Field(ge=1)
    long_days: int = Field(ge=1)
    pseudo_count: float = Field(default=0.5, gt=0)

    @model_validator(mode="after")
    def _windows(self) -> ChangeFeature:
        if self.short_days >= self.long_days:
            raise ValueError("short_days must be less than long_days")
        return self


class RateFeature(_Feature):
    kind: Literal["rate"]
    numerator: str
    denominator: str
    window_days: int = Field(ge=1)
    prior_rate: float = Field(ge=0, le=1)
    prior_weight: float = Field(gt=0)


class OverdueRatioFeature(_Feature):
    kind: Literal["overdue_ratio"]
    role: str
    lookback_days: int = Field(ge=1)
    cap: float = Field(gt=0)


class OutstandingFeature(_Feature):
    kind: Literal["outstanding"]
    open_role: str
    close_role: str
    min_age_days: float = Field(ge=0)


class TenureFeature(_Feature):
    kind: Literal["tenure"]


FeatureDef = Annotated[
    Union[RecencyFeature, CountFeature, ChangeFeature, RateFeature,
          OverdueRatioFeature, OutstandingFeature, TenureFeature],
    Field(discriminator="kind"),
]


class FeatureSpec(_Frozen):
    name: str
    features: tuple[FeatureDef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _names(self) -> FeatureSpec:
        names = [f.name for f in self.features]
        bad = [n for n in names if not _NAME_RE.fullmatch(n) or n in _KEY_COLUMNS]
        if bad:
            raise ValueError(f"invalid feature names: {bad}")
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate feature names: {duplicates}")
        return self

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def policy_dependent_names(self) -> list[str]:
        return [f.name for f in self.features if f.policy_dependent]

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))

    def roles_used(self) -> set[str]:
        return {getattr(f, field) for f in self.features for field in _ROLE_FIELDS if hasattr(f, field)}

    def validate_roles(self, roles: RoleMap) -> None:
        unknown = self.roles_used() - set(roles.selectors)
        if unknown:
            raise ValueError(f"feature spec references unknown roles: {sorted(unknown)}")


def load_feature_spec(*paths: str | Path, roles: RoleMap | None = None) -> FeatureSpec:
    spec = FeatureSpec.model_validate(load_yaml_layers(*paths))
    if roles is not None:
        spec.validate_roles(roles)
    return spec