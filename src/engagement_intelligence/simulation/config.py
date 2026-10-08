"""
Simulator configuration, validated at load time.

Latent traits and archetypes are declared in configuration, not code. Validation guarantees every archetype specifies
exactly the declared latents.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engagement_intelligence.core import ids
from engagement_intelligence.core.config import fingerprint, load_yaml_layers

_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,47}")

class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

class Calendar(_Frozen):
    start: date
    end: date

    @model_validator(mode="after")
    def _order(self) -> Calendar:
        if self.end <= self.start:
            raise ValueError("calendar.end must be after calendar.start")
        return self

class Population(_Frozen):
    n_accounts: int = Field(ge=1, le=999_999)
    tenant_id: str = "demo"
    initial_fraction: float = Field(ge=0.0, le=1.0)
    initial_window_days: int = Field(ge=1)
    arrivals_end: date

class Archetype(_Frozen):
    weight: float = Field(gt=0)
    sd: float = Field(gt=0)
    means: dict[str, float]

class SimConfig(_Frozen):
    seed: int = Field(ge=0)
    calendar: Calendar
    population: Population
    latents: tuple[str, ...]
    archetypes: dict[str, Archetype]

    @model_validator(mode="after")
    def _consistency(self) -> SimConfig:
        bad = [n for n in (*self.latents, *self.archetypes) if not _NAME_RE.fullmatch(n)]
        if bad:
            raise ValueError(f"invalid latent/archetype names: {bad}")
        if len(set(self.latents)) != len(self.latents):
            raise ValueError("duplicate latent names")
        if not self.latents or not self.archetypes:
            raise ValueError("at least one latent and one archetype are required.")

        declared = set(self.latents)
        for name, archetype in self.archetypes.items():
            missing = declared - set(archetype.means)
            extra = set(archetype.means) - declared
            if missing or extra:
                raise ValueError(
                    f"achetype {name!r}: missing latents {sorted(missing)}, "
                    f"undeclared latents {sorted(extra)}"
                )

        if not ids.is_valid_id(self.population.tenant_id):
            raise ValueError(f"invalid tenant_id: {self.population.tenant_id!r}")

        initial_end = self.calendar.start + timedelta(days=self.population.initial_window_days)
        if not initial_end < self.population.arrivals_end <= self.calendar.end:
            raise ValueError("arrivals_end must fall after the initial window and within the calendar")
        return self

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))

def load_sim_config(*paths: str | Path) -> SimConfig:
    return SimConfig.model_validate(load_yaml_layers(*paths))