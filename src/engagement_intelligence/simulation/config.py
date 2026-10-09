"""
Simulator configuration, validated at load time.

Latent traits, archetypes, and every pathway's coefficients are declared in configuration. The set of time-varying
state variables a pathway may reference is fixed by the simulator's mechanics (constants below); validation rejects
anything else.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engagement_intelligence.core import ids
from engagement_intelligence.core.config import fingerprint, load_yaml_layers

_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,47}")

# Time-varying state variables available to pathways (fixed by simulator mechanics).
PROPENSITY_INPUTS = frozenset({"satisfaction", "disrupted"})
DISENGAGEMENT_INPUTS = PROPENSITY_INPUTS | {"propensity"}
RESERVED_NAMES = DISENGAGEMENT_INPUTS | {"dip", "disengaged"}

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

class LinearPredictor(_Frozen):
    intercept: float = 0.0
    coefficients: dict[str, float] = Field(default_factory=dict)

class ArchetypeDynamics(_Frozen):
    satisfaction_drift_per_year: float = 0.0
    dip_rate_per_year: float = Field(default=0.0, ge=0)
    dip_mean_duration_days: float = Field(default=21.0, ge=1)

class Archetype(_Frozen):
    weight: float = Field(gt=0)
    sd: float = Field(gt=0)
    means: dict[str, float]
    dynamics: ArchetypeDynamics = Field(default_factory=ArchetypeDynamics)

class SatisfactionDynamics(_Frozen):
    anchor_latent: str
    reversion: float = Field(gt=0, le=1)
    noise_sd: float = Field(ge=0)

class DisruptionProcess(_Frozen):
    onset: LinearPredictor      # daily logit; latents only
    mean_duration_days: float = Field(ge=1)

class DisengagementProcess(_Frozen):
    hazard: LinearPredictor     # daily logit, while engaged
    recovery: LinearPredictor   # daily logit, while disengaged

class Dynamics(_Frozen):
    satisfaction: SatisfactionDynamics
    disruption: DisruptionProcess
    propensity: LinearPredictor
    disengagement: DisengagementProcess

class SimConfig(_Frozen):
    seed: int = Field(ge=0)
    calendar: Calendar
    population: Population
    latents: tuple[str, ...]
    archetypes: dict[str, Archetype]
    dynamics: Dynamics

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
        clash = declared & RESERVED_NAMES
        if clash:
            raise ValueError(f"latent names clash with state variables: {sorted(clash)}")
        
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

        dyn = self.dynamics
        if dyn.satisfaction.anchor_latent not in declared:
            raise ValueError(
                f"dynamics.satisfaction.anchor_latent {dyn.satisfaction.anchor_latent!r} "
                "is not a declared latent"
            )
        allowed = {
            "dynamics.propensity": (dyn.propensity, declared | PROPENSITY_INPUTS),
            "dynamics.disruption.onset": (dyn.disruption.onset, declared),
            "dynamics.disengagement.hazard": (dyn.disengagement.hazard, declared | DISENGAGEMENT_INPUTS),
            "dynamics.disengagement.recovery": (dyn.disengagement.recovery, declared | DISENGAGEMENT_INPUTS)
        }
        for where, (predictor, names) in allowed.items():
            unknown = set(predictor.coefficients) - names
            if unknown:
                raise ValueError(f"{where}: unknown references {sorted(unknown)}")
        return self

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))

def load_sim_config(*paths: str | Path) -> SimConfig:
    return SimConfig.model_validate(load_yaml_layers(*paths))