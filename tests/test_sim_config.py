import copy
from datetime import date

import pytest
from pydantic import ValidationError

from engagement_intelligence.core.config import deep_merge
from engagement_intelligence.simulation.config import SimConfig


def test_reference_config_is_valid(reference_cfg):
    assert reference_cfg.population.n_accounts == 2000
    assert len(reference_cfg.archetypes) == 7
    assert abs(sum(a.weight for a in reference_cfg.archetypes.values()) - 1.0) < 1e-9


def test_small_overlay_changes_only_population_size(reference_cfg, small_cfg):
    assert small_cfg.population.n_accounts == 300
    assert small_cfg.archetypes == reference_cfg.archetypes
    assert small_cfg.fingerprint != reference_cfg.fingerprint


def test_archetype_missing_latent_rejected(reference_raw):
    raw = copy.deepcopy(reference_raw)
    del raw["archetypes"]["stable_high"]["means"]["persistence"]
    with pytest.raises(ValidationError, match="persistence"):
        SimConfig.model_validate(raw)


def test_archetype_undeclared_latent_rejected(reference_raw):
    raw = copy.deepcopy(reference_raw)
    raw["archetypes"]["stable_high"]["means"]["mood"] = 0.0
    with pytest.raises(ValidationError, match="mood"):
        SimConfig.model_validate(raw)


def test_arrivals_outside_calendar_rejected(reference_raw):
    raw = deep_merge(copy.deepcopy(reference_raw), {"population": {"arrivals_end": date(2027, 1, 1)}})
    with pytest.raises(ValidationError, match="arrivals_end"):
        SimConfig.model_validate(raw)


def test_unknown_keys_rejected(reference_raw):
    raw = deep_merge(copy.deepcopy(reference_raw), {"population": {"n_acounts": 10}})  # typo
    with pytest.raises(ValidationError):
        SimConfig.model_validate(raw)