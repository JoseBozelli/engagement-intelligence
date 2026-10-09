from pathlib import Path

import pytest
import numpy as np

from engagement_intelligence.core.config import load_yaml_layers
from engagement_intelligence.simulation.config import load_sim_config
from engagement_intelligence.labels.spec import load_labeling_config
from engagement_intelligence.schemas.events import REFERENCE_VOCABULARY
from engagement_intelligence.labels.build import build_assessments
from engagement_intelligence.simulation.diagnostics import RunArtifacts
from engagement_intelligence.simulation.events import simulate_world
from engagement_intelligence.simulation.noise import apply_noise
from engagement_intelligence.simulation.population import sample_population, utc_midnight
from engagement_intelligence.core.roles import load_roles
from engagement_intelligence.features.spec import load_feature_spec
from engagement_intelligence.models.config import load_modeling_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

@pytest.fixture(scope="session")
def reference_raw():
    return load_yaml_layers(CONFIG_DIR / "sim_reference.yaml")

@pytest.fixture(scope="session")
def reference_cfg():
    return load_sim_config(CONFIG_DIR / "sim_reference.yaml")

@pytest.fixture(scope="session")
def small_cfg():
    return load_sim_config(CONFIG_DIR / "sim_reference.yaml", CONFIG_DIR / "sim_small.yaml")

@pytest.fixture(scope="session")
def roles_cfg():
    return load_roles(CONFIG_DIR / "roles_reference.yaml", vocabulary=REFERENCE_VOCABULARY)


@pytest.fixture(scope="session")
def labeling_cfg(roles_cfg):
    return load_labeling_config(CONFIG_DIR / "labeling_reference.yaml",
                                roles=roles_cfg, vocabulary=REFERENCE_VOCABULARY)


@pytest.fixture(scope="session")
def features_spec(roles_cfg):
    return load_feature_spec(CONFIG_DIR / "features_reference.yaml", roles=roles_cfg)

@pytest.fixture(scope="session")
def small_pipeline(small_cfg, labeling_cfg):
    """Full small run: population -> world -> noise -> labels (built once per test session)."""
    accounts = sample_population(small_cfg)
    _, episodes, clean = simulate_world(small_cfg, accounts)
    observable, noise_log = apply_noise(small_cfg, accounts, clean)
    as_of = utc_midnight(small_cfg.calendar.end)
    return RunArtifacts(
        sim=small_cfg,
        accounts=accounts,
        episodes=episodes,
        noise_log=noise_log,
        clean_event_types=np.array(clean.column("event_type").to_pylist()),
        observable=observable,
        assessments=build_assessments(labeling_cfg, observable, as_of),
        as_of=as_of,
    )

@pytest.fixture(scope="session")
def modeling_cfg(features_spec):
    return load_modeling_config(CONFIG_DIR / "models_reference.yaml", spec=features_spec)