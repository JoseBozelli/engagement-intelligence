from pathlib import Path

import pytest

from engagement_intelligence.core.config import load_yaml_layers
from engagement_intelligence.simulation.config import load_sim_config

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