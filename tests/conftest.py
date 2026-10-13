from pathlib import Path
from dataclasses import asdict

import pytest
import numpy as np
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from engagement_intelligence.service.app import create_app
from engagement_intelligence.service.settings import ServiceSettings
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
from engagement_intelligence.features.build import build_features
from engagement_intelligence.models.bundle import build_bundle
from engagement_intelligence.models.data import load_dataset
from engagement_intelligence.models.operating import OperatingConfig, derive_operating_point
from engagement_intelligence.models.train import FittedCandidate, fit_final

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

@pytest.fixture(scope="session")
def small_bundle(small_pipeline, roles_cfg, features_spec, labeling_cfg, modeling_cfg):
    """A real bundle trained on the small run (logistic, no_policy), plus its feature table."""
    features = build_features(features_spec, roles_cfg, small_pipeline.observable, small_pipeline.assessments)
    ds = load_dataset(features, small_pipeline.assessments, features_spec.names)
    names = modeling_cfg.feature_sets["no_policy"].resolve(features_spec)
    estimator = fit_final("logistic", {"C": 0.01, "max_iter": 2000}, None,
                          ds.subset(ds.split == modeling_cfg.splits.train), names,
                          modeling_cfg.log1p_features, modeling_cfg.seed)
    candidate = FittedCandidate("logistic[no_policy]", "logistic", "no_policy", names, {"C": 0.01}, None, [], estimator)
    operating = OperatingConfig(review_capacity=0.1, watch_capacity=0.25, ece_tolerance=0.02,
                                calibration_bins=10, tenure_feature="tenure_days", tenure_buckets_days=(90, 180, 365))
    validation = ds.subset(ds.split == modeling_cfg.splits.validation)
    point = {
        "selected_model": candidate.name,
        "calibration": {"method": "none", "slope": 1.0, "intercept": 0.0},
        "operating_point": asdict(derive_operating_point(candidate.predict(validation), operating)),
        "operating_config_fingerprint": operating.fingerprint,
        "provenance": {"feature spec fingerprint": features_spec.fingerprint},
        "validation_summary": {},
    }
    bundle = build_bundle(candidate, point, features_spec, roles_cfg, labeling_cfg, modeling_cfg, operating)
    return bundle, features

@pytest.fixture
def service_client(tmp_path, small_bundle, small_pipeline):
    """In-process HTTP client for a service built from the small bundle and small run."""
    from datetime import datetime, timezone

    bundle, _ = small_bundle
    events = tmp_path / "events.parquet"
    pq.write_table(small_pipeline.observable, events)
    settings = ServiceSettings(bundle_dir=bundle.save(tmp_path / "bundle"), events_path=events,
                               store_path=tmp_path / "store.jsonl", trace_path=tmp_path / "trace.jsonl",
                               default_as_of=datetime(2025, 12, 1, tzinfo=timezone.utc))
    return TestClient(create_app(settings))