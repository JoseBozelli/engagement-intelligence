"""Self-describing model bundle: the estimator plus everything needed to score raw events.

A bundle directory holds model.joblib and bundle.json. The manifest records the target
(eligibility rule), roles, feature spec, selected model, calibration, operating point,
training window, and provenance fingerprints. Loading verifies the estimator file's SHA-256
against the manifest. Joblib files can execute code when loaded: load bundles only from
trusted locations.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa

from engagement_intelligence.core.config import fingerprint
from engagement_intelligence.core.roles import RoleMap
from engagement_intelligence.features.spec import FeatureSpec
from engagement_intelligence.labels.spec import LabelingConfig
from engagement_intelligence.models.config import ModelingConfig
from engagement_intelligence.models.operating import Calibration, OperatingConfig, OperatingPoint, assign_bands
from engagement_intelligence.models.train import FittedCandidate

MODEL_FILE = "model.joblib"
MANIFEST_FILE = "bundle.json"
BUNDLE_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass
class ModelBundle:
    manifest: dict[str, Any]
    estimator: Any

    @property
    def version(self) -> str:
        return self.manifest["model_version"]

    @property
    def feature_names(self) -> list[str]:
        return list(self.manifest["features"])

    @property
    def calibration(self) -> Calibration:
        return Calibration(**self.manifest["calibration"])

    @property
    def operating_point(self) -> OperatingPoint:
        return OperatingPoint(**self.manifest["operating_point"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated lapse probability for a matrix whose columns follow feature_names."""
        return self.calibration.apply(self.estimator.predict_proba(X)[:, 1])

    def predict_table(self, features: pa.Table) -> np.ndarray:
        X = np.column_stack([features.column(n).to_numpy() for n in self.feature_names]).astype(float)
        return self.predict_proba(X)

    def bands(self, p: np.ndarray) -> np.ndarray:
        return assign_bands(p, self.operating_point)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, directory / MODEL_FILE)
        manifest = {**self.manifest, "model_sha256": _sha256(directory / MODEL_FILE)}
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> ModelBundle:
        directory = Path(directory)
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported bundle schema version {manifest.get('bundle_schema_version')!r}")
        if _sha256(directory / MODEL_FILE) != manifest.get("model_sha256"):
            raise ValueError("model file failed integrity check")
        return cls(manifest, joblib.load(directory / MODEL_FILE))


def build_bundle(candidate: FittedCandidate, point: dict[str, Any], spec: FeatureSpec, roles: RoleMap,
                 labeling: LabelingConfig, modeling: ModelingConfig, operating: OperatingConfig) -> ModelBundle:
    if point["selected_model"] != candidate.name:
        raise ValueError("operating point was derived for a different model")
    if point["provenance"].get("feature spec fingerprint") != spec.fingerprint:
        raise ValueError("operating point was derived with a different feature spec")
    if point.get("operating_config_fingerprint") != operating.fingerprint:
        raise ValueError("operating point was derived with a different operating config")
    train_window = labeling.splits.window(modeling.splits.train)
    content = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "target": labeling.target.model_dump(mode="json"),
        "roles": roles.to_dict(),
        "feature_spec": spec.model_dump(mode="json"),
        "features": list(candidate.feature_names),
        "policy_dependent_features_used": [n for n in candidate.feature_names if n in spec.policy_dependent_names],
        "model": {"name": candidate.name, "kind": candidate.kind,
                  "feature_set": candidate.feature_set, "params": candidate.params},
        "calibration": point["calibration"],
        "operating_point": point["operating_point"],
        "operating_config": operating.model_dump(mode="json"),
        "training": {"split": modeling.splits.train,
                     "start": train_window.start.isoformat(), "end": train_window.end.isoformat()},
        "provenance": {**point["provenance"],
                       "operating config fingerprint": operating.fingerprint,
                       "roles fingerprint": roles.fingerprint},
        "validation_summary": point["validation_summary"],
    }
    version = f"{_slug(labeling.target.name)}-{_slug(candidate.name)}-{fingerprint(content)[:10]}"
    return ModelBundle({**content, "model_version": version}, candidate.estimator)