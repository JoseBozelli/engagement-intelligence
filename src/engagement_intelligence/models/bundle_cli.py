"""Build the model bundle for the selected candidate.

Usage:
  uv run python -m engagement_intelligence.models.bundle_cli \
      --artifacts artifacts/model_development --point reports/experiments/operating_point.json \
      --labeling-config configs/labeling_reference.yaml --roles configs/roles_reference.yaml \
      --feature-spec configs/features_reference.yaml --config configs/models_reference.yaml \
      --operating configs/operating_reference.yaml --out-dir artifacts/models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from engagement_intelligence.core.roles import load_roles
from engagement_intelligence.features.spec import load_feature_spec
from engagement_intelligence.labels.spec import load_labeling_config
from engagement_intelligence.models.bundle import build_bundle
from engagement_intelligence.models.config import load_modeling_config
from engagement_intelligence.models.operating import load_operating_config
from engagement_intelligence.schemas.events import REFERENCE_VOCABULARY


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the model bundle for the selected candidate.")
    for flag in ("--artifacts", "--point", "--out-dir"):
        parser.add_argument(flag, required=True, type=Path)
    for flag in ("--labeling-config", "--roles", "--feature-spec", "--config", "--operating"):
        parser.add_argument(flag, action="append", required=True)
    args = parser.parse_args(argv)

    roles = load_roles(*args.roles, vocabulary=REFERENCE_VOCABULARY)
    labeling = load_labeling_config(*args.labeling_config, roles=roles, vocabulary=REFERENCE_VOCABULARY)
    spec = load_feature_spec(*args.feature_spec, roles=roles)
    modeling = load_modeling_config(*args.config, spec=spec)
    operating = load_operating_config(*args.operating)
    point = json.loads(args.point.read_text(encoding="utf-8"))
    if point["provenance"].get("modeling config fingerprint") != modeling.fingerprint:
        raise ValueError("operating point was derived from a different modeling run")

    candidates = joblib.load(args.artifacts / "candidates.joblib")
    candidate = next(c for c in candidates if c.name == point["selected_model"])
    bundle = build_bundle(candidate, point, spec, roles, labeling, modeling, operating)
    path = bundle.save(args.out_dir / bundle.version)

    op = bundle.operating_point
    print(f"model version: {bundle.version}")
    print(f"bundle path:   {path}")
    print(f"features ({len(bundle.feature_names)}): {', '.join(bundle.feature_names)}")
    print(f"policy-dependent features used: {bundle.manifest['policy_dependent_features_used'] or 'none'}")
    print(f"calibration: {bundle.calibration.method}")
    print(f"bands: HIGH >= {op.high_threshold:.4f} ({op.review_capacity:.0%}), "
          f"MEDIUM >= {op.medium_threshold:.4f} ({op.watch_capacity:.0%})")


if __name__ == "__main__":
    main()