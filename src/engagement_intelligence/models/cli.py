"""Model development on the development splits: tune, fit, compare, report.

Usage:
  uv run python -m engagement_intelligence.models.cli \
      --features data/reference/features/features.parquet \
      --assessments data/reference/labels/assessments.parquet \
      --labeling-config configs/labeling_reference.yaml --roles configs/roles_reference.yaml \
      --feature-spec configs/features_reference.yaml --config configs/models_reference.yaml \
      --report reports/model_comparison.md --record reports/experiments/model_comparison.json \
      --artifacts artifacts/model_development
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from engagement_intelligence.core.arrow_io import read_parquet_metadata, write_parquet
from engagement_intelligence.core.roles import load_roles
from engagement_intelligence.features.spec import load_feature_spec
from engagement_intelligence.labels.spec import load_labeling_config
from engagement_intelligence.models.config import load_modeling_config
from engagement_intelligence.models.data import load_dataset
from engagement_intelligence.models.report import evaluate_candidates, render_comparison
from engagement_intelligence.models.train import run_experiments
from engagement_intelligence.schemas.events import REFERENCE_VOCABULARY


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Model development on the development splits.")
    for flag in ("--features", "--assessments", "--report", "--record", "--artifacts"):
        parser.add_argument(flag, required=True, type=Path)
    for flag in ("--labeling-config", "--roles", "--feature-spec", "--config"):
        parser.add_argument(flag, action="append", required=True)
    args = parser.parse_args(argv)

    roles = load_roles(*args.roles, vocabulary=REFERENCE_VOCABULARY)
    labeling = load_labeling_config(*args.labeling_config, roles=roles, vocabulary=REFERENCE_VOCABULARY)
    spec = load_feature_spec(*args.feature_spec, roles=roles)
    cfg = load_modeling_config(*args.config, spec=spec)

    features_meta = read_parquet_metadata(args.features)
    labels_meta = read_parquet_metadata(args.assessments)
    if features_meta.get("features.spec_fingerprint") != spec.fingerprint:
        raise ValueError("features were built with a different feature spec")
    if features_meta.get("features.labels_fingerprint") != labeling.fingerprint:
        raise ValueError("features were built against different labels")
    if labels_meta.get("labels.config_fingerprint") != labeling.fingerprint:
        raise ValueError("assessments were built with a different labeling config")

    ds = load_dataset(pq.read_table(args.features), pq.read_table(args.assessments), spec.names)
    dev = ds.subset(np.isin(ds.split, [cfg.splits.train, cfg.splits.validation]))
    candidates = run_experiments(cfg, spec, dev, labeling.target.horizon_days)
    results = evaluate_candidates(cfg, candidates, dev)

    provenance = {
        "modeling config fingerprint": cfg.fingerprint,
        "feature spec fingerprint": spec.fingerprint,
        "labeling config fingerprint": labeling.fingerprint,
        "source run fingerprint": features_meta.get("features.source_fingerprint", ""),
    }
    report = render_comparison(cfg, candidates, results, provenance)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    record = {
        "provenance": provenance,
        "candidates": [{"name": c.name, "kind": c.kind, "feature_set": c.feature_set,
                        "features": c.feature_names, "params": c.params,
                        "cv_log_loss": c.cv_log_loss, "cv_results": c.cv_results} for c in candidates],
        "validation": results["metrics"],
        "comparisons": results["comparisons"],
        "selection": results["selection"],
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    args.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidates, args.artifacts / "candidates.joblib")
    val = dev.subset(dev.split == cfg.splits.validation)
    predictions = pa.table({
        "account_id": val.account_id.tolist(),
        "assessed_at": pa.array(val.assessed_us.astype("datetime64[us]"), pa.timestamp("us", tz="UTC")),
        "label": val.y,
        "cold_start": val.cold_start,
        **{name: p for name, p in results["predictions"].items()},
    })
    write_parquet(predictions, args.artifacts / "validation_predictions.parquet",
                  {k.replace(" ", "_"): v for k, v in provenance.items()})
    print(report)


if __name__ == "__main__":
    main()