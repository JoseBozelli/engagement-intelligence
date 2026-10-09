"""Compute point-in-time features for labeled assessments.

Usage:
  uv run python -m engagement_intelligence.features.cli \
      --events data/reference/observable/events.parquet \
      --assessments data/reference/labels/assessments.parquet \
      --roles configs/roles_reference.yaml --spec configs/features_reference.yaml \
      --out data/reference/features/features.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from engagement_intelligence.core.arrow_io import check_schema, read_parquet_metadata, write_parquet
from engagement_intelligence.core.roles import load_roles
from engagement_intelligence.features.build import build_features
from engagement_intelligence.features.spec import FeatureSpec, load_feature_spec
from engagement_intelligence.schemas.events import REFERENCE_VOCABULARY


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute point-in-time features.")
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--assessments", required=True, type=Path)
    parser.add_argument("--roles", action="append", required=True)
    parser.add_argument("--spec", action="append", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    roles = load_roles(*args.roles, vocabulary=REFERENCE_VOCABULARY)
    spec = load_feature_spec(*args.spec, roles=roles)
    events = pq.read_table(args.events)
    check_schema(events)
    assessments = pq.read_table(args.assessments)

    features = build_features(spec, roles, events, assessments)
    metadata = {
        "features.spec_fingerprint": spec.fingerprint,
        "features.roles_fingerprint": roles.fingerprint,
        "features.policy_dependent": ",".join(spec.policy_dependent_names),
        "features.source_fingerprint": read_parquet_metadata(args.events).get("sim.config_fingerprint", ""),
        "features.labels_fingerprint": read_parquet_metadata(args.assessments).get("labels.config_fingerprint", ""),
    }
    write_parquet(features, args.out, metadata)
    _summarize(spec, features, assessments)


def _summarize(spec: FeatureSpec, features: pa.Table, assessments: pa.Table) -> None:
    labels = np.array(assessments.column("label").to_pylist())
    splits = np.array(assessments.column("split").to_pylist())
    split_names = list(dict.fromkeys(splits.tolist()))
    print(f"feature spec fingerprint: {spec.fingerprint}   rows: {features.num_rows}   "
          f"features: {len(spec.features)}")
    header = " ".join(f"{'AUC ' + s:>16}" for s in split_names)
    print(f"{'feature':<26} {'policy':>6} {'mean':>8} {'p50':>8} {'p95':>8} {header}")
    for f in spec.features:
        x = features.column(f.name).to_numpy()
        aucs = []
        for s in split_names:
            m = splits == s
            ok = 0 < labels[m].sum() < m.sum()
            aucs.append(roc_auc_score(labels[m], x[m]) if ok else float("nan"))
        print(f"{f.name:<26} {'yes' if f.policy_dependent else '':>6} {x.mean():8.2f} "
              f"{np.median(x):8.2f} {np.percentile(x, 95):8.2f} "
              + " ".join(f"{v:16.3f}" for v in aucs))


if __name__ == "__main__":
    main()