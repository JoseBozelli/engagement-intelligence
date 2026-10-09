"""Build point-in-time assessments and labels from an observable event store.

Usage:
  uv run python -m engagement_intelligence.labels.cli \
      --events data/reference/observable/events.parquet \
      --config configs/labeling_reference.yaml --as-of 2026-07-01 \
      --out data/reference/labels/assessments.parquet
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from engagement_intelligence.core.arrow_io import check_schema, read_parquet_metadata, write_parquet
from engagement_intelligence.labels.build import build_assessments
from engagement_intelligence.labels.spec import LabelingConfig, load_labeling_config
from engagement_intelligence.schemas.events import REFERENCE_VOCABULARY


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build point-in-time assessments and labels.")
    parser.add_argument("--events", required=True, type=Path, help="observable events parquet")
    parser.add_argument("--config", action="append", required=True, help="labeling YAML layer(s)")
    parser.add_argument("--as-of", required=True, help="snapshot date (YYYY-MM-DD, 00:00 UTC)")
    parser.add_argument("--out", required=True, type=Path, help="output parquet path")
    args = parser.parse_args(argv)

    cfg = load_labeling_config(*args.config, vocabulary=REFERENCE_VOCABULARY)
    events = pq.read_table(args.events)
    check_schema(events)
    as_of = datetime.combine(date.fromisoformat(args.as_of), time(), tzinfo=timezone.utc)

    table = build_assessments(cfg, events, as_of)
    source = read_parquet_metadata(args.events)
    metadata = {
        "labels.config_fingerprint": cfg.fingerprint,
        "labels.as_of": as_of.isoformat(),
        "labels.source_fingerprint": source.get("sim.config_fingerprint", ""),
    }
    write_parquet(table, args.out, metadata)
    _summarize(cfg, table)


def _summarize(cfg: LabelingConfig, table: pa.Table) -> None:
    splits = np.array(table.column("split").to_pylist())
    labels = np.array(table.column("label").to_pylist())
    cold = np.array(table.column("cold_start").to_pylist())
    accounts = np.array(table.column("account_id").to_pylist())
    times = table.column("assessed_at").to_numpy()

    def rate(mask: np.ndarray) -> float:
        return labels[mask].mean() if mask.any() else float("nan")

    print(f"labeling config fingerprint: {cfg.fingerprint}")
    print(f"assessments: {table.num_rows}   accounts: {len(set(accounts.tolist()))}   "
          f"overall prevalence: {labels.mean():.3f}")
    print(f"{'split':<12} {'rows':>7} {'accounts':>9} {'weeks':>6} {'prevalence':>11} "
          f"{'cold_rows':>10} {'cold_prev':>10}")
    for w in cfg.splits.windows:
        m = splits == w.name
        print(f"{w.name:<12} {int(m.sum()):7d} {len(set(accounts[m].tolist())):9d} "
              f"{len(set(times[m].tolist())):6d} {rate(m):11.3f} "
              f"{int((m & cold).sum()):10d} {rate(m & cold):10.3f}")


if __name__ == "__main__":
    main()