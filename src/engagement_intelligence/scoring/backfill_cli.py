"""Score every weekly assessment time in a window and persist the history (demo backfill).

Optionally verifies that live scores equal the bundle's predictions on precomputed feature
rows for the same (account, week), an end-to-end consistency check.

Usage:
  uv run python -m engagement_intelligence.scoring.backfill_cli \
      --bundle artifacts/models/<version> --events data/reference/observable/events.parquet \
      --store data/reference/scoring/assessments.jsonl --start 2025-12-01 --end 2026-07-01 \
      --check-features data/reference/features/features.parquet --overwrite
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from engagement_intelligence.labels.build import assessment_grid, column_us, to_us
from engagement_intelligence.models.bundle import ModelBundle
from engagement_intelligence.scoring.engine import Scorer, from_us
from engagement_intelligence.scoring.sources import TableEventSource
from engagement_intelligence.scoring.store import JsonlAssessmentStore


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill weekly assessments into a store.")
    for flag in ("--bundle", "--events", "--store"):
        parser.add_argument(flag, required=True, type=Path)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--check-features", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.store.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.store} exists; pass --overwrite to rebuild it")
        args.store.unlink()

    bundle = ModelBundle.load(args.bundle)
    scorer = Scorer(bundle, TableEventSource.from_parquet(args.events), JsonlAssessmentStore(args.store))
    print(f"model version: {bundle.version}")
    print(f"{'week':<12} {'scored':>7} {'not_elig':>9} {'high':>6} {'medium':>7} "
          f"{'escalated':>10} {'de-escal':>9} {'in_capacity':>12}")
    scores: dict[tuple[str, int], float] = {}
    for t in assessment_grid(scorer.target, args.start, args.end):
        as_of = from_us(int(t))
        results = scorer.assess(as_of)
        scored = [a for a in results if a.status == "scored"]
        bands = Counter(a.risk_band for a in scored)
        moves = Counter(a.risk_change.transition for a in scored)
        n = max(len(scored), 1)
        print(f"{as_of.date()!s:<12} {len(scored):7d} {len(results) - len(scored):9d} "
              f"{bands['high'] / n:6.3f} {bands['medium'] / n:7.3f} {moves['escalated']:10d} "
              f"{moves['de-escalated']:9d} {sum(bool(a.within_review_capacity) for a in scored):12d}")
        scores.update({(a.account_id, int(t)): a.risk_score for a in scored})

    if args.check_features:
        features = pq.read_table(args.check_features)
        weeks = set(int(t) for t in assessment_grid(scorer.target, args.start, args.end))
        times = column_us(features, "assessed_at")
        rows = features.filter(np.isin(times, list(weeks)))
        keys = list(zip(rows.column("account_id").to_pylist(), column_us(rows, "assessed_at").tolist()))
        missing = [k for k in keys if k not in scores]
        expected = bundle.predict_table(rows)
        diffs = [abs(scores[k] - e) for k, e in zip(keys, expected) if k in scores]
        print(f"consistency: {len(keys)} feature rows checked, {len(missing)} missing, "
              f"max |score difference| = {max(diffs, default=0.0):.2e}")
        if missing or max(diffs, default=0.0) > 1e-9:
            raise SystemExit("live scores disagree with bundle predictions on precomputed features")


if __name__ == "__main__":
    main()