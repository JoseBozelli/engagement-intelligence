"""One-time held-out evaluation of a model bundle (D-041).

The held-out split was not used for any modeling decision. A second run requires
--rerun-reason; every run and its reason are kept in the record.

Usage:
  uv run python -m engagement_intelligence.models.evaluate_cli \
      --bundle artifacts/models/<model_version> \
      --features data/reference/features/features.parquet \
      --assessments data/reference/labels/assessments.parquet \
      --candidates artifacts/model_development/candidates.joblib \
      --report reports/test_evaluation.md --record reports/experiments/test_evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

from engagement_intelligence.core.arrow_io import read_parquet_metadata
from engagement_intelligence.evaluation.metrics import binary_log_loss, clustered_bootstrap, expected_calibration_error, metric_suite
from engagement_intelligence.evaluation.slices import bucket_labels, bucket_names, slice_table
from engagement_intelligence.models.bundle import ModelBundle
from engagement_intelligence.models.data import load_dataset
from engagement_intelligence.models.operating import calibration_line

_KEYS = ("account_id", "assessed_at")


def check_first_run(record_path: Path, rerun_reason: str | None) -> list[dict]:
    """Return previous runs; refuse a repeat evaluation without a recorded reason."""
    if not record_path.exists():
        return []
    if not rerun_reason:
        raise ValueError(f"held-out evaluation already recorded in {record_path}; pass --rerun-reason to run again")
    return json.loads(record_path.read_text(encoding="utf-8"))["runs"]


def _f(x: float, d: int = 3) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.{d}f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One-time held-out evaluation of a model bundle.")
    for flag in ("--bundle", "--features", "--assessments", "--candidates", "--report", "--record"):
        parser.add_argument(flag, required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20241001)
    parser.add_argument("--rerun-reason", default=None)
    args = parser.parse_args(argv)

    previous = check_first_run(args.record, args.rerun_reason)
    bundle = ModelBundle.load(args.bundle)
    prov = bundle.manifest["provenance"]
    meta = read_parquet_metadata(args.features)
    if meta.get("features.spec_fingerprint") != prov["feature spec fingerprint"]:
        raise ValueError("features were built with a different feature spec than the bundle")
    if meta.get("features.labels_fingerprint") != prov["labeling config fingerprint"]:
        raise ValueError("features were built against different labels than the bundle")

    features = pq.read_table(args.features)
    names = [c for c in features.column_names if c not in _KEYS]
    ds = load_dataset(features, pq.read_table(args.assessments), names)
    test = ds.subset(ds.split == args.split)
    if not len(test.y):
        raise ValueError(f"no rows in split {args.split!r}")

    baseline = next(c for c in joblib.load(args.candidates) if c.name == "baseline")
    model_p = bundle.predict_proba(test.columns(bundle.feature_names))
    base_p = baseline.predict(test)
    op = bundle.operating_point
    ocfg = bundle.manifest["operating_config"]
    bins = ocfg["calibration_bins"]
    y, weeks, accounts = test.y, test.assessed_us, test.account_id

    def ci(stat) -> tuple[float, float]:
        return clustered_bootstrap(accounts, stat, args.bootstrap, args.seed)

    metrics = {}
    for name, p in (("model", model_p), ("baseline", base_p)):
        m = metric_suite(y, p, weeks, (op.review_capacity,), bins)
        m["log_loss_ci"] = ci(lambda i, p=p: binary_log_loss(y[i], p[i]))
        metrics[name] = m
    diff = binary_log_loss(y, model_p) - binary_log_loss(y, base_p)
    diff_ci = ci(lambda i: binary_log_loss(y[i], model_p[i]) - binary_log_loss(y[i], base_p[i]))
    slope, intercept = calibration_line(y, model_p)

    flagged = bundle.bands(model_p) == "high"
    week_ids = np.unique(weeks)
    weekly_high = np.array([flagged[weeks == w].mean() for w in week_ids])
    per_k = {"flagged": 1000 * flagged.mean(), "lapses caught": 1000 * (flagged & y).mean(),
             "false alarms": 1000 * (flagged & ~y).mean(), "lapses missed": 1000 * (~flagged & y).mean()}
    precision_high = float(y[flagged].mean())
    recall_high = float((y & flagged).sum() / y.sum())
    tenure = test.columns([ocfg["tenure_feature"]])[:, 0]
    edges = ocfg["tenure_buckets_days"]
    table = slice_table(
        y, model_p, flagged,
        {"cold_start": np.where(test.cold_start, "cold", "warm"), "tenure_days": bucket_labels(tenure, edges)},
        orders={"cold_start": ["warm", "cold"], "tenure_days": bucket_names(edges)},
    )

    cap = f"{op.review_capacity:.0%}"
    lines = ["# Held-out evaluation (test split)", "",
             "Synthetic data for methodology evaluation only. The held-out split was not used for any "
             "modeling decision; this evaluation runs once (reruns require a recorded reason).", "",
             f"- Bundle: `{bundle.version}`",
             f"- Run: {args.rerun_reason or 'first run'} (previous runs: {len(previous)})"]
    lines += [f"- {k}: `{v}`" for k, v in prov.items()]
    lines += ["", "## 1. Selected model vs recency baseline", "",
              f"| model | log loss [95% CI] | PR-AUC | ROC-AUC | Brier | ECE | precision@{cap} | recall@{cap} |",
              "|---|---|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        lo, hi = m["log_loss_ci"]
        lines.append(f"| {name} | {_f(m['log_loss'], 4)} [{lo:.4f}, {hi:.4f}] | {_f(m['pr_auc'])} | {_f(m['roc_auc'])} "
                     f"| {_f(m['brier'], 4)} | {_f(m['ece'], 4)} | {_f(m[f'precision@{cap}'])} | {_f(m[f'recall@{cap}'])} |")
    lines += ["", f"- Log loss difference (model - baseline): {diff:.4f} [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]",
              "", "## 2. Calibration on the held-out split", "",
              f"- ECE: {_f(expected_calibration_error(y, model_p, bins), 4)}; slope / intercept: "
              f"{_f(slope)} / {_f(intercept)} (ideal 1 / 0); calibration applied: {bundle.calibration.method}", "",
              "| week | rows | mean predicted | observed |", "|---|---|---|---|"]
    for w in week_ids:
        m = weeks == w
        lines.append(f"| {np.datetime64(int(w), 'us').astype('datetime64[D]')} | {int(m.sum())} "
                     f"| {_f(model_p[m].mean())} | {_f(y[m].mean())} |")
    lines += ["", "## 3. Validation-derived operating point applied unchanged (demo)", "",
              f"- HIGH: score >= {op.high_threshold:.4f} (set for {cap} weekly review capacity on validation)",
              f"- Realized weekly HIGH share: mean {weekly_high.mean():.3f}, range "
              f"{weekly_high.min():.3f}-{weekly_high.max():.3f}",
              f"- At HIGH: precision {precision_high:.3f}, recall {recall_high:.3f}, lift {precision_high / y.mean():.2f}",
              "", "Per 1,000 eligible accounts per week:", "",
              "| flagged HIGH | lapses caught | false alarms | lapses missed |", "|---|---|---|---|",
              "| " + " | ".join(f"{v:.1f}" for v in per_k.values()) + " |",
              "", "## 4. Observable slices", "",
              "| slice | value | rows | prevalence | mean predicted | gap | PR-AUC | HIGH share | precision | recall |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['slice']} | {r['value']} | {r['n']} | {_f(r['prevalence'])} | {_f(r['mean_predicted'])} "
                     f"| {_f(r['calibration_gap'])} | {_f(r['pr_auc'])} | {_f(r['flagged_share'])} "
                     f"| {_f(r['precision'])} | {_f(r['recall'])} |")
    report = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    run = {"reason": args.rerun_reason or "first run", "bundle": bundle.version, "split": args.split,
           "metrics": metrics, "log_loss_diff_vs_baseline": diff, "log_loss_diff_ci": diff_ci,
           "calibration_slope": slope, "calibration_intercept": intercept,
           "weekly_high_share": {"mean": weekly_high.mean(), "min": weekly_high.min(), "max": weekly_high.max()},
           "precision_high": precision_high, "recall_high": recall_high, "per_1000_weekly": per_k}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps({"runs": previous + [run]}, indent=2, default=float), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()