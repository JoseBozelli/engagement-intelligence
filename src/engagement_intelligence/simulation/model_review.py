"""Model review on validation: calibration decision, operating point, and error analysis.

Evaluation harness for simulated runs. Truth (archetypes, latent disengagement) is used only
to explain errors; the operating point derives from observable predictions and labels.

Usage:
  uv run python -m engagement_intelligence.simulation.model_review \
      --run data/reference --labeling-config configs/labeling_reference.yaml \
      --roles configs/roles_reference.yaml --operating configs/operating_reference.yaml \
      --artifacts artifacts/model_development --record reports/experiments/model_comparison.json \
      --report reports/model_review.md --out reports/experiments/operating_point.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from engagement_intelligence.core.roles import load_roles
from engagement_intelligence.evaluation.metrics import expected_calibration_error, metric_suite, reliability_table
from engagement_intelligence.evaluation.slices import bucket_labels, bucket_names, slice_table
from engagement_intelligence.labels.build import column_us
from engagement_intelligence.labels.spec import load_labeling_config
from engagement_intelligence.models.operating import (
    assign_bands,
    calibration_line,
    choose_calibration,
    derive_operating_point,
    load_operating_config,
)
from engagement_intelligence.schemas.events import REFERENCE_VOCABULARY
from engagement_intelligence.simulation.diagnostics import annotate, load_run


def _f(x: float, d: int = 3) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.{d}f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Model review on validation (evaluation only).")
    for flag in ("--run", "--artifacts", "--record", "--report", "--out"):
        parser.add_argument(flag, required=True, type=Path)
    for flag in ("--labeling-config", "--roles", "--operating"):
        parser.add_argument(flag, action="append", required=True)
    args = parser.parse_args(argv)

    roles = load_roles(*args.roles, vocabulary=REFERENCE_VOCABULARY)
    labeling = load_labeling_config(*args.labeling_config, roles=roles, vocabulary=REFERENCE_VOCABULARY)
    op_cfg = load_operating_config(*args.operating)
    run = load_run(args.run, labeling)
    record = json.loads(args.record.read_text(encoding="utf-8"))
    if record["provenance"]["labeling config fingerprint"] != labeling.fingerprint:
        raise ValueError("model record was built against different labels")
    selected = record["selection"]["selected"]

    preds = pq.read_table(args.artifacts / "validation_predictions.parquet")
    features = pq.read_table(args.run / "features" / "features.parquet")
    if not features.column("account_id").equals(run.assessments.column("account_id")):
        raise ValueError("features are not aligned with assessments")
    keys = zip(run.assessments.column("account_id").to_pylist(), column_us(run.assessments, "assessed_at").tolist())
    row_of = {k: i for i, k in enumerate(keys)}
    rows = np.array([row_of[k] for k in zip(preds.column("account_id").to_pylist(),
                                            column_us(preds, "assessed_at").tolist())])

    annotated = annotate(labeling, run.accounts, run.episodes, run.observable, run.assessments)
    archetype = annotated["archetype"][rows]
    latent = annotated["latent_positive"][rows]
    tenure = features.column(op_cfg.tenure_feature).to_numpy()[rows]
    y = np.array(preds.column("label").to_pylist(), dtype=bool)
    cold = np.array(preds.column("cold_start").to_pylist(), dtype=bool)
    weeks = column_us(preds, "assessed_at")
    p_raw = preds.column(selected).to_numpy()

    calibration = choose_calibration(y, p_raw, op_cfg)
    p = calibration.apply(p_raw)
    op = derive_operating_point(p, op_cfg)
    bands = assign_bands(p, op)
    flagged = bands == "high"

    report, summary = _render(selected, record, op_cfg, calibration, op, y, p_raw, p, bands, flagged,
                              weeks, cold, tenure, archetype, latent)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    point = {
        "selected_model": selected,
        "calibration": asdict(calibration),
        "operating_point": asdict(op),
        "operating_config_fingerprint": op_cfg.fingerprint,
        "provenance": record["provenance"],
        "validation_summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(point, indent=2), encoding="utf-8")
    print(report)


def _render(selected, record, op_cfg, calibration, op, y, p_raw, p, bands, flagged,
            weeks, cold, tenure, archetype, latent) -> tuple[str, dict]:
    bins = op_cfg.calibration_bins
    slope, intercept = calibration_line(y, p_raw)
    ece_raw = expected_calibration_error(y, p_raw, bins)
    ece_cal = expected_calibration_error(y, p, bins)
    lines = ["# Model review (validation)", "",
             "Synthetic data for methodology evaluation only. Thresholds are demo values, not operating "
             "thresholds for any real deployment. The held-out test split was not used.", "",
             f"- Selected model: `{selected}`",
             f"- Operating config fingerprint: `{op_cfg.fingerprint}`"]
    lines += [f"- {k}: `{v}`" for k, v in record["provenance"].items()]

    lines += ["", "## 1. Calibration", "",
              f"- ECE (raw): {_f(ece_raw, 4)}; tolerance: {op_cfg.ece_tolerance}",
              f"- Calibration slope / intercept (raw): {_f(slope)} / {_f(intercept)} (ideal 1 / 0)",
              f"- Decision: **{calibration.method}**"
              + ("" if calibration.method == "none" else f" (slope {calibration.slope:.3f}, intercept {calibration.intercept:.3f})"),
              f"- ECE (after decision): {_f(ece_cal, 4)}", "",
              "| bin | mean predicted | observed | count |", "|---|---|---|---|"]
    for i, r in enumerate(reliability_table(y, p, bins), start=1):
        lines.append(f"| {i} | {_f(r['mean_predicted'])} | {_f(r['observed'])} | {r['count']} |")
    lines += ["", "Weekly calibration (mean predicted vs observed):", "",
              "| week | rows | mean predicted | observed |", "|---|---|---|---|"]
    for w in np.unique(weeks):
        m = weeks == w
        day = np.datetime64(int(w), "us").astype("datetime64[D]")
        lines.append(f"| {day} | {int(m.sum())} | {_f(p[m].mean())} | {_f(y[m].mean())} |")

    weekly_high = [flagged[weeks == w].mean() for w in np.unique(weeks)]
    weekly_watch = [(bands[weeks == w] != "low").mean() for w in np.unique(weeks)]
    per_k = {
        "flagged": 1000 * flagged.mean(),
        "lapses caught": 1000 * (flagged & y).mean(),
        "false alarms": 1000 * (flagged & ~y).mean(),
        "lapses missed": 1000 * (~flagged & y).mean(),
    }
    suite = metric_suite(y, p, weeks, (op.review_capacity,), bins)
    precision_high = float(y[flagged].mean())
    recall_high = float((y & flagged).sum() / y.sum())
    lines += ["", "## 2. Operating point (demo)", "",
              f"- HIGH: score >= {op.high_threshold:.4f} (review capacity {op.review_capacity:.0%} of eligible accounts per week)",
              f"- MEDIUM: score >= {op.medium_threshold:.4f} (watch capacity {op.watch_capacity:.0%})",
              f"- Realized weekly HIGH share: mean {np.mean(weekly_high):.3f}, range "
              f"{np.min(weekly_high):.3f}-{np.max(weekly_high):.3f}",
              f"- Realized weekly HIGH+MEDIUM share: mean {np.mean(weekly_watch):.3f}, range "
              f"{np.min(weekly_watch):.3f}-{np.max(weekly_watch):.3f}",
              f"- At HIGH: precision {precision_high:.3f}, recall {recall_high:.3f}, lift "
              f"{precision_high / y.mean():.2f}; PR-AUC {suite['pr_auc']:.3f}", "",
              "Per 1,000 eligible accounts per week:", "",
              "| flagged HIGH | lapses caught | false alarms | lapses missed |", "|---|---|---|---|",
              "| " + " | ".join(f"{v:.1f}" for v in per_k.values()) + " |", "",
              "False alarms consume review time and send outreach to accounts that would have stayed "
              "engaged; missed lapses are accounts that disengage without timely contact. The capacity "
              "setting trades one for the other."]

    tenure_buckets = bucket_labels(tenure, op_cfg.tenure_buckets_days)
    table = slice_table(
        y, p, flagged,
        {"cold_start": np.where(cold, "cold", "warm"), "tenure_days": tenure_buckets, "archetype (truth)": archetype},
        orders={"cold_start": ["warm", "cold"], "tenure_days": bucket_names(op_cfg.tenure_buckets_days)},
    )
    lines += ["", "## 3. Error analysis by slice", "",
              "| slice | value | rows | prevalence | mean predicted | gap | PR-AUC | HIGH share | precision | recall |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['slice']} | {r['value']} | {r['n']} | {_f(r['prevalence'])} | {_f(r['mean_predicted'])} "
                     f"| {_f(r['calibration_gap'])} | {_f(r['pr_auc'])} | {_f(r['flagged_share'])} "
                     f"| {_f(r['precision'])} | {_f(r['recall'])} |")

    false_alarm = flagged & ~y
    missed = ~flagged & y
    lines += ["", "## 4. HIGH flags vs latent truth (evaluation only)", "",
              f"- Label precision among HIGH: {_f(precision_high)}",
              f"- Latent precision among HIGH (latent-disengaged or onset within horizon): {_f(latent[flagged].mean())}",
              f"- Label false alarms that are latent-positive: {_f(latent[false_alarm].mean())} "
              f"({int((false_alarm & latent).sum())} of {int(false_alarm.sum())})", "",
              "| archetype | share of false alarms | share of missed lapses |", "|---|---|---|"]
    fa, ms = Counter(archetype[false_alarm].tolist()), Counter(archetype[missed].tolist())
    for name in sorted(set(archetype.tolist())):
        lines.append(f"| {name} | {_f(fa[name] / max(false_alarm.sum(), 1))} | {_f(ms[name] / max(missed.sum(), 1))} |")

    summary = {"ece_raw": ece_raw, "ece_after": ece_cal, "calibration_slope_raw": slope,
               "calibration_intercept_raw": intercept, "precision_high": precision_high,
               "recall_high": recall_high, "per_1000_weekly": per_k}
    return "\n".join(lines) + "\n", summary


if __name__ == "__main__":
    main()