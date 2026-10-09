"""Day-1 checkpoint diagnostics: observable labels compared with simulator truth.

Evaluation only. This module reads truth tables, which nothing on the modeling path may
do, and writes a markdown report. Provenance is checked so a report cannot mix runs.

Usage:
  uv run python -m engagement_intelligence.simulation.diagnostics \
      --run data/reference --labeling-config configs/labeling_reference.yaml \
      --out reports/day1_checkpoint.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from engagement_intelligence.core.arrow_io import read_parquet_metadata
from engagement_intelligence.labels.build import DAY_US, column_us, engagement_blocks, latest_known
from engagement_intelligence.labels.spec import LabelingConfig, load_labeling_config
from engagement_intelligence.schemas.events import REFERENCE_VOCABULARY
from engagement_intelligence.simulation.config import SimConfig

_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
_OPEN_END = np.iinfo(np.int64).max


@dataclass(frozen=True)
class RunArtifacts:
    sim: SimConfig
    accounts: pa.Table
    episodes: pa.Table
    noise_log: pa.Table
    clean_event_types: np.ndarray
    observable: pa.Table
    assessments: pa.Table
    as_of: datetime


def check_provenance(labels_meta: Mapping[str, str], sim_fingerprint: str, labeling: LabelingConfig) -> None:
    if labels_meta.get("labels.config_fingerprint") != labeling.fingerprint:
        raise ValueError("assessments were built with a different labeling config")
    if labels_meta.get("labels.source_fingerprint") != sim_fingerprint:
        raise ValueError("assessments were built from a different simulation run")


def load_run(run_dir: Path, labeling: LabelingConfig) -> RunArtifacts:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    sim = SimConfig.model_validate(manifest["config"])
    if sim.fingerprint != manifest["config_fingerprint"]:
        raise ValueError("manifest config does not match its recorded fingerprint")
    labels_path = run_dir / "labels" / "assessments.parquet"
    meta = read_parquet_metadata(labels_path)
    check_provenance(meta, sim.fingerprint, labeling)
    truth = run_dir / "truth"
    clean_types = pq.read_table(truth / "events_clean.parquet", columns=["event_type"])
    return RunArtifacts(
        sim=sim,
        accounts=pq.read_table(truth / "accounts.parquet"),
        episodes=pq.read_table(truth / "disengagement_episodes.parquet"),
        noise_log=pq.read_table(truth / "noise_log.parquet"),
        clean_event_types=np.array(clean_types.column("event_type").to_pylist()),
        observable=pq.read_table(run_dir / "observable" / "events.parquet"),
        assessments=pq.read_table(labels_path),
        as_of=datetime.fromisoformat(meta["labels.as_of"]),
    )


def annotate(labeling: LabelingConfig, accounts: pa.Table, episodes: pa.Table,
             observable: pa.Table, assessments: pa.Table) -> dict[str, np.ndarray]:
    """Per-assessment diagnostics: point-in-time heuristic score and latent truth."""
    acc = np.array(assessments.column("account_id").to_pylist())
    if len(acc) and not np.all(acc[:-1] <= acc[1:]):
        raise ValueError("assessments must be sorted by account_id")
    t_us = column_us(assessments, "assessed_at")
    n = len(acc)
    unique, starts, counts = np.unique(acc, return_index=True, return_counts=True)
    rows_of = {str(a): slice(int(s), int(s + k)) for a, s, k in zip(unique, starts, counts)}

    days_since = np.full(n, np.nan)
    for account, occurred, recorded in engagement_blocks(labeling.target, observable):
        rows = rows_of.get(account)
        if rows is not None:
            days_since[rows] = (t_us[rows] - latest_known(occurred, recorded, t_us[rows])) / DAY_US

    t_ord = t_us // DAY_US + _EPOCH_ORDINAL
    horizon = labeling.target.horizon_days
    spans_of: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for a, s, e in zip(episodes.column("account_id").to_pylist(),
                       episodes.column("start").to_pylist(),
                       episodes.column("end").to_pylist()):
        spans_of[a].append((s.toordinal(), _OPEN_END if e is None else e.toordinal()))
    disengaged = np.zeros(n, dtype=bool)
    onset = np.zeros(n, dtype=bool)
    for account, spans in spans_of.items():
        rows = rows_of.get(account)
        if rows is None:
            continue
        s = np.array([x[0] for x in spans])
        e = np.array([x[1] for x in spans])
        d = t_ord[rows][:, None]
        disengaged[rows] = ((s <= d) & (d < e)).any(axis=1)
        onset[rows] = ((s > d) & (s <= d + horizon)).any(axis=1)

    archetype_of = dict(zip(accounts.column("account_id").to_pylist(), accounts.column("archetype").to_pylist()))
    return {
        "account_id": acc,
        "split": np.array(assessments.column("split").to_pylist()),
        "cold_start": np.array(assessments.column("cold_start").to_pylist()),
        "label": np.array(assessments.column("label").to_pylist()),
        "archetype": np.array([archetype_of[a] for a in acc]),
        "days_since_engagement": days_since,
        "latent_positive": disengaged | onset,
    }


def _rate(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else float("nan")


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if 0 < labels.sum() < labels.size else float("nan")


def _auc_reading(auc: float) -> str:
    if np.isnan(auc):
        return "n/a"
    if auc > 0.90:
        return "investigate: possibly trivial"
    if auc < 0.60:
        return "investigate: weak heuristic signal"
    return "plausible"


def _f(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{x:.3f}"


def render_report(run: RunArtifacts, labeling: LabelingConfig, a: Mapping[str, np.ndarray]) -> str:
    splits = [w.name for w in labeling.splits.windows]
    eval_split = splits[1] if len(splits) > 1 else splits[0]
    archetypes = list(run.sim.archetypes)
    label, cold, latent = a["label"], a["cold_start"], a["latent_positive"]
    split, arch, days = a["split"], a["archetype"], a["days_since_engagement"]

    lines = [
        "# Day-1 checkpoint report", "",
        "Synthetic data for methodology evaluation only; values are not estimates of any real population.", "",
        f"- Simulation config fingerprint: `{run.sim.fingerprint}`",
        f"- Labeling config fingerprint: `{labeling.fingerprint}`",
        f"- Snapshot (as_of): {run.as_of.isoformat()}",
        f"- Assessments: {label.size} rows, {len(set(a['account_id'].tolist()))} accounts", "",
        "## 1. Observable lapse prevalence (diagnostic, D-007)", "",
        "| split | rows | prevalence | warm prevalence | cold-start rows | cold-start prevalence |",
        "|---|---|---|---|---|---|",
    ]
    for s in splits:
        m = split == s
        lines.append(f"| {s} | {int(m.sum())} | {_f(_rate(label[m]))} | {_f(_rate(label[m & ~cold]))} "
                     f"| {int((m & cold).sum())} | {_f(_rate(label[m & cold]))} |")
    lines += ["", "By archetype (truth; evaluation only):", "",
              "| archetype | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
    for name in archetypes:
        cells = [_f(_rate(label[(arch == name) & (split == s)])) for s in splits]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## 2. Heuristic diagnostic: days since last known engagement (D-008)", "",
              "| split | AUC all | AUC warm | AUC cold-start | reading |", "|---|---|---|---|---|"]
    aucs = {}
    for s in splits:
        m = split == s
        aucs[s] = _auc(label[m], days[m])
        lines.append(f"| {s} | {_f(aucs[s])} | {_f(_auc(label[m & ~cold], days[m & ~cold]))} "
                     f"| {_f(_auc(label[m & cold], days[m & cold]))} | {_auc_reading(aucs[s])} |")

    lines += ["", "## 3. Observable label vs latent truth", "",
              "Latent positive: latent-disengaged at assessment, or latent onset within the horizon.", "",
              "| group | rows | label rate | latent+ rate | P(label \\| latent+) | P(label \\| latent-) "
              "| P(latent+ \\| label) |", "|---|---|---|---|---|---|---|"]
    groups = [("all", np.ones(label.size, dtype=bool))] + [(name, arch == name) for name in archetypes]
    for name, g in groups:
        lines.append(f"| {name} | {int(g.sum())} | {_f(_rate(label[g]))} | {_f(_rate(latent[g]))} "
                     f"| {_f(_rate(label[g & latent]))} | {_f(_rate(label[g & ~latent]))} "
                     f"| {_f(_rate(latent[g & label]))} |")

    nz = run.sim.noise
    exempt = np.isin(run.clean_event_types, list(nz.exempt_types))
    log = run.noise_log
    measured = {
        "drop": _rate(np.array(log.column("dropped").to_pylist())[~exempt]),
        "delay": _rate((log.column("delay_hours").to_numpy() > 0)[~exempt]),
        "duplicate": _rate(np.array(log.column("duplicated").to_pylist())[~exempt]),
    }
    configured = {"drop": nz.drop_prob, "delay": nz.delay_prob, "duplicate": nz.duplicate_prob}
    tolerance = {"drop": 0.005, "delay": 0.01, "duplicate": 0.002}
    lines += ["", "## 4. Observation noise: measured vs configured", "",
              "| mechanism | configured | measured |", "|---|---|---|"]
    lines += [f"| {k} | {configured[k]:.3f} | {measured[k]:.4f} |" for k in configured]

    overall = _rate(label)
    checks = [
        ("All archetypes present in assessments", set(archetypes) <= set(arch.tolist())),
        ("Overall prevalence within 10-25% (diagnostic, D-007)", 0.10 <= overall <= 0.25),
        (f"Heuristic AUC on '{eval_split}' within 0.60-0.90 (diagnostic, D-008)",
         0.60 <= aucs[eval_split] <= 0.90),
        ("Label noise measurable: P(label \\| latent-) > 0 and P(label \\| latent+) < 1",
         _rate(label[~latent]) > 0 and _rate(label[latent]) < 1),
        ("Noise rates within tolerance of configuration",
         all(abs(measured[k] - configured[k]) <= tolerance[k] for k in configured)),
    ]
    lines += ["", "## 5. Day-1 stop criteria", "", "| criterion | status |", "|---|---|"]
    lines += [f"| {name} | {'PASS' if ok else 'REVIEW'} |" for name, ok in checks]
    lines.append("| Determinism and leakage controls | covered by test suite |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Day-1 checkpoint report (evaluation only).")
    parser.add_argument("--run", required=True, type=Path, help="simulation run directory")
    parser.add_argument("--labeling-config", action="append", required=True, help="labeling YAML layer(s)")
    parser.add_argument("--out", required=True, type=Path, help="markdown report path")
    args = parser.parse_args(argv)

    labeling = load_labeling_config(*args.labeling_config, vocabulary=REFERENCE_VOCABULARY)
    run = load_run(args.run, labeling)
    annotated = annotate(labeling, run.accounts, run.episodes, run.observable, run.assessments)
    report = render_report(run, labeling, annotated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()