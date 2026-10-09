"""Validation evaluation, paired comparisons, provisional selection, and rendering."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from engagement_intelligence.evaluation.metrics import binary_log_loss, clustered_bootstrap, metric_suite
from engagement_intelligence.models.config import ModelingConfig
from engagement_intelligence.models.data import Dataset
from engagement_intelligence.models.train import FittedCandidate

_SIMPLICITY = {"logistic": 0, "hist_gb": 1, "xgboost": 1}
_NAN = float("nan")


def _reading(lo: float, hi: float) -> str:
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    return "equivalent"


def evaluate_candidates(cfg: ModelingConfig, candidates: list[FittedCandidate], ds: Dataset) -> dict[str, Any]:
    val = ds.subset(ds.split == cfg.splits.validation)
    ev = cfg.evaluation
    y, accounts = val.y, val.account_id
    preds = {c.name: c.predict(val) for c in candidates}

    def ci(stat) -> tuple[float, float]:
        return clustered_bootstrap(accounts, stat, ev.bootstrap_replicates, cfg.seed)

    def loss_diff_ci(a: str, b: str) -> tuple[float, float]:
        return ci(lambda i: binary_log_loss(y[i], preds[a][i]) - binary_log_loss(y[i], preds[b][i]))

    metrics: dict[str, dict[str, Any]] = {}
    for name, p in preds.items():
        m: dict[str, Any] = metric_suite(y, p, val.assessed_us, ev.capacities, ev.calibration_bins)
        m["log_loss_ci"] = ci(lambda i, p=p: binary_log_loss(y[i], p[i]))
        m["pr_auc_ci"] = ci(lambda i, p=p: average_precision_score(y[i], p[i]))
        for label, mask in (("warm", ~val.cold_start), ("cold", val.cold_start)):
            m[f"{label}_log_loss"] = binary_log_loss(y[mask], p[mask]) if mask.any() else _NAN
            ok = 0 < y[mask].sum() < mask.sum()
            m[f"{label}_pr_auc"] = float(average_precision_score(y[mask], p[mask])) if ok else _NAN
        metrics[name] = m

    kinds, sets = list(cfg.candidates), list(cfg.feature_sets)
    pairs = [(c.name, "baseline", "vs recency baseline") for c in candidates if c.name != "baseline"]
    pairs += [(f"{k}[{s}]", f"{kinds[0]}[{s}]", f"{k} vs {kinds[0]}") for s in sets for k in kinds[1:]]
    pairs += [(f"{k}[{s}]", f"{k}[{sets[0]}]", f"{s} vs {sets[0]}") for k in kinds for s in sets[1:]]
    comparisons = []
    for a, b, question in pairs:
        if a in preds and b in preds:
            lo, hi = loss_diff_ci(a, b)
            comparisons.append({"question": question, "a": a, "b": b,
                                "log_loss_diff": binary_log_loss(y, preds[a]) - binary_log_loss(y, preds[b]),
                                "ci": (lo, hi), "reading": _reading(lo, hi)})

    learned = [c for c in candidates if c.name != "baseline"]
    best = min(learned, key=lambda c: metrics[c.name]["log_loss"])
    selected, selected_ci = best, None
    by_simplicity = sorted(learned, key=lambda c: (_SIMPLICITY.get(c.kind, 9), len(c.feature_names),
                                                   metrics[c.name]["log_loss"]))
    for c in by_simplicity:
        if c is best:
            break
        lo, hi = loss_diff_ci(c.name, best.name)
        if lo <= 0:  # not significantly worse than the best
            selected, selected_ci = c, (lo, hi)
            break
    selection = {
        "best_by_log_loss": best.name,
        "selected": selected.name,
        "difference_vs_best_ci": selected_ci,
        "rule": "simplest candidate not significantly worse than the best by validation log loss "
                "(paired, account-clustered bootstrap)",
    }
    return {"metrics": metrics, "comparisons": comparisons, "selection": selection, "predictions": preds}


def _f(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _ci(t: tuple[float, float] | None, digits: int = 3) -> str:
    return "n/a" if t is None else f"[{t[0]:.{digits}f}, {t[1]:.{digits}f}]"


def render_comparison(cfg: ModelingConfig, candidates: list[FittedCandidate],
                      results: dict[str, Any], provenance: dict[str, str]) -> str:
    m = results["metrics"]
    caps = list(cfg.evaluation.capacities)
    mid = caps[len(caps) // 2]
    lines = ["# Model comparison (validation)", "",
             "Synthetic data for methodology evaluation only. The held-out test split was not used.", ""]
    lines += [f"- {key}: `{value}`" for key, value in provenance.items()]

    lines += ["", "## 1. Tuning (purged walk-forward CV within train)", "",
              "| candidate | features | selected parameters | CV log loss |", "|---|---|---|---|"]
    for c in candidates:
        grid = cfg.candidates[c.kind].grid if c.name != "baseline" else {}
        shown = [f"{k}={v}" for k, v in sorted(c.params.items()) if k in grid]
        best = min(c.cv_results, key=lambda r: r["cv_log_loss"]) if c.cv_results else None
        if best and best["rounds"]:
            shown.append(f"rounds={best['rounds']}")
        lines.append(f"| {c.name} | {len(c.feature_names)} | {', '.join(shown) or '-'} | {_f(c.cv_log_loss, 4)} |")

    cap_cols = [f"precision@{c:.0%}" for c in caps]
    lines += ["", "## 2. Validation metrics", "",
              "| candidate | log loss [95% CI] | PR-AUC [95% CI] | ROC-AUC | Brier | ECE | "
              + " | ".join(cap_cols) + f" | recall@{mid:.0%} |",
              "|---|" + "---|" * (5 + len(caps) + 1)]
    for c in candidates:
        r = m[c.name]
        lines.append(
            f"| {c.name} | {_f(r['log_loss'], 4)} {_ci(r['log_loss_ci'], 4)} | {_f(r['pr_auc'])} {_ci(r['pr_auc_ci'])} "
            f"| {_f(r['roc_auc'])} | {_f(r['brier'], 4)} | {_f(r['ece'], 4)} | "
            + " | ".join(_f(r[col]) for col in cap_cols) + f" | {_f(r[f'recall@{mid:.0%}'])} |")

    lines += ["", "## 3. Cold-start vs warm (validation)", "",
              "| candidate | warm log loss | cold log loss | warm PR-AUC | cold PR-AUC |", "|---|---|---|---|---|"]
    for c in candidates:
        r = m[c.name]
        lines.append(f"| {c.name} | {_f(r['warm_log_loss'], 4)} | {_f(r['cold_log_loss'], 4)} "
                     f"| {_f(r['warm_pr_auc'])} | {_f(r['cold_pr_auc'])} |")

    lines += ["", "## 4. Paired comparisons (log loss difference a - b; negative favors a)", "",
              "| question | a | b | difference [95% CI] | reading |", "|---|---|---|---|---|"]
    for cmp in results["comparisons"]:
        lines.append(f"| {cmp['question']} | {cmp['a']} | {cmp['b']} "
                     f"| {_f(cmp['log_loss_diff'], 4)} {_ci(cmp['ci'], 4)} | {cmp['reading']} |")

    s = results["selection"]
    lines += ["", "## 5. Provisional selection", "",
              f"- Best by validation log loss: `{s['best_by_log_loss']}`",
              f"- Selected: `{s['selected']}`",
              f"- Difference vs best [95% CI]: {_ci(s['difference_vs_best_ci'], 4)}",
              f"- Rule: {s['rule']}",
              "- Calibration and the operating threshold are finalized in the next step."]
    return "\n".join(lines) + "\n"