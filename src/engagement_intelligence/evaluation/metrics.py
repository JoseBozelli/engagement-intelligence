"""Metrics for probabilistic binary predictions.

Capacity metrics express operating points in business units: the share of eligible accounts
an operator can review per assessment group (e.g., per week). Uncertainty comes from an
account-clustered bootstrap, because rows from one account are correlated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

_EPS = 1e-7


def binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    y = y.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _quantile_bins(p: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int) -> list[dict[str, float]]:
    """Equal-frequency bins: mean predicted probability vs observed rate."""
    idx = _quantile_bins(p, bins)
    return [
        {"mean_predicted": float(p[idx == b].mean()), "observed": float(y[idx == b].mean()),
         "count": int((idx == b).sum())}
        for b in range(bins) if (idx == b).any()
    ]


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int) -> float:
    n = len(p)
    return float(sum(r["count"] / n * abs(r["mean_predicted"] - r["observed"])
                     for r in reliability_table(y, p, bins)))


def capacity_flags(p: np.ndarray, groups: np.ndarray, capacity: float) -> np.ndarray:
    """Flag the top ceil(capacity * n) scores within each group."""
    flags = np.zeros(len(p), dtype=bool)
    order = np.lexsort((-p, groups))
    g = groups[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(p)]
    for s, e in zip(starts, ends):
        flags[order[s:s + math.ceil(capacity * (e - s))]] = True
    return flags


def capacity_metrics(y: np.ndarray, p: np.ndarray, groups: np.ndarray, capacity: float) -> dict[str, float]:
    flags = capacity_flags(p, groups, capacity)
    prevalence = float(y.mean())
    precision = float(y[flags].mean()) if flags.any() else float("nan")
    recall = float(y[flags].sum() / y.sum()) if y.any() else float("nan")
    return {"precision": precision, "recall": recall,
            "lift": precision / prevalence if prevalence else float("nan")}


def metric_suite(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                 capacities: Sequence[float], bins: int) -> dict[str, float]:
    y = y.astype(bool)
    both = 0 < y.sum() < len(y)
    out = {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "log_loss": binary_log_loss(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, bins),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
    }
    for c in capacities:
        m = capacity_metrics(y, p, groups, c)
        for key, value in m.items():
            out[f"{key}@{c:.0%}"] = value
    return out


def clustered_bootstrap(clusters: np.ndarray, statistic: Callable[[np.ndarray], float],
                        replicates: int, seed: int) -> tuple[float, float]:
    """95% percentile interval of statistic(row_indices) over resamples of whole clusters."""
    unique, inverse = np.unique(clusters, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        pick = rng.integers(0, len(unique), len(unique))
        values.append(statistic(np.concatenate([members[k] for k in pick])))
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))