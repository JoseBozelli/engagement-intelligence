"""Slice tables for error analysis (model- and vocabulary-agnostic)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

_NAN = float("nan")


def bucket_names(edges: Sequence[int]) -> list[str]:
    return [f"<{edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}"]


def bucket_labels(values: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    return np.array(bucket_names(edges))[np.digitize(values, edges)]


def slice_table(y: np.ndarray, p: np.ndarray, flagged: np.ndarray, slices: Mapping[str, np.ndarray],
                orders: Mapping[str, Sequence[str]] | None = None) -> list[dict[str, Any]]:
    rows = []
    for name, values in slices.items():
        values = np.asarray(values).astype(str)
        present = set(values.tolist())
        order = [v for v in (orders or {}).get(name, sorted(present)) if v in present]
        for v in order:
            m = values == v
            yy, pp, ff = y[m], p[m], flagged[m]
            both = 0 < yy.sum() < m.sum()
            rows.append({
                "slice": name, "value": v, "n": int(m.sum()),
                "prevalence": float(yy.mean()), "mean_predicted": float(pp.mean()),
                "calibration_gap": float(pp.mean() - yy.mean()),
                "pr_auc": float(average_precision_score(yy, pp)) if both else _NAN,
                "flagged_share": float(ff.mean()),
                "precision": float(yy[ff].mean()) if ff.any() else _NAN,
                "recall": float((yy & ff).sum() / yy.sum()) if yy.any() else _NAN,
            })
    return rows