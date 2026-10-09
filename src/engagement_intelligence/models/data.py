"""Modeling dataset: point-in-time features aligned with labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa

from engagement_intelligence.labels.build import column_us


@dataclass(frozen=True)
class Dataset:
    feature_names: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    account_id: np.ndarray
    assessed_us: np.ndarray
    split: np.ndarray
    cold_start: np.ndarray

    def subset(self, mask: np.ndarray) -> Dataset:
        return replace(self, X=self.X[mask], y=self.y[mask], account_id=self.account_id[mask],
                       assessed_us=self.assessed_us[mask], split=self.split[mask],
                       cold_start=self.cold_start[mask])

    def columns(self, names: Sequence[str]) -> np.ndarray:
        index = {n: i for i, n in enumerate(self.feature_names)}
        return self.X[:, [index[n] for n in names]]


def load_dataset(features: pa.Table, assessments: pa.Table, feature_names: Sequence[str]) -> Dataset:
    if features.num_rows != assessments.num_rows:
        raise ValueError("features and assessments have different row counts")
    for key in ("account_id", "assessed_at"):
        if not features.column(key).equals(assessments.column(key)):
            raise ValueError(f"features and assessments are not aligned on {key}")
    return Dataset(
        feature_names=tuple(feature_names),
        X=np.column_stack([features.column(n).to_numpy() for n in feature_names]).astype(float),
        y=np.array(assessments.column("label").to_pylist(), dtype=bool),
        account_id=np.array(assessments.column("account_id").to_pylist()),
        assessed_us=column_us(assessments, "assessed_at"),
        split=np.array(assessments.column("split").to_pylist()),
        cold_start=np.array(assessments.column("cold_start").to_pylist(), dtype=bool),
    )