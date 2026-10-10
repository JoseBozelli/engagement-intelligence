"""Driver attribution.

Exact for linear models: a feature's contribution is its coefficient times its transformed,
standardized value, i.e. its additive share of the log-odds relative to an average training
account. Contributions refer to the model's raw log-odds (before any calibration step).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from engagement_intelligence.models.bundle import ModelBundle
from engagement_intelligence.scoring.schemas import Driver


class Explainer(Protocol):
    intercept: float

    def contributions(self, X: np.ndarray) -> np.ndarray: ...


class LinearExplainer:
    def __init__(self, pipeline) -> None:
        self._transform = pipeline[:-1]
        model = pipeline[-1]
        self.coef = model.coef_[0]
        self.intercept = float(model.intercept_[0])

    def contributions(self, X: np.ndarray) -> np.ndarray:
        return self._transform.transform(X) * self.coef


def make_explainer(bundle: ModelBundle) -> Explainer:
    kind = bundle.manifest["model"]["kind"]
    if kind != "logistic":
        raise NotImplementedError(f"no driver explainer available for model kind {kind!r}")
    return LinearExplainer(bundle.estimator)


def top_drivers(contributions: np.ndarray, values: np.ndarray, names: Sequence[str],
                descriptions: Mapping[str, str], k: int) -> tuple[Driver, ...]:
    """The k largest risk-increasing contributions, descending."""
    order = np.argsort(-contributions, kind="stable")[:k]
    return tuple(
        Driver(feature=names[j], description=descriptions[names[j]],
               value=float(values[j]), contribution=float(contributions[j]))
        for j in order if contributions[j] > 0
    )