"""Operating policy: calibration decision, risk-band thresholds, band assignment.

Business-facing values are configuration in business units (D-025): review capacity is the
share of eligible accounts a team can review per assessment week. Thresholds derive from
observable predictions only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.linear_model import LogisticRegression

from engagement_intelligence.core.config import fingerprint, load_yaml_layers
from engagement_intelligence.evaluation.metrics import expected_calibration_error

BANDS = ("low", "medium", "high")
_EPS = 1e-7


class OperatingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    review_capacity: float = Field(gt=0, lt=1)
    watch_capacity: float = Field(gt=0, lt=1)
    ece_tolerance: float = Field(gt=0)
    calibration_bins: int = Field(ge=2)
    tenure_feature: str
    tenure_buckets_days: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistency(self) -> OperatingConfig:
        if self.watch_capacity <= self.review_capacity:
            raise ValueError("watch_capacity must exceed review_capacity")
        b = self.tenure_buckets_days
        if b[0] <= 0 or any(x >= y for x, y in zip(b, b[1:])):
            raise ValueError("tenure_buckets_days must be positive and strictly increasing")
        return self

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


def load_operating_config(*paths: str | Path) -> OperatingConfig:
    return OperatingConfig.model_validate(load_yaml_layers(*paths))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


@dataclass(frozen=True)
class Calibration:
    method: str            # "none" or "platt"
    slope: float = 1.0
    intercept: float = 0.0

    def apply(self, p: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return np.asarray(p, dtype=float)
        return 1.0 / (1.0 + np.exp(-(self.slope * _logit(p) + self.intercept)))


def calibration_line(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of the outcome log-odds on logit(p); (1, 0) means calibrated."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p)[:, None], y.astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def choose_calibration(y: np.ndarray, p: np.ndarray, cfg: OperatingConfig) -> Calibration:
    if expected_calibration_error(y, p, cfg.calibration_bins) <= cfg.ece_tolerance:
        return Calibration("none")
    slope, intercept = calibration_line(y, p)
    return Calibration("platt", slope, intercept)


@dataclass(frozen=True)
class OperatingPoint:
    high_threshold: float
    medium_threshold: float
    review_capacity: float
    watch_capacity: float


def derive_operating_point(p: np.ndarray, cfg: OperatingConfig) -> OperatingPoint:
    return OperatingPoint(
        high_threshold=float(np.quantile(p, 1 - cfg.review_capacity)),
        medium_threshold=float(np.quantile(p, 1 - cfg.watch_capacity)),
        review_capacity=cfg.review_capacity,
        watch_capacity=cfg.watch_capacity,
    )


def assign_bands(p: np.ndarray, op: OperatingPoint) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.where(p >= op.high_threshold, "high", np.where(p >= op.medium_threshold, "medium", "low"))