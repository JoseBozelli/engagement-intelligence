"""Purged walk-forward tuning within train, and final fits on the full train split.

Only the configured development splits may be passed in; held-out rows are never read here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from engagement_intelligence.evaluation.metrics import binary_log_loss
from engagement_intelligence.features.spec import FeatureSpec
from engagement_intelligence.labels.build import DAY_US
from engagement_intelligence.models.config import CandidateConfig, Fold, ModelingConfig
from engagement_intelligence.models.data import Dataset
from engagement_intelligence.models.estimators import make_estimator


@dataclass
class FittedCandidate:
    name: str
    kind: str
    feature_set: str
    feature_names: list[str]
    params: dict[str, Any]
    cv_log_loss: float | None
    cv_results: list[dict[str, Any]]
    estimator: Any

    def predict(self, ds: Dataset) -> np.ndarray:
        return self.estimator.predict_proba(ds.columns(self.feature_names))[:, 1]


def _date_us(d: date) -> int:
    return int(np.datetime64(d, "D").astype("datetime64[us]").astype(np.int64))


def fold_masks(ds: Dataset, folds: Sequence[Fold], horizon_days: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(fit, score) masks: fit rows precede the scoring window, purged by the label horizon."""
    masks = []
    for f in folds:
        start, end = _date_us(f.score_start), _date_us(f.score_end)
        fit = ds.assessed_us + horizon_days * DAY_US <= start
        score = (ds.assessed_us >= start) & (ds.assessed_us < end)
        if not fit.any() or not score.any():
            raise ValueError(f"fold {f.score_start}..{f.score_end} has no fit rows or no score rows")
        masks.append((fit, score))
    return masks


def _fit(kind: str, estimator, X: np.ndarray, y: np.ndarray, eval_set=None):
    if kind == "xgboost" and eval_set is not None:
        estimator.fit(X, y.astype(int), eval_set=[(eval_set[0], eval_set[1].astype(int))], verbose=False)
    else:
        estimator.fit(X, y.astype(int))
    return estimator


def tune(kind: str, candidate: CandidateConfig, train: Dataset, names: Sequence[str],
         folds: Sequence[Fold], horizon_days: int, log1p: Sequence[str], seed: int) -> list[dict[str, Any]]:
    X = train.columns(names)
    results = []
    for params in candidate.configurations():
        losses, rounds = [], []
        for fit, score in fold_masks(train, folds, horizon_days):
            estimator = _fit(kind, make_estimator(kind, params, names, log1p, seed),
                             X[fit], train.y[fit], eval_set=(X[score], train.y[score]))
            losses.append(binary_log_loss(train.y[score], estimator.predict_proba(X[score])[:, 1]))
            if kind == "xgboost":
                rounds.append(estimator.best_iteration + 1)
        results.append({
            "params": params,
            "cv_log_loss": float(np.mean(losses)),
            "fold_log_loss": [float(v) for v in losses],
            "rounds": int(round(float(np.mean(rounds)))) if rounds else None,
        })
    return results


def fit_final(kind: str, params: dict[str, Any], rounds: int | None, train: Dataset,
              names: Sequence[str], log1p: Sequence[str], seed: int):
    params = dict(params)
    if kind == "xgboost":
        params.pop("early_stopping_rounds", None)
        params["n_estimators"] = rounds
    return _fit(kind, make_estimator(kind, params, names, log1p, seed), train.columns(names), train.y)


def run_experiments(cfg: ModelingConfig, spec: FeatureSpec, ds: Dataset, horizon_days: int) -> list[FittedCandidate]:
    allowed = {cfg.splits.train, cfg.splits.validation}
    if not set(ds.split.tolist()) <= allowed:
        raise ValueError(f"only development splits {sorted(allowed)} may be used here")
    train = ds.subset(ds.split == cfg.splits.train)
    log1p = cfg.log1p_features

    base_names = [cfg.baseline.feature]
    candidates = [FittedCandidate(
        "baseline", "logistic", "baseline", base_names, dict(cfg.baseline.params), None, [],
        fit_final("logistic", cfg.baseline.params, None, train, base_names, log1p, cfg.seed),
    )]
    for set_name, feature_set in cfg.feature_sets.items():
        names = feature_set.resolve(spec)
        for kind, candidate in cfg.candidates.items():
            results = tune(kind, candidate, train, names, cfg.cv_folds, horizon_days, log1p, cfg.seed)
            best = min(results, key=lambda r: r["cv_log_loss"])
            estimator = fit_final(kind, best["params"], best["rounds"], train, names, log1p, cfg.seed)
            candidates.append(FittedCandidate(f"{kind}[{set_name}]", kind, set_name, names,
                                              best["params"], best["cv_log_loss"], results, estimator))
    return candidates