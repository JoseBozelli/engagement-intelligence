"""Estimator factory. Every estimator exposes fit and predict_proba (scikit-learn API)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

KINDS = ("logistic", "xgboost", "hist_gb")


def log1p_columns(X: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    X = np.array(X, dtype=float, copy=True)
    X[:, columns] = np.log1p(np.maximum(X[:, columns], 0.0))
    return X


def make_estimator(kind: str, params: dict[str, Any], feature_names: Sequence[str],
                   log1p_features: Sequence[str], seed: int):
    if kind == "logistic":
        wanted = set(log1p_features)
        columns = [i for i, n in enumerate(feature_names) if n in wanted]
        return Pipeline([
            ("log1p", FunctionTransformer(log1p_columns, kw_args={"columns": columns})),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**params)),
        ])
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=seed, **params)
    if kind == "hist_gb":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"unknown model kind {kind!r}")