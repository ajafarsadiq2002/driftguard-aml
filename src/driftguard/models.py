"""Model builders and decision-threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from driftguard import config

MODEL_NAMES = ["trivial", "logreg", "rf", "xgb"]


def build_model(name: str, seed: int):
    if name == "trivial":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed),
        )
    if name == "rf":
        return RandomForestClassifier(**config.RF_PARAMS, random_state=seed)
    if name == "xgb":
        return XGBClassifier(**config.XGB_PARAMS, random_state=seed, eval_metric="logloss")
    raise ValueError(f"unknown model {name!r}")


def fit_model(name: str, X: np.ndarray, y: np.ndarray, seed: int, sample_weight=None, n_jobs: int | None = None):
    model = build_model(name, seed)
    if n_jobs is not None and "n_jobs" in model.get_params():
        model.set_params(n_jobs=n_jobs)
    if sample_weight is None:
        model.fit(X, y.astype(int))
    elif hasattr(model, "steps"):  # sklearn Pipeline: route weights to the final estimator
        model.fit(X, y.astype(int), **{f"{model.steps[-1][0]}__sample_weight": sample_weight})
    else:
        model.fit(X, y.astype(int), sample_weight=sample_weight)
    return model


def predict_scores(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def select_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Decision threshold that maximises illicit-class F1 (call on validation data only)."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if len(thresholds) == 0:
        return 0.5
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])
