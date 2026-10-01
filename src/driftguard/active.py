"""Query strategies (which K transactions to label) and the simulated analyst."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from driftguard.splits import LeakageError

STRATEGY_NAMES = ["random", "uncertainty", "novelty", "hybrid"]


def fit_novelty(X_train: np.ndarray, seed: int) -> IsolationForest:
    """IsolationForest on pre-deployment features (steps 1-34 only)."""
    return IsolationForest(n_estimators=200, random_state=seed, n_jobs=1).fit(X_train)


def novelty_scores(iso: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous relative to the training data."""
    return -iso.score_samples(X)


def _top(values: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(values[candidates], kind="stable")
    return candidates[order[:k]]


def select(
    strategy: str,
    k: int,
    scores: np.ndarray,
    threshold: float,
    novelty: np.ndarray,
    available: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return positions (into the step's labeled rows) to query. `available` = boolean mask of queryable rows."""
    cand = np.flatnonzero(available)
    k = min(k, len(cand))
    if k == 0:
        return np.array([], dtype=int)
    if strategy == "random":
        return rng.choice(cand, size=k, replace=False)
    if strategy == "uncertainty":
        return _top(np.abs(scores - threshold), cand, k)
    if strategy == "novelty":
        return _top(-novelty, cand, k)
    if strategy == "hybrid":
        unc = _top(np.abs(scores - threshold), cand, k // 2)
        rest = np.setdiff1d(cand, unc)
        nov = _top(-novelty, rest, k - len(unc))
        return np.concatenate([unc, nov])
    raise ValueError(f"unknown strategy {strategy!r}")


class SimulatedAnalyst:
    """Reveals ground-truth labels, but only for labeled transactions of the step currently being processed."""

    def __init__(self) -> None:
        self.log: list[dict] = []

    def label(self, step_rows: pd.DataFrame, positions: np.ndarray, step: int, reason: str) -> pd.DataFrame:
        picked = step_rows.iloc[positions]
        if (picked["step"] != step).any():
            raise LeakageError("analyst asked to label a transaction from another step")
        if picked["y"].isna().any():
            raise LeakageError("analyst asked to label a transaction without ground truth")
        self.log += [{"txId": int(t), "step": step, "reason": reason} for t in picked["txId"]]
        return picked
