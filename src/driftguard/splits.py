"""Time-based split helpers and leakage assertions."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from driftguard import config


class LeakageError(AssertionError):
    """Raised when data from a step >= the evaluated step would be used for training or tuning."""


def in_steps(df: pd.DataFrame, steps: Iterable[int]) -> pd.DataFrame:
    return df[df["step"].isin(list(steps))]


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["y"].notna()]


def assert_no_leakage(fit_steps: Iterable[int], eval_steps: Iterable[int]) -> None:
    """Everything used to fit or tune must come strictly before everything evaluated."""
    fit, ev = set(fit_steps), set(eval_steps)
    if not fit or not ev:
        raise LeakageError("empty fit or eval step set")
    if max(fit) >= min(ev):
        raise LeakageError(f"fit steps reach {max(fit)} but evaluation starts at {min(ev)}")


def static_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Labeled train (1-29), validation (30-34) and test (35-49) frames for static baselines."""
    assert_no_leakage(config.TRAIN_STEPS, config.VAL_STEPS)
    assert_no_leakage(list(config.TRAIN_STEPS) + list(config.VAL_STEPS), config.TEST_STEPS)
    lab = labeled(df)
    return {
        "train": in_steps(lab, config.TRAIN_STEPS),
        "val": in_steps(lab, config.VAL_STEPS),
        "test": in_steps(lab, config.TEST_STEPS),
    }
