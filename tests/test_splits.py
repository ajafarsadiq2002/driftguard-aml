"""No temporal leakage in the static splits."""

import numpy as np
import pandas as pd
import pytest

from driftguard import config
from driftguard.splits import LeakageError, assert_no_leakage, static_splits


@pytest.fixture
def toy_df():
    steps = np.repeat(np.arange(1, 50), 4)
    y = np.tile([1.0, 0.0, np.nan, 0.0], 49)
    return pd.DataFrame({"txId": np.arange(len(steps)), "step": steps, "y": y, "f": 0.0})


def test_config_splits_do_not_overlap():
    assert not set(config.TRAIN_STEPS) & set(config.VAL_STEPS)
    assert not set(config.TRAIN_STEPS) & set(config.TEST_STEPS)
    assert not set(config.VAL_STEPS) & set(config.TEST_STEPS)


def test_static_splits_respect_step_ranges(toy_df):
    sp = static_splits(toy_df)
    assert set(sp["train"]["step"]) == set(config.TRAIN_STEPS)
    assert set(sp["val"]["step"]) == set(config.VAL_STEPS)
    assert set(sp["test"]["step"]) == set(config.TEST_STEPS)


def test_no_test_step_in_static_training_or_tuning_data(toy_df):
    sp = static_splits(toy_df)
    fit_steps = set(sp["train"]["step"]) | set(sp["val"]["step"])
    assert max(fit_steps) < min(config.TEST_STEPS)
    assert not fit_steps & set(sp["test"]["step"])


def test_static_splits_use_labeled_rows_only(toy_df):
    for frame in static_splits(toy_df).values():
        assert frame["y"].notna().all()


def test_assert_no_leakage():
    assert_no_leakage(range(1, 30), range(30, 35))
    with pytest.raises(LeakageError):
        assert_no_leakage(range(1, 36), range(35, 50))
    with pytest.raises(LeakageError):
        assert_no_leakage([40], [40])
