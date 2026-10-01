"""Sanity checks on the experiment configuration."""

import pytest

from driftguard import config
from driftguard.pipeline import parse_args


def test_splits_do_not_overlap():
    train, val, test = set(config.TRAIN_STEPS), set(config.VAL_STEPS), set(config.TEST_STEPS)
    assert not train & val
    assert not train & test
    assert not val & test


def test_splits_are_ordered_in_time_and_cover_all_steps():
    assert max(config.TRAIN_STEPS) < min(config.VAL_STEPS) < min(config.TEST_STEPS)
    covered = set(config.TRAIN_STEPS) | set(config.VAL_STEPS) | set(config.TEST_STEPS)
    assert covered == set(range(1, config.EXPECTED_STEPS + 1))


def test_eval_windows_are_inside_test_steps():
    test = set(config.TEST_STEPS)
    for window in config.EVAL_WINDOWS.values():
        assert set(window) <= test
    assert min(config.EVAL_WINDOWS["post_shutdown"]) == config.SHUTDOWN_STEP


def test_five_seeds():
    assert len(config.SEEDS) == 5 == len(set(config.SEEDS))


def test_cli_requires_a_stage():
    with pytest.raises(SystemExit):
        parse_args([])
    assert parse_args(["--all"]).all
