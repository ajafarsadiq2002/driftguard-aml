"""Query strategies, simulated analyst and the stream's test-then-train guarantee."""

import numpy as np
import pandas as pd
import pytest

from driftguard import config
from driftguard.active import SimulatedAnalyst, select
from driftguard.splits import LeakageError
from driftguard.stream import AlarmConfig, prepare_seed, run_one

SCORES = np.array([0.05, 0.48, 0.9, 0.52, 0.2, 0.7])
NOVELTY = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.05])


@pytest.mark.parametrize("strategy", ["random", "uncertainty", "novelty", "hybrid"])
def test_select_respects_k_and_availability(strategy):
    available = np.array([True, True, True, False, True, True])
    pick = select(strategy, 3, SCORES, 0.5, NOVELTY, available, np.random.default_rng(0))
    assert len(pick) == 3 == len(set(pick))
    assert available[pick].all()


def test_uncertainty_picks_scores_closest_to_threshold():
    pick = select("uncertainty", 2, SCORES, 0.5, NOVELTY, np.ones(6, bool), np.random.default_rng(0))
    assert set(pick) == {1, 3}


def test_novelty_and_hybrid():
    allow = np.ones(6, bool)
    assert set(select("novelty", 2, SCORES, 0.5, NOVELTY, allow, np.random.default_rng(0))) == {3, 4}
    # 1 most uncertain (index 1, stable tie-break with index 3) + 1 most novel of the rest (index 3)
    assert set(select("hybrid", 2, SCORES, 0.5, NOVELTY, allow, np.random.default_rng(0))) == {1, 3}
    hy = select("hybrid", 4, SCORES, 0.5, NOVELTY, allow, np.random.default_rng(0))
    assert len(set(hy)) == 4


def test_analyst_refuses_other_steps_and_unlabeled_rows():
    rows = pd.DataFrame({"txId": [1, 2, 3], "step": [40, 40, 41], "y": [1.0, np.nan, 0.0]})
    analyst = SimulatedAnalyst()
    analyst.label(rows, np.array([0]), 40, "query")
    with pytest.raises(LeakageError):
        analyst.label(rows, np.array([2]), 40, "query")
    with pytest.raises(LeakageError):
        analyst.label(rows, np.array([1]), 40, "query")


@pytest.fixture(scope="module")
def toy_stream():
    """Small synthetic dataset with all 49 steps; illicit pattern flips after step 43."""
    rng = np.random.default_rng(0)
    frames = []
    for s in range(1, 50):
        n = 120
        y = (rng.uniform(size=n) < 0.15).astype(float)
        sign = 1.0 if s < config.SHUTDOWN_STEP else -1.0
        X = rng.normal(size=(n, 4)) + sign * y[:, None] * 2.0
        y[rng.uniform(size=n) < 0.3] = np.nan  # some unknown rows
        f = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        f.insert(0, "y", y)
        f.insert(0, "step", s)
        f.insert(0, "txId", s * 1000 + np.arange(n))
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    ctx = prepare_seed(df, "logreg", ["a", "b", "c", "d"], seed=0)
    alarms = AlarmConfig(unsupervised_fired={}, audit_thr=0.0, audit_size=5)
    return df, ctx, alarms


@pytest.mark.parametrize(
    "policy,strategy,k",
    [("static", "none", 0), ("always", "hybrid", 5), ("drift_triggered", "random", 5), ("full_retrain", "none", 0)],
)
def test_stream_predicts_step_t_with_model_trained_before_t(toy_stream, policy, strategy, k):
    df, ctx, alarms = toy_stream
    res = run_one(ctx, alarms, policy, strategy, k)
    for r in res["per_step"]:
        assert r["train_max_step"] < r["step"]


def test_stream_queries_come_from_current_step_labeled_rows(toy_stream):
    df, ctx, alarms = toy_stream
    res = run_one(ctx, alarms, "drift_triggered", "hybrid", 5)
    q = pd.DataFrame(res["queries"])
    assert len(q) > 0
    truth = df.set_index("txId")
    assert (truth.loc[q["txId"], "step"].to_numpy() == q["step"].to_numpy()).all()
    assert truth.loc[q["txId"], "y"].notna().all()
    assert q["step"].min() >= min(config.TEST_STEPS)
    per_step = pd.DataFrame(res["per_step"])
    assert (per_step["labels_cum"].iloc[-1]) == len(q)


def test_static_policy_uses_no_labels(toy_stream):
    _, ctx, alarms = toy_stream
    res = run_one(ctx, alarms, "static")
    assert res["per_step"][-1]["labels_cum"] == 0 and res["queries"] == []
