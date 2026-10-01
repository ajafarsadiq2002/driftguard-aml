"""SHAP alert helpers: top-5 extraction, outcome labels, and no raw feature values in the output."""

import numpy as np
import pandas as pd

from driftguard.explain import outcome, top_contributions


def test_top_contributions_orders_by_absolute_shap():
    sv = np.array([[0.1, -0.9, 0.3, 0.0, 0.05, -0.2], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    names = ["a", "b", "c", "d", "e", "f"]
    top = top_contributions(sv, names)
    assert [top[0][f"feature_{i}"] for i in range(1, 6)] == ["b", "c", "f", "a", "e"]
    assert top[0]["shap_1"] == -0.9
    assert top[1]["feature_1"] == "f"
    assert set(top[0]) == {f"feature_{i}" for i in range(1, 6)} | {f"shap_{i}" for i in range(1, 6)}


def test_outcome_labels():
    y = np.array([1, 0, 1, 0])
    flagged = np.array([True, True, False, False])
    assert outcome(y, flagged).tolist() == ["true_positive", "false_positive", "missed_illicit", "true_negative"]


def test_committed_alerts_contain_no_raw_feature_columns():
    from driftguard import config

    path = config.ARTIFACTS_DIR / "alerts.parquet"
    if not path.exists():
        return
    cols = set(pd.read_parquet(path).columns)
    assert not any(c.startswith(("local_", "agg_")) for c in cols)
    assert {"txId", "step", "score", "y", "feature_1", "shap_1"} <= cols
