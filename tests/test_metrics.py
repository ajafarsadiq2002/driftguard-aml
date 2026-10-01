"""Metric functions match scikit-learn; threshold selection maximises F1."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from driftguard.evaluate import compute_metrics, per_step_and_window
from driftguard.models import select_threshold

Y = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
S = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.05, 0.3, 0.5])


def test_compute_metrics_matches_sklearn():
    m = compute_metrics(Y, S, threshold=0.5)
    pred = (S >= 0.5).astype(int)
    assert m["pr_auc"] == pytest.approx(average_precision_score(Y, S))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(Y, S))
    assert m["f1"] == pytest.approx(f1_score(Y, pred))
    assert m["precision"] == pytest.approx(precision_score(Y, pred))
    assert m["recall"] == pytest.approx(recall_score(Y, pred))
    assert m["caught"] == int((pred & Y).sum())
    assert m["n_illicit"] == 4


def test_single_class_step_is_handled():
    m = compute_metrics(np.zeros(5), np.linspace(0, 1, 5), threshold=0.5)
    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"])
    assert m["recall"] == 0.0


def test_select_threshold_maximises_f1():
    thr = select_threshold(Y, S)
    best = f1_score(Y, (S >= thr).astype(int))
    for t in np.unique(S):
        assert f1_score(Y, (S >= t).astype(int)) <= best + 1e-12


def test_windows_pool_the_right_steps():
    steps = np.array([35] * 5 + [43] * 5)
    per_step, windows = per_step_and_window(steps, Y, S, 0.5)
    assert [r["step"] for r in per_step] == [35, 43]
    w = {r["window"]: r for r in windows}
    assert w["pre_shutdown"]["n"] == 5 and w["post_shutdown"]["n"] == 5 and w["all_test"]["n"] == 10


def test_recovery_efficiency_is_extra_catches_per_label_vs_same_seed_static():
    import pandas as pd

    from driftguard.evaluate import add_recovery_efficiency, summarise_stream

    rows = []
    for seed, (static_caught, adaptive_caught) in enumerate([(3, 13), (5, 9)]):
        for policy, strategy, k, caught, labels in [("static", "none", 0, static_caught, 0),
                                                    ("always", "random", 10, adaptive_caught, 100)]:
            base = {"policy": policy, "strategy": strategy, "k": k, "seed": seed, "pr_auc": 0.1}
            rows.append({**base, "window": "post_shutdown", "caught": caught, "labels_used": labels // 2})
            rows.append({**base, "window": "all_test", "caught": caught + 10, "labels_used": labels})
    out = add_recovery_efficiency(pd.DataFrame(rows))
    ad = out[(out.policy == "always") & (out.window == "post_shutdown")].sort_values("seed")
    assert ad["extra_caught_vs_static"].tolist() == [10, 4]
    assert ad["recovery_efficiency"].tolist() == pytest.approx([0.10, 0.04])  # divided by total labels (100)
    st = out[(out.policy == "static") & (out.window == "post_shutdown")]
    assert st["recovery_efficiency"].isna().all()  # 0 labels -> undefined
    summ = summarise_stream(out)
    row = summ[(summ.policy == "always") & (summ.window == "post_shutdown")].iloc[0]
    assert row["recovery_efficiency_mean"] == pytest.approx(0.07)
