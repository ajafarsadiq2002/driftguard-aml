"""v4 casework: graph boosting, fair budgets, and reviewed rows never evaluated."""

import numpy as np
import pandas as pd

from driftguard.casework import build_adjacency, casework_step

# Toy step: tx 1 (illicit, top score) connects to tx 4 (illicit, low score). tx 2/3/5 licit.
TX = np.array([1, 2, 3, 4, 5])
Y = np.array([1, 0, 0, 1, 0])
S = np.array([0.9, 0.8, 0.7, 0.01, 0.02])
ADJ = build_adjacency(pd.DataFrame({"txId1": [1, 2], "txId2": [4, 3]}))


def test_adjacency_is_undirected():
    assert ADJ[1] == {4} and ADJ[4] == {1} and ADJ[3] == {2}


def test_graph_arm_follows_the_money_in_reviews():
    out = casework_step(TX, Y, S, ADJ, k=2, alert_budget=0, graph=True)
    assert out["review_hits"] == 2  # reviews tx1 (illicit) then its neighbour tx4 (illicit)
    assert out["reviewed_mask"].tolist() == [True, False, False, True, False]


def test_control_arm_reviews_by_score_only():
    out = casework_step(TX, Y, S, ADJ, k=2, alert_budget=0, graph=False)
    assert out["review_hits"] == 1
    assert out["reviewed_mask"].tolist() == [True, True, False, False, False]


def test_boosted_neighbours_rank_first_in_remaining_alerts():
    out = casework_step(TX, Y, S, ADJ, k=1, alert_budget=1, graph=True)
    assert out["review_hits"] == 1 and out["alert_hits"] == 1  # tx4 jumps ahead of tx2 (0.8)
    ctrl = casework_step(TX, Y, S, ADJ, k=1, alert_budget=1, graph=False)
    assert ctrl["alert_hits"] == 0


def test_same_workload_in_both_arms_and_reviewed_rows_excluded():
    for graph in (True, False):
        out = casework_step(TX, Y, S, ADJ, k=2, alert_budget=2, graph=graph)
        assert out["reviewed_mask"].sum() == 2 and out["alerts_reviewed"] == 2
