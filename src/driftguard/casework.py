"""Post-hoc v4: 'follow the money' casework within each time step (see config.V4_NOTE).

Per step t, with the frozen base model:
1. Score every labelled transaction of step t.
2. An analyst reviews K transactions, highest score first. In the graph arm, each confirmed illicit case moves its
   unreviewed 1-hop neighbours (either edge direction) to the front of the review queue.
3. The analyst then works the top-N remaining alerts. In the graph arm, remaining neighbours of confirmed illicit
   cases rank above everything else.

Only labels the analyst paid for (step t, labelled rows) are used, and reviewed rows are never evaluated.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from driftguard import config
from driftguard.evaluate import compute_metrics
from driftguard.models import predict_scores
from driftguard.splits import LeakageError
from driftguard.stream import SeedContext


def build_adjacency(edges: pd.DataFrame) -> dict[int, set[int]]:
    """Undirected 1-hop neighbourhoods from the directed payment edges."""
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in zip(edges["txId1"].to_numpy(), edges["txId2"].to_numpy(), strict=True):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def casework_step(tx: np.ndarray, y: np.ndarray, scores: np.ndarray, adj: dict[int, set[int]], k: int,
                  alert_budget: int, graph: bool) -> dict:
    """Run reviews + alert triage for one step. Returns counts and the evaluation mask (unreviewed rows)."""
    pos_of = {int(t): i for i, t in enumerate(tx)}
    reviewed = np.zeros(len(tx), dtype=bool)
    boosted = np.zeros(len(tx), dtype=bool)
    order = list(np.argsort(-scores, kind="stable"))
    review_hits = 0
    for _ in range(min(k, len(tx))):
        cand = [i for i in np.flatnonzero(boosted & ~reviewed)] if graph else []
        if cand:
            i = max(cand, key=lambda j: scores[j])
        else:
            i = next(j for j in order if not reviewed[j])
        reviewed[i] = True
        if y[i] == 1:  # analyst confirms illicit (ground truth of a labelled row in step t)
            review_hits += 1
            if graph:
                for nb in adj.get(int(tx[i]), ()):
                    j = pos_of.get(nb)
                    if j is not None and not reviewed[j]:
                        boosted[j] = True
    rest = ~reviewed
    final = scores + (boosted.astype(float) if graph else 0.0)  # scores <= 1, so boosted rows rank first
    rest_idx = np.flatnonzero(rest)
    top = rest_idx[np.argsort(-final[rest_idx], kind="stable")[:alert_budget]]
    return {"reviewed_mask": reviewed, "eval_scores": final, "review_hits": review_hits,
            "alert_hits": int((y[top] == 1).sum()), "alerts_reviewed": len(top), "boosted": int(boosted[rest].sum())}


def run_casework(ctx: SeedContext, adj: dict[int, set[int]], k: int, graph: bool,
                 alert_budget: int = config.ALERT_BUDGET) -> tuple[list[dict], list[dict]]:
    """All test steps for one seed and arm. Per-step rows + pooled window rows (on unreviewed transactions)."""
    model = ctx.base_model
    rows, ys, ss, steps = [], [], [], []
    for sd in ctx.steps:
        if (sd.rows["step"] != sd.step).any() or sd.rows["y"].isna().any():
            raise LeakageError("casework may only review labelled transactions of the current step")
        y = sd.rows["y"].to_numpy().astype(int)
        scores = predict_scores(model, sd.X)
        out = casework_step(sd.rows["txId"].to_numpy(), y, scores, adj, k, alert_budget, graph)
        keep = ~out["reviewed_mask"]
        m = compute_metrics(y[keep], out["eval_scores"][keep], ctx.threshold)
        rows.append({"step": sd.step, **m, "review_hits": out["review_hits"], "alert_hits": out["alert_hits"],
                     "alerts_reviewed": out["alerts_reviewed"], "labels_step": int(out["reviewed_mask"].sum()),
                     "boosted_remaining": out["boosted"],
                     "identified": out["review_hits"] + out["alert_hits"], "n_illicit_step": int(y.sum())})
        ys.append(y[keep])
        ss.append(out["eval_scores"][keep])
        steps.append(np.full(keep.sum(), sd.step))
    y_all, s_all, st_all = np.concatenate(ys), np.concatenate(ss), np.concatenate(steps)
    windows = []
    for name, rng in config.EVAL_WINDOWS.items():
        mask = np.isin(st_all, list(rng))
        in_w = [r for r in rows if r["step"] in rng]
        windows.append({"window": name, **compute_metrics(y_all[mask], s_all[mask], ctx.threshold),
                        **{c: sum(r[c] for r in in_w) for c in ("review_hits", "alert_hits", "identified",
                                                                 "labels_step", "n_illicit_step")}})
    return rows, windows


def run_casework_grid(contexts: list[SeedContext], adj: dict[int, set[int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_step, windows = [], []
    arms = [("static_alerts", 0, False)]
    arms += [(arm, k, graph) for k in config.K_VALUES for arm, graph in (("control", False), ("v4_graph", True))]
    for ctx in contexts:
        for arm, k, graph in arms:
            key = {"arm": arm, "k": k, "seed": ctx.seed}
            rows, wins = run_casework(ctx, adj, k, graph)
            per_step += [{**key, **r} for r in rows]
            windows += [{**key, **w} for w in wins]
    return pd.DataFrame(per_step), pd.DataFrame(windows)


def summarise_casework(windows: pd.DataFrame) -> list[dict]:
    cols = ["pr_auc", "recall", "identified", "review_hits", "alert_hits", "labels_step", "n_illicit_step"]
    agg = windows.groupby(["arm", "k", "window"], sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index().to_dict(orient="records")
