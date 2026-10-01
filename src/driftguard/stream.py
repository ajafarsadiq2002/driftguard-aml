"""Prequential (test-then-train) streaming loop over test steps 35-49.

For every step t: (1) predict with the current model, (2) record metrics, (3) check drift, (4) if the policy
says so, the simulated analyst labels transactions from step t and the model is retrained. Labels from step t
can therefore only influence predictions for steps > t; `run_one` asserts this before every prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from driftguard import config
from driftguard.active import SimulatedAnalyst, fit_novelty, novelty_scores, select
from driftguard.drift import audit_misses, draw_audit
from driftguard.evaluate import compute_metrics, per_step_and_window
from driftguard.models import fit_model, predict_scores, select_threshold
from driftguard.splits import LeakageError, in_steps, labeled


@dataclass
class StepData:
    step: int
    rows: pd.DataFrame  # labeled rows of this step: txId, step, y (features kept in X)
    X: np.ndarray
    novelty: np.ndarray


@dataclass
class SeedContext:
    seed: int
    model_name: str
    X_base: np.ndarray  # labeled steps 1-34
    y_base: np.ndarray
    base_model: object
    threshold: float  # frozen: chosen on validation with a model trained on steps 1-29
    steps: list[StepData]


@dataclass
class AlarmConfig:
    unsupervised_fired: dict  # step -> bool (seed independent, computed by the frozen monitor)
    audit_thr: float  # random audit (pre-registered drift_triggered policy)
    audit_thr_uncertainty: float = np.inf  # uncertainty audit (post-hoc v2 policy)
    audit_size: int = config.AUDIT_SIZE


def prepare_seed(df: pd.DataFrame, model_name: str, cols: list[str], seed: int) -> SeedContext:
    """Threshold from a 1-29 model on validation, base model on 1-34, novelty model on 1-34 features."""
    tr = labeled(in_steps(df, config.TRAIN_STEPS))
    va = labeled(in_steps(df, config.VAL_STEPS))
    m29 = fit_model(model_name, tr[cols].to_numpy(), tr["y"].to_numpy(), seed)
    threshold = select_threshold(va["y"].to_numpy(), predict_scores(m29, va[cols].to_numpy()))

    base = labeled(in_steps(df, list(config.TRAIN_STEPS) + list(config.VAL_STEPS)))
    X_base, y_base = base[cols].to_numpy(), base["y"].to_numpy()
    base_model = fit_model(model_name, X_base, y_base, seed)

    pre_all = in_steps(df, list(config.TRAIN_STEPS) + list(config.VAL_STEPS))
    iso = fit_novelty(pre_all[cols].to_numpy(), seed)

    steps = []
    for s in config.TEST_STEPS:
        rows = labeled(df[df["step"] == s])
        X = rows[cols].to_numpy()
        steps.append(StepData(s, rows[["txId", "step", "y"]].reset_index(drop=True), X, novelty_scores(iso, X)))
    return SeedContext(seed, model_name, X_base, y_base, base_model, threshold, steps)


def run_one(ctx: SeedContext, alarms: AlarmConfig, policy: str, strategy: str = "none", k: int = 0,
            n_jobs_model: int = 1, on_predict=None) -> dict:
    """Run one policy/strategy/K/seed configuration through the stream.

    `on_predict(step_data, model, scores, threshold)` is called right after each step is scored (used for SHAP).
    """
    rng_audit = np.random.default_rng([ctx.seed, 1])
    rng_query = np.random.default_rng([ctx.seed, 2])
    analyst = SimulatedAnalyst()
    model, model_max_step = ctx.base_model, max(config.VAL_STEPS)
    pool_X, pool_y, pool_steps = [], [], []
    labels_cum = 0
    rows, all_y, all_scores, all_steps = [], [], [], []

    def retrain():
        X = np.vstack([ctx.X_base, *pool_X])
        y = np.concatenate([ctx.y_base, *pool_y])
        w = np.concatenate([np.ones(len(ctx.y_base)), np.full(len(y) - len(ctx.y_base), config.QUERY_WEIGHT)])
        m = fit_model(ctx.model_name, X, y, ctx.seed, sample_weight=w, n_jobs=n_jobs_model)
        return m, max(pool_steps)

    for sd in ctx.steps:
        t = sd.step
        # 1-2. predict with the current model and record metrics BEFORE any label from step t is used
        if model_max_step >= t:
            raise LeakageError(f"model trained on step {model_max_step} used to predict step {t}")
        scores = predict_scores(model, sd.X)
        y = sd.rows["y"].to_numpy()
        metrics = compute_metrics(y, scores, ctx.threshold)
        predicted_with_max_step = model_max_step
        if on_predict is not None:
            on_predict(sd, model, scores, ctx.threshold)
        top = np.argsort(-scores, kind="stable")[: config.ALERT_BUDGET]
        all_y.append(y)
        all_scores.append(scores)
        all_steps.append(np.full(len(y), t))

        # 3-4. drift check and analyst queries
        available = np.ones(len(y), dtype=bool)
        new_positions, n_audit, misses, fired = [], 0, np.nan, False
        if policy in ("drift_triggered", "drift_triggered_v2"):
            if policy == "drift_triggered":  # pre-registered: uniform random audit
                audit = draw_audit(len(y), alarms.audit_size, rng_audit)
                audit_thr = alarms.audit_thr
            else:  # post-hoc v2: audit the transactions closest to the decision threshold
                audit = select("uncertainty", alarms.audit_size, scores, ctx.threshold, sd.novelty, available,
                               rng_audit)
                audit_thr = alarms.audit_thr_uncertainty
            analyst.label(sd.rows, audit, t, "audit")
            misses = audit_misses(y[audit], scores[audit], ctx.threshold)
            fired = bool(alarms.unsupervised_fired.get(t, False) or misses > audit_thr)
            available[audit] = False
            new_positions.append(audit)
            n_audit = len(audit)
            if fired:
                q = select(strategy, k, scores, ctx.threshold, sd.novelty, available, rng_query)
                analyst.label(sd.rows, q, t, "query")
                new_positions.append(q)
        elif policy == "always":
            q = select(strategy, k, scores, ctx.threshold, sd.novelty, available, rng_query)
            analyst.label(sd.rows, q, t, "query")
            new_positions.append(q)
        elif policy == "full_retrain":
            new_positions.append(np.arange(len(y)))
        elif policy != "static":
            raise ValueError(f"unknown policy {policy!r}")

        pos = np.concatenate(new_positions).astype(int) if new_positions else np.array([], dtype=int)
        if len(pos):
            pool_X.append(sd.X[pos])
            pool_y.append(y[pos])
            pool_steps.append(t)
            labels_cum += len(pos)
        retrained = policy in ("always", "full_retrain") or (policy.startswith("drift_triggered") and fired)
        if retrained and pool_X:
            model, model_max_step = retrain()

        rows.append({
            "step": t, **metrics, "labels_step": len(pos), "labels_cum": labels_cum,
            "labeled_illicit": int((y[pos] == 1).sum()), "audit_size": n_audit,
            "alerts_reviewed": len(top), "alert_hits": int((y[top] == 1).sum()),
            "audit_misses": misses, "unsup_fired": bool(alarms.unsupervised_fired.get(t, False)),
            "fired": fired, "retrained": bool(retrained and pool_X),
            "train_max_step": predicted_with_max_step,
        })

    key = {"policy": policy, "strategy": strategy, "k": k, "seed": ctx.seed}
    y_all, s_all, st_all = np.concatenate(all_y), np.concatenate(all_scores), np.concatenate(all_steps)
    _, windows = per_step_and_window(st_all, y_all, s_all, ctx.threshold)
    for w in windows:
        in_window = [r for r in rows if r["step"] in config.EVAL_WINDOWS[w["window"]]]
        for col in ("labels_step", "alert_hits", "alerts_reviewed"):
            w["labels_used" if col == "labels_step" else col] = sum(r[col] for r in in_window)
    post = np.isin(st_all, list(config.EVAL_WINDOWS["post_shutdown"]))
    return {"per_step": [{**key, **r} for r in rows], "windows": [{**key, **w} for w in windows],
            "queries": [{**key, **q} for q in analyst.log], "post_scores": (y_all[post], s_all[post]),
            "final_model": model}


def grid_configs() -> list[tuple[str, str, int]]:
    cfgs = [("static", "none", 0), ("full_retrain", "none", 0)]
    for policy in ("drift_triggered", "always"):
        cfgs += [(policy, s, k) for s in config.STRATEGIES for k in config.K_VALUES]
    cfgs += [("drift_triggered_v2", "uncertainty", k) for k in config.K_VALUES]  # post-hoc, see config.V2_NOTE
    return cfgs


def run_grid(contexts: list[SeedContext], alarms: AlarmConfig, n_jobs: int = -1, verbose: int = 5):
    """All configurations x seeds, in parallel (each model fit single-threaded)."""
    jobs = [(ctx, cfg) for ctx in contexts for cfg in grid_configs()]
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_run_light)(ctx, alarms, *cfg) for ctx, cfg in jobs
    )
    per_step = pd.DataFrame([r for res in results for r in res["per_step"]])
    windows = pd.DataFrame([w for res in results for w in res["windows"]])
    queries = pd.DataFrame([q for res in results for q in res["queries"]])
    post_scores = {(r["per_step"][0]["policy"], r["per_step"][0]["strategy"], r["per_step"][0]["k"]): r["post_scores"]
                   for r in results if r["per_step"][0]["seed"] == config.SEEDS[0]}
    return per_step, windows, queries, post_scores


def _run_light(ctx, alarms, policy, strategy, k):
    res = run_one(ctx, alarms, policy, strategy, k)
    res.pop("final_model")
    return res
