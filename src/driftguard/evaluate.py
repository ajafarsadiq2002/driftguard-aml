"""Metrics, per-step / per-window tables and the static baseline experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from driftguard import config
from driftguard.data import FEATURE_SETS
from driftguard.models import MODEL_NAMES, fit_model, predict_scores, select_threshold
from driftguard.splits import static_splits

METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall"]


def compute_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Illicit-class metrics. PR-AUC is the primary metric; ROC-AUC is NaN if only one class is present."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    has_pos = y.sum() > 0
    return {
        "pr_auc": float(average_precision_score(y, scores)) if has_pos else np.nan,
        "roc_auc": float(roc_auc_score(y, scores)) if both else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "n": int(len(y)),
        "n_illicit": int(y.sum()),
        "caught": int((pred & y).sum()),
        "flagged": int(pred.sum()),
    }


def per_step_and_window(steps: np.ndarray, y: np.ndarray, scores: np.ndarray, threshold: float):
    """Metrics for every test step, and pooled over each evaluation window."""
    steps = np.asarray(steps)
    per_step = [
        {"step": int(s), **compute_metrics(y[steps == s], scores[steps == s], threshold)}
        for s in np.unique(steps)
    ]
    windows = []
    for name, rng in config.EVAL_WINDOWS.items():
        mask = np.isin(steps, list(rng))
        windows.append({"window": name, **compute_metrics(y[mask], scores[mask], threshold)})
    return per_step, windows


def run_static_baselines(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Train on steps 1-29, pick threshold on 30-34, freeze, evaluate on 35-49 (5 seeds).

    Runs every model on the full feature set, plus the XGBoost feature-set ablation.
    """
    sp = static_splits(df)
    runs = [(m, "local_agg_graph") for m in MODEL_NAMES]
    runs += [("xgb", fs) for fs in FEATURE_SETS if fs != "local_agg_graph"]

    step_rows, window_rows, val_rows = [], [], []
    for model_name, fs in runs:
        cols = FEATURE_SETS[fs]
        X_tr, y_tr = sp["train"][cols].to_numpy(), sp["train"]["y"].to_numpy()
        X_va, y_va = sp["val"][cols].to_numpy(), sp["val"]["y"].to_numpy()
        X_te, y_te = sp["test"][cols].to_numpy(), sp["test"]["y"].to_numpy()
        steps_te = sp["test"]["step"].to_numpy()
        seeds = [config.SEEDS[0]] if model_name == "trivial" else config.SEEDS
        for seed in seeds:
            model = fit_model(model_name, X_tr, y_tr, seed)
            val_scores = predict_scores(model, X_va)
            thr = select_threshold(y_va, val_scores)
            val_rows.append(
                {"model": model_name, "feature_set": fs, "seed": seed, "threshold": thr,
                 **compute_metrics(y_va, val_scores, thr)}
            )
            per_step, windows = per_step_and_window(steps_te, y_te, predict_scores(model, X_te), thr)
            key = {"model": model_name, "feature_set": fs, "seed": seed}
            step_rows += [{**key, **r} for r in per_step]
            window_rows += [{**key, **r} for r in windows]
        if verbose:
            post = pd.DataFrame(window_rows).query(
                "model == @model_name and feature_set == @fs and window == 'post_shutdown'"
            )
            pre = pd.DataFrame(window_rows).query(
                "model == @model_name and feature_set == @fs and window == 'pre_shutdown'"
            )
            print(
                f"  {model_name:8s} {fs:16s} PR-AUC pre {pre.pr_auc.mean():.3f}  "
                f"post {post.pr_auc.mean():.3f}  recall pre {pre.recall.mean():.3f} post {post.recall.mean():.3f}",
                flush=True,
            )

    val = pd.DataFrame(val_rows)
    per_step = pd.DataFrame(step_rows)
    windows = pd.DataFrame(window_rows)
    return {
        "val": val,
        "per_step": per_step,
        "windows": windows,
        "summary": summarise(windows, ["model", "feature_set", "window"]),
        "val_summary": summarise(val, ["model", "feature_set"]),
    }


def summarise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and std over seeds for each metric (std is 0 for single-seed rows)."""
    cols = [c for c in METRICS + ["threshold", "caught", "n_illicit", "flagged"] if c in df.columns]
    agg = df.groupby(keys, sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.fillna({c: 0.0 for c in agg.columns if c.endswith("_std")}).reset_index()


def pick_best_static(val_summary: pd.DataFrame) -> dict:
    """Best model on validation PR-AUC (full feature set comparison across model families)."""
    full = val_summary[val_summary["model"] != "trivial"]
    best = full.sort_values("pr_auc_mean", ascending=False).iloc[0]
    return {"model": best["model"], "feature_set": best["feature_set"], "val_pr_auc": float(best["pr_auc_mean"])}
