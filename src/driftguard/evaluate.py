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


# ---------------------------------------------------------------------------
# Streaming grid aggregation (phase 5)
# ---------------------------------------------------------------------------
CFG_KEYS = ["policy", "strategy", "k"]
STREAM_COLS = METRICS + ["caught", "n_illicit", "flagged", "labels_used", "alert_hits", "alerts_reviewed"]


def add_recovery_efficiency(windows: pd.DataFrame) -> pd.DataFrame:
    """Per seed: (illicit caught on 43-49 by config - by static, same seed) / labels used over the whole stream."""
    post = windows[windows["window"] == "post_shutdown"].set_index(CFG_KEYS + ["seed"])
    total_labels = windows[windows["window"] == "all_test"].set_index(CFG_KEYS + ["seed"])["labels_used"]
    static = post.xs(("static", "none", 0), level=CFG_KEYS)["caught"]
    extra = post["caught"] - static.reindex(post.index.get_level_values("seed")).to_numpy()
    eff = extra / total_labels.reindex(post.index).replace(0, np.nan)
    out = windows.copy()
    key = out[CFG_KEYS + ["seed"]].apply(tuple, axis=1)
    is_post = out["window"] == "post_shutdown"
    out["extra_caught_vs_static"] = np.where(is_post, key.map(extra), np.nan)
    out["recovery_efficiency"] = np.where(is_post, key.map(eff), np.nan)
    out["total_labels"] = key.map(total_labels)
    return out


def summarise_stream(windows: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds per configuration and window."""
    cols = [c for c in STREAM_COLS + ["extra_caught_vs_static", "recovery_efficiency", "total_labels"]
            if c in windows.columns]
    agg = windows.groupby(CFG_KEYS + ["window"], sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def alarm_timeline(per_step: pd.DataFrame) -> pd.DataFrame:
    """Fraction of seeds in which each drift-triggered configuration's alarm fired, per step."""
    dt = per_step[per_step["policy"].str.startswith("drift_triggered")]
    return (dt.groupby(CFG_KEYS + ["step"], as_index=False)
              .agg(fired_share=("fired", "mean"), audit_misses_mean=("audit_misses", "mean")))


def _pick(summary: pd.DataFrame, policy: str, strategy: str, k: int, window: str = "post_shutdown") -> dict:
    row = summary.query("policy == @policy and strategy == @strategy and k == @k and window == @window")
    if row.empty:
        raise KeyError((policy, strategy, k, window))
    r = row.iloc[0].to_dict()
    return {k_: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k_, v in r.items()}


def headline(summary: pd.DataFrame) -> dict:
    """Headline numbers for the fixed comparisons (config.HEADLINE / HEADLINE_V2), plus best-in-grid for context."""
    adaptive = summary[(summary["window"] == "post_shutdown")
                       & summary["policy"].isin(["drift_triggered", "always", "drift_triggered_v2"])]
    best = adaptive.sort_values("pr_auc_mean", ascending=False).iloc[0]
    return {
        "window": "post_shutdown (steps 43-49)",
        "static": _pick(summary, "static", "none", 0),
        "driftguard": {**_pick(summary, **config.HEADLINE), "label": "pre-registered"},
        "driftguard_v2": {**_pick(summary, **config.HEADLINE_V2), "label": "post-hoc", "note": config.V2_NOTE},
        "full_retrain": _pick(summary, "full_retrain", "none", 0),
        "best_in_grid": {
            **_pick(summary, best["policy"], best["strategy"], int(best["k"])),
            "note": "Best adaptive configuration by post-shutdown PR-AUC, selected on test results: context only, "
                    "not a pre-registered claim.",
        },
    }


def git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=config.ROOT, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_results(windows: pd.DataFrame, per_step: pd.DataFrame) -> dict:
    """Assemble artifacts/results.json from pipeline outputs only."""
    import json
    from datetime import UTC, datetime

    art = config.ARTIFACTS_DIR
    windows = add_recovery_efficiency(windows)
    summary = summarise_stream(windows)
    static_meta = json.loads((art / "static_baselines.json").read_text())
    drift_meta = json.loads((art / "drift.json").read_text())
    dataset = json.loads((art / "dataset_stats.json").read_text())
    timeline = alarm_timeline(per_step)
    fired_steps = (timeline[timeline["fired_share"] > 0].groupby(CFG_KEYS)["step"].apply(list)
                   .reset_index().to_dict(orient="records"))
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": {
            "train_steps": [min(config.TRAIN_STEPS), max(config.TRAIN_STEPS)],
            "val_steps": [min(config.VAL_STEPS), max(config.VAL_STEPS)],
            "test_steps": [min(config.TEST_STEPS), max(config.TEST_STEPS)],
            "shutdown_step": config.SHUTDOWN_STEP,
            "windows": {k: [min(v), max(v)] for k, v in config.EVAL_WINDOWS.items()},
            "seeds": config.SEEDS,
            "k_values": config.K_VALUES,
            "policies": config.POLICIES + ["drift_triggered_v2"],
            "strategies": config.STRATEGIES,
            "audit_size": config.AUDIT_SIZE,
            "query_weight": config.QUERY_WEIGHT,
            "alert_budget": config.ALERT_BUDGET,
            "xgb_params": config.XGB_PARAMS,
            "headline_config": config.HEADLINE,
            "headline_v2_config": config.HEADLINE_V2,
        },
        "dataset": dataset,
        "static_baselines": {k: static_meta[k] for k in ("protocol", "best_static", "validation", "test_windows")},
        "drift": drift_meta,
        "headline": headline(summary),
        "adaptive_grid": summary.to_dict(orient="records"),
        "alarm_fired_steps": fired_steps,
    }


# ---------------------------------------------------------------------------
# Figures (PNG, light surface; categorical slots from a CVD-validated palette)
# ---------------------------------------------------------------------------
SERIES = {
    "Static": "#8a8985",
    "DriftGuard (pre-registered)": "#2a78d6",
    "DriftGuard v2 (post-hoc)": "#eb6834",
    "Full retrain (oracle)": "#1baf7a",
}
ALARM = "#d03b3b"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def _style(ax, title: str, ylabel: str, xlabel: str = "Time step") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _shutdown(ax) -> None:
    ax.axvline(config.SHUTDOWN_STEP - 0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.text(config.SHUTDOWN_STEP - 0.4, 0.98, "dark-market shutdown", transform=ax.get_xaxis_transform(),
            color=MUTED, fontsize=8, va="top")


def headline_series(per_step: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cfgs = {
        "Static": {"policy": "static", "strategy": "none", "k": 0},
        "DriftGuard (pre-registered)": config.HEADLINE,
        "DriftGuard v2 (post-hoc)": config.HEADLINE_V2,
        "Full retrain (oracle)": {"policy": "full_retrain", "strategy": "none", "k": 0},
    }
    out = {}
    for label, c in cfgs.items():
        sub = per_step[(per_step["policy"] == c["policy"]) & (per_step["strategy"] == c["strategy"])
                       & (per_step["k"] == c["k"])]
        out[label] = sub.groupby("step", as_index=False).agg(
            pr_auc=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"), recall=("recall", "mean"),
            fired=("fired", "mean"))
    return out


def make_figures(per_step: pd.DataFrame, summary: pd.DataFrame, drift: pd.DataFrame, pr: pd.DataFrame) -> list:
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = config.FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    series = headline_series(per_step)

    # 1. Timeline: PR-AUC and recall per step, alarms marked
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, facecolor=SURFACE)
    for metric, ax, title in (("pr_auc", axes[0], "PR-AUC (illicit) per time step"),
                              ("recall", axes[1], "Recall at the frozen threshold")):
        for label, s in series.items():
            wide = label == "Static"  # drawn wider underneath: DriftGuard lines sit almost exactly on top of it
            ax.plot(s["step"], s[metric], color=SERIES[label], linewidth=6 if wide else 2, alpha=0.6 if wide else 1,
                    marker=None if wide else "o", markersize=4, label=label, zorder=1 if wide else 2)
        _style(ax, title, "PR-AUC" if metric == "pr_auc" else "Recall", "")
        ax.set_ylim(0, 1.02)
        _shutdown(ax)
    fired = pd.concat([series[lbl][series[lbl]["fired"] > 0]
                       for lbl in ("DriftGuard (pre-registered)", "DriftGuard v2 (post-hoc)")])
    axes[0].scatter(fired["step"], fired["pr_auc"], s=110, facecolors="none", edgecolors=ALARM, linewidths=2,
                    zorder=5, label="Drift alarm fired (in any seed)")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    axes[1].set_xlabel("Time step", color=MUTED)
    fig.suptitle("Static vs DriftGuard vs full retrain (mean of 5 seeds)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    path = fig_dir / "timeline.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)

    # 2. Drift statistics (two charts, one axis each)
    test = drift.sort_values("step")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor=SURFACE)
    thr = json.loads((config.ARTIFACTS_DIR / "drift.json").read_text())
    for ax, col, t, title in ((axes[0], "psi", thr["psi_thr"], "Score PSI vs reference (steps 30-34)"),
                              (axes[1], "ks_mean", thr["ks_mean_thr"], "Mean KS statistic, top-20 features")):
        for split, color in (("train", "#8a8985"), ("val", "#4a3aa7"), ("test", "#2a78d6")):
            part = test[test["split"] == split]
            ax.plot(part["step"], part[col], color=color, linewidth=2, marker="o", markersize=4, label=split)
        ax.axhline(t, color=ALARM, linestyle=":", linewidth=1.5, label="alarm threshold (95th pct, steps 1-34)")
        _style(ax, title, col)
        _shutdown(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    path = fig_dir / "drift.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)

    # 3. Label budget: post-shutdown PR-AUC vs labels used
    post = summary[summary["window"] == "post_shutdown"]
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    groups = [("always", "Always query", "#2a78d6", "o"), ("drift_triggered", "Drift-triggered (random audit)",
              "#eb6834", "s"), ("drift_triggered_v2", "Drift-triggered v2 (post-hoc)", "#1baf7a", "D")]
    for policy, label, color, marker in groups:
        sub = post[post["policy"] == policy]
        ax.errorbar(sub["total_labels_mean"], sub["pr_auc_mean"], yerr=sub["pr_auc_std"], fmt=marker, color=color,
                    markersize=8, capsize=3, label=label, linestyle="none")
    for _, r in post[post["policy"] == "always"].iterrows():
        if r["k"] == 50:
            ax.annotate(r["strategy"], (r["total_labels_mean"], r["pr_auc_mean"]), textcoords="offset points",
                        xytext=(6, 0), fontsize=8, color=MUTED)
    st = post[post["policy"] == "static"].iloc[0]
    ax.axhline(st["pr_auc_mean"], color="#8a8985", linestyle="--", linewidth=1.2, label="Static (0 labels)")
    ax.set_xscale("log")
    _style(ax, "Post-shutdown PR-AUC (steps 43-49) vs analyst labels used", "PR-AUC", "Labels used over steps 35-49")
    fr = post[post["policy"] == "full_retrain"].iloc[0]
    ax.scatter([fr["total_labels_mean"]], [fr["pr_auc_mean"]], color="#4a3aa7", marker="*", s=180,
               label="Full retrain (oracle)", zorder=5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    path = fig_dir / "label_budget.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)

    # 4. Post-shutdown precision-recall curves (seed 0)
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
    for label, grp in pr.groupby("config", sort=False):
        ax.plot(grp["recall"], grp["precision"], color=SERIES.get(label, "#4a3aa7"), linewidth=2, label=label)
    _style(ax, "Precision-recall, steps 43-49 (seed 0)", "Precision", "Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = fig_dir / "pr_curves.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)
    return written
