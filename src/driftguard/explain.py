"""SHAP explanations for alerts, using the model that actually scored each test step.

Stored per transaction: txId, step, score, true label, outcome and the top-5 features by |SHAP| with their SHAP
contributions. Raw feature values are never written (Elliptic licence: no redistribution).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from driftguard import config
from driftguard.stream import AlarmConfig, SeedContext, run_one

TOP_N = 5


def top_contributions(shap_values: np.ndarray, feature_names: list[str], n: int = TOP_N) -> list[dict]:
    """Top-n features by absolute SHAP value for every row (feature name + signed contribution)."""
    order = np.argsort(-np.abs(shap_values), axis=1, kind="stable")[:, :n]
    out = []
    for i, idx in enumerate(order):
        row = {}
        for j, f in enumerate(idx, start=1):
            row[f"feature_{j}"] = feature_names[f]
            row[f"shap_{j}"] = float(shap_values[i, f])
        out.append(row)
    return out


def outcome(y: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    return np.select([flagged & (y == 1), flagged & (y == 0), ~flagged & (y == 1)],
                     ["true_positive", "false_positive", "missed_illicit"], "true_negative")


def explain_stream(ctx: SeedContext, alarms: AlarmConfig, cfg: dict, feature_names: list[str]):
    """Re-run one configuration (deterministic for a seed) and explain flagged + missed-illicit rows per step.

    Returns (alerts DataFrame, mean |SHAP| per feature for pre/post-shutdown illicit transactions).
    """
    alerts, importance = [], []

    def on_predict(sd, model, scores, threshold):
        y = sd.rows["y"].to_numpy().astype(int)
        flagged = scores >= threshold
        keep = flagged | (y == 1)
        if not keep.any():
            return
        sv = shap.TreeExplainer(model).shap_values(sd.X[keep])
        top = top_contributions(sv, feature_names)
        kept = sd.rows[keep].reset_index(drop=True)
        outs = outcome(y[keep], flagged[keep])
        for i, r in enumerate(kept.itertuples(index=False)):
            alerts.append({"txId": int(r.txId), "step": int(r.step), "score": float(scores[keep][i]),
                           "y": int(r.y), "outcome": outs[i], "flagged": bool(flagged[keep][i]), **top[i]})
        ill = y[keep] == 1
        if ill.any():
            window = "post_shutdown" if sd.step >= config.SHUTDOWN_STEP else "pre_shutdown"
            importance.append(pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(sv[ill]).mean(0),
                                            "n": int(ill.sum()), "window": window}))

    run_one(ctx, alarms, cfg["policy"], cfg["strategy"], cfg["k"], n_jobs_model=-1, on_predict=on_predict)
    imp = pd.concat(importance, ignore_index=True)
    imp = (imp.assign(w=imp["mean_abs_shap"] * imp["n"]).groupby(["window", "feature"], as_index=False)
              .agg(w=("w", "sum"), n=("n", "sum")))
    imp["mean_abs_shap"] = imp["w"] / imp["n"]
    return pd.DataFrame(alerts).sort_values(["step", "score"], ascending=[True, False]), imp.drop(columns="w")


def shap_figure(imp: pd.DataFrame, path, top: int = 15) -> None:
    """Mean |SHAP| for illicit transactions before vs after the shutdown (no raw feature values)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from driftguard.evaluate import GRID, INK, MUTED, SURFACE

    wide = imp.pivot(index="feature", columns="window", values="mean_abs_shap").fillna(0)
    wide = wide.sort_values("pre_shutdown", ascending=True).tail(top)
    y = np.arange(len(wide))
    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=SURFACE)
    ax.barh(y + 0.2, wide["pre_shutdown"], height=0.38, color="#2a78d6", label="Illicit, steps 35-42")
    ax.barh(y - 0.2, wide["post_shutdown"], height=0.38, color="#eb6834", label="Illicit, steps 43-49")
    ax.set_yticks(y, wide.index)
    ax.set_facecolor(SURFACE)
    ax.set_title("What drives the model's score for illicit transactions (mean |SHAP|)", loc="left", color=INK)
    ax.set_xlabel("Mean |SHAP value| (log-odds)", color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
