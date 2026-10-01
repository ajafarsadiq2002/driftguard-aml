"""DriftGuard AML dashboard. Reads only the committed files in artifacts/ (no raw data)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
SHUTDOWN_STEP = 43
LABELS = {"trivial": "Trivial (prior)", "logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}

st.set_page_config(page_title="DriftGuard AML", page_icon="🛡️", layout="wide")
st.title("🛡️ DriftGuard AML")
st.caption(
    "Research prototype only. Not a financial, compliance, or AML product; not financial advice. "
    "No real funds or live systems are involved."
)

st.markdown(
    "Fraud models are usually trained once and assumed to keep working. On the Elliptic Bitcoin dataset, a "
    "dark-market shutdown around **time step 43** changes criminal behaviour, and models trained on earlier "
    "steps stop catching illicit transactions. DriftGuard watches for that drift and asks an analyst for a "
    "small number of labels to recover."
)


@st.cache_data
def load_static():
    meta = json.loads((ARTIFACTS / "static_baselines.json").read_text())
    per_step = pd.read_parquet(ARTIFACTS / "static_per_step.parquet")
    return meta, per_step


meta, per_step = load_static()
windows = pd.DataFrame(meta["test_windows"])
best = meta["best_static"]

pre = windows.query("model == @best['model'] and feature_set == @best['feature_set'] and window == 'pre_shutdown'")
post = windows.query("model == @best['model'] and feature_set == @best['feature_set'] and window == 'post_shutdown'")

c1, c2, c3 = st.columns(3)
c1.metric("Static PR-AUC, steps 35–42", f"{pre.pr_auc_mean.iloc[0]:.3f}", help="Before the shutdown")
c2.metric(
    "Static PR-AUC, steps 43–49",
    f"{post.pr_auc_mean.iloc[0]:.3f}",
    delta=f"{post.pr_auc_mean.iloc[0] - pre.pr_auc_mean.iloc[0]:.3f}",
    help="After the shutdown",
)
c3.metric("Static recall, steps 43–49", f"{post.recall_mean.iloc[0]:.1%}")
st.caption(
    f"Best static model on validation: {LABELS[best['model']]} ({best['feature_set']} features). Mean of 5 seeds."
)

st.subheader("PR-AUC per time step (static models, full feature set)")
full = per_step[per_step["feature_set"] == "local_agg_graph"]
curve = full.groupby(["model", "step"], as_index=False)["pr_auc"].mean()
fig = go.Figure()
for model, grp in curve.groupby("model"):
    fig.add_trace(go.Scatter(x=grp["step"], y=grp["pr_auc"], mode="lines+markers", name=LABELS[model]))
fig.add_vline(x=SHUTDOWN_STEP - 0.5, line_dash="dash", annotation_text="dark-market shutdown")
fig.update_layout(xaxis_title="Time step", yaxis_title="PR-AUC (illicit)", yaxis_range=[0, 1.02], height=420)
st.plotly_chart(fig, width="stretch")

st.subheader("Static baselines (test windows, mean ± std over 5 seeds)")
table = windows.assign(
    Model=windows["model"].map(LABELS),
    Features=windows["feature_set"],
    Window=windows["window"],
    **{
        "PR-AUC": windows.apply(lambda r: f"{r.pr_auc_mean:.3f} ± {r.pr_auc_std:.3f}", axis=1),
        "F1": windows.apply(lambda r: f"{r.f1_mean:.3f} ± {r.f1_std:.3f}", axis=1),
        "Recall": windows.apply(lambda r: f"{r.recall_mean:.3f} ± {r.recall_std:.3f}", axis=1),
    },
)[["Model", "Features", "Window", "PR-AUC", "F1", "Recall"]]
st.dataframe(table, hide_index=True, width="stretch")

st.caption(
    "Data: Elliptic Data Set (Weber et al., 2019), CC BY-NC-ND 4.0, not redistributed here. "
    "Only derived metrics are shown."
)
