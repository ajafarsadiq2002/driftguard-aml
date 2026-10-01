"""Per-step performance with drift alarms, and the drift statistics underneath."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import common as c

c.page_header()
R = c.results()
per_step = c.parquet("per_step")
drift = c.parquet("drift")

st.title("📈 Timeline")
st.markdown("Each test step is scored **before** any of its labels are used (test-then-train). Mean of 5 seeds.")

metric = st.radio("Metric", ["pr_auc", "recall", "precision", "f1"], horizontal=True,
                  format_func=lambda m: {"pr_auc": "PR-AUC", "recall": "Recall", "precision": "Precision",
                                         "f1": "F1"}[m])

configs = c.headline_configs()
with st.expander("Add another configuration to the chart"):
    opts = (per_step[["policy", "strategy", "k"]].drop_duplicates()
            .query("policy in ['always', 'drift_triggered', 'drift_triggered_v2']")
            .sort_values(["policy", "strategy", "k"]).to_dict("records"))
    extra = st.selectbox("Configuration", [None, *opts], format_func=lambda o: "(none)" if o is None else
                         f"{c.POLICY_LABELS[o['policy']]} · {o['strategy']} · K={o['k']}")
    if extra:
        configs["Selected configuration"] = extra

fig = go.Figure()
alarm_x, alarm_y = [], []
for label, cfg in configs.items():
    sub = c.cfg_rows(per_step, **cfg).groupby("step", as_index=False).agg(
        v=(metric, "mean"), sd=(metric, "std"), fired=("fired", "mean"))
    color = c.COLORS.get(label, c.COLORS["extra"])
    fig.add_trace(go.Scatter(
        x=sub["step"], y=sub["v"], name=label, mode="lines+markers",
        line=dict(color=color, width=5 if label == "Static" else 2.5),
        opacity=0.55 if label == "Static" else 1, marker=dict(size=7),
        customdata=sub["sd"], hovertemplate=f"{label}: %{{y:.3f}} ± %{{customdata:.3f}}<extra></extra>",
    ))
    if cfg["policy"].startswith("drift_triggered"):
        f = sub[sub["fired"] > 0]
        alarm_x += f["step"].tolist()
        alarm_y += f["v"].tolist()
if alarm_x:
    fig.add_trace(go.Scatter(x=alarm_x, y=alarm_y, mode="markers", name="Drift alarm fired (any seed)",
                             marker=dict(symbol="circle-open", size=18, color=c.ALARM, line=dict(width=3)),
                             hoverinfo="skip"))
c.shutdown_marker(fig)
fig.update_yaxes(range=[0, 1.05], title=metric.replace("_", "-").upper() if metric == "pr_auc" else metric.title())
fig.update_xaxes(title="Time step", dtick=1)
st.plotly_chart(c.style(fig, height=430), width="stretch")
st.caption("The Static line is drawn wide underneath: the DriftGuard lines sit almost exactly on top of it, "
           "because their alarms rarely fire.")

st.subheader("Drift statistics (no labels needed)")
d = R["drift"]
cols = st.columns(2)
for col, stat, thr, title in ((cols[0], "psi", d["psi_thr"], "Score PSI vs reference (steps 30–34)"),
                              (cols[1], "ks_mean", d["ks_mean_thr"], "Mean KS statistic, top-20 features")):
    f2 = go.Figure()
    split_colors = (("train", c.MUTED), ("val", c.COLORS["extra"]), ("test", c.COLORS["DriftGuard (pre-registered)"]))
    for split, color in split_colors:
        part = drift[drift["split"] == split].sort_values("step")
        f2.add_trace(go.Scatter(x=part["step"], y=part[stat], name=f"{split} steps", mode="lines+markers",
                                line=dict(color=color, width=2), marker=dict(size=6)))
    f2.add_hline(y=thr, line_dash="dot", line_color=c.ALARM, annotation_text="alarm threshold",
                 annotation_font_color=c.ALARM)
    c.shutdown_marker(f2)
    f2.update_layout(title=title)
    f2.update_xaxes(title="Time step")
    col.plotly_chart(c.style(f2, height=360, legend=dict(orientation="h", y=-0.3, x=0, bgcolor="rgba(0,0,0,0)"),
                             margin=dict(l=10, r=10, t=50, b=10)), width="stretch")

st.info(
    f"Thresholds are the 95th percentile of each statistic over steps 1–34 only (PSI {d['psi_thr']:.3f}, mean "
    f"KS {d['ks_mean_thr']:.3f}). Neither fires at step 43: illicit transactions are about 1–2% of a step, so "
    "the overall score and feature distributions barely move when their behaviour changes. The share of features "
    "with KS p < 0.01 (the originally specified rule) saturates at 0.65–0.95 even between validation steps, so it is "
    "recorded but not used."
)
c.citation()
