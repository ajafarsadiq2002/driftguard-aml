"""Flagged (and missed) test transactions with their top-5 SHAP reasons."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import common as c

c.page_header()
R = c.results()
alerts = c.parquet("alerts")
cfg = R["config"]["headline_config"]

st.title("🚨 Alert Queue")
st.markdown(
    f"Test transactions scored by the pre-registered DriftGuard configuration ({cfg['policy']}, {cfg['strategy']}, "
    f"K={cfg['k']}, seed {R['config']['seeds'][0]}). Every row is explained with **SHAP** using the model that "
    "actually scored that step. Only feature names and SHAP contributions are stored, never raw feature values."
)

OUTCOMES = {"true_positive": "✅ Flagged, illicit", "false_positive": "⚠️ Flagged, licit",
            "missed_illicit": "❌ Missed illicit"}
f1, f2 = st.columns([2, 3])
steps = sorted(alerts["step"].unique())
step_sel = f1.select_slider("Time steps", options=steps, value=(steps[0], steps[-1]))
VIEWS = {"Flagged alerts": ["true_positive", "false_positive"], "Missed illicit": ["missed_illicit"],
         "All": list(OUTCOMES)}
out_sel = VIEWS[f2.radio("Show", list(VIEWS), horizontal=True)]

view = alerts[alerts["step"].between(*step_sel) & alerts["outcome"].isin(out_sel)]
view = view.sort_values("score", ascending=False).reset_index(drop=True)

m1, m2, m3 = st.columns(3)
m1.metric("Rows shown", f"{len(view):,}")
m2.metric("Flagged that were illicit", f"{(view['outcome'] == 'true_positive').sum():,}")
m3.metric("Missed illicit in range",
          f"{((alerts['outcome'] == 'missed_illicit') & alerts['step'].between(*step_sel)).sum():,}")

table = view.assign(
    Outcome=view["outcome"].map(OUTCOMES), Risk=view["score"].round(4),
    **{"Top reason": view["feature_1"] + " (" + view["shap_1"].map("{:+.2f}".format) + ")"},
)[["txId", "step", "Risk", "Outcome", "Top reason"]].rename(columns={"step": "Step"})
st.caption("Select a row to see why the model scored it that way.")
event = st.dataframe(table, hide_index=True, width="stretch", height=360, on_select="rerun",
                     selection_mode="single-row",
                     column_config={"Risk": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1,
                                                                            format="%.4f")})

rows = event.selection.rows if event and event.selection else []
if not len(view):
    st.warning("No transactions match the filters.")
else:
    r = view.iloc[rows[0] if rows else 0]
    st.subheader(f"Why transaction {r['txId']} (step {r['step']}) scored {r['score']:.4f}")
    if not rows:
        st.caption("Showing the highest-risk row. Select a row in the table to change it.")
    feats = [r[f"feature_{i}"] for i in range(1, 6)][::-1]
    vals = [r[f"shap_{i}"] for i in range(1, 6)][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker_color=[c.ALARM if v > 0 else c.COLORS["DriftGuard (pre-registered)"] for v in vals],
        text=[f"{v:+.2f}" for v in vals], textposition="outside",
        hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>",
    ))
    span = max(abs(v) for v in vals) * 1.35
    fig.update_xaxes(range=[-span, span], title="SHAP contribution (log-odds); red pushes towards illicit")
    st.plotly_chart(c.style(fig, height=300, hovermode="closest", showlegend=False), width="stretch")
    st.markdown(f"**Outcome:** {OUTCOMES[r['outcome']]} · **true label:** {'illicit' if r['y'] == 1 else 'licit'}")

st.subheader("Why the misses happen")
imp = c.parquet("shap_importance")
wide = imp.pivot(index="feature", columns="window", values="mean_abs_shap").fillna(0)
wide = wide.sort_values("pre_shutdown", ascending=False).head(12).iloc[::-1]
fig = go.Figure()
fig.add_trace(go.Bar(y=wide.index, x=wide["pre_shutdown"], orientation="h", name="Illicit, steps 35–42",
                     marker_color=c.COLORS["DriftGuard (pre-registered)"]))
fig.add_trace(go.Bar(y=wide.index, x=wide["post_shutdown"], orientation="h", name="Illicit, steps 43–49",
                     marker_color=c.COLORS["DriftGuard v2 (post-hoc)"]))
fig.update_xaxes(title="Mean |SHAP| for illicit transactions")
st.plotly_chart(c.style(fig, height=420, barmode="group", hovermode="y unified"), width="stretch")
missed = alerts[alerts["outcome"] == "missed_illicit"]
miss_pre = missed[missed.step < c.SHUTDOWN_STEP].score.median()
miss_post = missed[missed.step >= c.SHUTDOWN_STEP].score.median()
st.markdown(
    "The features that signalled illicit activity before the shutdown contribute far less afterwards. "
    f"The median score of a missed illicit transaction is **{miss_pre:.4f}** "
    f"before and **{miss_post:.4f}** after: the model is confidently "
    "wrong, not uncertain."
)
c.citation()
