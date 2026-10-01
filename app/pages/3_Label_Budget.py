"""How much does each analyst-label budget buy after the shutdown?"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import common as c

c.page_header()
R = c.results()
G = c.grid()
post = G[G["window"] == "post_shutdown"]
static = c.cfg_rows(post, "static", "none", 0).iloc[0]
oracle = c.cfg_rows(post, "full_retrain", "none", 0).iloc[0]

st.title("🏷️ Label Budget")
st.markdown(
    "Choose when the analyst is asked (policy), which transactions they label (strategy) and how many per query "
    "(K). Results are for the post-shutdown window, steps 43–49, mean ± std over 5 seeds."
)

c1, c2, c3 = st.columns([2, 2, 1.4])
policy = c1.radio("When to query", ["drift_triggered", "always", "drift_triggered_v2"],
                  format_func=c.POLICY_LABELS.get)
strategies = sorted(post[post["policy"] == policy]["strategy"].unique())
strategy = c2.radio("Which transactions", strategies, horizontal=True,
                    index=strategies.index("hybrid") if "hybrid" in strategies else 0)
k = c3.select_slider("K per query", options=R["config"]["k_values"], value=25)
sel = c.cfg_rows(post, policy, strategy, k).iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("PR-AUC, steps 43–49", c.pm(sel["pr_auc_mean"], sel["pr_auc_std"]),
          delta=f"{sel['pr_auc_mean'] - static['pr_auc_mean']:+.3f} vs static", delta_color="off")
m2.metric("Recall at frozen threshold", f"{sel['recall_mean']:.1%}",
          delta=f"{(sel['recall_mean'] - static['recall_mean']) * 100:+.1f} pts vs static", delta_color="off")
m3.metric("Labels used (steps 35–49)", f"{sel['total_labels_mean']:,.0f}")
m4.metric(f"Illicit in top-{R['config']['alert_budget']} alerts / step, summed", f"{sel['alert_hits_mean']:.1f}",
          delta=f"{sel['alert_hits_mean'] - static['alert_hits_mean']:+.1f} vs static", delta_color="off")
eff = sel["recovery_efficiency_mean"]
st.caption(
    f"Recovery efficiency (extra illicit caught vs static ÷ labels used): **{eff:.4f}** "
    f"(≈ {1 / eff:,.0f} labels per extra illicit catch). " if eff and eff > 0 else
    "Recovery efficiency: no extra illicit caught at the frozen threshold compared with static. "
)
if policy == "drift_triggered_v2":
    st.warning(R["drift"]["v2_note"])

fig = go.Figure()
groups = [("always", "circle", c.COLORS["DriftGuard (pre-registered)"]),
          ("drift_triggered", "square", c.COLORS["DriftGuard v2 (post-hoc)"]),
          ("drift_triggered_v2", "diamond", c.COLORS["Full retrain (oracle)"])]
for pol, symbol, color in groups:
    sub = post[post["policy"] == pol]
    fig.add_trace(go.Scatter(
        x=sub["total_labels_mean"], y=sub["pr_auc_mean"], mode="markers", name=c.POLICY_LABELS[pol],
        marker=dict(symbol=symbol, size=11, color=color, line=dict(width=1, color="#0b1426")),
        error_y=dict(type="data", array=sub["pr_auc_std"], color=color, thickness=1),
        customdata=sub[["strategy", "k"]],
        hovertemplate="%{customdata[0]}, K=%{customdata[1]}<br>labels %{x:,.0f}<br>PR-AUC %{y:.3f}<extra></extra>",
    ))
fig.add_trace(go.Scatter(x=[sel["total_labels_mean"]], y=[sel["pr_auc_mean"]], mode="markers", name="Your selection",
                         marker=dict(symbol="circle-open", size=24, color=c.INK, line=dict(width=2)),
                         hoverinfo="skip"))
fig.add_trace(go.Scatter(x=[oracle["total_labels_mean"]], y=[oracle["pr_auc_mean"]], mode="markers",
                         name="Full retrain (oracle)", marker=dict(symbol="star", size=18, color=c.COLORS["extra"]),
                         hovertemplate="Full retrain: %{x:,.0f} labels, PR-AUC %{y:.3f}<extra></extra>"))
fig.add_hline(y=static["pr_auc_mean"], line_dash="dash", line_color=c.MUTED,
              annotation_text=f"static {static['pr_auc_mean']:.3f}", annotation_font_color=c.MUTED)
ticks = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
fig.update_xaxes(type="log", title="Analyst labels used over steps 35–49 (log scale)", tickvals=ticks,
                 ticktext=[f"{v:,}" for v in ticks])
fig.update_yaxes(title="PR-AUC, steps 43–49")
st.plotly_chart(c.style(fig, height=460, hovermode="closest"), width="stretch")

st.subheader("All configurations")
tbl = post.assign(
    Policy=post["policy"].map(c.POLICY_LABELS), Strategy=post["strategy"], K=post["k"].astype(int),
    **{"PR-AUC": [c.pm(m, s) for m, s in zip(post["pr_auc_mean"], post["pr_auc_std"], strict=True)],
       "Recall": post["recall_mean"].map("{:.1%}".format),
       "Illicit caught": post["caught_mean"].round(1),
       "Labels used": post["total_labels_mean"].round(0).astype(int),
       "Top-alert hits": post["alert_hits_mean"].round(1)},
)[["Policy", "Strategy", "K", "PR-AUC", "Recall", "Illicit caught", "Labels used", "Top-alert hits"]]
st.dataframe(tbl, hide_index=True, width="stretch")
best = R["headline"]["best_in_grid"]
st.caption(f"Best adaptive configuration by post-shutdown PR-AUC: {c.POLICY_LABELS[best['policy']]}, "
           f"{best['strategy']}, K={int(best['k'])}. {best['note']}")
c.citation()
