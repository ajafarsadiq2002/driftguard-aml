"""Home page: problem, headline numbers and findings. Reads only artifacts/."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import common as c

c.page_header()
R = c.results()
G = c.grid()
H = R["headline"]

static_pre = c.cfg_rows(G, "static", "none", 0).query("window == 'pre_shutdown'").iloc[0]
static_post, dg, v2, oracle = H["static"], H["driftguard"], H["driftguard_v2"], H["full_retrain"]
alerts = c.parquet("alerts")
missed = alerts[alerts["outcome"] == "missed_illicit"]
miss_pre = missed[missed["step"] < c.SHUTDOWN_STEP]["score"].median()
miss_post = missed[missed["step"] >= c.SHUTDOWN_STEP]["score"].median()
drift = R["drift"]

st.title("🛡️ DriftGuard AML")
st.markdown("#### Your AML model can fail silently. We measured how badly, and what it takes to recover.")
st.markdown(
    "Fraud models are usually trained once and assumed to keep working. On the Elliptic Bitcoin dataset, a "
    "real dark-market shutdown around **time step 43** changes criminal behaviour. DriftGuard trains a detector on "
    "early time steps, streams through later ones, watches for drift, asks a simulated analyst for a small "
    "number of labels, retrains, and reports honestly whether that recovers the lost performance."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Static PR-AUC, steps 35–42", f"{static_pre['pr_auc_mean']:.3f}", help="Before the shutdown")
k2.metric("Static PR-AUC, steps 43–49", f"{static_post['pr_auc_mean']:.3f}",
          delta=f"{static_post['pr_auc_mean'] - static_pre['pr_auc_mean']:.3f}", help="After the shutdown")
k3.metric("DriftGuard PR-AUC, steps 43–49", c.pm(dg["pr_auc_mean"], dg["pr_auc_std"]),
          delta=f"{dg['pr_auc_mean'] - static_post['pr_auc_mean']:+.3f} vs static", delta_color="off",
          help=f"Pre-registered configuration: {dg['policy']}, {dg['strategy']}, K={int(dg['k'])}")
k4.metric("DriftGuard analyst labels", f"{dg['total_labels_mean']:.0f}",
          delta=f"{dg['extra_caught_vs_static_mean']:+.1f} illicit caught vs static", delta_color="off",
          help="Analyst labels over steps 35–49; extra illicit caught on 43–49 vs static (same seed)")
st.caption("Pooled over the window, mean ± std over 5 seeds. Every number on this page comes from results.json.")

st.subheader("What we found")
st.markdown(
    f"""
1. **The model collapses.** Static XGBoost goes from **{static_pre['pr_auc_mean']:.3f}** PR-AUC before the
   shutdown to **{static_post['pr_auc_mean']:.3f}** after it, catching **{static_post['caught_mean']:.1f}** of
   **{static_post['n_illicit_mean']:.0f}** illicit transactions.
2. **It fails silently.** Label-free drift monitors (score PSI, feature KS) fired on
   **{len(drift['unsupervised_fired_test_steps'])}** of 15 test steps. A random 10-label audit per step rarely
   contains an illicit case once they become rare, so the pre-registered DriftGuard alarm almost never fires.
3. **The misses are confident.** The median score of a missed illicit transaction drops from
   **{miss_pre:.4f}** before the shutdown to **{miss_post:.4f}** after it, so lowering the threshold cannot help, and
   uncertainty-based audits (our post-hoc v2) go quiet exactly when the model breaks.
4. **Recovery needs real label budgets.** DriftGuard (pre-registered) reaches **{c.pm(dg['pr_auc_mean'],
   dg['pr_auc_std'])}** PR-AUC with {dg['total_labels_mean']:.0f} labels. The oracle that retrains on every
   label reaches **{c.pm(oracle['pr_auc_mean'], oracle['pr_auc_std'])}** with
   {oracle['total_labels_mean']:,.0f} labels. See *Label Budget* for everything in between.
"""
)

st.subheader("Post-shutdown PR-AUC (steps 43–49)")
bars = [("Static", static_post), ("DriftGuard (pre-registered)", dg), ("DriftGuard v2 (post-hoc)", v2),
        ("Full retrain (oracle)", oracle)]
fig = go.Figure(go.Bar(
    x=[b[1]["pr_auc_mean"] for b in bars], y=[b[0] for b in bars], orientation="h",
    error_x=dict(type="data", array=[b[1]["pr_auc_std"] for b in bars], color=c.MUTED),
    marker_color=[c.COLORS[b[0]] for b in bars],
    text=[f"  {b[1]['pr_auc_mean']:.3f}  ·  {b[1]['total_labels_mean']:,.0f} labels" for b in bars],
    textposition="outside", hovertemplate="%{y}: PR-AUC %{x:.3f}<extra></extra>",
))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(range=[0, max(b[1]["pr_auc_mean"] for b in bars) * 1.45], title="PR-AUC (illicit)")
st.plotly_chart(c.style(fig, height=300, hovermode="closest", showlegend=False), width="stretch")

st.markdown(
    "**Explore:** *Timeline* shows the collapse step by step with drift alarms. *Alert Queue* explains every "
    "flagged and missed transaction with SHAP. *Label Budget* compares all policies, strategies and K. "
    "*Methodology* covers splits, leakage safeguards, baselines and limitations."
)
c.citation()
