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
best = H["best_in_grid"]
ex = R["explanations"]["windows"]
miss_pre = ex["pre_shutdown"]["missed_illicit_median_score"]
miss_post = ex["post_shutdown"]["missed_illicit_median_score"]
drift = R["drift"]
n_test = R["config"]["test_steps"][1] - R["config"]["test_steps"][0] + 1

st.title("🛡️ DriftGuard AML")
st.markdown("#### A blind-spot auditor for AML models: when the model is confidently wrong, why, and what it "
            "takes to recover.")
st.markdown(
    "Fraud models are usually trained once and assumed to keep working. On the Elliptic Bitcoin dataset, a "
    "real dark-market shutdown around **time step 43** changes criminal behaviour. DriftGuard replays that shift "
    "step by step with no look-ahead: it **audits** whether the model still catches illicit activity, **explains** "
    "every alert and every miss with SHAP, **checks** whether standard drift alarms notice, and **plans** how many "
    "analyst reviews it would take to recover."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Model PR-AUC: before → after shutdown",
          f"{static_pre['pr_auc_mean']:.3f} → {static_post['pr_auc_mean']:.3f}",
          help="Static XGBoost, steps 35–42 vs 43–49, pooled, mean of 5 seeds")
k2.metric("Median score of a missed illicit tx", f"{miss_post:.4f}",
          delta=f"was {miss_pre:.4f} before the shutdown", delta_color="off",
          help="Confident misses: the model is sure these are licit, so lowering the threshold cannot help")
k3.metric("Label-free drift alarms fired", f"{len(drift['unsupervised_fired_test_steps'])} of {n_test} steps",
          help="Score PSI + mean KS statistic, thresholds calibrated on steps 1–34 only")
k4.metric(f"Illicit in top-{R['config']['alert_budget']} alerts, steps 43–49",
          f"{static_post['alert_hits_mean']:.0f} → {best['alert_hits_mean']:.0f}",
          delta=f"with {best['k']:.0f} uncertainty reviews/step", delta_color="off",
          help="Static vs the best adaptive configuration (always query, uncertainty, K=50). Selected on test "
               "results, so this is context for planning, not a pre-registered claim.")
st.caption("Pooled over the window, mean over 5 seeds. Every number on this page comes from results.json.")

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
