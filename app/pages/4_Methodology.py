"""Splits, leakage safeguards, baselines, ablations, limitations, citation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import common as c

c.page_header()
R = c.results()
cfg, ds, d = R["config"], R["dataset"], R["drift"]

st.title("📚 Methodology & Results")

st.header("Data")
a, b, e, f = st.columns(4)
a.metric("Transactions", f"{ds['rows']:,}")
b.metric("Time steps", ds["time_steps"])
e.metric("Illicit / licit", f"{ds['illicit']:,} / {ds['licit']:,}")
f.metric("Labelled share", f"{ds['labeled_share']:.1%}")
st.markdown(
    "Elliptic Bitcoin transaction graph: 93 local + 72 aggregated anonymised features per transaction, plus "
    "in/out-degree from the payment edge list. Supervised training and evaluation use labelled rows only. Drift "
    "statistics use every transaction, because they need no labels."
)

st.header("Time-based splits")
st.table(pd.DataFrame({
    "Split": ["Train", "Validation", "Test / stream"],
    "Steps": [f"{cfg['train_steps'][0]}–{cfg['train_steps'][1]}", f"{cfg['val_steps'][0]}–{cfg['val_steps'][1]}",
              f"{cfg['test_steps'][0]}–{cfg['test_steps'][1]}"],
    "Used for": ["Fitting models", "Model selection, decision threshold, drift reference",
                 "Final evaluation and the streaming loop only"],
}))

st.header("Leakage safeguards")
st.markdown(
    f"""
- Static models are trained on steps {cfg['train_steps'][0]}–{cfg['train_steps'][1]}. The decision threshold
  (maximising illicit F1) and model choice use validation only. Test steps never touch fitting or tuning, and
  `assert_no_leakage` enforces this.
- **Test-then-train streaming:** each step is scored and its metrics recorded *before* any of its labels are
  used. The loop raises an error if a model trained on step ≥ t predicts step t, and a test checks this for
  every policy.
- The simulated analyst can only label ground-truth rows **from the current step**; tests enforce this.
- Drift thresholds are the {d['quantile']:.0%} quantile over steps 1–34 only, using out-of-fold scores for
  training steps and leave-one-step-out for validation steps. Calibration refuses any step ≥ 35.
- The headline configuration, audit size ({cfg['audit_size']}), label weight ({cfg['query_weight']:g}) and alert
  budget ({cfg['alert_budget']}) were fixed before aggregating test results.
- Five seeds; every stochastic number is reported as mean ± std.
"""
)

st.header("Static baselines")
st.caption(R["static_baselines"]["protocol"] + ". Mean ± std over 5 seeds.")
sw = pd.DataFrame(R["static_baselines"]["test_windows"])
tbl = sw.assign(
    Model=sw["model"].map(c.MODEL_LABELS), Features=sw["feature_set"], Window=sw["window"].map(c.WINDOW_LABELS),
    **{"PR-AUC": [c.pm(m, s) for m, s in zip(sw["pr_auc_mean"], sw["pr_auc_std"], strict=True)],
       "F1": [c.pm(m, s) for m, s in zip(sw["f1_mean"], sw["f1_std"], strict=True)],
       "Recall": [c.pm(m, s) for m, s in zip(sw["recall_mean"], sw["recall_std"], strict=True)]},
)[["Model", "Features", "Window", "PR-AUC", "F1", "Recall"]]
st.dataframe(tbl, hide_index=True, width="stretch")
best = R["static_baselines"]["best_static"]
st.markdown(
    f"**Base model:** {c.MODEL_LABELS[best['model']]} on `{best['feature_set']}` features, chosen by validation "
    f"PR-AUC ({best['val_pr_auc']:.3f}). The feature-set ablation (local → + aggregated → + graph degree) changes "
    "pre-shutdown PR-AUC by about 0.02 and does not prevent the post-shutdown collapse. The full-feature model "
    "scores slightly higher on test, but choosing on test would be leakage."
)

st.header("Drift detection")
st.markdown(
    f"""
- **Unsupervised monitor:** score PSI (threshold {d['psi_thr']:.3f}) and mean KS statistic over the top-20
  features (threshold {d['ks_mean_thr']:.3f}), measured against steps 30–34. Fired on test steps:
  **{d['unsupervised_fired_test_steps'] or 'none'}**.
- The originally specified rule (share of features with KS p < 0.01) saturates with thousands of
  transactions per step, at 0.65–0.95 even between validation steps. We replaced it with the mean KS statistic,
  justified on steps 1–34 alone.
- **Audit alarm (pre-registered):** {d['audit_size']} random labelled transactions per step. It fires when more
  than {d['audit_misses_thr']:.0f} of them are illicit and missed.
- **v2 (post-hoc):** {d['v2_note']} Threshold: more than {d['v2_uncertainty_audit_misses_thr']:.0f} misses.
"""
)

st.header("Honest limitations")
st.markdown(
    """
- **The adaptive method does not recover performance.** The pre-registered DriftGuard is within noise of
  static after the shutdown. Only large label budgets (always querying 50 per step, or full retraining) help.
- Features are anonymised, so SHAP reasons name features (`local_53`) but cannot be read as business rules.
- The analyst is simulated. Real labels arrive late, cost money and can be wrong.
- Only about 23% of transactions have labels, and after the shutdown some steps contain as few as 2 illicit
  cases, so per-step metrics are noisy. We therefore report pooled windows.
- Single dataset and single shift event; post-deployment labels get a fixed sample weight of 10.
- The v2 design was created after seeing test results and is reported separately for that reason.
"""
)

st.header("Dataset, licence, disclosure")
st.markdown(
    "Weber, M. et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks "
    "for Financial Forensics.* KDD Workshop on Anomaly Detection in Finance. Elliptic Data Set, **CC BY-NC-ND "
    "4.0**. The raw data is not redistributed; this app shows only derived metrics.\n\n"
    "Most of this project's code was written by Claude Code (an AI coding agent) under the team's direction. See "
    "`docs/AI_DISCLOSURE.md`. Code licence: MIT."
)
st.caption(f"Results generated {R['generated_at']} from commit {R['git_commit']}.")
