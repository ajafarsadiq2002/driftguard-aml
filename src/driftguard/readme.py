"""Regenerate the README's numeric sections from artifacts/results.json (never hand-edit numbers).

Generated blocks sit between `<!-- BEGIN:name -->` and `<!-- END:name -->` markers in README.md.
"""

from __future__ import annotations

import json
import re

from driftguard import config

README = config.ROOT / "README.md"
MODEL_LABELS = {"trivial": "Trivial (prior)", "logreg": "Logistic Regression", "rf": "Random Forest",
                "xgb": "XGBoost"}
POLICY_LABELS = {"static": "Static (no labels)", "drift_triggered": "Drift-triggered, random audit",
                 "drift_triggered_v2": "Drift-triggered v2, uncertainty audit", "always": "Always query",
                 "full_retrain": "Full retrain (oracle)"}


def _pm(r: dict, key: str, fmt: str = ".3f") -> str:
    return f"{r[f'{key}_mean']:{fmt}} ± {r[f'{key}_std']:{fmt}}"


def _steps(steps: list) -> str:
    steps = [int(s) for s in steps]
    if not steps:
        return "no test step"
    words = ", ".join(map(str, steps[:-1])) + (" and " if len(steps) > 1 else "") + str(steps[-1])
    return ("step " if len(steps) == 1 else "steps ") + words


def _cfg(r: dict) -> str:
    if r["policy"] in ("static", "full_retrain"):
        return POLICY_LABELS[r["policy"]]
    return f"{POLICY_LABELS[r['policy']]}, {r['strategy']}, K={int(r['k'])}"


def render_headline(res: dict) -> str:
    h = res["headline"]
    grid = res["adaptive_grid"]
    static_pre = next(g for g in grid if g["policy"] == "static" and g["window"] == "pre_shutdown")
    rows = [("Static", h["static"]), ("**DriftGuard (pre-registered)**", h["driftguard"]),
            ("DriftGuard v2 (post-hoc)", h["driftguard_v2"]), ("Best in grid (selected on test, context only)",
                                                               h["best_in_grid"]),
            ("Full retrain (oracle)", h["full_retrain"])]
    budget = res["config"]["alert_budget"]
    lines = [
        f"Post-shutdown window (steps 43–49), pooled, mean ± std over {len(res['config']['seeds'])} seeds. "
        f"For reference, the same static model scores **{_pm(static_pre, 'pr_auc')}** PR-AUC on steps 35–42.",
        "",
        f"| Configuration | Setting | PR-AUC | Recall | Illicit caught | Analyst labels | "
        f"Illicit in top-{budget} alerts/step (sum) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in rows:
        lines.append(
            f"| {name} | {_cfg(r)} | {_pm(r, 'pr_auc')} | {r['recall_mean']:.1%} | "
            f"{r['caught_mean']:.1f} of {r['n_illicit_mean']:.0f} | {r['total_labels_mean']:,.0f} | "
            f"{r['alert_hits_mean']:.1f} |"
        )
    dg = h["driftguard"]
    eff = dg.get("recovery_efficiency_mean")
    lines += ["", f"Recovery efficiency of the pre-registered DriftGuard (extra illicit caught vs static on 43–49 ÷ "
                  f"labels used): **{eff:.4f}**." if eff == eff and eff is not None else ""]
    return "\n".join(lines).rstrip()


def render_findings(res: dict) -> str:
    h, d, ex = res["headline"], res["drift"], res.get("explanations") or {}
    grid = res["adaptive_grid"]
    static_pre = next(g for g in grid if g["policy"] == "static" and g["window"] == "pre_shutdown")
    st, dg, v2, orc, best = (h[k] for k in ("static", "driftguard", "driftguard_v2", "full_retrain", "best_in_grid"))
    fired = res.get("alarm_fired_steps", [])
    dg_fired = next((f["step"] for f in fired if f["policy"] == dg["policy"] and f["strategy"] == dg["strategy"]
                     and f["k"] == dg["k"]), [])
    v2_fired = next((f["step"] for f in fired if f["policy"] == v2["policy"] and f["k"] == v2["k"]), [])
    wins = ex.get("windows", {})
    miss_pre = wins.get("pre_shutdown", {}).get("missed_illicit_median_score")
    miss_post = wins.get("post_shutdown", {}).get("missed_illicit_median_score")
    feats = ex.get("top_features_mean_abs_shap_illicit", [])
    feat_txt = ", ".join(f"`{f['feature']}` {f['pre_shutdown']:.2f} → {f['post_shutdown']:.2f}" for f in feats[:4])
    strat = {g["strategy"]: g for g in grid if g["policy"] == "always" and g["k"] == 50
             and g["window"] == "post_shutdown"}
    order = sorted(strat, key=lambda s: strat[s]["pr_auc_mean"], reverse=True)
    strat_txt = " > ".join(f"{s} ({strat[s]['pr_auc_mean']:.3f})" for s in order)
    return "\n".join([
        f"1. **Standard models collapse after the shutdown.** Static XGBoost falls from "
        f"{static_pre['pr_auc_mean']:.3f} to {st['pr_auc_mean']:.3f} PR-AUC and catches {st['caught_mean']:.1f} of "
        f"{st['n_illicit_mean']:.0f} illicit transactions on steps 43–49.",
        f"2. **The failure is silent.** Label-free monitors (score PSI, mean KS over the top-20 features), "
        f"calibrated on steps 1–34, fired on {_steps(d['unsupervised_fired_test_steps'])}. The "
        f"pre-registered random-audit alarm fired on {_steps(dg_fired)} (in any seed), never at step 43, "
        f"because a 10-transaction audit rarely contains an illicit case once they become rare.",
        f"3. **The misses are confident, not uncertain.** The median score of a missed illicit transaction is "
        f"{miss_pre:.4f} before the shutdown and {miss_post:.4f} after it, so lowering the threshold cannot help. "
        f"Our post-hoc v2 audits the transactions closest to the threshold; those contain *fewer* illicit cases "
        f"after the shutdown, so v2 fired on {_steps(v2_fired)}."
        if miss_pre is not None and miss_post is not None else "3. (explanations not available)",
        f"4. **The old fingerprints fade.** Mean |SHAP| for illicit transactions, before → after: {feat_txt}.",
        f"5. **Budget beats triggers.** The pre-registered DriftGuard is within noise of static "
        f"({_pm(dg, 'pr_auc')} vs {_pm(st, 'pr_auc')}, {dg['total_labels_mean']:,.0f} labels). The best adaptive "
        f"configuration ({_cfg(best)}) reaches {_pm(best, 'pr_auc')} with {best['total_labels_mean']:,.0f} labels "
        f"(selected on test results, so context only).",
        f"6. **Which transactions to label matters.** Always querying K=50, ranked by post-shutdown PR-AUC: "
        f"{strat_txt}. Novelty sampling (IsolationForest) is the weakest. This contradicts our prior that novelty "
        f"or hybrid would find the new illicit patterns.",
        f"7. **Recovery is expensive even with every label.** The full-retrain oracle uses "
        f"{orc['total_labels_mean']:,.0f} labels and reaches {_pm(orc, 'pr_auc')} PR-AUC "
        f"({orc['recall_mean']:.1%} recall).",
    ])


def render_baselines(res: dict) -> str:
    sb = res["static_baselines"]
    rows = {(r["model"], r["feature_set"], r["window"]): r for r in sb["test_windows"]}
    val = {(r["model"], r["feature_set"]): r for r in sb["validation"]}
    lines = ["| Model | Features | Validation PR-AUC | PR-AUC 35–42 | PR-AUC 43–49 | Recall 43–49 |",
             "|---|---|---|---|---|---|"]
    for (m, fs) in val:
        pre, post = rows[(m, fs, "pre_shutdown")], rows[(m, fs, "post_shutdown")]
        lines.append(f"| {MODEL_LABELS[m]} | `{fs}` | {_pm(val[(m, fs)], 'pr_auc')} | {_pm(pre, 'pr_auc')} | "
                     f"{_pm(post, 'pr_auc')} | {post['recall_mean']:.1%} |")
    best = sb["best_static"]
    lines += ["", f"Protocol: {sb['protocol']}. Base model for the stream: **{MODEL_LABELS[best['model']]} on "
                  f"`{best['feature_set']}` features** (best validation PR-AUC, {best['val_pr_auc']:.3f})."]
    return "\n".join(lines)


def render_runtime(res: dict) -> str:
    path = config.ARTIFACTS_DIR / "runtimes.json"
    if not path.exists():
        return "_Runtimes are recorded in `artifacts/runtimes.json` after a full run._"
    rt = json.loads(path.read_text())
    s = rt["stages_seconds"]
    parts = ", ".join(f"`--{k}` {v / 60:.1f} min" if v >= 60 else f"`--{k}` {v:.0f} s" for k, v in s.items())
    m = rt["machine"]
    return (f"Last full run on a {m['cpu_count']}-thread laptop CPU ({m['os']}, Python {m['python']}): "
            f"{parts}; total ≈ {sum(s.values()) / 60:.0f} min. Seeds: {res['config']['seeds']}.")


BLOCKS = {"headline": render_headline, "findings": render_findings, "baselines": render_baselines,
          "runtime": render_runtime}


def render(readme_text: str, res: dict) -> str:
    for name, fn in BLOCKS.items():
        pattern = re.compile(rf"(<!-- BEGIN:{name} -->).*?(<!-- END:{name} -->)", re.S)
        if not pattern.search(readme_text):
            raise ValueError(f"README is missing the {name} block markers")
        readme_text = pattern.sub(lambda m, fn=fn: f"{m.group(1)}\n{fn(res)}\n{m.group(2)}", readme_text)
    return readme_text


def update_readme(res: dict | None = None) -> None:
    res = res or json.loads((config.ARTIFACTS_DIR / "results.json").read_text())
    README.write_text(render(README.read_text(encoding="utf-8"), res), encoding="utf-8")


if __name__ == "__main__":
    update_readme()
