"""Cross-platform CLI entry point: python -m driftguard.pipeline --all"""

from __future__ import annotations

import argparse
import sys
import time

STAGES = ["data", "train", "stream", "eval"]


def run_data() -> None:
    import json

    from driftguard import config
    from driftguard.data import build_dataset

    _, stats = build_dataset(save=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print("[driftguard] data validated: " + ", ".join(f"{k}={v}" for k, v in stats.items()), flush=True)


def run_train() -> None:
    import json
    from datetime import UTC, datetime

    from driftguard import config
    from driftguard.data import load_processed
    from driftguard.evaluate import pick_best_static, run_static_baselines

    res = run_static_baselines(load_processed())
    best = pick_best_static(res["val_summary"])
    print(f"[driftguard] best static model on validation: {best}", flush=True)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    res["per_step"].to_parquet(config.ARTIFACTS_DIR / "static_per_step.parquet", index=False)
    out = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": "train steps 1-29, threshold + model selection on 30-34, frozen evaluation on 35-49",
        "seeds": config.SEEDS,
        "best_static": best,
        "validation": res["val_summary"].to_dict(orient="records"),
        "test_windows": res["summary"].to_dict(orient="records"),
    }
    (config.ARTIFACTS_DIR / "static_baselines.json").write_text(json.dumps(out, indent=2))


def load_best_static() -> dict:
    import json

    from driftguard import config

    path = config.ARTIFACTS_DIR / "static_baselines.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m driftguard.pipeline --train")
    return json.loads(path.read_text())["best_static"]


def run_drift():
    import json

    import pandas as pd

    from driftguard import config
    from driftguard.data import FEATURE_SETS, load_processed
    from driftguard.drift import build_monitor, unsupervised_timeline

    df = load_processed()
    best = load_best_static()
    mon = build_monitor(df, best["model"], FEATURE_SETS[best["feature_set"]])
    timeline = unsupervised_timeline(df, mon)
    d = mon.detector
    print(f"[driftguard] unsupervised thresholds: psi_thr={d.psi_thr:.4f} ks_thr={d.ks_thr:.4f}; "
          f"audit_thr={mon.audit_thr:.2f} misses per {config.AUDIT_SIZE}", flush=True)
    print(timeline.to_string(index=False, float_format="%.3f"), flush=True)

    pd.concat([mon.calib, timeline], ignore_index=True).to_parquet(config.ARTIFACTS_DIR / "drift.parquet", index=False)
    meta = {
        "monitor_model": best,
        "seed": config.BASE_SEED,
        "reference_steps": [min(config.VAL_STEPS), max(config.VAL_STEPS)],
        "calibration_steps": [min(config.TRAIN_STEPS), max(config.VAL_STEPS)],
        "quantile": config.DRIFT_THRESHOLD_QUANTILE,
        "decision_threshold": mon.threshold,
        "psi_thr": d.psi_thr,
        "ks_mean_thr": d.ks_thr,
        "ks_features": mon.ks_cols,
        "unsupervised_fired_test_steps": timeline.loc[timeline["fired"], "step"].tolist(),
        "audit_size": config.AUDIT_SIZE,
        "audit_misses_thr": mon.audit_thr,
        "audit_calibration_miss_counts": mon.audit_calib["misses"].value_counts().sort_index().to_dict(),
        "v2_uncertainty_audit_misses_thr": mon.audit_thr_uncertainty,
        "v2_uncertainty_audit_calibration_miss_counts":
            mon.audit_calib_uncertainty["misses"].value_counts().sort_index().to_dict(),
        "v2_note": config.V2_NOTE,
    }
    (config.ARTIFACTS_DIR / "drift.json").write_text(json.dumps(meta, indent=2, default=int))
    return df, best, mon, timeline


def save_pr_curves(post_scores: dict) -> None:
    """Post-shutdown precision-recall curves (seed 0) for headline configs, downsampled; no raw features."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_curve

    from driftguard import config

    keep = {("static", "none", 0): "Static", ("full_retrain", "none", 0): "Full retrain (oracle)",
            tuple(config.HEADLINE.values()): "DriftGuard (pre-registered)",
            tuple(config.HEADLINE_V2.values()): "DriftGuard v2 (post-hoc)"}
    rows = []
    for key, label in keep.items():
        y, s = post_scores[key]
        p, r, _ = precision_recall_curve(y.astype(int), s)
        idx = np.unique(np.linspace(0, len(p) - 1, min(len(p), 300)).astype(int))
        rows += [{"config": label, "precision": float(p[i]), "recall": float(r[i])} for i in idx]
    pd.DataFrame(rows).to_parquet(config.ARTIFACTS_DIR / "pr_curves.parquet", index=False)


def run_stream() -> None:
    from driftguard import config
    from driftguard.data import FEATURE_SETS
    from driftguard.stream import AlarmConfig, prepare_seed, run_grid

    df, best, mon, timeline = run_drift()
    alarms = AlarmConfig(unsupervised_fired=dict(zip(timeline["step"], timeline["fired"], strict=True)),
                         audit_thr=mon.audit_thr, audit_thr_uncertainty=mon.audit_thr_uncertainty)
    cols = FEATURE_SETS[best["feature_set"]]
    contexts = [prepare_seed(df, best["model"], cols, seed) for seed in config.SEEDS]
    per_step, windows, _, post_scores = run_grid(contexts, alarms)
    per_step.to_parquet(config.ARTIFACTS_DIR / "per_step.parquet", index=False)
    windows.to_parquet(config.ARTIFACTS_DIR / "windows.parquet", index=False)
    save_pr_curves(post_scores)
    print(f"[driftguard] stream grid: {len(per_step)} step rows written to per_step.parquet", flush=True)


def run_explain() -> None:
    import json

    from driftguard import config
    from driftguard.data import FEATURE_SETS, load_processed
    from driftguard.explain import explain_stream, shap_figure
    from driftguard.stream import AlarmConfig, prepare_seed

    art = config.ARTIFACTS_DIR
    drift = json.loads((art / "drift.json").read_text())
    best = load_best_static()
    cols = FEATURE_SETS[best["feature_set"]]
    alarms = AlarmConfig(unsupervised_fired={s: True for s in drift["unsupervised_fired_test_steps"]},
                         audit_thr=drift["audit_misses_thr"],
                         audit_thr_uncertainty=drift["v2_uncertainty_audit_misses_thr"])
    ctx = prepare_seed(load_processed(), best["model"], cols, config.SEEDS[0])
    alerts, imp = explain_stream(ctx, alarms, config.HEADLINE, cols)
    alerts.to_parquet(art / "alerts.parquet", index=False)
    imp.to_parquet(art / "shap_importance.parquet", index=False)
    shap_figure(imp, config.FIGURES_DIR / "shap_summary.png")
    print(f"[driftguard] alerts.parquet: {len(alerts)} rows "
          f"({alerts['outcome'].value_counts().to_dict()})", flush=True)


def run_eval() -> None:
    import json

    import pandas as pd

    from driftguard import config
    from driftguard.evaluate import add_recovery_efficiency, build_results, make_figures, summarise_stream

    art = config.ARTIFACTS_DIR
    per_step = pd.read_parquet(art / "per_step.parquet")
    windows = pd.read_parquet(art / "windows.parquet")
    results = build_results(windows, per_step)
    (art / "results.json").write_text(json.dumps(results, indent=2, default=float))
    summary = summarise_stream(add_recovery_efficiency(windows))
    drift, pr = pd.read_parquet(art / "drift.parquet"), pd.read_parquet(art / "pr_curves.parquet")
    figs = make_figures(per_step, summary, drift, pr)
    h = results["headline"]
    for key in ("static", "driftguard", "driftguard_v2", "full_retrain", "best_in_grid"):
        r = h[key]
        print(f"  {key:14s} {r['policy']:19s} {r['strategy']:11s} K={int(r['k']):<3d} "
              f"PR-AUC {r['pr_auc_mean']:.3f}+-{r['pr_auc_std']:.3f}  recall {r['recall_mean']:.3f}  "
              f"caught {r['caught_mean']:.1f}  labels {r['total_labels_mean']:.0f}", flush=True)
    print(f"[driftguard] wrote results.json and {len(figs)} figures", flush=True)
    run_explain()


RUNNERS = {"data": run_data, "train": run_train, "stream": run_stream, "eval": run_eval}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="driftguard.pipeline", description=__doc__)
    parser.add_argument("--data", action="store_true", help="load, validate and save processed data")
    parser.add_argument("--train", action="store_true", help="train and evaluate static baselines")
    parser.add_argument("--stream", action="store_true", help="drift detection + active-learning stream")
    parser.add_argument("--eval", action="store_true", help="aggregate results, figures and SHAP alerts")
    parser.add_argument("--all", action="store_true", help="run every stage in order")
    args = parser.parse_args(argv)
    if not (args.all or any(getattr(args, s) for s in STAGES)):
        parser.error("choose at least one of --data --train --stream --eval --all")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stages = STAGES if args.all else [s for s in STAGES if getattr(args, s)]
    for stage in stages:
        start = time.perf_counter()
        print(f"[driftguard] stage '{stage}' ...", flush=True)
        RUNNERS[stage]()
        print(f"[driftguard] stage '{stage}' done in {time.perf_counter() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
