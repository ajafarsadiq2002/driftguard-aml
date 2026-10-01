"""Cross-platform CLI entry point: python -m driftguard.pipeline --all"""

from __future__ import annotations

import argparse
import sys
import time

STAGES = ["data", "train", "stream", "eval"]


def run_data() -> None:
    from driftguard.data import build_dataset

    _, stats = build_dataset(save=True)
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
    }
    (config.ARTIFACTS_DIR / "drift.json").write_text(json.dumps(meta, indent=2, default=int))
    return df, best, mon, timeline


def run_stream() -> None:
    from driftguard import config
    from driftguard.data import FEATURE_SETS
    from driftguard.stream import AlarmConfig, prepare_seed, run_grid

    df, best, mon, timeline = run_drift()
    alarms = AlarmConfig(unsupervised_fired=dict(zip(timeline["step"], timeline["fired"], strict=True)),
                         audit_thr=mon.audit_thr)
    cols = FEATURE_SETS[best["feature_set"]]
    contexts = [prepare_seed(df, best["model"], cols, seed) for seed in config.SEEDS]
    per_step, _ = run_grid(contexts, alarms)
    per_step.to_parquet(config.ARTIFACTS_DIR / "per_step.parquet", index=False)
    print(f"[driftguard] stream grid: {len(per_step)} step rows written to per_step.parquet", flush=True)


def run_eval() -> None:
    raise NotImplementedError("--eval is implemented in phases 5-6")


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
