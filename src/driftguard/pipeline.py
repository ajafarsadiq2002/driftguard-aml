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
    raise NotImplementedError("--train is implemented in phase 2")


def run_stream() -> None:
    raise NotImplementedError("--stream is implemented in phases 3-4")


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
