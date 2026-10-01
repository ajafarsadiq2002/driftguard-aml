"""Load, validate and merge the Elliptic Bitcoin dataset, plus cheap graph degree features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from driftguard import config

LOCAL_COLS = [f"local_{i}" for i in range(1, config.N_LOCAL + 1)]
AGG_COLS = [f"agg_{i}" for i in range(1, config.N_AGG + 1)]
GRAPH_COLS = ["in_degree", "out_degree"]
FEATURE_COLS = LOCAL_COLS + AGG_COLS + GRAPH_COLS

FEATURE_SETS = {
    "local": LOCAL_COLS,
    "local_agg": LOCAL_COLS + AGG_COLS,
    "local_agg_graph": FEATURE_COLS,
}


class DataValidationError(RuntimeError):
    """Raised when the raw data does not match the published Elliptic dataset."""


def find_raw_file(name: str, search_dirs: list[Path] | None = None) -> Path:
    """Search the configured folders (recursively) for a raw CSV instead of assuming its path."""
    dirs = config.RAW_SEARCH_DIRS if search_dirs is None else search_dirs
    for d in dirs:
        if d.is_dir():
            hits = sorted(d.rglob(name))
            if hits:
                return hits[0]
    searched = ", ".join(str(d) for d in dirs)
    raise FileNotFoundError(
        f"{name} not found in: {searched}. Run scripts/download_data.sh "
        "(kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw --unzip)."
    )


def read_features(path: Path) -> pd.DataFrame:
    """Features CSV has no header: txId, time step, 93 local, 72 aggregated features."""
    names = ["txId", "step", *LOCAL_COLS, *AGG_COLS]
    dtypes = {"txId": np.int64, "step": np.int16} | dict.fromkeys(LOCAL_COLS + AGG_COLS, np.float32)
    df = pd.read_csv(path, header=None, names=names, dtype=dtypes, engine="pyarrow")
    if df.shape[1] != 2 + config.N_LOCAL + config.N_AGG:
        raise DataValidationError(f"features file has {df.shape[1]} columns, expected 167")
    return df


def read_classes(path: Path) -> pd.DataFrame:
    """Classes CSV: '1' = illicit -> 1, '2' = licit -> 0, 'unknown' -> NaN."""
    classes = pd.read_csv(path, dtype={"txId": np.int64, "class": str})
    unexpected = set(classes["class"].unique()) - {"1", "2", "unknown"}
    if unexpected:
        raise DataValidationError(f"unexpected class values: {sorted(unexpected)}")
    classes["y"] = classes["class"].map({"1": 1.0, "2": 0.0, "unknown": np.nan})
    return classes[["txId", "y"]]


def read_edges(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"txId1": np.int64, "txId2": np.int64})


def add_degree_features(df: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """In-degree (payments received from) and out-degree (payments sent to) per transaction."""
    out_deg = edges["txId1"].value_counts()
    in_deg = edges["txId2"].value_counts()
    df = df.copy()
    df["in_degree"] = df["txId"].map(in_deg).fillna(0).astype(np.float32)
    df["out_degree"] = df["txId"].map(out_deg).fillna(0).astype(np.float32)
    return df


def merge(features: pd.DataFrame, classes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    df = features.merge(classes, on="txId", how="left", validate="one_to_one")
    df = add_degree_features(df, edges)
    cols = ["txId", "step", "y", *FEATURE_COLS]
    return df[cols].sort_values(["step", "txId"], kind="stable").reset_index(drop=True)


def validate(
    df: pd.DataFrame,
    edges: pd.DataFrame,
    expected_rows: int = config.EXPECTED_ROWS,
    expected_steps: int = config.EXPECTED_STEPS,
    expected_illicit: int = config.EXPECTED_ILLICIT,
    expected_licit: int = config.EXPECTED_LICIT,
) -> dict:
    """Fail loudly if the merged data does not match the published dataset statistics."""
    errors = []
    n_rows = len(df)
    steps = sorted(df["step"].unique().tolist())
    n_illicit = int((df["y"] == 1).sum())
    n_licit = int((df["y"] == 0).sum())
    n_unknown = int(df["y"].isna().sum())

    if n_rows != expected_rows:
        errors.append(f"rows: got {n_rows}, expected {expected_rows}")
    if steps != list(range(1, expected_steps + 1)):
        errors.append(f"time steps: got {len(steps)} distinct, expected 1..{expected_steps}")
    if n_illicit != expected_illicit:
        errors.append(f"illicit: got {n_illicit}, expected {expected_illicit}")
    if n_licit != expected_licit:
        errors.append(f"licit: got {n_licit}, expected {expected_licit}")
    if df["txId"].duplicated().any():
        errors.append("duplicate txIds in features")
    if df[FEATURE_COLS].isna().any().any():
        errors.append("missing feature values")

    step_of = df.set_index("txId")["step"]
    s1 = edges["txId1"].map(step_of)
    s2 = edges["txId2"].map(step_of)
    if s1.isna().any() or s2.isna().any():
        errors.append("edge list references txIds missing from features")
    elif (s1 != s2).any():
        errors.append(f"{int((s1 != s2).sum())} edges cross time steps")

    if errors:
        raise DataValidationError("Elliptic data validation failed:\n  - " + "\n  - ".join(errors))

    return {
        "rows": n_rows,
        "time_steps": len(steps),
        "illicit": n_illicit,
        "licit": n_licit,
        "unknown": n_unknown,
        "labeled_share": round((n_illicit + n_licit) / n_rows, 4),
        "edges": len(edges),
        "n_features": len(FEATURE_COLS),
    }


def build_dataset(save: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load raw CSVs, merge, add degree features, validate, and save processed Parquet."""
    features = read_features(find_raw_file(config.FEATURES_CSV))
    classes = read_classes(find_raw_file(config.CLASSES_CSV))
    edges = read_edges(find_raw_file(config.EDGES_CSV))
    df = merge(features, classes, edges)
    stats = validate(df, edges)
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.PROCESSED_FILE, index=False)
    return df, stats


def load_processed() -> pd.DataFrame:
    """Load the processed Parquet produced by build_dataset()."""
    if not config.PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"{config.PROCESSED_FILE} missing. Run: python -m driftguard.pipeline --data"
        )
    return pd.read_parquet(config.PROCESSED_FILE)
