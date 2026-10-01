"""Single source of truth for paths, splits, seeds and experiment grid."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_FILE = PROCESSED_DIR / "elliptic.parquet"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Raw CSVs are searched for (recursively) in these folders, in order.
RAW_SEARCH_DIRS = [DATA_DIR / "raw", ROOT / "elliptic_bitcoin_dataset"]
FEATURES_CSV = "elliptic_txs_features.csv"
CLASSES_CSV = "elliptic_txs_classes.csv"
EDGES_CSV = "elliptic_txs_edgelist.csv"

# ---------------------------------------------------------------------------
# Dataset expectations (validated on load -- fail loudly if they do not match)
# ---------------------------------------------------------------------------
EXPECTED_ROWS = 203_769
EXPECTED_STEPS = 49
EXPECTED_ILLICIT = 4_545
EXPECTED_LICIT = 42_019
N_LOCAL = 93
N_AGG = 72

# ---------------------------------------------------------------------------
# Time-based splits (inclusive step ranges)
# ---------------------------------------------------------------------------
TRAIN_STEPS = range(1, 30)  # 1-29
VAL_STEPS = range(30, 35)  # 30-34
TEST_STEPS = range(35, 50)  # 35-49
SHUTDOWN_STEP = 43

EVAL_WINDOWS = {
    "pre_shutdown": range(35, 43),  # 35-42
    "post_shutdown": range(43, 50),  # 43-49
    "all_test": range(35, 50),  # 35-49
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
BASE_SEED = 42
SEEDS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
PSI_BINS = 10
KS_TOP_FEATURES = 20
KS_PVALUE = 0.01
DRIFT_THRESHOLD_QUANTILE = 0.95  # calibrated on steps 1-34 only
AUDIT_SIZE = 10  # random analyst audit per test step (fixed before seeing test results)

# ---------------------------------------------------------------------------
# Active learning grid
# ---------------------------------------------------------------------------
K_VALUES = [10, 25, 50]
POLICIES = ["static", "drift_triggered", "always", "full_retrain"]
STRATEGIES = ["random", "uncertainty", "novelty", "hybrid"]
# Sample weight for labels collected after deployment (audit, queries, full retrain), identical for every
# policy. Fixed before seeing test results so a handful of new labels can move a model trained on ~30k rows.
QUERY_WEIGHT = 10.0

# Analyst capacity: alerts reviewed per step for the alert-budget metric (top-N scores among labeled rows).
ALERT_BUDGET = 50

# Headline comparison, fixed before aggregation: pre-registered DriftGuard configuration.
HEADLINE = {"policy": "drift_triggered", "strategy": "hybrid", "k": 25}
# Post-hoc v2 (designed AFTER seeing phase-4 test results; always reported separately and labelled as such).
HEADLINE_V2 = {"policy": "drift_triggered_v2", "strategy": "uncertainty", "k": 25}
V2_NOTE = (
    "Post-hoc: designed after seeing phase-4 test results. Replaces the random audit with an uncertainty audit "
    "(10 transactions closest to the decision threshold); alarm threshold calibrated on steps 1-34 only."
)

# ---------------------------------------------------------------------------
# Models (kept modest so the full grid runs on a laptop CPU)
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": -1,
}
RF_PARAMS = {"n_estimators": 300, "max_features": 50, "n_jobs": -1}
