"""Drift detection.

Two alarms, both calibrated on steps 1-34 only:

1. Unsupervised monitor (no labels): score PSI and mean KS statistic over the top-20 features, step t vs the
   validation reference (steps 30-34). Uses one frozen scorer (base model trained on steps 1-29) so statistics
   stay comparable across the stream. The share of features with KS p < 0.01 is recorded for transparency but
   not used: with thousands of transactions per step it saturates (0.65-0.90 even between validation steps).
2. Audit alarm (few labels): an analyst labels a small random audit of step t; the statistic is the number of
   audited transactions that are illicit but scored below the decision threshold (missed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupKFold

from driftguard import config
from driftguard.active import select
from driftguard.models import fit_model, predict_scores, select_threshold
from driftguard.splits import LeakageError, in_steps, labeled

EPS = 1e-4


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def psi(reference: np.ndarray, current: np.ndarray, bins: int = config.PSI_BINS) -> float:
    """Population Stability Index with quantile bins taken from the reference distribution."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    inner = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    ref_p = np.bincount(np.searchsorted(inner, reference, side="right"), minlength=len(inner) + 1) / len(reference)
    cur_p = np.bincount(np.searchsorted(inner, current, side="right"), minlength=len(inner) + 1) / len(current)
    ref_p, cur_p = np.clip(ref_p, EPS, None), np.clip(cur_p, EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _ks(reference: np.ndarray, current: np.ndarray) -> list:
    return [ks_2samp(reference[:, j], current[:, j]) for j in range(reference.shape[1])]


def ks_mean_stat(reference: np.ndarray, current: np.ndarray) -> float:
    """Mean two-sample KS statistic (effect size, not p-value) across feature columns."""
    return float(np.mean([r.statistic for r in _ks(reference, current)]))


def ks_drift_share(reference: np.ndarray, current: np.ndarray, pvalue: float = config.KS_PVALUE) -> float:
    """Fraction of feature columns whose KS test rejects equality at `pvalue` (recorded, not used to alarm)."""
    return float(np.mean([r.pvalue < pvalue for r in _ks(reference, current)]))


def audit_misses(y: np.ndarray, scores: np.ndarray, threshold: float) -> int:
    """Audited transactions that are illicit but scored below the decision threshold."""
    return int(((np.asarray(y) == 1) & (np.asarray(scores) < threshold)).sum())


def draw_audit(n_rows: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Positions of a uniform random audit sample (without replacement)."""
    return rng.choice(n_rows, size=min(size, n_rows), replace=False)


def top_features(model, feature_cols: list[str], k: int = config.KS_TOP_FEATURES) -> list[str]:
    order = np.argsort(np.asarray(model.feature_importances_))[::-1][:k]
    return [feature_cols[i] for i in order]


# ---------------------------------------------------------------------------
# Detectors and calibration
# ---------------------------------------------------------------------------
@dataclass
class DriftDetector:
    """Unsupervised monitor: compares a step's scores and top-feature distributions with a fixed reference."""

    ref_scores: np.ndarray
    ref_X: np.ndarray
    psi_thr: float = np.inf
    ks_thr: float = np.inf

    def stats(self, scores: np.ndarray, X: np.ndarray) -> dict:
        return {
            "psi": psi(self.ref_scores, scores),
            "ks_mean": ks_mean_stat(self.ref_X, X),
            "ks_share": ks_drift_share(self.ref_X, X),
        }

    def fires(self, stats: dict) -> bool:
        return bool(stats["psi"] > self.psi_thr or stats["ks_mean"] > self.ks_thr)


def _check_calibration_steps(calib: pd.DataFrame) -> None:
    if calib["step"].max() >= min(config.TEST_STEPS):
        raise LeakageError("drift thresholds must be calibrated on steps before the test window only")


def calibrate_thresholds(calib: pd.DataFrame, q: float = config.DRIFT_THRESHOLD_QUANTILE) -> dict:
    """Unsupervised thresholds = q-quantile of each statistic over calibration steps (steps < first test step)."""
    _check_calibration_steps(calib)
    return {"psi_thr": float(np.quantile(calib["psi"], q)), "ks_thr": float(np.quantile(calib["ks_mean"], q))}


def calibrate_audit_threshold(audits: pd.DataFrame, q: float = config.DRIFT_THRESHOLD_QUANTILE) -> float:
    """Audit alarm fires when misses exceed the q-quantile of simulated audits on calibration steps."""
    _check_calibration_steps(audits)
    return float(np.quantile(audits["misses"], q))


def _oof_train_scores(df: pd.DataFrame, model_name: str, cols: list[str], seed: int) -> pd.Series:
    """Out-of-fold scores for every transaction in steps 1-29 (each step scored by a model that never saw it)."""
    train_all = in_steps(df, config.TRAIN_STEPS)
    out = pd.Series(np.nan, index=train_all.index)
    for fit_idx, score_idx in GroupKFold(n_splits=5).split(train_all, groups=train_all["step"]):
        fit_rows = labeled(in_steps(train_all, set(train_all.iloc[fit_idx]["step"])))
        model = fit_model(model_name, fit_rows[cols].to_numpy(), fit_rows["y"].to_numpy(), seed)
        rows = in_steps(train_all, set(train_all.iloc[score_idx]["step"]))
        out.loc[rows.index] = predict_scores(model, rows[cols].to_numpy())
    return out


@dataclass
class Monitor:
    detector: DriftDetector
    scorer: object
    cols: list
    ks_cols: list
    threshold: float
    audit_thr: float
    calib: pd.DataFrame
    audit_calib: pd.DataFrame
    audit_thr_uncertainty: float = np.inf  # post-hoc v2
    audit_calib_uncertainty: pd.DataFrame | None = None


def build_monitor(df: pd.DataFrame, model_name: str, cols: list[str], seed: int = config.BASE_SEED) -> Monitor:
    """Fit the frozen monitor scorer on steps 1-29 and calibrate both alarms on steps 1-34."""
    train_lab = labeled(in_steps(df, config.TRAIN_STEPS))
    scorer = fit_model(model_name, train_lab[cols].to_numpy(), train_lab["y"].to_numpy(), seed)
    ks_cols = top_features(scorer, cols)

    val_all = in_steps(df, config.VAL_STEPS)
    val_scores = predict_scores(scorer, val_all[cols].to_numpy())
    val_lab_mask = val_all["y"].notna().to_numpy()
    threshold = select_threshold(val_all["y"].to_numpy()[val_lab_mask], val_scores[val_lab_mask])
    detector = DriftDetector(ref_scores=val_scores, ref_X=val_all[ks_cols].to_numpy())

    # Out-of-sample scores for every calibration step: OOF for 1-29, the 1-29 scorer for 30-34.
    train_all = in_steps(df, config.TRAIN_STEPS)
    calib_frame = pd.concat([train_all.assign(score=_oof_train_scores(df, model_name, cols, seed)),
                             val_all.assign(score=val_scores)])

    rows = []
    val_steps = val_all["step"].to_numpy()
    val_X = val_all[ks_cols].to_numpy()
    for s in list(config.TRAIN_STEPS) + list(config.VAL_STEPS):
        cur = calib_frame[calib_frame["step"] == s]
        if s in config.VAL_STEPS:  # leave-one-step-out reference so a step is never compared with itself
            rest = val_steps != s
            ref = DriftDetector(ref_scores=val_scores[rest], ref_X=val_X[rest])
        else:
            ref = detector
        split = "val" if s in config.VAL_STEPS else "train"
        rows.append({"step": s, "split": split, **ref.stats(cur["score"].to_numpy(), cur[ks_cols].to_numpy())})
    calib = pd.DataFrame(rows)
    thr = calibrate_thresholds(calib)
    detector.psi_thr, detector.ks_thr = thr["psi_thr"], thr["ks_thr"]
    calib["fired"] = [detector.fires(r) for r in calib.to_dict("records")]

    audit_rows = []
    calib_lab = labeled(calib_frame)
    for seed_i in config.SEEDS:
        rng = np.random.default_rng(seed_i)
        for s, grp in calib_lab.groupby("step"):
            pick = draw_audit(len(grp), config.AUDIT_SIZE, rng)
            m = audit_misses(grp["y"].to_numpy()[pick], grp["score"].to_numpy()[pick], threshold)
            audit_rows.append({"step": int(s), "seed": seed_i, "misses": m})
    audit_calib = pd.DataFrame(audit_rows)
    audit_thr = calibrate_audit_threshold(audit_calib)

    # Post-hoc v2: deterministic uncertainty audit (closest to the threshold), same calibration steps.
    unc_rows = []
    for s, grp in calib_lab.groupby("step"):
        sc = grp["score"].to_numpy()
        pick = select("uncertainty", config.AUDIT_SIZE, sc, threshold, sc, np.ones(len(grp), bool),
                      np.random.default_rng(0))
        unc_rows.append({"step": int(s), "misses": audit_misses(grp["y"].to_numpy()[pick], sc[pick], threshold)})
    audit_calib_unc = pd.DataFrame(unc_rows)
    audit_thr_unc = calibrate_audit_threshold(audit_calib_unc)

    return Monitor(detector, scorer, cols, ks_cols, threshold, audit_thr, calib, audit_calib,
                   audit_thr_unc, audit_calib_unc)


def unsupervised_timeline(df: pd.DataFrame, mon: Monitor) -> pd.DataFrame:
    """Unsupervised drift statistics for every test step using all transactions (labeled + unknown)."""
    rows = []
    for s in config.TEST_STEPS:
        step_df = df[df["step"] == s]
        st = mon.detector.stats(predict_scores(mon.scorer, step_df[mon.cols].to_numpy()),
                                step_df[mon.ks_cols].to_numpy())
        rows.append({"step": s, "split": "test", **st, "fired": mon.detector.fires(st)})
    return pd.DataFrame(rows)
