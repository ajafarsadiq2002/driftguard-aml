"""PSI / KS / audit sanity on synthetic data and leakage-safe threshold calibration."""

import numpy as np
import pandas as pd
import pytest

from driftguard.drift import (
    DriftDetector,
    audit_misses,
    calibrate_audit_threshold,
    calibrate_thresholds,
    draw_audit,
    ks_drift_share,
    ks_mean_stat,
    psi,
)
from driftguard.splits import LeakageError

rng = np.random.default_rng(0)


def test_psi_identical_distribution_is_near_zero():
    x = rng.normal(size=20_000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)
    assert psi(x, rng.normal(size=20_000)) < 0.01


def test_psi_shifted_distribution_is_large():
    assert psi(rng.normal(size=20_000), rng.normal(loc=1.0, size=20_000)) > 0.2


def test_psi_handles_tied_scores():
    ref = np.r_[np.zeros(9_000), rng.uniform(size=1_000)]
    assert np.isfinite(psi(ref, ref))
    assert psi(ref, rng.uniform(size=5_000)) > 0.2


def test_ks_statistics_identical_vs_shifted():
    ref = rng.normal(size=(5_000, 10))
    same = rng.normal(size=(5_000, 10))
    shifted = rng.normal(size=(5_000, 10))
    shifted[:, :6] += 1.0
    assert ks_drift_share(ref, same) <= 0.1
    assert ks_drift_share(ref, shifted) == pytest.approx(0.6)
    assert ks_mean_stat(ref, same) < 0.05
    assert ks_mean_stat(ref, shifted) > 0.2


def test_detector_fires_on_either_statistic():
    det = DriftDetector(ref_scores=np.zeros(1), ref_X=np.zeros((1, 1)), psi_thr=0.2, ks_thr=0.3)
    assert det.fires({"psi": 0.3, "ks_mean": 0.0})
    assert det.fires({"psi": 0.0, "ks_mean": 0.4})
    assert not det.fires({"psi": 0.1, "ks_mean": 0.2, "ks_share": 1.0})  # share is informational only


def test_audit_misses_counts_illicit_below_threshold():
    y = np.array([1, 1, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.05, 0.4, 0.8])
    assert audit_misses(y, s, threshold=0.5) == 2


def test_draw_audit_is_without_replacement_and_capped():
    pick = draw_audit(100, 10, np.random.default_rng(1))
    assert len(set(pick)) == 10 and pick.max() < 100
    assert len(draw_audit(4, 10, np.random.default_rng(1))) == 4


def test_calibration_refuses_test_steps():
    ok = pd.DataFrame({"step": range(1, 35), "psi": np.linspace(0, 1, 34), "ks_mean": np.linspace(0, 1, 34),
                       "misses": np.arange(34) % 3})
    thr = calibrate_thresholds(ok, q=0.95)
    assert thr["psi_thr"] == pytest.approx(np.quantile(ok["psi"], 0.95))
    assert calibrate_audit_threshold(ok, q=0.95) == pytest.approx(np.quantile(ok["misses"], 0.95))
    bad = pd.concat([ok, pd.DataFrame({"step": [35], "psi": [5.0], "ks_mean": [1.0], "misses": [9]})])
    with pytest.raises(LeakageError):
        calibrate_thresholds(bad)
    with pytest.raises(LeakageError):
        calibrate_audit_threshold(bad)
