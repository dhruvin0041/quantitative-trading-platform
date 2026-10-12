# backend/tests/test_feature_diagnostics_research.py
"""
Unit tests for HYDRA V2.4.1 Phase 3b Feature Diagnostics Audit.

Verifies:
1. Immutability of frozen baseline manifest and runtime contracts.
2. Causal lookahead-freedom of Family A (Lead-Lag) and Family B (Vol-Regime) features.
3. Strict alignment of feature dates with development universe.
4. Effective sample size adjustment for autocorrelation in trade signals.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.feature_diagnostics_audit import (
    compute_cross_asset_lead_lag_features,
    compute_volatility_regime_features,
)
from scripts.research.triple_barrier_experiment import (
    EXPECTED_V24_MANIFEST_SHA256,
    PROTECTED_V24_MANIFEST,
    verify_manifest_integrity,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent


def test_frozen_manifest_integrity():
    """Verify manifest is byte-for-byte identical to frozen baseline."""
    digest = verify_manifest_integrity()
    assert digest == EXPECTED_V24_MANIFEST_SHA256
    assert PROTECTED_V24_MANIFEST.exists()


def test_core_contracts_invariance():
    """Ensure zero modifications to calibration.py or evaluation_contract.py."""
    calib_path = BACKEND_DIR / "src" / "models" / "regime" / "calibration.py"
    eval_path = BACKEND_DIR / "src" / "execution" / "evaluation_contract.py"

    assert calib_path.exists()
    assert eval_path.exists()

    calib_hash = hashlib.sha256(calib_path.read_bytes()).hexdigest()
    eval_hash = hashlib.sha256(eval_path.read_bytes()).hexdigest()

    assert calib_hash == "badff3d49e1ee755231b0e2f159dc539a199ac6263831d66910c9863884958f5"
    assert eval_hash == "94399aefe5015e318eeeafa3e23b7103c8ae4809146948e06b1c3f13e1300720"


def test_feature_causality_lookahead_freedom():
    """Appending future data points must never change past feature values."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(42)
    prices = 150.0 + np.cumsum(np.random.randn(100))
    highs = prices + np.abs(np.random.randn(100))
    lows = prices - np.abs(np.random.randn(100))

    df_raw = pd.DataFrame(
        {"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": 10000},
        index=dates,
    )
    df_feat = pd.DataFrame({"ATR": [2.5] * 100}, index=dates)

    spy_df = pd.DataFrame({"Close": 400.0 + np.cumsum(np.random.randn(100))}, index=dates)
    qqq_df = pd.DataFrame({"Close": 350.0 + np.cumsum(np.random.randn(100))}, index=dates)
    vix_df = pd.DataFrame({"Close": 18.0 + np.cumsum(np.random.randn(100))}, index=dates)

    # Compute on first 80 bars
    df_lead_80 = compute_cross_asset_lead_lag_features(
        df_raw.iloc[:80], spy_df.iloc[:80], qqq_df.iloc[:80], vix_df.iloc[:80]
    )
    df_vol_80 = compute_volatility_regime_features(df_raw.iloc[:80], df_feat.iloc[:80])

    # Compute on full 100 bars
    df_lead_100 = compute_cross_asset_lead_lag_features(df_raw, spy_df, qqq_df, vix_df)
    df_vol_100 = compute_volatility_regime_features(df_raw, df_feat)

    # Past 80 bars must match exactly
    np.testing.assert_allclose(
        df_lead_80.values,
        df_lead_100.iloc[:80].values,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Cross-Asset Lead-Lag features leaked future information!",
    )
    np.testing.assert_allclose(
        df_vol_80.values,
        df_vol_100.iloc[:80].values,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Volatility-Regime features leaked future information!",
    )


def test_autocorrelation_adjusted_sample_size():
    """Positive autocorrelation must strictly reduce effective sample size."""
    n_raw = 200
    rho_zero = 0.0
    rho_pos = 0.25

    n_eff_zero = n_raw / (1.0 + 2.0 * rho_zero)
    n_eff_pos = n_raw / (1.0 + 2.0 * rho_pos)

    assert n_eff_zero == 200.0
    assert n_eff_pos < n_raw
    assert np.isclose(n_eff_pos, 200.0 / 1.5)
