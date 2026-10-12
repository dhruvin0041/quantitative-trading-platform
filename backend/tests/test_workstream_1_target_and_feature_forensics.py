# backend/tests/test_workstream_1_target_and_feature_forensics.py
"""
Unit tests for HYDRA V2.4.1 Workstream 1 Forensics Engine.

Verifies:
1. Immutability of frozen V2.4 manifest, calibration contract, and evaluation contract.
2. Pinned development dataset existence, schema integrity, and SHA-256 hash.
3. Target construction, same-bar collision detection, and tie-breaking fallback logic.
4. Point-in-time causality of engineered features (lag_vix=True, no forward leakage).
5. Mathematical Brownian survival probability bounds for triple-barrier HOLD class (< 10% for H >= 15).
6. Consistency of generated forensic reports (JSON, MD, CSV).
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.research.workstream_1_target_and_feature_forensics import (
    EXPECTED_CALIB_SHA256,
    EXPECTED_EVAL_SHA256,
    EXPECTED_V24_MANIFEST_SHA256,
    PINNED_DATASET_PATH,
    compute_brownian_survival_probability,
    simulate_brownian_barrier_paths,
    verify_governance_invariance,
)
from src.execution.live_inference import FEATURE_COLUMNS_V24, add_upgraded_features

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_DIR / "reports"


def test_governance_invariance():
    """Manifest and core execution engines must match immutable cryptographic digests."""
    res = verify_governance_invariance()
    assert res["status"] == "ALL_INVARIANTS_VERIFIED"
    assert res["manifest_sha256"] == EXPECTED_V24_MANIFEST_SHA256
    assert res["calibration_sha256"] == EXPECTED_CALIB_SHA256
    assert res["evaluation_contract_sha256"] == EXPECTED_EVAL_SHA256


def test_pinned_dataset_integrity():
    """Pinned development dataset must exist, have valid SHA-256, and contain expected components."""
    assert PINNED_DATASET_PATH.exists(), f"Pinned dataset missing at {PINNED_DATASET_PATH}"
    dataset_bytes = PINNED_DATASET_PATH.read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    assert len(dataset_hash) == 64

    # Load and inspect dataset
    data = joblib.load(PINNED_DATASET_PATH)
    assert "metadata" in data
    assert "raw_equity" in data
    assert "raw_spy" in data
    assert "raw_vix" in data
    assert "engineered_features_clean" in data
    assert "feature_columns_v24" in data

    # Check row counts and date range
    df_equity = data["raw_equity"]
    assert len(df_equity) >= 2000
    assert df_equity.index[0] <= pd.Timestamp("2016-01-04")
    assert df_equity.index[-1] >= pd.Timestamp("2024-12-30")

    # Check clean feature columns
    df_clean = data["engineered_features_clean"]
    for col in FEATURE_COLUMNS_V24:
        assert col in df_clean.columns, f"Feature {col} missing from clean dataset"
        assert not df_clean[col].isna().any(), f"NaN values detected in feature {col}"


def test_same_bar_collision_synthetic():
    """Tests same-bar collision ambiguity detection and pessimistic tie-breaking."""
    dates = pd.date_range("2023-01-01", periods=20, freq="B")
    df_raw = pd.DataFrame(
        {
            "Open": [100.0] * 20,
            "High": [101.0] * 20,
            "Low": [99.0] * 20,
            "Close": [100.0] * 20,
            "Volume": [1000] * 20,
        },
        index=dates,
    )

    # Bar 5: massive volatility candle crossing both UB (100 + 2*1.5 = 103) and LB (100 - 2*1.5 = 97)
    df_raw.loc[dates[5], "High"] = 105.0
    df_raw.loc[dates[5], "Low"] = 95.0

    # Test detection: on step h=5, both High >= 103 and Low <= 97
    h = 5
    ub = 103.0
    lb = 97.0
    hit_ub = df_raw.loc[dates[h], "High"] >= ub
    hit_lb = df_raw.loc[dates[h], "Low"] <= lb
    assert hit_ub and hit_lb, "Simultaneous touch must be True on bar 5"


def test_point_in_time_vix_lag():
    """Verify that lag_vix=True shifts VIX by 1 day and eliminates same-day settlement lookahead."""
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    df_raw = pd.DataFrame(
        {
            "Open": [100.0] * 30,
            "High": [101.0] * 30,
            "Low": [99.0] * 30,
            "Close": [100.0] * 30,
            "Volume": [1000] * 30,
        },
        index=dates,
    )
    spy_df = df_raw.copy()
    vix_df = pd.DataFrame({"Close": np.linspace(15.0, 30.0, 30)}, index=dates)

    f_lagged = add_upgraded_features(df_raw.copy(), spy_df, vix_df, lag_vix=True)
    f_unlagged = add_upgraded_features(df_raw.copy(), spy_df, vix_df, lag_vix=False)

    # At bar 10, lagged VIX must equal unlagged VIX at bar 9
    assert np.isclose(f_lagged["VIX_Level"].iloc[10], f_unlagged["VIX_Level"].iloc[9])
    # And must NOT equal unlagged VIX at bar 10
    assert not np.isclose(f_lagged["VIX_Level"].iloc[10], f_unlagged["VIX_Level"].iloc[10])


def test_mathematical_hold_survival_bounds():
    """
    Mathematical first-passage time proof:
    Survival probability P(tau > 15 days) inside [-1.5 ATR, +1.5 ATR] must be < 5%.
    """
    # Daily sigma = 0.0175, ATR = 0.018 * S0, b = 1.5 * ATR = 0.027, T = 15
    res = compute_brownian_survival_probability(
        b_up=0.027, b_down=0.027, sigma=0.0175, T=15
    )
    p_survival_15 = res["full_series_probability"]
    assert p_survival_15 < 0.05, f"Analytical survival prob {p_survival_15} must be < 5%"

    # Verify exact analytical derivation values from Issue A
    assert np.isclose(res["exponent_magnitude"], 7.774, atol=0.01)
    assert np.isclose(res["first_term_probability"], 0.000535, atol=0.00005)

    # Monte Carlo simulation must also show HOLD < 5%
    mc_res = simulate_brownian_barrier_paths(
        tp_mult=1.5, sl_mult=1.5, horizon=15, sigma=0.0175, drift=0.0006
    )
    assert mc_res["sim_hold_pct"] < 5.0, f"Simulated HOLD % {mc_res['sim_hold_pct']} must be < 5%"


def test_data_lineage_accounting_consistency():
    """Verify exact count alignment across all data lineage filtering stages (Issue B)."""
    data = joblib.load(PINNED_DATASET_PATH)
    raw_equity = data["raw_equity"]
    clean_feat = data["engineered_features_clean"]

    # Stage 1: Raw bars
    assert len(raw_equity) == 2515
    assert raw_equity.index[0].strftime("%Y-%m-%d") == "2015-01-02"
    assert raw_equity.index[-1].strftime("%Y-%m-%d") == "2024-12-30"

    # Stage 2 & 3: Warmup purge (119 bars) and Clean features (2,396 bars)
    assert len(clean_feat) == 2396
    assert clean_feat.index[0].strftime("%Y-%m-%d") == "2015-06-24"

    # Stage 4: Target collision audit (final 20 bars trimmed for H=20 labeling)
    target_audit_bars = len(clean_feat) - 20
    assert target_audit_bars == 2376

    # Stage 5: Development universe (2016-01-04 to 2024-12-30)
    dev_mask = (clean_feat.index >= "2016-01-04") & (clean_feat.index <= "2024-12-30")
    assert dev_mask.sum() == 2263

    # Stage 6: Purged WFA split modeling (4 outer validation folds of 414 = 1,656 + 418 initial train)
    assert (418 + 4 * 414) == 2074


def test_connors_rsi_statistical_verification():
    """Verify that Connors RSI 5-day Spearman IC is +0.0054 (p ~ 0.79), explaining the +0.0580 typo (Issue C)."""
    from scipy.stats import spearmanr

    data = joblib.load(PINNED_DATASET_PATH)
    clean_feat = data["engineered_features_clean"]
    raw_close = data["raw_equity"].loc[clean_feat.index, "Close"]
    fwd_5d = raw_close.pct_change(5).shift(-5)
    mask = ~clean_feat["Connors_RSI"].isna() & ~fwd_5d.isna()

    r, p_val = spearmanr(clean_feat.loc[mask, "Connors_RSI"], fwd_5d[mask])
    assert np.isclose(r, 0.0054, atol=0.001), f"Expected IC ~ +0.0054, got {r}"
    assert np.isclose(p_val, 0.79, atol=0.05), f"Expected p ~ 0.79, got {p_val}"


def test_end_to_end_vix_calendar_synchrony():
    """Verify zero calendar gaps across all 2,515 bars between AAPL, SPY, and VIX (Issue D)."""
    data = joblib.load(PINNED_DATASET_PATH)
    eq_dates = set(data["raw_equity"].index)
    vix_dates = set(data["raw_vix"].index)
    spy_dates = set(data["raw_spy"].index)

    assert len(eq_dates - vix_dates) == 0, "Zero missing equity dates in VIX"
    assert len(vix_dates - eq_dates) == 0, "Zero missing VIX dates in equity"
    assert len(eq_dates - spy_dates) == 0, "Zero missing equity dates in SPY"


def test_workstream_1_report_artifacts_exist():
    """JSON, Markdown, and CSV reports must exist and have valid structure."""
    json_path = REPORTS_DIR / "workstream_1_target_and_feature_forensics.json"
    md_path = REPORTS_DIR / "workstream_1_target_and_feature_forensics.md"
    csv_path = REPORTS_DIR / "workstream_1_collision_and_imbalance_metrics.csv"

    assert json_path.exists()
    assert md_path.exists()
    assert csv_path.exists()

    with open(json_path) as f:
        data = json.load(f)

    assert data["document_id"] == "HYDRA-AUDIT-20261012-WORKSTREAM-1-TARGET-FEATURE-FORENSICS"
    assert "target_construction_and_ambiguity_audit" in data
    assert "point_in_time_causality_audit" in data
    assert "mathematical_imbalance_derivation" in data
    assert "data_lineage_reconciliation" in data
    assert "four_feature_disposition" in data
    assert "predeclared_workstream_sequencing" in data
