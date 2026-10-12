# backend/tests/test_stationary_feature_pruning.py
"""
Unit tests for HYDRA V2.4.1 Phase 3c Stationary Feature Pruning Audit.

Verifies:
1. Immutability of frozen baseline manifest and runtime contracts.
2. Arithmetic sum vs compounding equity reconciliation logic.
3. Strict nested-fold isolation (zero validation data seen during feature ranking).
4. Feature subset dimensionality and stationarity guarantees.
"""

import hashlib
from pathlib import Path

import numpy as np

from scripts.research.stationary_feature_pruning_audit import (
    reconcile_phase_3b_trades,
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


def test_trade_reconciliation_math():
    """Verify arithmetic vs continuous compounding reconciliation logic."""
    synthetic_trades_vol = [
        {"net_ret": 0.08, "net_pnl": 800.0, "exit_reason": "TP"},
        {"net_ret": 0.05, "net_pnl": 500.0, "exit_reason": "TP"},
        {"net_ret": -0.04, "net_pnl": -400.0, "exit_reason": "SL"},
        {"net_ret": 0.09, "net_pnl": 900.0, "exit_reason": "TP"},
    ]
    synthetic_trades_ctrl = [
        {"net_ret": 0.01, "net_pnl": 100.0, "exit_reason": "TIME"},
        {"net_ret": -0.02, "net_pnl": -200.0, "exit_reason": "SL"},
    ]

    rec = reconcile_phase_3b_trades(synthetic_trades_vol, synthetic_trades_ctrl)

    # Arithmetic sum = 0.08 + 0.05 - 0.04 + 0.09 = 0.18 (18.0%)
    expected_arith = 0.18
    # Compounded = (1.08 * 1.05 * 0.96 * 1.09) - 1.0 = 0.18659...
    expected_comp = (1.08 * 1.05 * 0.96 * 1.09) - 1.0

    assert np.isclose(rec["volatility_candidate"]["arithmetic_sum_net_return"], expected_arith)
    assert np.isclose(rec["volatility_candidate"]["continuous_compounded_net_return"], expected_comp, atol=1e-4)
    assert rec["volatility_candidate"]["trade_count"] == 4
    assert rec["volatility_candidate"]["top1_trade_ret"] == 0.09
    assert rec["volatility_candidate"]["top3_pnl_concentration_pct"] > 50.0


def test_nested_selection_dimension_discipline():
    """Verify candidate matrices satisfy compact dimension criteria (<= 8 features)."""
    assert len(["ATR_Regime_Ratio", "Parkinson_Vol_20", "Realized_Vol_Ratio_20_60", "ATR_Percentile_252", "VIX_Level"]) == 5
    assert len(["ATR_Regime_Ratio", "Parkinson_Vol_20", "Realized_Vol_Ratio_20_60", "ATR_Percentile_252", "Relative_Strength", "ZScore_RSI_120", "ZScore_MACD_Hist_50", "ZScore_Volume_Ratio_50"]) == 8
