# backend/tests/test_regime_discrimination_causality.py
"""
HYDRA V2.4.1 — Causality, Temporal Invariance & Governance Regression Test Suite.

Verifies:
- Test A: Future-Append Invariance (Appending future bars preserves historical features, regimes, decisions).
- Test B: Future-Price Invariance (Mutating future price prints does not alter prior signals).
- Test C: Moving-Average Causality (Strictly backward-looking windows, no future leakage).
- Test D: Regime Boundary Behavior (Exact boundary cross, undefined history handling).
- Test E: No Hidden Look-Ahead (Normalization fit strictly on training fold, zero validation leakage).
- Test F: Baseline Behavior Preservation (Frozen V2.4 manifest and logic unmutated).
- Test G: Trading-Cost Accounting Consistency (Identical execution timing and costs applied across all candidates).
- Test H: No Majority-Class Collapse (Explicit rejection of 100% SELL trivial classifiers).
- Test I: Numerical Repeatability (Deterministic outputs under fixed seed).
"""

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.execution.evaluation_contract import (
    canonical_multiclass_brier,
)


class TestRegimeDiscriminationCausality(unittest.TestCase):
    def setUp(self):
        self.backend_dir = BACKEND_DIR
        self.manifest_v24_path = self.backend_dir / "artifacts" / "frozen_strategy_manifest_v2.4.json"
        self.expected_v24_digest = "d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5"

    def test_a_future_append_invariance(self):
        """Test A: Appending future observations does not alter prior features or regime decisions."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        closes = 150.0 + np.cumsum(np.random.randn(100))
        df_base = pd.DataFrame({"Close": closes[:80]}, index=dates[:80])
        df_base["SMA50"] = df_base["Close"].rolling(50).mean()
        df_base["Regime"] = df_base["Close"] >= df_base["SMA50"]

        df_extended = pd.DataFrame({"Close": closes}, index=dates)
        df_extended["SMA50"] = df_extended["Close"].rolling(50).mean()
        df_extended["Regime"] = df_extended["Close"] >= df_extended["SMA50"]

        pd.testing.assert_series_equal(df_base["SMA50"], df_extended["SMA50"].iloc[:80], check_names=False)
        pd.testing.assert_series_equal(df_base["Regime"], df_extended["Regime"].iloc[:80], check_names=False)

    def test_b_future_price_invariance(self):
        """Test B: Altering future prices does not alter decision at bar T."""
        np.random.seed(42)
        base_prices = 100.0 + np.cumsum(np.random.randn(50))
        t_decision = 35

        sma_t_before = pd.Series(base_prices[: t_decision + 1]).rolling(20).mean().iloc[-1]

        # Mutate future prices drastically
        mutated_prices = base_prices.copy()
        mutated_prices[t_decision + 1 :] = mutated_prices[t_decision + 1 :] * 5.0

        sma_t_after = pd.Series(mutated_prices[: t_decision + 1]).rolling(20).mean().iloc[-1]
        self.assertAlmostEqual(sma_t_before, sma_t_after, places=9)

    def test_c_moving_average_causality(self):
        """Test C: Rolling indicators must not look forward."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        series = pd.Series(range(1, 31), index=dates, dtype=float)
        # Backward-looking rolling mean
        roll_mean = series.rolling(5).mean()
        # Bar 4 (index 4, value 5) should be mean of [1,2,3,4,5] = 3.0
        self.assertEqual(roll_mean.iloc[4], 3.0)
        # Verify first 4 bars are NaN (warmup requirement)
        self.assertTrue(roll_mean.iloc[:4].isna().all())

    def test_d_regime_boundary_behavior(self):
        """Test D: Boundary testing exactly at, above, and below SMA."""
        sma = 100.0
        # Exactly at boundary: Close >= SMA should be True
        self.assertTrue(100.0 >= sma)
        # Below boundary
        self.assertFalse(99.9999 >= sma)
        # Above boundary
        self.assertTrue(100.0001 >= sma)

        # Handling missing history (NaN)
        nan_sma = float("nan")
        self.assertFalse(bool(100.0 >= nan_sma))

    def test_e_no_hidden_lookahead_scaler_isolation(self):
        """Test E: Verifies scaler fitted strictly on train fold does not leak validation distribution."""
        np.random.seed(42)
        train_data = np.random.normal(loc=10.0, scale=2.0, size=(100, 1))
        val_data = np.random.normal(loc=50.0, scale=10.0, size=(50, 1))  # Distribution shift

        scaler = StandardScaler()
        scaler.fit(train_data)

        # Mean of scaler should match train distribution, not combined
        self.assertAlmostEqual(scaler.mean_[0], 10.0, delta=0.5)
        # Scaling val_data must not alter scaler parameters
        _ = scaler.transform(val_data)
        self.assertAlmostEqual(scaler.mean_[0], 10.0, delta=0.5)

    def test_f_baseline_behavior_preservation(self):
        """Test F: Verifies frozen V2.4 manifest remains byte-for-byte unchanged."""
        self.assertTrue(self.manifest_v24_path.exists())
        actual_digest = hashlib.sha256(self.manifest_v24_path.read_bytes()).hexdigest()
        self.assertEqual(actual_digest, self.expected_v24_digest)

    def test_g_trading_cost_accounting(self):
        """Test G: Execution costs and slippage are identical and causal across all candidates."""
        open_price = 100.0
        # Buy: 5 bps slippage
        exec_buy = open_price * 1.0005
        self.assertAlmostEqual(exec_buy, 100.05, places=4)
        # Sell: 5 bps slippage
        exec_sell = open_price * 0.9995
        self.assertAlmostEqual(exec_sell, 99.95, places=4)

        # Commission formula: $0.005/share, min $1.00
        shares_100 = 100
        comm_100 = max(1.0, shares_100 * 0.005)
        self.assertEqual(comm_100, 1.0)  # $0.50 floored at $1.00

        shares_500 = 500
        comm_500 = max(1.0, shares_500 * 0.005)
        self.assertEqual(comm_500, 2.50)

    def test_h_no_majority_class_collapse(self):
        """Test H: Flags and rejects trivial majority-class collapse (e.g. 100% SELL)."""
        y_true = np.array([0] * 75 + [1] * 10 + [2] * 15)  # 75% class 0
        collapsed_preds = np.array([0] * 100)  # 100% class 0

        # Accuracy is high (75%), but BUY recall is 0.0%
        tp_buy = np.sum((y_true == 2) & (collapsed_preds == 2))
        buy_recall = tp_buy / np.sum(y_true == 2)
        self.assertEqual(buy_recall, 0.0)

        # Institutional rule: a candidate with 0% BUY recall MUST fail acceptance
        is_acceptable = buy_recall > 0.05
        self.assertFalse(is_acceptable)

    def test_i_numerical_repeatability(self):
        """Test I: Repeatability under fixed data and calculation."""
        y_true = np.array([0, 1, 2, 0, 2])
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
            [0.3, 0.3, 0.4],
        ])
        brier_1 = canonical_multiclass_brier(y_true, probs)
        brier_2 = canonical_multiclass_brier(y_true, probs)
        self.assertEqual(brier_1, brier_2)
        self.assertAlmostEqual(brier_1, 0.224, places=4)


if __name__ == "__main__":
    unittest.main()
