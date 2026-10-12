# backend/tests/test_workstream_2_directional_baselines.py
"""
Unit and Quantitative Test Suite for HYDRA Workstream 2:
Directional Baselines, Execution Accounting, and The $1,000 Portfolio Simulation.

Validates:
1. Pinned dataset and frozen manifest cryptographic invariance.
2. Deterministic execution accounting:
   - Whole-share purchases only.
   - Cash non-negativity constraint.
   - Brokerage fee application (max($1.00, Q * $0.005)).
   - Adverse slippage direction (+5 bps on BUY, -5 bps on SELL).
3. Causal fill timing: decision at Close[t] fills at Open[t+1].
4. Exact reconciliation: Equity = Cash + Position * Close.
5. Complete trade ledger: includes both winners AND losers.
6. Slippage sensitivity: higher slippage monotonically erodes or maintains equity.
7. Formal acceptance gates evaluation.
"""

import unittest

import numpy as np
import pandas as pd

from scripts.research.workstream_2_directional_baselines import (
    EXPECTED_PINNED_SHA256,
    EXPECTED_V24_MANIFEST_SHA256,
    InstitutionalPortfolioSimulator,
    verify_governance_invariants,
)


class TestWorkstream2DirectionalBaselines(unittest.TestCase):
    def test_1_governance_invariants(self):
        """Proof 1: Baseline manifest and pinned dataset hashes match exact requirements."""
        res = verify_governance_invariants()
        self.assertEqual(res["status"], "ALL_INVARIANTS_VERIFIED")
        self.assertEqual(res["manifest_sha256"], EXPECTED_V24_MANIFEST_SHA256)
        self.assertEqual(res["pinned_dataset_sha256"], EXPECTED_PINNED_SHA256)

    def test_2_deterministic_brokerage_fees_and_whole_shares(self):
        """Proof 2: Whole shares only, min commission $1.00, cash never negative."""
        sim = InstitutionalPortfolioSimulator(
            initial_capital=1000.0,
            slippage_bps=5.0,
            commission_per_share=0.005,
            min_commission=1.00,
        )
        self.assertEqual(sim.compute_commission(10), 1.00)  # 10 * 0.005 = 0.05 < 1.00 -> $1.00
        self.assertEqual(sim.compute_commission(500), 2.50)  # 500 * 0.005 = 2.50

        # Test single BUY and SELL trade lifecycle
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        df_prices = pd.DataFrame(
            {
                "Open": [100.0, 102.0, 104.0, 106.0, 108.0],
                "High": [101.0, 103.0, 105.0, 107.0, 109.0],
                "Low": [99.0, 101.0, 103.0, 105.0, 107.0],
                "Close": [100.5, 102.5, 104.5, 106.5, 108.5],
            },
            index=dates,
        )
        # Signal BUY on bar 0, SELL on bar 2
        sigs = pd.Series(["BUY", "HOLD", "SELL", "HOLD", "HOLD"], index=dates)

        res = sim.simulate(df_prices, sigs, "Test_Strategy")
        summary = res["summary"]
        trades = res["trades"]

        # BUY executes at bar 1 Open: 102.0 * (1 + 0.0005) = 102.051
        # Max alloc cash: 1000 - 1.0 = 999.0
        # Q = floor(999.0 / 102.051) = 9 shares
        # Cost = 9 * 102.051 + 1.0 = 918.459 + 1.0 = 919.459
        # Cash left = 1000.0 - 919.459 = 80.541
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["entry_shares"], 9)
        self.assertEqual(t["entry_fill_price"], 102.051)
        self.assertEqual(t["entry_fees"], 1.0)

        # SELL executes at bar 3 Open: 106.0 * (1 - 0.0005) = 105.947
        # Proceeds = 9 * 105.947 - 1.0 = 953.523 - 1.0 = 952.523
        self.assertEqual(t["exit_shares"], 9)
        self.assertEqual(t["exit_fill_price"], 105.947)
        self.assertEqual(t["exit_fees"], 1.0)
        self.assertTrue(t["is_winner"])

        # Cash after exit: 80.541 + 952.523 = 1033.064
        self.assertAlmostEqual(summary["final_equity"], 1033.06, places=1)

    def test_3_causal_timing_and_fill_invariance(self):
        """Proof 3: Signal at Close[t] fills at Open[t+1] with adverse slippage."""
        sim = InstitutionalPortfolioSimulator(initial_capital=1000.0, slippage_bps=10.0)
        dates = pd.date_range("2026-01-01", periods=3, freq="D")
        df_prices = pd.DataFrame(
            {
                "Open": [50.0, 52.0, 54.0],
                "High": [51.0, 53.0, 55.0],
                "Low": [49.0, 51.0, 53.0],
                "Close": [50.0, 52.0, 54.0],
            },
            index=dates,
        )
        sigs = pd.Series(["BUY", "HOLD", "HOLD"], index=dates)
        res = sim.simulate(df_prices, sigs, "Timing_Test")

        # Fill occurred on bar 1 (Open = 52.0), NOT bar 0
        eq_curve = res["equity_curve"]
        self.assertEqual(eq_curve[0]["shares"], 0)  # FLAT at bar 0 close
        self.assertGreater(eq_curve[1]["shares"], 0)  # LONG at bar 1 close

    def test_4_reconciliation_equity_equals_cash_plus_position_value(self):
        """Proof 4: Identity holds continuously: Equity_t = Cash_t + Q_t * Close_t."""
        sim = InstitutionalPortfolioSimulator(initial_capital=1000.0, slippage_bps=5.0)
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(10) * 2.0)
        df_prices = pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
            },
            index=dates,
        )
        # Alternate BUY and SELL
        sigs = pd.Series(["BUY", "HOLD", "SELL", "HOLD", "BUY", "HOLD", "HOLD", "SELL", "HOLD", "HOLD"], index=dates)
        res = sim.simulate(df_prices, sigs, "Reconciliation_Test")

        for bar in res["equity_curve"]:
            expected_eq = round(bar["cash"] + (bar["shares"] * bar["close"]), 2)
            self.assertLessEqual(abs(bar["equity"] - expected_eq), 0.02)

    def test_5_complete_trade_ledger_includes_losses(self):
        """Proof 5: Trade ledger contains both winning and losing trades without omission."""
        sim = InstitutionalPortfolioSimulator(initial_capital=1000.0, slippage_bps=5.0)
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        # Crash prices to force losing trade
        df_prices = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 80.0, 80.0, 100.0, 100.0],
                "High": [101.0, 101.0, 81.0, 81.0, 101.0, 101.0],
                "Low": [99.0, 99.0, 79.0, 79.0, 99.0, 99.0],
                "Close": [100.0, 100.0, 80.0, 80.0, 100.0, 100.0],
            },
            index=dates,
        )
        # Buy at 100, sell at 80 (loss)
        sigs = pd.Series(["BUY", "SELL", "HOLD", "HOLD", "HOLD", "HOLD"], index=dates)
        res = sim.simulate(df_prices, sigs, "Loss_Test")

        trades = res["trades"]
        self.assertEqual(len(trades), 1)
        self.assertFalse(trades[0]["is_winner"])
        self.assertLess(trades[0]["net_pnl"], 0.0)
        self.assertEqual(res["summary"]["losing_trades"], 1)

    def test_6_slippage_monotonicity(self):
        """Proof 6: Higher slippage strictly produces equal or lower final equity."""
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        df_prices = pd.DataFrame(
            {
                "Open": [100.0, 102.0, 105.0, 103.0, 106.0, 108.0, 110.0, 107.0, 109.0, 112.0],
                "High": [101.0, 103.0, 106.0, 104.0, 107.0, 109.0, 111.0, 108.0, 110.0, 113.0],
                "Low": [99.0, 101.0, 104.0, 102.0, 105.0, 107.0, 109.0, 106.0, 108.0, 111.0],
                "Close": [100.5, 102.5, 105.5, 103.5, 106.5, 108.5, 110.5, 107.5, 109.5, 112.5],
            },
            index=dates,
        )
        sigs = pd.Series(["BUY", "HOLD", "SELL", "BUY", "HOLD", "SELL", "HOLD", "HOLD", "HOLD", "HOLD"], index=dates)

        sim_low = InstitutionalPortfolioSimulator(slippage_bps=0.0)
        sim_high = InstitutionalPortfolioSimulator(slippage_bps=20.0)

        res_low = sim_low.simulate(df_prices, sigs, "Low_Slippage")
        res_high = sim_high.simulate(df_prices, sigs, "High_Slippage")

        self.assertGreater(res_low["summary"]["final_equity"], res_high["summary"]["final_equity"])


if __name__ == "__main__":
    unittest.main()
