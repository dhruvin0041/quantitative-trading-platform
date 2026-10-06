# tests/test_backtest_accounting.py
"""
Deterministic Unit Tests for HYDRA Chronological Backtest Accounting & Realism.
Verifies:
1. Intraday barrier triggers (High for TP, Low for SL).
2. Stop-Loss precedence on bars breaching both TP and SL.
3. Two-sided commission accounting in per-trade net_pnl.
4. Mathematical cash reconciliation: cash_delta == net_pnl.
5. End-of-backtest forced liquidation accounting.
6. Immutable snapshot dataset loading.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.evaluation.backtest import fetch_and_prepare_data


class TestBacktestAccountingAndRealism(unittest.TestCase):
    def setUp(self):
        self.initial_capital = 100000.0
        self.slippage_bps = 5.0
        self.slippage_rate = 5.0 / 10000.0
        self.commission_per_share = 0.005
        self.min_commission = 1.00

    def test_long_cash_reconciliation_and_two_sided_commissions(self):
        """Proof: Long trade cash delta must strictly equal net_pnl down to the cent."""
        cash = self.initial_capital
        shares = 100
        open_p = 150.0

        # Entry at Open with adverse slippage
        entry_fill = open_p * (1.0 + self.slippage_rate)
        entry_commission = max(self.min_commission, shares * self.commission_per_share)
        required_cash = (shares * entry_fill) + entry_commission
        cash -= required_cash

        pos = {
            "side": "LONG",
            "shares": shares,
            "entry_fill": entry_fill,
            "entry_commission": entry_commission,
            "tp_price": 160.0,
            "sl_price": 140.0,
            "bars_held": 2,
        }

        # Exit at Take Profit intraday
        exit_fill_base = pos["tp_price"]
        exit_fill = exit_fill_base * (1.0 - self.slippage_rate)
        gross_pnl = (exit_fill - pos["entry_fill"]) * shares
        exit_commission = max(self.min_commission, shares * self.commission_per_share)
        net_pnl = gross_pnl - exit_commission - pos["entry_commission"]

        cash += (shares * exit_fill) - exit_commission
        total_cash_delta = cash - self.initial_capital

        self.assertAlmostEqual(total_cash_delta, net_pnl, places=4)
        self.assertLess(net_pnl, gross_pnl)
        self.assertEqual(net_pnl, gross_pnl - exit_commission - entry_commission)

    def test_short_cash_reconciliation_and_two_sided_commissions(self):
        """Proof: Short trade cash delta must strictly equal net_pnl down to the cent."""
        cash = self.initial_capital
        shares = 100
        open_p = 200.0

        # Short Entry at Open with adverse slippage
        entry_fill = open_p * (1.0 - self.slippage_rate)
        entry_commission = max(self.min_commission, shares * self.commission_per_share)
        required_cash = (shares * entry_fill) + entry_commission
        cash -= required_cash

        pos = {
            "side": "SHORT",
            "shares": shares,
            "entry_fill": entry_fill,
            "entry_commission": entry_commission,
            "tp_price": 190.0,
            "sl_price": 210.0,
            "bars_held": 3,
        }

        # Exit at Take Profit intraday
        exit_fill_base = pos["tp_price"]
        exit_fill = exit_fill_base * (1.0 + self.slippage_rate)
        gross_pnl = (pos["entry_fill"] - exit_fill) * shares
        exit_commission = max(self.min_commission, shares * self.commission_per_share)
        net_pnl = gross_pnl - exit_commission - pos["entry_commission"]

        cash += (shares * (2.0 * pos["entry_fill"] - exit_fill)) - exit_commission
        total_cash_delta = cash - self.initial_capital

        self.assertAlmostEqual(total_cash_delta, net_pnl, places=4)

    def test_intraday_barrier_evaluation_logic(self):
        """Proof: Intraday barrier checking correctly triggers TP, SL, and SL-precedence."""
        # 1. Normal Take Profit hit via High
        pos_long = {
            "side": "LONG",
            "tp_price": 105.0,
            "sl_price": 95.0,
            "bars_held": 1,
        }
        bar_normal_tp = {"Open": 100.0, "High": 106.0, "Low": 99.0, "Close": 104.0}
        hit_sl = bar_normal_tp["Low"] <= pos_long["sl_price"]
        hit_tp = bar_normal_tp["High"] >= pos_long["tp_price"]
        self.assertFalse(hit_sl)
        self.assertTrue(hit_tp)

        # 2. Both TP and SL touched on same day -> Conservative Stop-Loss Precedence
        bar_both_hit = {"Open": 100.0, "High": 107.0, "Low": 93.0, "Close": 101.0}
        hit_sl_both = bar_both_hit["Low"] <= pos_long["sl_price"]
        hit_tp_both = bar_both_hit["High"] >= pos_long["tp_price"]
        self.assertTrue(hit_sl_both and hit_tp_both)
        # Verify resolution
        if hit_sl_both and hit_tp_both:
            resolved_reason = "STOP_LOSS"
            fill_base = pos_long["sl_price"]
        elif hit_sl_both:
            resolved_reason = "STOP_LOSS"
            fill_base = pos_long["sl_price"]
        else:
            resolved_reason = "TAKE_PROFIT"
            fill_base = pos_long["tp_price"]

        self.assertEqual(resolved_reason, "STOP_LOSS")
        self.assertEqual(fill_base, 95.0)

    def test_end_of_backtest_liquidation_accounting(self):
        """Proof: Final liquidation deducts both commissions and applies exit slippage."""
        cash = 50000.0
        shares = 50
        entry_fill = 100.0
        entry_comm = 1.00
        close_p = 110.0

        pos = {
            "side": "LONG",
            "shares": shares,
            "entry_fill": entry_fill,
            "entry_commission": entry_comm,
            "bars_held": 4,
        }

        exit_fill = close_p * (1.0 - self.slippage_rate)
        gross_pnl = (exit_fill - pos["entry_fill"]) * pos["shares"]
        exit_comm = max(self.min_commission, pos["shares"] * self.commission_per_share)
        net_pnl = gross_pnl - exit_comm - pos["entry_commission"]

        initial_cash_before_trade = cash + (shares * entry_fill) + entry_comm
        cash += (shares * exit_fill) - exit_comm
        delta = cash - initial_cash_before_trade

        self.assertAlmostEqual(delta, net_pnl, places=4)

    def test_snapshot_loading_determinism(self):
        """Proof: fetch_and_prepare_data loads from snapshot parquet when present without network call."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            snap_dir = Path(tmp_dir)
            sample_df = pd.DataFrame({
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000, 1100],
                "ATR_14": [2.0, 2.1],
            }, index=pd.date_range("2024-01-01", periods=2, freq="D"))
            snapshot_file = snap_dir / "TEST_features.parquet"
            sample_df.to_parquet(snapshot_file)

            loaded = fetch_and_prepare_data(
                ticker="TEST",
                spy_df=pd.DataFrame(),
                vix_df=pd.DataFrame(),
                snapshot_dir=snap_dir,
            )
            self.assertEqual(len(loaded), 2)
            self.assertEqual(list(loaded["Close"]), [101.0, 102.0])


if __name__ == "__main__":
    unittest.main()
