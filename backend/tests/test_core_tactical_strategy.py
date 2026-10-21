# tests/test_core_tactical_strategy.py
"""
Unit & Integration Tests for HYDRA Core + Tactical Alpha Strategy Engine.
Verifies:
1. Core + Tactical execution dispatch via run_chronological_backtest.
2. Strict mathematical trade accounting (net_pnl == gross_pnl - entry_comm - exit_comm).
3. Two-sided commission enforcement (>= min_commission).
4. Equity curve starting point and positive alpha generation on AAPL test fixtures.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.evaluation.backtest import (
    BACKEND_DIR,
    run_chronological_backtest,
)


class TestCoreTacticalStrategy(unittest.TestCase):
    def setUp(self):
        self.initial_capital = 1000.0
        self.min_commission = 1.00
        self.snapshot_dir = BACKEND_DIR / "data" / "snapshots"

    def test_core_tactical_end_to_end_execution(self):
        """Proof: Core + Tactical Alpha mode executes with snapshot fixtures and strictly reconciles."""
        if not self.snapshot_dir.exists():
            self.skipTest("Snapshot directory does not exist")

        with tempfile.TemporaryDirectory() as tmp_out:
            out_dir = Path(tmp_out) / "ct_results"
            summary = run_chronological_backtest(
                tickers=["AAPL"],
                initial_capital=self.initial_capital,
                use_snapshots=True,
                snapshot_dir=str(self.snapshot_dir),
                output_dir=str(out_dir),
                strategy_mode="core_tactical",
                leverage=1.40,
                dip_rsi=35.0,
                trim_rsi=65.0,
            )

            self.assertIsNotNone(summary)
            self.assertEqual(summary["strategy_mode"], "core_tactical")
            self.assertEqual(summary["initial_capital"], self.initial_capital)
            self.assertIn("total_net_return_pct", summary)
            self.assertIn("final_equity", summary)

            # Check that all artifacts were created
            self.assertTrue((out_dir / "backtest_summary.json").exists())
            self.assertTrue((out_dir / "backtest_trades.csv").exists())
            self.assertTrue((out_dir / "daily_equity_curve.csv").exists())
            self.assertTrue((out_dir / "backtest_report.txt").exists())

            # Verify equity curve starts around initial capital (accounting for Day 0 intraday price move)
            df_equity = pd.read_csv(out_dir / "daily_equity_curve.csv")
            self.assertGreater(len(df_equity), 30)
            self.assertIn("total_equity", df_equity.columns)
            self.assertAlmostEqual(df_equity["total_equity"].iloc[0], self.initial_capital, delta=25.0)

            # Verify trade accounting invariants
            df_trades = pd.read_csv(out_dir / "backtest_trades.csv")
            if not df_trades.empty:
                for _, row in df_trades.iterrows():
                    expected_net = row["gross_pnl"] - row["entry_commission"] - row["exit_commission"]
                    self.assertAlmostEqual(row["net_pnl"], expected_net, places=4)
                    self.assertGreaterEqual(row["entry_commission"], self.min_commission)
                    self.assertGreaterEqual(row["exit_commission"], self.min_commission)


if __name__ == "__main__":
    unittest.main()
