import os
import sqlite3
import tempfile
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from execution.broker_interface import MockPaperBroker
from execution.paper_runner import DailyPaperRunner


class TestDailyPaperRunner(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.test_dir.name, "test_paper_runner.db")
        self.broker = MockPaperBroker(initial_capital=100000.0, state_file=self.db_path)
        self.runner = DailyPaperRunner(
            broker=self.broker,
            state_db_path=self.db_path,
            universe=["AAPL", "NVDA"],
            benchmark_tickers=["SPY"],
            target_risk_pct=0.01,
            initial_capital=100000.0,
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def _create_mock_df(
        self,
        close_val: float,
        sma_200_val: float,
        high_val: float,
        low_val: float,
        atr_val: float,
        ts_mult_val: float,
    ) -> pd.DataFrame:
        dates = pd.date_range(end=datetime.now(), periods=50, freq="D")
        df = pd.DataFrame(
            {
                "Open": [close_val] * 50,
                "High": [high_val] * 50,
                "Low": [low_val] * 50,
                "Close": [close_val] * 50,
                "Volume": [1000000] * 50,
                "SMA_200": [sma_200_val] * 50,
                "ATR": [atr_val] * 50,
                "ts_mult": [ts_mult_val] * 50,
                "vol_20d": [0.015] * 50,
            },
            index=dates,
        )
        return df

    def test_db_initialization(self):
        """Verify runner initializes the SQLite schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            self.assertIn("daily_execution_runs", tables)
            self.assertIn("trailing_stops", tables)
        finally:
            conn.close()

    def test_evaluate_macro_filters_bull_vs_bear(self):
        """Verify symmetric macro regime gating."""
        # SPY healthy: Close 450 >= SMA50 420
        spy_df = self._create_mock_df(450.0, 420.0, 455.0, 445.0, 4.0, 2.5)
        spy_df["SPY_SMA_50"] = 420.0

        # AAPL healthy: Close 160 >= SMA200 140 -> Bull (Long allowed, Short forbidden)
        aapl_df = self._create_mock_df(160.0, 140.0, 162.0, 158.0, 2.5, 3.0)

        # NVDA breakdown: Close 110 < SMA200 130 -> Non-bull (Long forbidden, Short allowed)
        nvda_df = self._create_mock_df(110.0, 130.0, 112.0, 108.0, 4.0, 4.0)

        mock_data = {"SPY": spy_df, "AAPL": aapl_df, "NVDA": nvda_df}
        filters = self.runner.evaluate_macro_filters(mock_data)

        self.assertTrue(filters["AAPL"]["long_allowed"])
        self.assertFalse(filters["AAPL"]["short_allowed"])

        self.assertFalse(filters["NVDA"]["long_allowed"])
        self.assertTrue(filters["NVDA"]["short_allowed"])

    def test_asymmetric_veto_signal_filtering(self):
        """Verify model predictions flow through Asymmetric Veto and macro filters."""
        spy_df = self._create_mock_df(450.0, 420.0, 455.0, 445.0, 4.0, 2.5)
        spy_df["SPY_SMA_50"] = 420.0
        aapl_df = self._create_mock_df(160.0, 140.0, 162.0, 158.0, 2.5, 3.0)
        nvda_df = self._create_mock_df(140.0, 130.0, 142.0, 138.0, 4.0, 4.0)

        mock_data = {"SPY": spy_df, "AAPL": aapl_df, "NVDA": nvda_df}
        macro_filters = self.runner.evaluate_macro_filters(mock_data)

        # AAPL: XGB predicts strong BUY (P=0.75), secondary models neutral -> Approved BUY
        # NVDA: XGB predicts strong BUY (P=0.75), DQN vetoes with strong SELL (P=0.70) -> Vetoed HOLD
        mock_preds = {
            "AAPL": {
                "XGB_AGENT": np.array([0.10, 0.15, 0.75]),
                "LGBM_AGENT": np.array([0.30, 0.40, 0.30]),
                "DQN_AGENT": np.array([0.30, 0.40, 0.30]),
            },
            "NVDA": {
                "XGB_AGENT": np.array([0.10, 0.15, 0.75]),
                "LGBM_AGENT": np.array([0.30, 0.40, 0.30]),
                "DQN_AGENT": np.array([0.70, 0.15, 0.15]),  # Veto
            },
        }

        signals = self.runner.generate_and_filter_signals(mock_data, macro_filters, mock_preds)

        self.assertEqual(signals["AAPL"]["signal"], "BUY")
        self.assertIn("Approved", signals["AAPL"]["signal_note"])

        self.assertEqual(signals["NVDA"]["signal"], "HOLD")
        self.assertIn("Vetoed", signals["NVDA"]["signal_note"])

    def test_asset_expectancy_filter_suspends_underperforming_symbol(self):
        """Verify AssetExpectancyFilter suspends new entries when trailing PF < 1.15."""
        spy_df = self._create_mock_df(450.0, 420.0, 455.0, 445.0, 4.0, 2.5)
        spy_df["SPY_SMA_50"] = 420.0
        nvda_df = self._create_mock_df(140.0, 130.0, 142.0, 138.0, 4.0, 4.0)
        mock_data = {"SPY": spy_df, "NVDA": nvda_df}
        macro_filters = self.runner.evaluate_macro_filters(mock_data)

        # Record 4 losing trades on NVDA to drop trailing PF to 0.50
        now = datetime.now()
        for _ in range(4):
            self.runner.expectancy_filter.record_trade("NVDA", now, -0.03)

        # Even with strong BUY signals, NVDA must be suppressed by the expectancy gate
        mock_preds = {
            "NVDA": {
                "XGB_AGENT": np.array([0.05, 0.15, 0.80]),
                "LGBM_AGENT": np.array([0.05, 0.15, 0.80]),
                "DQN_AGENT": np.array([0.05, 0.15, 0.80]),
            }
        }

        signals = self.runner.generate_and_filter_signals(mock_data, macro_filters, mock_preds)
        self.assertEqual(signals["NVDA"]["signal"], "HOLD")
        self.assertIn("Suspended by Expectancy Gate", signals["NVDA"]["signal_note"])

    def test_size_and_execute_orders_with_adaptive_stop(self):
        """Verify order sizing and volatility-adaptive trailing stop persistence."""
        aapl_df = self._create_mock_df(150.0, 140.0, 152.0, 148.0, 3.0, 3.5)
        mock_data = {"AAPL": aapl_df}

        signals = {
            "AAPL": {
                "signal": "BUY",
                "current_price": 150.0,
                "atr": 3.0,
                "ts_mult": 3.5,  # 3.5 * 3.0 = 10.5 stop distance
            }
        }

        orders = self.runner.size_and_execute_orders(signals, mock_data)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["symbol"], "AAPL")
        self.assertEqual(orders[0]["side"], "BUY")
        self.assertEqual(orders[0]["status"], "FILLED")

        # Verify trailing stop state persisted
        ts_state = self.runner._get_trailing_stop_state("AAPL")
        self.assertIsNotNone(ts_state)
        self.assertEqual(ts_state["side"], "LONG")
        self.assertAlmostEqual(ts_state["ts_mult"], 3.5, places=2)
        # stop_price = fill_price - 3.5 * 3.0
        self.assertAlmostEqual(ts_state["stop_price"], orders[0]["fill_price"] - 10.5, places=2)

    def test_trailing_stop_breach_triggers_exit(self):
        """Verify trailing stop breach exits position and updates expectancy filter."""
        # Open LONG on AAPL @ 150
        self.broker.submit_order("AAPL", qty=100, side="BUY", current_price=150.0)
        self.runner._save_trailing_stop_state(
            "AAPL", "LONG", 150.15, 150.15, 140.0, 3.0, 3.0
        )

        # Market plunges: Low is 135 <= 140 stop
        aapl_crash_df = self._create_mock_df(138.0, 140.0, 142.0, 135.0, 3.0, 3.0)
        mock_data = {"AAPL": aapl_crash_df}

        closed_orders = self.runner.update_existing_positions(mock_data)
        self.assertEqual(len(closed_orders), 1)
        self.assertEqual(closed_orders[0]["symbol"], "AAPL")
        self.assertEqual(closed_orders[0]["side"], "SELL")
        self.assertEqual(closed_orders[0]["status"], "FILLED")

        # Position should be closed
        self.assertEqual(len(self.broker.get_positions()), 0)
        # Trailing stop state should be removed
        self.assertIsNone(self.runner._get_trailing_stop_state("AAPL"))
        # Expectancy filter should have recorded the trade
        pf, count = self.runner.expectancy_filter.get_trailing_profit_factor("AAPL", datetime.now())
        self.assertEqual(count, 1)
        self.assertLess(pf, 1.0)

    def test_full_daily_cycle_e2e(self):
        """Verify complete end-to-end daily paper cycle execution."""
        spy_df = self._create_mock_df(450.0, 420.0, 455.0, 445.0, 4.0, 2.5)
        spy_df["SPY_SMA_50"] = 420.0
        aapl_df = self._create_mock_df(160.0, 140.0, 162.0, 158.0, 2.5, 3.0)
        nvda_df = self._create_mock_df(140.0, 130.0, 142.0, 138.0, 4.0, 4.0)
        mock_data = {"SPY": spy_df, "AAPL": aapl_df, "NVDA": nvda_df}

        mock_preds = {
            "AAPL": {
                "XGB_AGENT": np.array([0.05, 0.15, 0.80]),
                "LGBM_AGENT": np.array([0.10, 0.20, 0.70]),
                "DQN_AGENT": np.array([0.10, 0.20, 0.70]),
            },
            "NVDA": {
                "XGB_AGENT": np.array([0.20, 0.60, 0.20]),  # HOLD
                "LGBM_AGENT": np.array([0.20, 0.60, 0.20]),
                "DQN_AGENT": np.array([0.20, 0.60, 0.20]),
            },
        }

        summary = self.runner.run_daily_cycle(mock_data, mock_preds)

        self.assertIn("run_id", summary)
        self.assertEqual(summary["signals"]["AAPL"]["signal"], "BUY")
        self.assertEqual(summary["signals"]["NVDA"]["signal"], "HOLD")
        self.assertEqual(len(summary["new_orders"]), 1)
        self.assertEqual(summary["new_orders"][0]["symbol"], "AAPL")
        self.assertEqual(len(summary["positions"]), 1)

        # Verify persisted in SQLite
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT run_id, cash, equity FROM daily_execution_runs WHERE run_id = ?", (summary["run_id"],))
            row = cur.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], summary["run_id"])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
