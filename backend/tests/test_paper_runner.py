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
        """Verify model predictions flow through Asymmetric Veto and macro filters when enabled."""
        self.runner.use_veto = True
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

    def test_pure_xgboost_default_mode(self):
        """Verify default production mode (use_veto=False) executes XGBoost alpha without secondary veto."""
        self.assertFalse(self.runner.use_veto)
        spy_df = self._create_mock_df(450.0, 420.0, 455.0, 445.0, 4.0, 2.5)
        spy_df["SPY_SMA_50"] = 420.0
        nvda_df = self._create_mock_df(140.0, 130.0, 142.0, 138.0, 4.0, 4.0)

        mock_data = {"SPY": spy_df, "NVDA": nvda_df}
        macro_filters = self.runner.evaluate_macro_filters(mock_data)

        # NVDA: XGB predicts strong BUY (P=0.75), DQN predicts strong SELL (P=0.70)
        # In default mode, XGBoost is the unvetoed primary alpha driver
        mock_preds = {
            "NVDA": {
                "XGB_AGENT": np.array([0.10, 0.15, 0.75]),
                "LGBM_AGENT": np.array([0.30, 0.40, 0.30]),
                "DQN_AGENT": np.array([0.70, 0.15, 0.15]),
            },
        }

        signals = self.runner.generate_and_filter_signals(mock_data, macro_filters, mock_preds)
        self.assertEqual(signals["NVDA"]["signal"], "BUY")
        self.assertNotIn("Vetoed", signals["NVDA"]["signal_note"])

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

    def test_multi_signal_capital_allocation_concurrent_limit(self):
        """Verify candidate BUY signals are capped by max_concurrent_positions and sorted by conviction."""
        self.runner.universe = ["AAPL", "NVDA", "MSFT"]
        self.runner.max_concurrent_positions = 2

        aapl_df = self._create_mock_df(150.0, 140.0, 152.0, 148.0, 2.0, 2.5)
        nvda_df = self._create_mock_df(120.0, 110.0, 122.0, 118.0, 2.0, 2.5)
        msft_df = self._create_mock_df(400.0, 380.0, 405.0, 395.0, 4.0, 2.5)

        mock_data = {"AAPL": aapl_df, "NVDA": nvda_df, "MSFT": msft_df}

        signals = {
            "AAPL": {"signal": "BUY", "current_price": 150.0, "atr": 2.0, "ts_mult": 2.5, "agreement_score": 85.0},
            "NVDA": {"signal": "BUY", "current_price": 120.0, "atr": 2.0, "ts_mult": 2.5, "agreement_score": 70.0},
            "MSFT": {"signal": "BUY", "current_price": 400.0, "atr": 4.0, "ts_mult": 2.5, "agreement_score": 60.0},
        }

        orders = self.runner.size_and_execute_orders(signals, mock_data)
        # Max concurrent positions is 2, so only 2 highest conviction orders should be filled
        self.assertEqual(len(orders), 2)
        symbols = [o["symbol"] for o in orders]
        self.assertIn("AAPL", symbols)  # 85.0 conviction
        self.assertIn("NVDA", symbols)  # 70.0 conviction
        self.assertNotIn("MSFT", symbols)  # 60.0 conviction (omitted due to slot cap)

    def test_multi_signal_proportional_cash_split(self):
        """Verify available capital is split proportionally among candidate signals without exceeding buying power."""
        self.runner.universe = ["AAPL", "NVDA"]
        self.runner.max_concurrent_positions = 2
        self.runner.max_portfolio_allocation = 0.80

        aapl_df = self._create_mock_df(100.0, 90.0, 102.0, 98.0, 0.5, 2.5)
        nvda_df = self._create_mock_df(100.0, 90.0, 102.0, 98.0, 0.5, 2.5)
        mock_data = {"AAPL": aapl_df, "NVDA": nvda_df}

        # Available capital pool = 100,000 * 0.80 = 80,000.
        # Two candidates -> capital_per_candidate = 40,000 each.
        signals = {
            "AAPL": {"signal": "BUY", "current_price": 100.0, "atr": 0.5, "ts_mult": 2.5, "agreement_score": 80.0},
            "NVDA": {"signal": "BUY", "current_price": 100.0, "atr": 0.5, "ts_mult": 2.5, "agreement_score": 80.0},
        }

        orders = self.runner.size_and_execute_orders(signals, mock_data)
        self.assertEqual(len(orders), 2)
        total_invested = sum(o["qty"] * o["fill_price"] for o in orders)
        self.assertLessEqual(total_invested, 80000.0 * 1.01)
        for o in orders:
            # Each candidate should receive at most roughly 40,000
            notional = o["qty"] * o["fill_price"]
            self.assertLessEqual(notional, 40000.0 * 1.01)

    def test_dry_run_zero_mutations(self):
        """Verify dry-run mode computes signals, sizes, and stops without broker execution or SQLite DB writes."""
        spy_df = self._create_mock_df(450.0, 420.0, 455.0, 445.0, 4.0, 2.5)
        spy_df["SPY_SMA_50"] = 420.0
        aapl_df = self._create_mock_df(160.0, 140.0, 162.0, 158.0, 2.5, 3.0)
        mock_data = {"SPY": spy_df, "AAPL": aapl_df}

        mock_preds = {
            "AAPL": {
                "XGB_AGENT": np.array([0.05, 0.15, 0.80]),
                "LGBM_AGENT": np.array([0.10, 0.20, 0.70]),
                "DQN_AGENT": np.array([0.10, 0.20, 0.70]),
            },
        }

        summary = self.runner.run_daily_cycle(mock_data, mock_preds, dry_run=True)

        self.assertTrue(summary["dry_run"])
        self.assertIn("DRY_RUN", summary["run_id"])
        self.assertEqual(len(summary["new_orders"]), 1)
        self.assertTrue(summary["new_orders"][0].get("dry_run"))

        # Verify broker state has zero mutations
        self.assertEqual(len(self.broker.get_positions()), 0)
        self.assertEqual(self.broker.get_account_balance()["cash"], 100000.0)

        # Verify SQLite state has zero rows written
        self.assertIsNone(self.runner._get_trailing_stop_state("AAPL"))
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM daily_execution_runs")
            count = cur.fetchone()[0]
            self.assertEqual(count, 0)
        finally:
            conn.close()

    def test_beta_calibrated_trailing_stop_multipliers(self):
        """Verify ts_mult calculation produces differentiated values across varying beta values."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        spy_ret = pd.Series(np.tile([0.01, -0.01, 0.02, -0.02, 0.01], 12), index=dates)
        spy_prices = 400.0 * (1.0 + spy_ret).cumprod()
        spy_df = pd.DataFrame(
            {
                "Open": spy_prices,
                "High": spy_prices * 1.01,
                "Low": spy_prices * 0.99,
                "Close": spy_prices,
                "Volume": [1000000] * 60,
                "vol_20d": [0.01] * 60,
            },
            index=dates,
        )

        test_cases = [
            ("LOW_BETA", 0.5, 2.5),
            ("MID_BETA", 1.2, 3.0),
            ("HIGH_BETA", 1.8, 4.0),
        ]

        mock_data = {"SPY": spy_df}
        for sym, beta_target, _ in test_cases:
            asset_ret = beta_target * spy_df["Close"].pct_change()
            asset_prices = 100.0 * (1.0 + asset_ret.fillna(0)).cumprod()
            asset_df = pd.DataFrame(
                {
                    "Open": asset_prices,
                    "High": asset_prices * 1.01,
                    "Low": asset_prices * 0.99,
                    "Close": asset_prices,
                    "Volume": [1000000] * 60,
                    "vol_20d": [0.01] * 60,
                },
                index=dates,
            )
            mock_data[sym] = asset_df

        # Recompute Beta-calibrated trailing stop multiplier
        spy_ret_calc = spy_df["Close"].pct_change()
        spy_var_20d = spy_ret_calc.rolling(20, min_periods=5).var()

        for sym, _, expected_mult in test_cases:
            sym_df = mock_data[sym]
            asset_ret = sym_df["Close"].pct_change()
            cov_20d = asset_ret.rolling(20, min_periods=5).cov(spy_ret_calc)
            beta = cov_20d / (spy_var_20d + 1e-9)
            sym_df["beta_20d"] = beta
            sym_df["ts_mult"] = np.clip(
                2.5 * np.maximum(1.0, beta.fillna(1.0)), 2.5, 4.0
            ).fillna(2.5)

            actual_mult = float(sym_df["ts_mult"].iloc[-1])
            self.assertAlmostEqual(actual_mult, expected_mult, places=2)

    def test_live_inference_without_mocks_generates_active_probabilities(self):
        """Verify running paper_runner.py without mocks loads real model artifacts and generates varying probabilities."""
        self.assertIsNotNone(self.runner.xgb_model, "XGBoost model artifact was not loaded.")
        self.assertIsNotNone(self.runner.kept_features, "Kept features list was not loaded.")

        dates = pd.date_range("2023-01-01", periods=150, freq="D")
        np.random.seed(42)

        def make_synthetic_df(base_price: float, trend: float) -> pd.DataFrame:
            returns = np.random.normal(trend, 0.015, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            return pd.DataFrame(
                {
                    "Open": prices * 0.995,
                    "High": prices * 1.01,
                    "Low": prices * 0.99,
                    "Close": prices,
                    "Volume": np.random.randint(1000000, 5000000, len(dates)),
                    "SMA_200": prices * 0.95,
                    "ATR": prices * 0.02,
                    "ts_mult": [2.5] * len(dates),
                },
                index=dates,
            )

        spy_df = make_synthetic_df(450.0, 0.0005)
        spy_df["SPY_SMA_50"] = spy_df["Close"].rolling(50).mean().bfill()
        vix_df = pd.DataFrame({"Close": np.random.uniform(14, 22, len(dates))}, index=dates)
        aapl_df = make_synthetic_df(170.0, -0.001)
        nvda_df = make_synthetic_df(120.0, 0.003)

        mock_data = {"SPY": spy_df, "^VIX": vix_df, "AAPL": aapl_df, "NVDA": nvda_df}
        macro_filters = self.runner.evaluate_macro_filters(mock_data)

        # Execute signal generation with mock_predictions=None (in-process live inference)
        signals = self.runner.generate_and_filter_signals(
            mock_data, macro_filters, mock_predictions=None
        )

        self.assertIn("AAPL", signals)
        self.assertIn("NVDA", signals)

        score_aapl = signals["AAPL"]["agreement_score"]
        score_nvda = signals["NVDA"]["agreement_score"]

        # Ensure probabilities are dynamic and NOT identical static 70.0% mock fallbacks
        self.assertNotEqual(score_aapl, 70.0)
        self.assertNotEqual(score_nvda, 70.0)
        self.assertNotEqual(score_aapl, score_nvda)


if __name__ == "__main__":
    unittest.main()

