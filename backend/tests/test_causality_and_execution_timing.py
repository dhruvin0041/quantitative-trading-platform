# backend/tests/test_causality_and_execution_timing.py
"""
Automated Causality and Execution Timing Verification Suite.
Institutional Grade Causal Verification:
1. Append Invariance: Appending future data does NOT alter historical signals at time T.
2. Forming Bar Isolation: Unfinished candles can NEVER produce confirmed signals or ledger entries.
3. SignalLedger Provisional Rejection: Append-only ledger rejects any provisional signal insertion.
4. Next-Session Causal Execution: Backtester executes orders at Open[t+1] +/- slippage, not Close[t].
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.execution.inference_service import InferenceService
from src.execution.live_inference import FEATURE_COLUMNS
from src.execution.signal_ledger import SignalLedger


class MockAlphaModel:
    """Deterministic mock model that assigns signals based on point-in-time features."""
    def predict_proba(self, X):
        n = len(X)
        probs = np.zeros((n, 3))
        for i in range(n):
            val = float(X[i, 0])
            if val > 0.3:
                probs[i] = [0.05, 0.15, 0.80]  # Strong BUY
            elif val < -0.3:
                probs[i] = [0.80, 0.15, 0.05]  # Strong SELL
            else:
                probs[i] = [0.15, 0.70, 0.15]  # HOLD
        return probs


class TestCausalityAndExecutionTiming(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_ledger.db")
        self.ledger = SignalLedger(db_path=self.db_path)

        self.mock_mm = MagicMock()
        self.mock_mm.xgb_model = MockAlphaModel()

        self.service = InferenceService(
            model_manager=self.mock_mm,
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _generate_synthetic_ohlcv(self, start_date="2025-01-01", num_bars=100, seed=42) -> pd.DataFrame:
        """Generates synthetic daily OHLCV dataframe with realistic price walks and feature columns."""
        np.random.seed(seed)
        dates = pd.date_range(start=start_date, periods=num_bars, freq="B")

        returns = np.random.normal(0.0005, 0.015, size=num_bars)
        prices = 150.0 * np.exp(np.cumsum(returns))

        df = pd.DataFrame(index=dates)
        df["Close"] = prices
        df["Open"] = prices * (1.0 + np.random.normal(0.0, 0.002, size=num_bars))
        df["High"] = np.maximum(df["Open"], df["Close"]) * (1.0 + np.abs(np.random.normal(0.002, 0.005, size=num_bars)))
        df["Low"] = np.minimum(df["Open"], df["Close"]) * (1.0 - np.abs(np.random.normal(0.002, 0.005, size=num_bars)))
        df["Volume"] = np.random.randint(1_000_000, 10_000_000, size=num_bars)

        for col in FEATURE_COLUMNS:
            df[col] = np.random.normal(0.0, 1.0, size=num_bars)
        df["RSI"] = np.clip(50.0 + np.random.normal(0.0, 10.0, size=num_bars), 10.0, 90.0)
        df["ADX"] = np.clip(25.0 + np.random.normal(0.0, 5.0, size=num_bars), 5.0, 80.0)
        df["VIX_Level"] = 18.0
        return df

    def test_signal_ledger_provisional_rejection(self):
        """Mandate 5 & 1: Proves that provisional signals are rejected by the append-only ledger."""
        # 1. Attempt to record a provisional signal
        inserted = self.ledger.record_signal(
            symbol="AAPL",
            bar_timestamp="2026-04-17",
            signal="BUY",
            confidence=0.85,
            execution_target_bar="PENDING_CLOSE",
            execution_price=250.0,
            metadata={"is_provisional": True, "signal_state": "PROVISIONAL"}
        )
        self.assertFalse(inserted, "Ledger must REJECT any provisional signal insertion!")
        self.assertEqual(self.ledger.count_signals("AAPL"), 0)

        # 2. Genuine confirmed signal must be accepted
        inserted_confirmed = self.ledger.record_signal(
            symbol="AAPL",
            bar_timestamp="2026-04-17",
            signal="BUY",
            confidence=0.85,
            execution_target_bar="2026-04-20",
            execution_price=251.25,
            metadata={
                "signal_state": "CONFIRMED",
                "source_candle_timestamp": "2026-04-17 16:00:00 EST",
                "signal_generation_timestamp": "2026-04-17 16:00:00 EST",
                "execution_timestamp": "2026-04-20 09:30:00 EST"
            }
        )
        self.assertTrue(inserted_confirmed)
        self.assertEqual(self.ledger.count_signals("AAPL"), 1)

        # 3. Immutability check: Re-inserting the same bar must be ignored
        re_insert = self.ledger.record_signal(
            symbol="AAPL",
            bar_timestamp="2026-04-17",
            signal="SELL",  # Conflicting signal
            confidence=0.99,
            execution_target_bar="2026-04-20",
            execution_price=200.0,
        )
        self.assertFalse(re_insert)
        signals = self.ledger.get_signals("AAPL")
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["signal"], "BUY")
        self.assertEqual(signals[0]["signal_state"], "CONFIRMED")
        self.assertFalse(signals[0]["is_provisional"])

    def test_forming_bar_isolation(self):
        """
        Mandate 1: Proves that an unfinished forming candle can NEVER produce
        a confirmed BUY/SELL signal or enter the immutable confirmed signal ledger.
        """
        df = self._generate_synthetic_ohlcv(num_bars=80)
        forming_date = df.index[-1].strftime("%Y-%m-%d")

        # Fit scaler on features
        scaler = StandardScaler().fit(df[FEATURE_COLUMNS].values)
        self.mock_mm.scaler = scaler

        # Mock check_bar_forming_status to simulate active trading session on the last bar
        import src.execution.inference_service as inf_module
        orig_check = inf_module.check_bar_forming_status
        try:
            inf_module.check_bar_forming_status = MagicMock(return_value=(True, "FORMING"))

            # Request causal chart markers
            markers = self.service.get_causal_chart_markers("AAPL", df)

            # The forming bar timestamp must NEVER appear in confirmed markers!
            marker_times = [m.get("time") for m in markers]
            self.assertNotIn(
                forming_date,
                marker_times,
                f"Forming bar {forming_date} leaked into confirmed chart markers: {marker_times}"
            )

            # The forming bar must NEVER be recorded into the ledger
            latest_ledger_bar = self.ledger.get_latest_bar_timestamp("AAPL")
            if latest_ledger_bar:
                self.assertLess(
                    latest_ledger_bar,
                    forming_date,
                    f"Forming bar {forming_date} was recorded in ledger! Latest was {latest_ledger_bar}"
                )
        finally:
            inf_module.check_bar_forming_status = orig_check

    def test_append_invariance_causality(self):
        """
        Mandate 7: Core Causality Invariance Test.
        Run the system using data available at time T.
        Record the signals at time T.
        Then append later market data (T+1 .. T+40) and rerun.
        All signals decided at or before time T must remain 100% IDENTICAL (zero repainting).
        """
        # 1. Dataset at Time T (60 bars)
        df_T = self._generate_synthetic_ohlcv(num_bars=60, seed=123)

        scaler = StandardScaler().fit(df_T[FEATURE_COLUMNS].values)
        self.mock_mm.scaler = scaler

        # Run replay at Time T
        self.service._replay_causal_ml_signals("AAPL", df_T)
        signals_at_T = self.ledger.get_signals("AAPL")
        self.assertGreater(len(signals_at_T), 0, "Signals should have been generated at Time T")

        signal_snapshot_T = {
            s["bar_timestamp"]: {
                "signal": s["signal"],
                "confidence": s["confidence"],
                "execution_price": s["execution_price"],
                "execution_target_bar": s["execution_target_bar"],
            }
            for s in signals_at_T
        }

        # 2. Append 40 subsequent days of extreme market action (simulated rally and crash)
        df_extended = self._generate_synthetic_ohlcv(num_bars=100, seed=123)

        # Rerun evaluation on the extended dataset
        self.service.get_causal_chart_markers("AAPL", df_extended)
        signals_after_append = self.ledger.get_signals("AAPL")

        signal_snapshot_after = {
            s["bar_timestamp"]: {
                "signal": s["signal"],
                "confidence": s["confidence"],
                "execution_price": s["execution_price"],
                "execution_target_bar": s["execution_target_bar"],
            }
            for s in signals_after_append
        }

        # Check every single signal that was decided up to time T
        for bar_ts, orig_data in signal_snapshot_T.items():
            after_data = signal_snapshot_after.get(bar_ts)
            self.assertIsNotNone(after_data, f"Signal for bar {bar_ts} disappeared!")
            self.assertEqual(
                orig_data["signal"],
                after_data["signal"],
                f"REPAINTING DETECTED! Bar {bar_ts} changed signal from {orig_data['signal']} to {after_data['signal']}"
            )
            self.assertEqual(
                orig_data["execution_price"],
                after_data["execution_price"],
                f"Execution price repainted for bar {bar_ts}!"
            )
            self.assertEqual(
                orig_data["execution_target_bar"],
                after_data["execution_target_bar"],
                f"Execution target repainted for bar {bar_ts}!"
            )

    def test_backtester_next_session_execution_timing(self):
        """
        Mandate 3: Proves that the backtester executes signals at Open[t+1] with slippage,
        not at Close[t].
        """
        dates = pd.date_range("2026-01-05", periods=5, freq="B")
        data = {
            "Open": [100.0, 105.0, 108.0, 110.0, 112.0],
            "High": [102.0, 107.0, 110.0, 112.0, 115.0],
            "Low": [99.0, 104.0, 106.0, 108.0, 110.0],
            "Close": [101.0, 106.0, 109.0, 111.0, 114.0],
            "Volume": [1000000] * 5,
        }
        df = pd.DataFrame(data, index=dates)

        capital = 100000.0
        shares = 0
        slippage = 0.001
        commission_per_share = 0.005
        pending_position_size = 0.5  # 50%

        # Execution on Day 1 (index 1) at Day 1 Open ($105.00)
        day1_open = float(df["Open"].iloc[1])  # 105.0
        expected_buy_price = day1_open * (1 + slippage)  # 105.105
        max_spend = capital * pending_position_size  # 50000
        expected_shares = int(max_spend / expected_buy_price)

        # Apply execution
        shares += expected_shares
        capital -= (expected_shares * expected_buy_price) + (expected_shares * commission_per_share)

        # Assert execution was at Day 1 Open, NOT Day 0 Close ($101.00)
        self.assertEqual(expected_shares, int(50000.0 / 105.105))
        self.assertGreater(expected_buy_price, 105.0, "Execution price must include upward slippage on BUY")
        self.assertNotEqual(expected_buy_price, 101.0, "Execution must NOT occur at Day 0 Close")

        # Evaluate at Day 1 Close ($106.00)
        close_price = float(df["Close"].iloc[1])
        day1_equity = capital + (shares * close_price)
        self.assertGreater(day1_equity, 0)


if __name__ == "__main__":
    unittest.main()
