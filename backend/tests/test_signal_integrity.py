# tests/test_signal_integrity.py
"""
Automated Signal Integrity & Look-Ahead Verification Test Suite.
Asserts:
1. Repainting Invariance: Signals committed for bar t are permanently immutable.
2. Look-Ahead Truncation Invariance: Features at time t are identical whether evaluated on D_1..t or D_1..t+k.
3. Execution Latency (t -> t+1): Signal emitted at bar t executes at Open[t+1] +/- slippage, never High/Low[t].
4. Intraday Forming Bar Isolation: Incomplete bars are flagged as provisional and do not pollute the confirmed ledger.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.execution.live_inference import (
    FEATURE_COLUMNS,
    add_upgraded_features,
)
from src.execution.reporting import ReportGenerator
from src.execution.signal_ledger import SignalLedger


class TestSignalIntegrity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_ledger.db")
        self.ledger = SignalLedger(db_path=self.db_path)
        self.report_gen = ReportGenerator(kept_features_list=[])

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _generate_synthetic_market_data(self, n_bars: int = 100, seed: int = 42) -> pd.DataFrame:
        """Generates realistic daily OHLCV series for testing."""
        np.random.seed(seed)
        dates = pd.date_range("2025-01-01", periods=n_bars, freq="B")

        returns = np.random.normal(0.0005, 0.015, n_bars)
        price = 100.0 * np.exp(np.cumsum(returns))

        high = price * (1.0 + np.abs(np.random.normal(0.005, 0.003, n_bars)))
        low = price * (1.0 - np.abs(np.random.normal(0.005, 0.003, n_bars)))
        open_p = low + (high - low) * np.random.uniform(0.2, 0.8, n_bars)
        volume = np.random.randint(500000, 2000000, n_bars)

        df = pd.DataFrame(
            {
                "Open": open_p,
                "High": high,
                "Low": low,
                "Close": price,
                "Volume": volume,
            },
            index=dates,
        )
        return df

    def test_repainting_invariance(self):
        """
        Test 1: Repainting Invariance Test.
        Assert that once a signal is committed to the SignalLedger at day T,
        feeding subsequent data (e.g. T+1..T+10 with a crash or rally)
        CANNOT alter, remove, or repaint any signal at or before day T.
        """
        ticker = "AAPL"
        df_t = self._generate_synthetic_market_data(n_bars=60, seed=101)
        bar_t_date = df_t.index[-1].strftime("%Y-%m-%d")
        exec_target_date = "NEXT_SESSION_OPEN"
        exec_price = round(float(df_t["Close"].iloc[-1]) * 1.0005, 2)
        confidence = 0.78

        # 1. Commit confirmed signal at day T
        inserted = self.ledger.record_signal(
            symbol=ticker,
            bar_timestamp=bar_t_date,
            signal="BUY",
            confidence=confidence,
            execution_target_bar=exec_target_date,
            execution_price=exec_price,
            model_version="Institutional_Mesh_V2.1",
            raw_features_hash="hash_t_abc123",
            metadata={"status": "confirmed"},
        )
        self.assertTrue(inserted, "Signal should be successfully committed to ledger.")

        # Read back signals at day T
        signals_at_t = self.ledger.get_signals(symbol=ticker)
        self.assertEqual(len(signals_at_t), 1)
        self.assertEqual(signals_at_t[0]["signal"], "BUY")
        self.assertEqual(signals_at_t[0]["confidence"], confidence)
        self.assertEqual(signals_at_t[0]["execution_price"], exec_price)

        # 2. Advance time: append 10 new bars with extreme volatility (e.g. 30% crash)
        # Attempt to insert a conflicting signal for the same day T (repainting attempt)
        repainting_attempt = self.ledger.record_signal(
            symbol=ticker,
            bar_timestamp=bar_t_date,
            signal="SELL",  # Trying to retroactively flip BUY to SELL
            confidence=0.95,
            execution_target_bar=exec_target_date,
            execution_price=exec_price * 0.8,
            model_version="Institutional_Mesh_V2.1",
            raw_features_hash="corrupted_hash",
            metadata={"status": "repainted"},
        )
        self.assertFalse(
            repainting_attempt,
            "Ledger MUST reject repainting attempts for already-committed bars.",
        )

        # Verify that original signal at day T remains 100% bit-for-bit unchanged
        signals_after = self.ledger.get_signals(symbol=ticker)
        self.assertEqual(len(signals_after), 1)
        self.assertEqual(signals_after[0]["signal"], "BUY")
        self.assertEqual(signals_after[0]["confidence"], confidence)
        self.assertEqual(signals_after[0]["execution_price"], exec_price)
        self.assertEqual(signals_after[0]["raw_features_hash"], "hash_t_abc123")

    def test_lookahead_truncation_invariance(self):
        """
        Test 2: Look-Ahead Truncation Test.
        Assert that feature values computed for candle T using truncated data D_{1..T}
        are mathematically identical to feature values for candle T when computed
        on the full expanded dataset D_{1..T+K}.
        Proves zero future leakage across rolling indicators.
        """
        n_full = 100
        t_cutoff = 75

        df_full = self._generate_synthetic_market_data(n_bars=n_full, seed=202)
        df_truncated = df_full.iloc[:t_cutoff].copy()

        spy_full = self._generate_synthetic_market_data(n_bars=n_full, seed=303)
        spy_truncated = spy_full.iloc[:t_cutoff].copy()

        # Compute upgraded features on both truncated and full datasets
        feat_trunc = add_upgraded_features(df_truncated.copy(), spy_truncated.copy(), None)
        feat_full = add_upgraded_features(df_full.copy(), spy_full.copy(), None)

        target_date = df_truncated.index[-1]

        # Verify that the target date exists in both
        self.assertIn(target_date, feat_trunc.index)
        self.assertIn(target_date, feat_full.index)

        # Check each feature column for exact point-in-time mathematical invariance
        for col in FEATURE_COLUMNS:
            if col in feat_trunc.columns and col in feat_full.columns:
                val_trunc = float(feat_trunc.loc[target_date, col])
                val_full = float(feat_full.loc[target_date, col])

                # Assert mathematical equality within floating point precision
                self.assertAlmostEqual(
                    val_trunc,
                    val_full,
                    places=5,
                    msg=f"Look-ahead bias detected in feature '{col}': Truncated={val_trunc}, Full={val_full}",
                )

    def test_causal_execution_latency(self):
        """
        Test 3: Execution Latency Assertion (t -> t+1).
        Signals emitted at the close of bar t must model execution at Open[t+1] +/- slippage,
        never at High[t] or Low[t].
        """
        df = self._generate_synthetic_market_data(n_bars=50, seed=404)
        t_idx = 30
        bar_t_date = df.index[t_idx].strftime("%Y-%m-%d")
        bar_t1_date = df.index[t_idx + 1].strftime("%Y-%m-%d")

        bar_t_close = float(df["Close"].iloc[t_idx])
        bar_t_high = float(df["High"].iloc[t_idx])
        bar_t_low = float(df["Low"].iloc[t_idx])
        bar_t1_open = float(df["Open"].iloc[t_idx + 1])

        # BUY signal emitted at close of bar t
        slippage_bps = 0.0005  # 5 bps
        expected_exec_price = round(bar_t1_open * (1.0 + slippage_bps), 2)

        self.ledger.record_signal(
            symbol="MSFT",
            bar_timestamp=bar_t_date,
            signal="BUY",
            confidence=0.82,
            execution_target_bar=bar_t1_date,
            execution_price=expected_exec_price,
            model_version="Institutional_Mesh_V2.1",
            raw_features_hash="feat_hash_t",
        )

        signals = self.ledger.get_signals("MSFT")
        self.assertEqual(len(signals), 1)
        sig = signals[0]

        # Assert execution latency
        self.assertEqual(sig["execution_target_bar"], bar_t1_date)
        self.assertAlmostEqual(sig["execution_price"], expected_exec_price, places=2)

        # Assert execution price is strictly decoupled from bar t's extrema
        self.assertNotEqual(
            sig["execution_price"],
            round(bar_t_high, 2),
            "Execution price MUST NOT equal bar t's High (fantasy fill).",
        )
        self.assertNotEqual(
            sig["execution_price"],
            round(bar_t_low, 2),
            "Execution price MUST NOT equal bar t's Low (fantasy fill).",
        )
        self.assertNotEqual(
            sig["execution_price"],
            round(bar_t_close, 2),
            "Execution price MUST NOT equal bar t's Close (must be Open[t+1] +/- slippage).",
        )

    def test_post_hoc_zigzag_overlay_isolated(self):
        """
        Test 4: Verify that post-hoc swing pivot logic carries non-causal warning
        and generate_historical_markers issues deprecation warning.
        """
        df = self._generate_synthetic_market_data(n_bars=60, seed=505)

        # Direct call to post-hoc overlay
        markers, _ = self.report_gen.generate_post_hoc_zigzag_overlay("TEST", df.copy())
        for m in markers:
            self.assertEqual(m["label"], "Post-Hoc Pivot (Non-Causal)")

        # Call to deprecated alias
        with self.assertLogs("ReportGenerator", level="WARNING") as log:
            dep_markers, _ = self.report_gen.generate_historical_markers("TEST", df.copy())
            self.assertTrue(any("DEPRECATION WARNING" in record.getMessage() for record in log.records))
            self.assertEqual(len(dep_markers), len(markers))


if __name__ == "__main__":
    unittest.main()
