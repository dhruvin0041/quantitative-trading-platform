# tests/test_strategy_freeze_and_prospective.py
"""
Comprehensive Validation Test Suite for:
1. Provisional-signal rejection from confirmed ledger
2. Forming candle isolation
3. Point-in-time replay vs full-history append invariance
4. Immutability of original prospective signal fields during outcome evaluation
5. Anti-overfitting lock enforcement and detection of mutated parameters/models
6. VIX[t-1] timestamp purity (no 16:00-16:15 ET lookahead)
7. Next-session causal execution timing at Open[t+1]
8. Strict segregation between Preliminary Historical Evidence and Untouched Forward Validation datasets
"""
import copy
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.execution.live_inference import (
    FEATURE_COLUMNS,
    add_upgraded_features,
    check_bar_forming_status,
)
from src.execution.signal_ledger import SignalLedger
from src.execution.strategy_governance import (
    StrategyGovernanceEngine,
    StrategyLockError,
)


class TestStrategyFreezeAndProspective(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_ledger.db")
        self.ledger = SignalLedger(self.db_path)

    def tearDown(self):
        # Allow sqlite connection cleanup
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_1_provisional_signal_cannot_enter_confirmed_ledger(self):
        """Proof 1: A provisional signal cannot enter the confirmed ledger or prospective ledger."""
        # 1a. Test historical ledger table
        meta = {"is_provisional": True, "signal_state": "PROVISIONAL"}
        res = self.ledger.record_signal(
            symbol="AAPL",
            bar_timestamp="2026-09-30",
            signal="BUY",
            confidence=0.85,
            execution_target_bar="NEXT_SESSION_OPEN",
            execution_price=225.0,
            metadata=meta,
        )
        self.assertFalse(res, "Provisional signal must be rejected from historical ledger")
        self.assertEqual(self.ledger.count_signals("AAPL"), 0)

        # 1b. Test prospective ledger table
        res_prop = self.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-09-30 16:00:00 EST",
            signal_generation_timestamp="2026-09-30 16:00:00 EST",
            signal="BUY",
            probability=0.85,
            confidence=0.85,
            feature_hash="hash123",
            model_hash="model123",
            execution_target_timestamp="2026-10-01 09:30:00 EST",
            execution_price=225.0,
            is_provisional=True,
        )
        self.assertIsNone(res_prop, "Provisional signal must be rejected from prospective ledger")
        self.assertEqual(len(self.ledger.get_prospective_signals("AAPL")), 0)

    def test_2_forming_candle_isolation(self):
        """Proof 2: A forming intraday candle is recognized and excluded from confirmed signals."""
        import zoneinfo
        ny_tz = zoneinfo.ZoneInfo("America/New_York")
        now_ny = pd.Timestamp.now(tz=ny_tz)

        # Simulate a dataframe whose last row is today
        dates = pd.date_range(end=now_ny.floor("D"), periods=10, freq="D")
        df = pd.DataFrame(
            {
                "Open": np.linspace(100, 110, 10),
                "High": np.linspace(105, 115, 10),
                "Low": np.linspace(95, 105, 10),
                "Close": np.linspace(102, 112, 10),
                "Volume": np.ones(10) * 10000,
            },
            index=dates,
        )

        is_forming, status = check_bar_forming_status(df)
        if now_ny.hour < 16:
            self.assertTrue(is_forming)
            self.assertEqual(status, "FORMING")
        else:
            self.assertFalse(is_forming)
            self.assertEqual(status, "CONFIRMED")

    def test_3_append_invariance_and_zero_repainting(self):
        """Proof 3: Appending future candles cannot alter past signals (point-in-time vs full history)."""
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        df_full = pd.DataFrame(
            {
                "Open": 100 + np.sin(np.linspace(0, 20, 100)) * 10,
                "High": 105 + np.sin(np.linspace(0, 20, 100)) * 10,
                "Low": 95 + np.sin(np.linspace(0, 20, 100)) * 10,
                "Close": 102 + np.sin(np.linspace(0, 20, 100)) * 10,
                "Volume": 10000 + np.random.RandomState(42).randint(0, 5000, 100),
            },
            index=dates,
        )
        spy_full = pd.DataFrame(
            {"Close": 450 + np.linspace(0, 50, 100)},
            index=dates,
        )
        vix_full = pd.DataFrame(
            {"Close": 18 + np.sin(np.linspace(0, 10, 100)) * 5},
            index=dates,
        )

        # Slice at T=80
        t80 = dates[80]
        df_t80 = df_full.loc[:t80].copy()
        spy_t80 = spy_full.loc[:t80].copy()
        vix_t80 = vix_full.loc[:t80].copy()

        # Compute features at T=80 point-in-time
        feat_pit = add_upgraded_features(df_t80, spy_t80, vix_t80, lag_vix=True)
        # Compute features on full history up to T=100
        feat_full = add_upgraded_features(df_full.copy(), spy_full.copy(), vix_full.copy(), lag_vix=True)

        # Verify that feature vector at bar T=80 is 100% IDENTICAL
        pit_row = feat_pit.loc[t80, FEATURE_COLUMNS].values
        full_row = feat_full.loc[t80, FEATURE_COLUMNS].values

        np.testing.assert_allclose(
            pit_row,
            full_row,
            rtol=1e-7,
            atol=1e-7,
            err_msg="Feature repainting detected! Appending future bars altered historical features.",
        )

    def test_4_future_outcome_calculation_cannot_alter_original_signal(self):
        """Proof 4: Future outcome calculation appends returns but NEVER modifies original signal fields."""
        sig_id = self.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-09-30 16:00:00 EST",
            signal_generation_timestamp="2026-09-30 16:00:00 EST",
            signal="BUY",
            probability=0.78,
            confidence=0.78,
            feature_hash="orig_feat_hash",
            model_hash="orig_model_hash",
            execution_target_timestamp="2026-10-01 09:30:00 EST",
            execution_price=225.50,
        )
        self.assertIsNotNone(sig_id)

        # Record snapshot of original fields
        before = self.ledger.get_prospective_signals("AAPL")[0]

        # Simulate subsequent price bars
        dates = ["2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07"]
        price_df = pd.DataFrame(
            {
                "Open": [224.0, 226.0, 227.0, 228.0, 229.0, 230.0],
                "High": [225.0, 228.0, 229.0, 230.0, 231.0, 232.0],
                "Low": [223.0, 225.0, 226.0, 227.0, 228.0, 229.0],
                "Close": [225.0, 227.0, 228.0, 229.0, 230.0, 231.0],
            },
            index=dates,
        )

        # Run outcome evaluation
        evaluated = self.ledger.evaluate_prospective_outcomes("AAPL", price_df)
        self.assertEqual(evaluated, 1)

        after = self.ledger.get_prospective_signals("AAPL")[0]

        # Invariant checks: original generation fields must be strictly identical
        self.assertEqual(after["signal_id"], before["signal_id"])
        self.assertEqual(after["symbol"], before["symbol"])
        self.assertEqual(after["source_candle_timestamp"], before["source_candle_timestamp"])
        self.assertEqual(after["signal_generation_timestamp"], before["signal_generation_timestamp"])
        self.assertEqual(after["signal"], before["signal"])
        self.assertEqual(after["probability"], before["probability"])
        self.assertEqual(after["confidence"], before["confidence"])
        self.assertEqual(after["feature_hash"], before["feature_hash"])
        self.assertEqual(after["model_hash"], before["model_hash"])
        self.assertEqual(after["execution_target_timestamp"], before["execution_target_timestamp"])
        self.assertEqual(after["execution_price"], before["execution_price"])

        # Outcome fields successfully populated
        self.assertEqual(after["actual_market_open"], 226.0)
        self.assertIsNotNone(after["return_1d"])
        self.assertEqual(after["status"], "COMPLETED")
        self.assertEqual(after["outcome"], "WIN")

    def test_5_anti_overfitting_lock_detects_mutations(self):
        """Proof 5 & 6: Strategy lock detects any changed parameter or model file."""
        gov = StrategyGovernanceEngine()
        valid, violations = gov.verify_integrity()
        self.assertTrue(valid, f"Initial manifest must be valid. Violations: {violations}")

        # Simulate tampering with a manifest or config
        temp_manifest = copy.deepcopy(gov.load_manifest())
        temp_manifest["model_hashes"]["xgb_ensemble.json"] = "TAMPERED_HASH_9999"
        tampered_path = os.path.join(self.temp_dir, "tampered_manifest.json")
        with open(tampered_path, "w") as f:
            json.dump(temp_manifest, f)

        tampered_gov = StrategyGovernanceEngine(tampered_path)
        t_valid, t_violations = tampered_gov.verify_integrity()
        self.assertFalse(t_valid, "Tampered manifest must fail verification")
        self.assertIn("xgb_ensemble.json mutated", t_violations[0])

        with self.assertRaises(StrategyLockError):
            tampered_gov.enforce_anti_overfitting_lock()

    def test_6_vix_timestamp_purity(self):
        """Proof 7: VIX[t-1] prevents 16:00-16:15 ET lookahead information from leaking into day t."""
        dates = pd.date_range("2026-09-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [102.0, 103.0, 104.0, 105.0, 106.0],
                "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Volume": [1000, 1000, 1000, 1000, 1000],
            },
            index=dates,
        )
        spy_df = pd.DataFrame({"Close": [500.0, 501.0, 502.0, 503.0, 504.0]}, index=dates)
        # VIX on day 2 spikes to 35.0 post-close (16:15 ET)
        vix_df = pd.DataFrame({"Close": [15.0, 16.0, 35.0, 17.0, 18.0]}, index=dates)

        res = add_upgraded_features(df.copy(), spy_df, vix_df, lag_vix=True)
        # At day index 2 (2026-09-03), VIX_Level must be day index 1 value (16.0), NOT 35.0!
        self.assertEqual(res["VIX_Level"].iloc[2], 16.0)
        # The 35.0 spike is only visible on day index 3 (2026-09-04)
        self.assertEqual(res["VIX_Level"].iloc[3], 35.0)

    def test_7_next_session_causal_execution(self):
        """Proof 8: Confirmed signals execute at next session Open[t+1] with slippage."""
        sig_id = self.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-09-30 16:00:00 EST",
            signal_generation_timestamp="2026-09-30 16:00:00 EST",
            signal="BUY",
            probability=0.72,
            confidence=0.72,
            feature_hash="h1",
            model_hash="m1",
            execution_target_timestamp="2026-10-01 09:30:00 EST",
            execution_price=225.50,
        )
        self.assertIsNotNone(sig_id)

        # When evaluated with next day's open
        df_next = pd.DataFrame(
            {
                "Open": [224.0, 226.10],
                "High": [225.0, 227.0],
                "Low": [223.0, 225.5],
                "Close": [225.0, 226.8],
            },
            index=["2026-09-30", "2026-10-01"],
        )
        self.ledger.evaluate_prospective_outcomes("AAPL", df_next)

        rec = self.ledger.get_prospective_signals("AAPL")[0]
        self.assertEqual(rec["actual_market_open"], 226.10)
        self.assertEqual(rec["execution_target_timestamp"], "2026-10-01 09:30:00 EST")

    def test_8_historical_and_prospective_datasets_remain_strictly_separate(self):
        """Proof 9: Historical and prospective datasets remain completely separate in distinct tables and tags."""
        # Insert historical signal
        self.ledger.record_signal(
            symbol="AAPL",
            bar_timestamp="2024-05-15",
            signal="BUY",
            confidence=0.70,
            execution_target_bar="2024-05-16",
            execution_price=189.50,
            dataset="PRELIMINARY_HISTORICAL_EVIDENCE",
        )

        # Insert prospective signal
        self.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-09-30 16:00:00 EST",
            signal_generation_timestamp="2026-09-30 16:00:00 EST",
            signal="BUY",
            probability=0.70,
            confidence=0.70,
            feature_hash="h2",
            model_hash="m2",
            execution_target_timestamp="2026-10-01 09:30:00 EST",
            execution_price=225.50,
        )

        hist = self.ledger.get_signals("AAPL")
        prop = self.ledger.get_prospective_signals("AAPL")

        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["dataset"], "PRELIMINARY_HISTORICAL_EVIDENCE")
        self.assertEqual(hist[0]["bar_timestamp"], "2024-05-15")

        self.assertEqual(len(prop), 1)
        self.assertEqual(prop[0]["dataset"], "UNTOUCHED_FORWARD_VALIDATION")
        self.assertEqual(prop[0]["source_candle_timestamp"], "2026-09-30 16:00:00 EST")

        summary = self.ledger.get_prospective_summary("AAPL")
        self.assertEqual(summary["total_signals_generated"], 1)
        self.assertEqual(summary["dataset_label"], "UNTOUCHED FORWARD VALIDATION")


if __name__ == "__main__":
    unittest.main()
