# tests/test_strategy_freeze_and_prospective.py
"""
Comprehensive Validation Test Suite for:
1. Provisional-signal rejection from confirmed ledger
2. Forming candle isolation
3. Point-in-time replay vs full-history append invariance
4. Immutability of original prospective signal fields during outcome evaluation
5. Anti-overfitting lock enforcement and detection of mutated parameters/models
6. VIX[t-1] timestamp purity (no 16:00-16:15 ET lookahead)
7. Next-session causal execution timing at Open[t+1] with frozen 5-bps formula
8. Strict segregation between Preliminary Historical Evidence and Untouched Forward Validation datasets
9. Freeze timestamp and prospective sequence mathematical integrity (UTC vs America/New_York)
10. Strict execution price semantics (NULL fills at generation time, 5-bps formula at next session open)
11. Prospective ledger semantic field integrity (signal_reference_price, modeled_fill_price, immutability)
"""
import copy
import json
import os
import shutil
import tempfile
import unittest
import zoneinfo

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
from src.utils.timezone_utils import (
    format_new_york_display,
    parse_to_utc,
)


class TestStrategyFreezeAndProspective(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_ledger.db")
        self.ledger = SignalLedger(self.db_path)

    def tearDown(self):
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
            source_candle_timestamp="2026-09-30 16:00:00 EDT",
            signal_generation_timestamp="2026-09-30 16:00:00 EDT",
            signal="BUY",
            probability=0.85,
            confidence=0.85,
            feature_hash="hash123",
            model_hash="model123",
            execution_target_timestamp="2026-10-01 09:30:00 EDT",
            signal_reference_price=225.0,
            is_provisional=True,
        )
        self.assertIsNone(res_prop, "Provisional signal must be rejected from prospective ledger")
        self.assertEqual(len(self.ledger.get_prospective_signals("AAPL")), 0)

    def test_2_forming_candle_isolation(self):
        """Proof 2: A forming intraday candle is recognized and excluded from confirmed signals."""
        ny_tz = zoneinfo.ZoneInfo("America/New_York")
        now_ny = pd.Timestamp.now(tz=ny_tz)

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

        t80 = dates[80]
        df_t80 = df_full.loc[:t80].copy()
        spy_t80 = spy_full.loc[:t80].copy()
        vix_t80 = vix_full.loc[:t80].copy()

        feat_pit = add_upgraded_features(df_t80, spy_t80, vix_t80, lag_vix=True)
        feat_full = add_upgraded_features(df_full.copy(), spy_full.copy(), vix_full.copy(), lag_vix=True)

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
            source_candle_timestamp="2026-09-30 16:00:00 EDT",
            signal_generation_timestamp="2026-09-30 16:00:00 EDT",
            signal="BUY",
            probability=0.78,
            confidence=0.78,
            feature_hash="orig_feat_hash",
            model_hash="orig_model_hash",
            execution_target_timestamp="2026-10-01 09:30:00 EDT",
            signal_reference_price=225.50,
        )
        self.assertIsNotNone(sig_id)

        # Record snapshot of original fields
        before = self.ledger.get_prospective_signals("AAPL")[0]
        # At generation time, fill-related fields MUST BE NULL
        self.assertIsNone(before["market_open_price"])
        self.assertIsNone(before["modeled_fill_price"])
        self.assertIsNone(before["slippage_assumption_bps"])
        self.assertIsNone(before["commission_assumption"])
        self.assertEqual(before["signal_reference_price"], 225.50)
        self.assertEqual(before["status"], "PENDING_EXECUTION")

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
        self.assertEqual(after["signal_reference_price"], before["signal_reference_price"])

        # OBSERVED MARKET DATA populated:
        self.assertEqual(after["market_open_price"], 226.0)

        # MODELED EXECUTION ASSUMPTIONS populated with 5-bps formula:
        # BUY: Open[t+1] * (1 + 0.0005) = 226.0 * 1.0005 = 226.113
        self.assertEqual(after["modeled_fill_price"], 226.113)
        self.assertEqual(after["slippage_assumption_bps"], 5.0)
        self.assertEqual(after["commission_assumption"], 0.005)
        self.assertIsNotNone(after["return_1d"])
        self.assertEqual(after["status"], "COMPLETED")
        self.assertEqual(after["outcome"], "WIN")

    def test_5_anti_overfitting_lock_detects_mutations(self):
        """Proof 5 & 6: Strategy lock detects any changed parameter or model file."""
        gov = StrategyGovernanceEngine()
        valid, violations = gov.verify_integrity()
        self.assertTrue(valid, f"Initial manifest must be valid. Violations: {violations}")

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
        vix_df = pd.DataFrame({"Close": [15.0, 16.0, 35.0, 17.0, 18.0]}, index=dates)

        res = add_upgraded_features(df.copy(), spy_df, vix_df, lag_vix=True)
        self.assertEqual(res["VIX_Level"].iloc[2], 16.0)
        self.assertEqual(res["VIX_Level"].iloc[3], 35.0)

    def test_7_next_session_causal_execution(self):
        """Proof 8: Confirmed signals execute at next session Open[t+1] with 5-bps execution formula."""
        sig_id = self.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-09-30 16:00:00 EDT",
            signal_generation_timestamp="2026-09-30 16:00:00 EDT",
            signal="BUY",
            probability=0.72,
            confidence=0.72,
            feature_hash="h1",
            model_hash="m1",
            execution_target_timestamp="2026-10-01 09:30:00 EDT",
            signal_reference_price=225.50,
        )
        self.assertIsNotNone(sig_id)

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
        self.assertEqual(rec["market_open_price"], 226.10)
        # BUY: 226.10 * 1.0005 = 226.213
        self.assertEqual(rec["modeled_fill_price"], round(226.10 * 1.0005, 4))
        self.assertEqual(rec["slippage_assumption_bps"], 5.0)
        self.assertEqual(rec["commission_assumption"], 0.005)
        self.assertEqual(rec["execution_target_display"], "2026-10-01 09:30:00 EDT")

    def test_8_historical_and_prospective_datasets_remain_strictly_separate(self):
        """Proof 9: Historical and prospective datasets remain completely separate in distinct tables and tags."""
        self.ledger.record_signal(
            symbol="AAPL",
            bar_timestamp="2024-05-15",
            signal="BUY",
            confidence=0.70,
            execution_target_bar="2024-05-16",
            execution_price=189.50,
            dataset="PRELIMINARY_HISTORICAL_EVIDENCE",
        )

        self.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-09-30 16:00:00 EDT",
            signal_generation_timestamp="2026-09-30 16:00:00 EDT",
            signal="BUY",
            probability=0.70,
            confidence=0.70,
            feature_hash="h2",
            model_hash="m2",
            execution_target_timestamp="2026-10-01 09:30:00 EDT",
            signal_reference_price=225.50,
        )

        hist = self.ledger.get_signals("AAPL")
        prop = self.ledger.get_prospective_signals("AAPL")

        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["dataset"], "PRELIMINARY_HISTORICAL_EVIDENCE")
        self.assertEqual(hist[0]["bar_timestamp"], "2024-05-15")

        self.assertEqual(len(prop), 1)
        self.assertEqual(prop[0]["dataset"], "UNTOUCHED_FORWARD_VALIDATION")
        self.assertEqual(prop[0]["source_candle_timestamp"], "2026-09-30T20:00:00Z")
        self.assertEqual(prop[0]["source_candle_display"], "2026-09-30 16:00:00 EDT")

        summary = self.ledger.get_prospective_summary("AAPL")
        self.assertEqual(summary["total_signals_generated"], 1)
        self.assertEqual(summary["dataset_label"], "UNTOUCHED FORWARD VALIDATION")

    def test_9_freeze_timestamp_and_prospective_sequence_integrity(self):
        """Proof 10: Freeze timestamp and prospective sequence represent identical instants in UTC and Eastern."""
        # 1. Verify Preserved V1.0 Manifest
        from pathlib import Path
        backend_dir = Path(__file__).resolve().parent.parent
        v1_path = backend_dir / "artifacts" / "frozen_strategy_manifest_v1.0.json"
        if v1_path.exists():
            gov_v1 = StrategyGovernanceEngine(manifest_path=str(v1_path))
            status_v1 = gov_v1.get_governance_status()
            self.assertEqual(status_v1["freeze_timestamp_utc"], "2026-09-30T10:39:57Z")
            self.assertEqual(status_v1["freeze_timestamp_new_york"], "2026-09-30T06:39:57-04:00")
            dt_utc_v1 = parse_to_utc(status_v1["freeze_timestamp_utc"])
            dt_ny_v1 = parse_to_utc(status_v1["freeze_timestamp_new_york"])
            self.assertEqual(dt_utc_v1, dt_ny_v1)
            self.assertEqual(format_new_york_display(status_v1["freeze_timestamp_utc"]), "2026-09-30 06:39:57 EDT")

        # 2. Verify Active Governance Engine Manifest
        gov = StrategyGovernanceEngine()
        status = gov.get_governance_status()

        utc_freeze = status["freeze_timestamp_utc"]
        ny_freeze = status["freeze_timestamp_new_york"]

        # Mathematical verification of identical instant
        dt_utc = parse_to_utc(utc_freeze)
        dt_ny = parse_to_utc(ny_freeze)
        self.assertEqual(dt_utc, dt_ny, "UTC freeze and New York freeze must represent the exact same epoch instant")

        # Mathematical verification of prospective sequence
        seq = status["prospective_sequence"]
        candle_utc = seq["first_eligible_completed_candle_utc"]
        candle_ny = seq["first_eligible_completed_candle_new_york"]
        self.assertEqual(parse_to_utc(candle_utc), parse_to_utc(candle_ny))

        exec_utc = seq["first_next_session_execution_utc"]
        exec_ny = seq["first_next_session_execution_new_york"]
        self.assertEqual(parse_to_utc(exec_utc), parse_to_utc(exec_ny))

    def test_10_execution_formula_5bps_strict_precision(self):
        """Proof 11: 5-bps execution formula applies Open[t+1] * (1 ± 0.0005) for BUY and SELL."""
        # Test BUY
        buy_id = self.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-09-30 16:00:00 EDT",
            signal_generation_timestamp="2026-09-30 16:00:00 EDT",
            signal="BUY",
            probability=0.75,
            confidence=0.75,
            feature_hash="fb",
            model_hash="mb",
            execution_target_timestamp="2026-10-01 09:30:00 EDT",
            signal_reference_price=220.00,
        )
        self.assertIsNotNone(buy_id)

        # Test SELL
        sell_id = self.ledger.record_prospective_signal(
            symbol="MSFT",
            source_candle_timestamp="2026-09-30 16:00:00 EDT",
            signal_generation_timestamp="2026-09-30 16:00:00 EDT",
            signal="SELL",
            probability=0.80,
            confidence=0.80,
            feature_hash="fs",
            model_hash="ms",
            execution_target_timestamp="2026-10-01 09:30:00 EDT",
            signal_reference_price=450.00,
        )
        self.assertIsNotNone(sell_id)

        df_exec = pd.DataFrame(
            {
                "Open": [220.0, 200.0],
                "High": [221.0, 202.0],
                "Low": [219.0, 198.0],
                "Close": [220.5, 201.0],
            },
            index=["2026-09-30", "2026-10-01"],
        )
        self.ledger.evaluate_prospective_outcomes("AAPL", df_exec)

        df_msft = pd.DataFrame(
            {
                "Open": [450.0, 400.0],
                "High": [452.0, 402.0],
                "Low": [448.0, 395.0],
                "Close": [451.0, 398.0],
            },
            index=["2026-09-30", "2026-10-01"],
        )
        self.ledger.evaluate_prospective_outcomes("MSFT", df_msft)

        aapl_sig = self.ledger.get_prospective_signals("AAPL")[0]
        msft_sig = self.ledger.get_prospective_signals("MSFT")[0]

        # BUY formula: Open[t+1] * (1 + 0.0005) = 200.0 * 1.0005 = 200.1000
        self.assertEqual(aapl_sig["modeled_fill_price"], 200.1)
        self.assertEqual(aapl_sig["slippage_assumption_bps"], 5.0)
        self.assertEqual(aapl_sig["slippage_amount"], 0.1)
        self.assertEqual(aapl_sig["commission_assumption"], 0.005)

        # SELL formula: Open[t+1] * (1 - 0.0005) = 400.0 * 0.9995 = 399.8000
        self.assertEqual(msft_sig["modeled_fill_price"], 399.8)
        self.assertEqual(msft_sig["slippage_assumption_bps"], 5.0)
        self.assertEqual(msft_sig["slippage_amount"], 0.2)
        self.assertEqual(msft_sig["commission_assumption"], 0.005)

    def test_11_prospective_ledger_semantic_field_integrity(self):
        """
        Proof 12: Comprehensive semantic field integrity of the prospective ledger.

        Verifies:
        1. No future market-open price exists in a signal at generation time
        2. signal_reference_price equals Close[t]
        3. modeled_fill_price is NULL until next-session open
        4. modeled_fill_price uses exactly ±5 bps
        5. commission_assumption is $0.005/share
        6. Original signal fields remain immutable after outcome evaluation
        7. Correct semantic distinction between OBSERVED MARKET DATA and MODELED EXECUTION ASSUMPTIONS
        """
        close_t = 300.00  # Close[t] = $300.00
        open_t1 = 302.50  # Open[t+1] = $302.50

        # --- STEP 1: Record a BUY signal at generation time ---
        sig_id = self.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-09-30 16:00:00 EDT",
            signal_generation_timestamp="2026-09-30 16:00:00 EDT",
            signal="BUY",
            probability=0.82,
            confidence=0.82,
            feature_hash="semantic_test_feat",
            model_hash="semantic_test_model",
            execution_target_timestamp="2026-10-01 09:30:00 EDT",
            signal_reference_price=close_t,
        )
        self.assertIsNotNone(sig_id, "Signal must be successfully recorded")

        gen_snapshot = self.ledger.get_prospective_signals("AAPL")[0]

        # --- VERIFICATION 1: signal_reference_price == Close[t] ---
        self.assertEqual(
            gen_snapshot["signal_reference_price"], close_t,
            f"signal_reference_price must equal Close[t] = {close_t}"
        )

        # --- VERIFICATION 2: All fill-related fields are NULL at generation time ---
        self.assertIsNone(
            gen_snapshot["market_open_price"],
            "market_open_price (OBSERVED) must be NULL at signal generation time"
        )
        self.assertIsNone(
            gen_snapshot["modeled_fill_price"],
            "modeled_fill_price (MODELED) must be NULL at signal generation time"
        )
        self.assertIsNone(
            gen_snapshot["slippage_assumption_bps"],
            "slippage_assumption_bps must be NULL at signal generation time"
        )
        self.assertIsNone(
            gen_snapshot["slippage_amount"],
            "slippage_amount must be NULL at signal generation time"
        )
        self.assertIsNone(
            gen_snapshot["commission_assumption"],
            "commission_assumption must be NULL at signal generation time"
        )

        # --- VERIFICATION 3: Status is PENDING_EXECUTION ---
        self.assertEqual(gen_snapshot["status"], "PENDING_EXECUTION")

        # --- STEP 2: Evaluate outcomes with next-session data ---
        price_df = pd.DataFrame(
            {
                "Open": [299.0, open_t1, 303.0, 304.0, 305.0, 306.0],
                "High": [301.0, 304.0, 305.0, 306.0, 307.0, 308.0],
                "Low": [298.0, 301.0, 302.0, 303.0, 304.0, 305.0],
                "Close": [close_t, 303.0, 304.0, 305.0, 306.0, 307.0],
            },
            index=["2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07"],
        )
        n_evaluated = self.ledger.evaluate_prospective_outcomes("AAPL", price_df)
        self.assertEqual(n_evaluated, 1)

        eval_snapshot = self.ledger.get_prospective_signals("AAPL")[0]

        # --- VERIFICATION 4: OBSERVED MARKET DATA populated correctly ---
        self.assertEqual(
            eval_snapshot["market_open_price"], open_t1,
            f"market_open_price must equal Open[t+1] = {open_t1}"
        )

        # --- VERIFICATION 5: MODELED EXECUTION uses exactly ±5 bps ---
        expected_modeled_fill = round(open_t1 * 1.0005, 4)  # BUY: Open[t+1] * (1 + 0.0005)
        self.assertEqual(
            eval_snapshot["modeled_fill_price"], expected_modeled_fill,
            f"modeled_fill_price must be Open[t+1] * 1.0005 = {expected_modeled_fill}"
        )

        # --- VERIFICATION 6: Slippage is exactly 5.0 bps ---
        self.assertEqual(
            eval_snapshot["slippage_assumption_bps"], 5.0,
            "slippage_assumption_bps must be exactly 5.0"
        )
        expected_slippage_amount = round(abs(expected_modeled_fill - open_t1), 4)
        self.assertEqual(
            eval_snapshot["slippage_amount"], expected_slippage_amount,
            f"slippage_amount must be |modeled_fill - market_open| = {expected_slippage_amount}"
        )

        # --- VERIFICATION 7: Commission assumption is $0.005/share ---
        self.assertEqual(
            eval_snapshot["commission_assumption"], 0.005,
            "commission_assumption must be $0.005/share"
        )

        # --- VERIFICATION 8: Original generation-time fields are immutable ---
        self.assertEqual(eval_snapshot["signal_reference_price"], gen_snapshot["signal_reference_price"],
                         "signal_reference_price must not change after evaluation")
        self.assertEqual(eval_snapshot["signal_id"], gen_snapshot["signal_id"],
                         "signal_id must not change after evaluation")
        self.assertEqual(eval_snapshot["signal"], gen_snapshot["signal"],
                         "signal must not change after evaluation")
        self.assertEqual(eval_snapshot["probability"], gen_snapshot["probability"],
                         "probability must not change after evaluation")
        self.assertEqual(eval_snapshot["confidence"], gen_snapshot["confidence"],
                         "confidence must not change after evaluation")
        self.assertEqual(eval_snapshot["feature_hash"], gen_snapshot["feature_hash"],
                         "feature_hash must not change after evaluation")
        self.assertEqual(eval_snapshot["model_hash"], gen_snapshot["model_hash"],
                         "model_hash must not change after evaluation")
        self.assertEqual(eval_snapshot["source_candle_timestamp"], gen_snapshot["source_candle_timestamp"],
                         "source_candle_timestamp must not change after evaluation")
        self.assertEqual(eval_snapshot["signal_generation_timestamp"], gen_snapshot["signal_generation_timestamp"],
                         "signal_generation_timestamp must not change after evaluation")
        self.assertEqual(eval_snapshot["execution_target_timestamp"], gen_snapshot["execution_target_timestamp"],
                         "execution_target_timestamp must not change after evaluation")

        # --- VERIFICATION 9: Semantic separation clarity ---
        # signal_reference_price is Close[t], NOT Open[t+1]
        self.assertNotEqual(
            eval_snapshot["signal_reference_price"], eval_snapshot["market_open_price"],
            "signal_reference_price (Close[t]) must differ from market_open_price (Open[t+1])"
        )
        # modeled_fill_price is NOT the raw market_open_price
        self.assertNotEqual(
            eval_snapshot["modeled_fill_price"], eval_snapshot["market_open_price"],
            "modeled_fill_price must include slippage and differ from raw market_open_price"
        )

    def test_12_legacy_parameter_backward_compatibility(self):
        """Proof 13: Legacy parameter names (expected_execution_price) still work via aliases."""
        sig_id = self.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-09-30 16:00:00 EDT",
            signal_generation_timestamp="2026-09-30 16:00:00 EDT",
            signal="BUY",
            probability=0.70,
            confidence=0.70,
            feature_hash="legacy_h",
            model_hash="legacy_m",
            execution_target_timestamp="2026-10-01 09:30:00 EDT",
            expected_execution_price=250.00,  # Legacy parameter name
        )
        self.assertIsNotNone(sig_id)

        rec = self.ledger.get_prospective_signals("AAPL")[0]
        # The legacy parameter should map to signal_reference_price
        self.assertEqual(rec["signal_reference_price"], 250.00)
        # Legacy aliases should also be exposed
        self.assertEqual(rec["expected_execution_price"], 250.00)
        self.assertEqual(rec["execution_price"], 250.00)


if __name__ == "__main__":
    unittest.main()
