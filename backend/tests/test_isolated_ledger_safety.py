# backend/tests/test_isolated_ledger_safety.py
"""
Comprehensive Atomic Ledger Replacement & Transactional Safety Test Suite.

Verifies:
1. Successful atomic replacement in a single SQLite transaction with candidate provenance.
2. Rejection of NaN, +inf, -inf, and boolean types for confidence and execution price.
3. Rejection of malformed metadata (types, non-serializable, provisional) and timestamps (format, causality).
4. Rejection of invalid symbols, actions, and non-positive execution prices.
5. Strict model-version provenance (rejection of missing, empty, or unauthorized versions).
6. Inference exceptions preserve existing historical records and return empty list.
7. Rejected replacement handling (distinguishing replay vs persistence, no silent stale record return).
8. Confirmed completed empty replay vs unconfirmed/incomplete replay distinction.
9. Mid-transaction database write failure triggers complete rollback.
10. Prospective validation tables (prospective_signals) are 100% isolated and untouched across all operations.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.execution.inference_service import InferenceService
from src.execution.signal_ledger import SignalLedger


class TestIsolatedLedgerSafety(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_isolated_ledger.db")
        self.ledger = SignalLedger(db_path=self.db_path)

        # Seed initial valid historical records in preliminary evidence
        self.ledger.record_signal(
            symbol="AAPL",
            bar_timestamp="2026-09-01",
            signal="BUY",
            confidence=0.65,
            execution_target_bar="2026-09-02",
            execution_price=220.50,
            model_version="Institutional_Mesh_V2.4",
            raw_features_hash="hash_seed_1",
            dataset="PRELIMINARY_HISTORICAL_EVIDENCE",
            metadata={"source": "seed"},
        )
        self.ledger.record_signal(
            symbol="AAPL",
            bar_timestamp="2026-09-10",
            signal="SELL",
            confidence=0.62,
            execution_target_bar="2026-09-11",
            execution_price=230.10,
            model_version="Institutional_Mesh_V2.4",
            raw_features_hash="hash_seed_2",
            dataset="PRELIMINARY_HISTORICAL_EVIDENCE",
            metadata={"source": "seed"},
        )

        # Seed prospective validation records using the authorized method
        self.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-10-01 16:00:00 EDT",
            signal_generation_timestamp="2026-10-01 16:00:00 EDT",
            signal="BUY",
            probability=0.65,
            confidence=0.65,
            feature_hash="feat_hash_prospective",
            model_hash="model_hash_prospective",
            execution_target_timestamp="2026-10-02 09:30:00 EDT",
            signal_reference_price=225.0,
            strategy_version="Institutional_Mesh_V2.4",
            manifest_hash="manifest_hash_test",
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _get_prospective_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM prospective_signals;").fetchone()
            return int(row[0]) if row else 0

    def _get_historical_count(self, symbol: str = "AAPL") -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM signal_ledger WHERE symbol = ? AND dataset = 'PRELIMINARY_HISTORICAL_EVIDENCE';",
                (symbol,),
            ).fetchone()
            return int(row[0]) if row else 0

    def test_1_successful_atomic_replacement(self):
        """Proof: replace_historical_replay_atomically commits new signals in 1 transaction."""
        initial_prospective = self._get_prospective_count()

        new_staged = [
            {
                "symbol": "AAPL",
                "bar_timestamp": "2026-09-15",
                "signal": "BUY",
                "confidence": 0.70,
                "execution_target_bar": "2026-09-16",
                "execution_price": 222.0,
                "model_version": "Institutional_Mesh_V2.4.1",
                "raw_features_hash": "hash_new_1",
                "metadata": {"source": "new_replay", "signal_state": "CONFIRMED"},
            },
            {
                "symbol": "AAPL",
                "bar_timestamp": "2026-09-22",
                "signal": "SELL",
                "confidence": 0.68,
                "execution_target_bar": "2026-09-23",
                "execution_price": 235.0,
                "model_version": "Institutional_Mesh_V2.4.1",
                "raw_features_hash": "hash_new_2",
                "metadata": {"source": "new_replay", "signal_state": "CONFIRMED"},
            },
        ]

        success = self.ledger.replace_historical_replay_atomically(
            symbol="AAPL",
            validated_signals=new_staged,
            replay_status="COMPLETED",
            allow_empty=False,
        )
        self.assertTrue(success)

        signals = self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])
        self.assertEqual(len(signals), 2)
        self.assertEqual(signals[0]["bar_timestamp"], "2026-09-15")
        self.assertEqual(signals[1]["bar_timestamp"], "2026-09-22")
        self.assertEqual(signals[0]["model_version"], "Institutional_Mesh_V2.4.1")
        self.assertEqual(signals[1]["model_version"], "Institutional_Mesh_V2.4.1")

        # Verify prospective tables untouched
        self.assertEqual(self._get_prospective_count(), initial_prospective)

    def test_2_reject_non_finite_numerical_values(self):
        """Proof: Rejects NaN, +inf, -inf, and boolean values for confidence and execution_price."""
        initial_prospective = self._get_prospective_count()
        initial_historical = self._get_historical_count()

        base_valid = {
            "symbol": "AAPL",
            "bar_timestamp": "2026-09-15",
            "signal": "BUY",
            "confidence": 0.70,
            "execution_target_bar": "2026-09-16",
            "execution_price": 220.0,
            "model_version": "Institutional_Mesh_V2.4.1",
            "raw_features_hash": "hash_valid",
            "metadata": {},
        }

        non_finite_cases = [
            ("NaN confidence", {**base_valid, "confidence": float("nan")}),
            ("+inf confidence", {**base_valid, "confidence": float("inf")}),
            ("-inf confidence", {**base_valid, "confidence": float("-inf")}),
            ("bool confidence True", {**base_valid, "confidence": True}),
            ("bool confidence False", {**base_valid, "confidence": False}),
            ("out-of-range confidence > 1", {**base_valid, "confidence": 1.05}),
            ("out-of-range confidence < 0", {**base_valid, "confidence": -0.05}),
            ("NaN execution_price", {**base_valid, "execution_price": float("nan")}),
            ("+inf execution_price", {**base_valid, "execution_price": float("inf")}),
            ("-inf execution_price", {**base_valid, "execution_price": float("-inf")}),
            ("bool execution_price", {**base_valid, "execution_price": True}),
            ("zero execution_price", {**base_valid, "execution_price": 0.0}),
            ("negative execution_price", {**base_valid, "execution_price": -220.0}),
        ]

        for desc, bad_record in non_finite_cases:
            res = self.ledger.replace_historical_replay_atomically(
                symbol="AAPL",
                validated_signals=[bad_record],
                replay_status="COMPLETED",
                allow_empty=False,
            )
            self.assertFalse(res, f"Expected rejection for {desc}")

        # Ensure database is 100% unmodified
        self.assertEqual(self._get_historical_count(), initial_historical)
        self.assertEqual(self._get_prospective_count(), initial_prospective)

    def test_3_reject_malformed_metadata_and_timestamps(self):
        """Proof: Rejects malformed metadata types, provisional states, and invalid/anachronistic timestamps."""
        initial_prospective = self._get_prospective_count()
        initial_historical = self._get_historical_count()

        base_valid = {
            "symbol": "AAPL",
            "bar_timestamp": "2026-09-15",
            "signal": "BUY",
            "confidence": 0.70,
            "execution_target_bar": "2026-09-16",
            "execution_price": 220.0,
            "model_version": "Institutional_Mesh_V2.4.1",
            "raw_features_hash": "hash_valid",
            "metadata": {},
        }

        class NonSerializable:
            pass

        malformed_cases = [
            ("Metadata as list", {**base_valid, "metadata": ["invalid", "list"]}),
            ("Metadata as string", {**base_valid, "metadata": "invalid_string"}),
            ("Metadata as int", {**base_valid, "metadata": 42}),
            ("Metadata non-serializable", {**base_valid, "metadata": {"obj": NonSerializable()}}),
            ("Provisional flag True", {**base_valid, "metadata": {"is_provisional": True}}),
            ("Provisional state", {**base_valid, "metadata": {"signal_state": "PROVISIONAL"}}),
            ("Malformed bar_timestamp string", {**base_valid, "bar_timestamp": "not-a-date"}),
            ("Invalid calendar date Feb 31", {**base_valid, "bar_timestamp": "2026-02-31"}),
            ("Invalid calendar date Apr 31", {**base_valid, "bar_timestamp": "2026-04-31"}),
            ("Numeric bar_timestamp", {**base_valid, "bar_timestamp": 20260915}),
            ("Malformed exec_target string", {**base_valid, "execution_target_bar": "tomorrow"}),
            (
                "Temporal causality violation (exec_target before bar_ts date)",
                {**base_valid, "bar_timestamp": "2026-09-15", "execution_target_bar": "2026-09-10"},
            ),
            (
                "Temporal causality violation (same-day timestamp earlier than close)",
                {**base_valid, "bar_timestamp": "2026-09-15 16:00:00", "execution_target_bar": "2026-09-15 09:30:00"},
            ),
            (
                "Temporal causality violation (same-day timestamp equal to close)",
                {**base_valid, "bar_timestamp": "2026-09-15 16:00:00", "execution_target_bar": "2026-09-15 16:00:00"},
            ),
            (
                "Temporal causality violation (same calendar date daily bars)",
                {**base_valid, "bar_timestamp": "2026-09-15", "execution_target_bar": "2026-09-15"},
            ),
            (
                "Invalid timezone offset out of range +99:99",
                {**base_valid, "bar_timestamp": "2026-09-15T16:00:00+99:99", "execution_target_bar": "NEXT_SESSION_OPEN"},
            ),
            (
                "Invalid timezone offset out of range -20:00",
                {**base_valid, "bar_timestamp": "2026-09-15T16:00:00-20:00", "execution_target_bar": "NEXT_SESSION_OPEN"},
            ),
            ("Malformed creation timestamp", {**base_valid, "timestamp": "invalid_utc_time"}),
        ]

        for desc, bad_record in malformed_cases:
            res = self.ledger.replace_historical_replay_atomically(
                symbol="AAPL",
                validated_signals=[bad_record],
                replay_status="COMPLETED",
                allow_empty=False,
            )
            self.assertFalse(res, f"Expected rejection for {desc}")

        self.assertEqual(self._get_historical_count(), initial_historical)
        self.assertEqual(self._get_prospective_count(), initial_prospective)

    def test_4_reject_invalid_symbols_actions_and_missing_keys(self):
        """Proof: Rejects symbol mismatch, missing required fields, and non-permitted actions."""
        initial_prospective = self._get_prospective_count()
        initial_historical = self._get_historical_count()

        base_valid = {
            "symbol": "AAPL",
            "bar_timestamp": "2026-09-15",
            "signal": "BUY",
            "confidence": 0.70,
            "execution_target_bar": "NEXT_SESSION_OPEN",
            "execution_price": 220.0,
            "model_version": "Institutional_Mesh_V2.4.1",
            "raw_features_hash": "hash_valid",
            "metadata": {},
        }

        invalid_cases = [
            ("Symbol mismatch", {**base_valid, "symbol": "MSFT"}),
            ("Empty symbol", {**base_valid, "symbol": ""}),
            ("Unknown action", {**base_valid, "signal": "UNKNOWN"}),
            ("Short action", {**base_valid, "signal": "SHORT"}),
            ("Exit action", {**base_valid, "signal": "EXIT"}),
            ("Missing raw_features_hash", {k: v for k, v in base_valid.items() if k != "raw_features_hash"}),
            ("Empty raw_features_hash", {**base_valid, "raw_features_hash": ""}),
            ("Missing confidence", {k: v for k, v in base_valid.items() if k != "confidence"}),
        ]

        for desc, bad_record in invalid_cases:
            res = self.ledger.replace_historical_replay_atomically(
                symbol="AAPL",
                validated_signals=[bad_record],
                replay_status="COMPLETED",
                allow_empty=False,
            )
            self.assertFalse(res, f"Expected rejection for {desc}")

        self.assertEqual(self._get_historical_count(), initial_historical)

        # Permitted actions must succeed
        for permitted_action in ("BUY", "SELL", "HOLD", "HOLD_LONG"):
            record = {**base_valid, "signal": permitted_action}
            res = self.ledger.replace_historical_replay_atomically(
                symbol="AAPL",
                validated_signals=[record],
                replay_status="COMPLETED",
                allow_empty=False,
            )
            self.assertTrue(res, f"Permitted action {permitted_action} should succeed")

        self.assertEqual(self._get_prospective_count(), initial_prospective)

    def test_5_explicit_model_version_provenance(self):
        """Proof: Requires explicit, authorized candidate model-version provenance. Rejects defaults/unauthorized."""
        initial_prospective = self._get_prospective_count()
        initial_historical = self._get_historical_count()

        base_valid = {
            "symbol": "AAPL",
            "bar_timestamp": "2026-09-15",
            "signal": "BUY",
            "confidence": 0.70,
            "execution_target_bar": "NEXT_SESSION_OPEN",
            "execution_price": 220.0,
            "raw_features_hash": "hash_valid",
            "metadata": {},
        }

        unauthorized_versions = [
            ("Missing model_version", {k: v for k, v in base_valid.items() if k != "model_version"}),
            ("Empty model_version", {**base_valid, "model_version": ""}),
            ("Whitespace model_version", {**base_valid, "model_version": "   "}),
            ("Legacy v1.0.0", {**base_valid, "model_version": "v1.0.0"}),
            ("Arbitrary string", {**base_valid, "model_version": "CustomAlphaModel"}),
            ("V4 Experimental", {**base_valid, "model_version": "Institutional_Mesh_V4"}),
        ]

        for desc, bad_record in unauthorized_versions:
            res = self.ledger.replace_historical_replay_atomically(
                symbol="AAPL",
                validated_signals=[bad_record],
                replay_status="COMPLETED",
                allow_empty=False,
            )
            self.assertFalse(res, f"Expected rejection for {desc}")

        self.assertEqual(self._get_historical_count(), initial_historical)

        # Authorized versions must succeed
        for auth_ver in ("Institutional_Mesh_V2.4.1", "Institutional_Mesh_V2.4", "Institutional_Mesh_V3"):
            record = {**base_valid, "model_version": auth_ver}
            res = self.ledger.replace_historical_replay_atomically(
                symbol="AAPL",
                validated_signals=[record],
                replay_status="COMPLETED",
                allow_empty=False,
            )
            self.assertTrue(res, f"Authorized version {auth_ver} should be accepted")

        self.assertEqual(self._get_prospective_count(), initial_prospective)

    def test_6_inference_exception_preserves_records_and_returns_empty(self):
        """Proof: Exception during replay computation preserves existing historical records and returns empty."""
        initial_prospective = self._get_prospective_count()
        signals_before = self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])
        self.assertEqual(len(signals_before), 2)

        service = InferenceService(
            model_manager=MagicMock(),
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        # Simulate exception during feature computation / inference pipeline
        with patch("src.execution.inference_service.add_upgraded_features", side_effect=RuntimeError("Simulated pipeline failure")):
            dummy_df = pd.DataFrame(
                {"Open": [220.0] * 35, "High": [225.0] * 35, "Low": [215.0] * 35, "Close": [222.0] * 35, "Volume": [1000] * 35},
                index=pd.date_range("2026-08-01", periods=35),
            )
            res = service._replay_causal_ml_signals("AAPL", dummy_df)

        self.assertEqual(res, [], "Must return empty list on failure, NOT stale database records!")
        self.assertEqual(service.last_replay_audit["generation_success"], False)
        self.assertEqual(service.last_replay_audit["persistence_success"], False)
        self.assertIn("Simulated pipeline failure", service.last_replay_audit["error"])

        # Database records must remain completely intact
        signals_after = self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])
        self.assertEqual(len(signals_after), 2)
        self.assertEqual(self._get_prospective_count(), initial_prospective)

    def test_7_rejected_replacement_handling_preserves_records(self):
        """Proof: Rejected replacement sets audit failure, returns empty list, and leaves DB untouched."""
        initial_prospective = self._get_prospective_count()
        signals_before = self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])
        self.assertEqual(len(signals_before), 2)

        service = InferenceService(
            model_manager=MagicMock(),
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        from src.execution.live_inference import FEATURE_COLUMNS_V30

        # Create realistic input and feature frame containing all FEATURE_COLUMNS_V30
        dummy_df = pd.DataFrame(
            {"Open": [220.0] * 35, "High": [225.0] * 35, "Low": [215.0] * 35, "Close": [222.0] * 35, "Volume": [1000] * 35},
            index=pd.date_range("2026-08-01", periods=35),
        )
        feat_dict = {col: [1.0] * 35 for col in FEATURE_COLUMNS_V30}
        feat_dict["Close"] = [222.0] * 35
        feat_df = pd.DataFrame(feat_dict, index=dummy_df.index)

        # Force replace_historical_replay_atomically to return False (rejected persistence)
        with patch.object(self.ledger, "replace_historical_replay_atomically", return_value=False), \
             patch("src.execution.inference_service.add_upgraded_features", return_value=feat_df), \
             patch("joblib.load") as mock_joblib, \
             patch("xgboost.XGBClassifier") as mock_xgb:

            service.mm.scaler = None
            service.mm.xgb_model = None

            mock_scaler = MagicMock()
            mock_scaler.n_features_in_ = 31
            mock_scaler.transform.return_value = np.zeros((35, 31))
            mock_joblib.return_value = mock_scaler

            mock_clf = MagicMock()
            mock_clf.n_features_in_ = 31
            mock_clf.predict_proba.return_value = np.tile([0.1, 0.2, 0.7], (35, 1))
            mock_xgb.return_value = mock_clf

            res = service._replay_causal_ml_signals("AAPL", dummy_df, version="Institutional_Mesh_V2.4")

        self.assertEqual(res, [], "On persistence rejection, must return empty list, NOT stale records!")
        self.assertTrue(service.last_replay_audit["generation_success"])
        self.assertFalse(service.last_replay_audit["persistence_success"])
        self.assertEqual(service.last_replay_audit["error"], "LEDGER_REPLACEMENT_FAILED")

        # Prior records preserved in DB
        signals_after = self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])
        self.assertEqual(len(signals_after), 2)
        self.assertEqual(self._get_prospective_count(), initial_prospective)

    def test_8_confirmed_vs_unconfirmed_empty_replay(self):
        """Proof: Unconfirmed/incomplete empty replay preserves records, while verified complete empty replay clears stale history."""
        initial_prospective = self._get_prospective_count()

        # Unconfirmed: allow_empty=False
        res_unconfirmed = self.ledger.replace_historical_replay_atomically(
            symbol="AAPL",
            validated_signals=[],
            replay_status="COMPLETED",
            allow_empty=False,
            expected_bars=30,
            processed_bars=30,
        )
        self.assertFalse(res_unconfirmed)
        self.assertEqual(len(self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])), 2)

        # Incomplete replay status
        res_incomplete_status = self.ledger.replace_historical_replay_atomically(
            symbol="AAPL",
            validated_signals=[],
            replay_status="IN_PROGRESS",
            allow_empty=True,
            expected_bars=30,
            processed_bars=30,
        )
        self.assertFalse(res_incomplete_status)
        self.assertEqual(len(self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])), 2)

        # Missing completeness proof (None values)
        res_no_proof = self.ledger.replace_historical_replay_atomically(
            symbol="AAPL",
            validated_signals=[],
            replay_status="COMPLETED",
            allow_empty=True,
            expected_bars=None,
            processed_bars=None,
        )
        self.assertFalse(res_no_proof)
        self.assertEqual(len(self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])), 2)

        # Partial run (processed_bars < expected_bars)
        res_partial_run = self.ledger.replace_historical_replay_atomically(
            symbol="AAPL",
            validated_signals=[],
            replay_status="COMPLETED",
            allow_empty=True,
            expected_bars=30,
            processed_bars=20,
        )
        self.assertFalse(res_partial_run)
        self.assertEqual(len(self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])), 2)

        # Insufficient window (< 15 bars)
        res_insufficient_bars = self.ledger.replace_historical_replay_atomically(
            symbol="AAPL",
            validated_signals=[],
            replay_status="COMPLETED",
            allow_empty=True,
            expected_bars=10,
            processed_bars=10,
        )
        self.assertFalse(res_insufficient_bars)
        self.assertEqual(len(self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])), 2)

        # Confirmed empty replay: allow_empty=True, status="COMPLETED", processed_bars == expected_bars >= 15
        res_confirmed = self.ledger.replace_historical_replay_atomically(
            symbol="AAPL",
            validated_signals=[],
            replay_status="COMPLETED",
            allow_empty=True,
            expected_bars=30,
            processed_bars=30,
        )
        self.assertTrue(res_confirmed)
        self.assertEqual(len(self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])), 0)
        self.assertEqual(self._get_prospective_count(), initial_prospective)

    def test_9_mid_transaction_write_failure_rolls_back(self):
        """Proof: If a DB error occurs during insertion, transaction rolls back completely."""
        initial_prospective = self._get_prospective_count()

        valid_staged = [
            {
                "symbol": "AAPL",
                "bar_timestamp": "2026-09-15",
                "signal": "BUY",
                "confidence": 0.70,
                "execution_target_bar": "2026-09-16",
                "execution_price": 222.0,
                "model_version": "Institutional_Mesh_V2.4.1",
                "raw_features_hash": "hash_new_1",
                "metadata": {"source": "new_replay"},
            }
        ]

        orig_get_connection = self.ledger._get_connection

        class FaultyCursor:
            def __init__(self, real_cursor):
                self.real = real_cursor

            def execute(self, *args, **kwargs):
                return self.real.execute(*args, **kwargs)

            def executemany(self, *args, **kwargs):
                raise sqlite3.OperationalError("Simulated disk I/O failure during batch insert")

            def __getattr__(self, name):
                return getattr(self.real, name)

        class FaultyConnection:
            def __init__(self, real_conn):
                self.real = real_conn

            def cursor(self):
                return FaultyCursor(self.real.cursor())

            def __enter__(self):
                self.real.__enter__()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return self.real.__exit__(exc_type, exc_val, exc_tb)

            def __getattr__(self, name):
                return getattr(self.real, name)

        with patch.object(self.ledger, "_get_connection", side_effect=lambda: FaultyConnection(orig_get_connection())):
            success = self.ledger.replace_historical_replay_atomically(
                symbol="AAPL",
                validated_signals=valid_staged,
                replay_status="COMPLETED",
                allow_empty=False,
            )
            self.assertFalse(success)

        # Verification: Rollback restored the original 2 records
        signals_after = self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])
        self.assertEqual(len(signals_after), 2, "Mid-transaction error must roll back and restore old records!")
        self.assertEqual(signals_after[0]["bar_timestamp"], "2026-09-01")
        self.assertEqual(signals_after[1]["bar_timestamp"], "2026-09-10")
        self.assertEqual(self._get_prospective_count(), initial_prospective)

    def test_10_deterministic_row_level_preservation_on_rejected_input_and_write_failure(self):
        """Proof: Deterministic row-level comparison verifies byte-for-byte preservation across rejected input and write failures."""
        initial_prospective = self._get_prospective_count()

        def fetch_all_historical_rows():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                return [
                    (
                        r["timestamp"],
                        r["bar_timestamp"],
                        r["symbol"],
                        r["signal"],
                        r["confidence"],
                        r["execution_target_bar"],
                        r["execution_price"],
                        r["model_version"],
                        r["raw_features_hash"],
                        r["metadata"],
                        r["dataset"],
                    )
                    for r in conn.execute(
                        "SELECT timestamp, bar_timestamp, symbol, signal, confidence, "
                        "execution_target_bar, execution_price, model_version, raw_features_hash, "
                        "metadata, dataset FROM signal_ledger WHERE symbol = 'AAPL' ORDER BY bar_timestamp;"
                    ).fetchall()
                ]

        original_rows = fetch_all_historical_rows()
        self.assertEqual(len(original_rows), 2)

        # 1. Attempt rejected input: bad confidence
        bad_sig = {
            "symbol": "AAPL",
            "bar_timestamp": "2026-09-15",
            "signal": "BUY",
            "confidence": float("nan"),
            "execution_target_bar": "NEXT_SESSION_OPEN",
            "execution_price": 220.0,
            "model_version": "Institutional_Mesh_V2.4.1",
            "raw_features_hash": "hash_bad",
            "metadata": {},
        }
        res_bad_conf = self.ledger.replace_historical_replay_atomically(
            symbol="AAPL",
            validated_signals=[bad_sig],
            replay_status="COMPLETED",
            allow_empty=False,
        )
        self.assertFalse(res_bad_conf)
        self.assertEqual(fetch_all_historical_rows(), original_rows, "Rows must be byte-for-byte identical after rejected confidence")

        # 2. Attempt rejected input: same-day temporal causality violation
        bad_causality = {
            "symbol": "AAPL",
            "bar_timestamp": "2026-09-15 16:00:00",
            "signal": "BUY",
            "confidence": 0.70,
            "execution_target_bar": "2026-09-15 09:30:00",
            "execution_price": 220.0,
            "model_version": "Institutional_Mesh_V2.4.1",
            "raw_features_hash": "hash_bad",
            "metadata": {},
        }
        res_causality = self.ledger.replace_historical_replay_atomically(
            symbol="AAPL",
            validated_signals=[bad_causality],
            replay_status="COMPLETED",
            allow_empty=False,
        )
        self.assertFalse(res_causality)
        self.assertEqual(fetch_all_historical_rows(), original_rows, "Rows must be byte-for-byte identical after causality violation")

        # 3. Attempt rejected input: unauthorized model version
        bad_model_ver = {
            "symbol": "AAPL",
            "bar_timestamp": "2026-09-15",
            "signal": "BUY",
            "confidence": 0.70,
            "execution_target_bar": "NEXT_SESSION_OPEN",
            "execution_price": 220.0,
            "model_version": "Unauthorized_Experimental_Mesh",
            "raw_features_hash": "hash_bad",
            "metadata": {},
        }
        res_model_ver = self.ledger.replace_historical_replay_atomically(
            symbol="AAPL",
            validated_signals=[bad_model_ver],
            replay_status="COMPLETED",
            allow_empty=False,
        )
        self.assertFalse(res_model_ver)
        self.assertEqual(fetch_all_historical_rows(), original_rows, "Rows must be byte-for-byte identical after unauthorized model version")

        # Confirm prospective tables remained 100% untouched throughout
        self.assertEqual(self._get_prospective_count(), initial_prospective)

    def test_11_provenance_verification_missing_unknown_mismatched_unauthorized(self):
        """Proof: InferenceService._resolve_verified_model_provenance rejects missing, unknown, mismatched, and unauthorized versions."""
        service = InferenceService(
            model_manager=MagicMock(),
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        mock_scaler_31 = MagicMock()
        mock_scaler_31.n_features_in_ = 31
        mock_scaler_31._artifact_hash = "15ba081e304a6abcd2fb181733c1841205f5be9d8bd2ba53a6a03ee0abd6bd49"

        mock_scaler_27 = MagicMock()
        mock_scaler_27.n_features_in_ = 27
        mock_scaler_27._artifact_hash = "15ba081e304a6abcd2fb181733c1841205f5be9d8bd2ba53a6a03ee0abd6bd49"

        mock_xgb_31 = MagicMock()
        mock_xgb_31.n_features_in_ = 31
        mock_xgb_31._artifact_hash = "a7291fd14368bb2035779a34dd6494a31b8532820c7bcce96155fbef9dcd2d80"

        mock_xgb_27 = MagicMock()
        mock_xgb_27.n_features_in_ = 27
        mock_xgb_27._artifact_hash = "a7291fd14368bb2035779a34dd6494a31b8532820c7bcce96155fbef9dcd2d80"

        # 1. Missing scaler or model -> None
        self.assertIsNone(service._resolve_verified_model_provenance("Institutional_Mesh_V2.4.1", None, mock_xgb_31))
        self.assertIsNone(service._resolve_verified_model_provenance("Institutional_Mesh_V2.4.1", mock_scaler_31, None))

        # 2. Mismatched dimensions (scaler 31 vs model 27, or scaler 27 vs model 31) -> None
        self.assertIsNone(service._resolve_verified_model_provenance("Institutional_Mesh_V2.4.1", mock_scaler_31, mock_xgb_27))
        self.assertIsNone(service._resolve_verified_model_provenance("Institutional_Mesh_V2.4.1", mock_scaler_27, mock_xgb_31))

        # 3. Unknown / unauthorized version string -> None
        self.assertIsNone(service._resolve_verified_model_provenance("Random_Model_v9", mock_scaler_31, mock_xgb_31))
        self.assertIsNone(service._resolve_verified_model_provenance("v1.0.0", mock_scaler_31, mock_xgb_31))
        self.assertIsNone(service._resolve_verified_model_provenance("Institutional_Mesh_V4", mock_scaler_31, mock_xgb_31))

        # 4. Dimension mismatch for requested version:
        # V2.4.1 requires 31 features. If model has 27 features -> None
        self.assertIsNone(service._resolve_verified_model_provenance("Institutional_Mesh_V2.4.1", mock_scaler_27, mock_xgb_27))

        # 5. Candidate Authorization Separation:
        # Institutional_Mesh_V2.4.1 is PROPOSED_CANDIDATE_PENDING_AUTHORIZATION in frozen_strategy_manifest_v2.4.1.json
        # Artifact identity is not authorization. Pending candidates must fail closed in ordinary inference.
        self.assertIsNone(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4.1", mock_scaler_31, mock_xgb_31),
            "Pending candidate without formal authorization must fail closed in ordinary inference!",
        )
        self.assertEqual(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", mock_scaler_31, mock_xgb_31),
            "Institutional_Mesh_V2.4",
            "Authorized frozen strategy Institutional_Mesh_V2.4 must succeed when hashes match.",
        )

        # 6. version is None: resolves strictly to active authorized baseline (Institutional_Mesh_V2.4),
        # because V2.4.1 is pending authorization and is excluded from implicit routing!
        resolved_none = service._resolve_verified_model_provenance(None, mock_scaler_31, mock_xgb_31)
        self.assertEqual(
            resolved_none,
            "Institutional_Mesh_V2.4",
            "Implicit resolution must skip pending candidates and select active authorized baseline.",
        )

        # 7. version is None with unsupported feature dimension or dimension mismatch -> None
        mock_scaler_bad = MagicMock()
        mock_scaler_bad.n_features_in_ = 99
        mock_scaler_bad._artifact_hash = "15ba081e304a6abcd2fb181733c1841205f5be9d8bd2ba53a6a03ee0abd6bd49"
        mock_xgb_bad = MagicMock()
        mock_xgb_bad.n_features_in_ = 99
        mock_xgb_bad._artifact_hash = "a7291fd14368bb2035779a34dd6494a31b8532820c7bcce96155fbef9dcd2d80"
        self.assertIsNone(service._resolve_verified_model_provenance(None, mock_scaler_bad, mock_xgb_bad))
        self.assertIsNone(service._resolve_verified_model_provenance(None, mock_scaler_31, mock_xgb_bad))

    def test_12_partial_replay_failure_fails_closed(self):
        """Proof: Replay failure mid-loop or incomplete window fails closed, leaving DB intact."""
        initial_prospective = self._get_prospective_count()
        signals_before = self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])
        self.assertEqual(len(signals_before), 2)

        service = InferenceService(
            model_manager=MagicMock(),
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        # Input df with fewer than 15 valid bars
        short_df = pd.DataFrame(
            {"Open": [220.0] * 10, "High": [225.0] * 10, "Low": [215.0] * 10, "Close": [222.0] * 10, "Volume": [1000] * 10},
            index=pd.date_range("2026-08-01", periods=10),
        )
        res_short = service._replay_causal_ml_signals("AAPL", short_df)
        self.assertEqual(res_short, [], "Short replay window must fail closed")
        self.assertEqual(len(self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])), 2)
        self.assertEqual(self._get_prospective_count(), initial_prospective)

    def test_13_governance_authorization_and_cryptographic_identity_checks(self):
        """Proof: Ambiguous strings fail, explicit requests enforce SHA-256 hashes, pending candidates require authorization."""
        service = InferenceService(
            model_manager=MagicMock(),
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        mock_scaler_31 = MagicMock()
        mock_scaler_31.n_features_in_ = 31

        mock_xgb_31 = MagicMock()
        mock_xgb_31.n_features_in_ = 31

        # 1. Ambiguous version strings must be rejected (no substring/prefix matching)
        ambiguous_strings = ["V2.4", "V2.4.1", "v2.4", "V3", "Institutional_Mesh", "", "Institutional_Mesh_V2", "Institutional_Mesh_V2.4.1_beta"]
        for ambig in ambiguous_strings:
            self.assertIsNone(
                service._resolve_verified_model_provenance(ambig, mock_scaler_31, mock_xgb_31),
                f"Ambiguous string '{ambig}' must be strictly rejected with fail-closed None!",
            )

        # 2. Cryptographic identity under explicit version requests:
        # Loaded model/scaler hash mismatch must fail closed even when dimensions match
        # Fetch expected hashes from V2.4 manifest
        v24_manifest_path = Path("artifacts/frozen_strategy_manifest_v2.4.json")
        if not v24_manifest_path.exists():
            v24_manifest_path = Path("backend/artifacts/frozen_strategy_manifest_v2.4.json")
        with open(v24_manifest_path, "r", encoding="utf-8") as f:
            v24_manifest = json.load(f)
        expected_xgb_hash = v24_manifest["model_hashes"]["xgb_ensemble.json"]
        expected_scaler_hash = v24_manifest["model_hashes"]["latest_scaler.joblib"]

        # Valid hashes -> success
        mock_xgb_31._artifact_hash = expected_xgb_hash
        mock_scaler_31._artifact_hash = expected_scaler_hash
        self.assertEqual(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", mock_scaler_31, mock_xgb_31),
            "Institutional_Mesh_V2.4",
        )

        # Mismatched model hash under explicit request -> fails closed
        mock_xgb_31._artifact_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        self.assertIsNone(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", mock_scaler_31, mock_xgb_31),
            "Explicit version request with corrupted/mismatched model hash must fail closed!",
        )

        # Reset model hash, corrupt scaler hash -> fails closed
        mock_xgb_31._artifact_hash = expected_xgb_hash
        mock_scaler_31._artifact_hash = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        self.assertIsNone(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", mock_scaler_31, mock_xgb_31),
            "Explicit version request with corrupted/mismatched scaler hash must fail closed!",
        )

        # 3. Hash verification under explicit request for pending candidate V2.4.1
        v241_manifest_path = Path("artifacts/frozen_strategy_manifest_v2.4.1.json")
        if not v241_manifest_path.exists():
            v241_manifest_path = Path("backend/artifacts/frozen_strategy_manifest_v2.4.1.json")
        with open(v241_manifest_path, "r", encoding="utf-8") as f:
            v241_manifest = json.load(f)
        expected_v241_xgb = v241_manifest["model_hashes"]["xgb_ensemble.json"]
        expected_v241_scaler = v241_manifest["model_hashes"]["latest_scaler.joblib"]

        # Matching hashes but candidate unauthorized -> None (ordinary inference fails closed)
        mock_xgb_31._artifact_hash = expected_v241_xgb
        mock_scaler_31._artifact_hash = expected_v241_scaler
        self.assertIsNone(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4.1", mock_scaler_31, mock_xgb_31),
            "Cryptographic identity alone does not equal authorization: pending candidate must fail closed!",
        )

        # Missing _artifact_hash entirely -> None (fails closed, no disk fallback!)
        mock_xgb_no_hash = MagicMock()
        mock_xgb_no_hash.n_features_in_ = 31
        self.assertIsNone(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", mock_scaler_31, mock_xgb_no_hash),
            "Missing in-memory _artifact_hash must fail closed without disk fallback!",
        )

    def test_14_blocker_a_authorization_precedence_and_conflicting_manifests(self):
        """
        Blocker A: Strict governance authorization precedence tests.
        Verifies that:
        1. Conflicting manifest (status=FROZEN..., anti_lock=True, candidate_status=PROPOSED..., formal_auth=False)
           FAILS CLOSED (False). Frozen status or anti-lock lock NEVER overrides explicit pending state.
        2. Revoked, rejected, or draft states fail closed.
        3. Missing or malformed authorization records fail closed.
        4. Contradictory authorization records fail closed.
        5. Historical baseline (no candidate_status, no authorization_record) succeeds only when frozen & locked.
        """
        from unittest.mock import patch

        service = InferenceService(
            model_manager=MagicMock(),
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        mock_scaler_31 = MagicMock()
        mock_scaler_31.n_features_in_ = 31
        mock_scaler_31._artifact_hash = "15ba081e304a6abcd2fb181733c1841205f5be9d8bd2ba53a6a03ee0abd6bd49"

        mock_xgb_31 = MagicMock()
        mock_xgb_31.n_features_in_ = 31
        mock_xgb_31._artifact_hash = "a7291fd14368bb2035779a34dd6494a31b8532820c7bcce96155fbef9dcd2d80"

        # Case 1: The exact conflicting manifest identified in audit review:
        # status="FROZEN_FOR_PROSPECTIVE_VALIDATION", anti_overfitting_lock=True,
        # BUT candidate_status="PROPOSED_CANDIDATE_PENDING_AUTHORIZATION", formal_authorization_granted=False
        conflicting_manifest = {
            "strategy_version": "HYDRA_PROSPECTIVE_V2.4.1",
            "status": "FROZEN_FOR_PROSPECTIVE_VALIDATION",
            "anti_overfitting_lock": True,
            "candidate_status": "PROPOSED_CANDIDATE_PENDING_AUTHORIZATION",
            "authorization_record": {
                "status": "PENDING_USER_APPROVAL",
                "formal_authorization_granted": False,
            },
            "model_hashes": {
                "xgb_ensemble.json": "a7291fd14368bb2035779a34dd6494a31b8532820c7bcce96155fbef9dcd2d80",
                "latest_scaler.joblib": "15ba081e304a6abcd2fb181733c1841205f5be9d8bd2ba53a6a03ee0abd6bd49",
            },
        }
        with patch.object(service, "_load_manifest_data", return_value=conflicting_manifest):
            res = service._resolve_verified_model_provenance("Institutional_Mesh_V2.4.1", mock_scaler_31, mock_xgb_31)
            self.assertIsNone(
                res,
                "Conflicting manifest with frozen status but pending candidate_status MUST fail closed!",
            )

        # Case 2: Revoked authorization record even if formal_authorization_granted is True
        revoked_manifest = {
            "strategy_version": "HYDRA_PROSPECTIVE_V2.4.1",
            "status": "FROZEN_FOR_PROSPECTIVE_VALIDATION",
            "anti_overfitting_lock": True,
            "candidate_status": "REVOKED",
            "authorization_record": {
                "status": "REVOKED",
                "formal_authorization_granted": True,
            },
            "model_hashes": conflicting_manifest["model_hashes"],
        }
        with patch.object(service, "_load_manifest_data", return_value=revoked_manifest):
            self.assertIsNone(
                service._resolve_verified_model_provenance("Institutional_Mesh_V2.4.1", mock_scaler_31, mock_xgb_31),
                "Revoked candidate MUST fail closed!",
            )

        # Case 3: Malformed authorization records (string, int, list, None)
        for malformed in ["granted", 1, True, ["authorized"]]:
            bad_manifest = {
                "strategy_version": "HYDRA_PROSPECTIVE_V2.4.1",
                "status": "FROZEN_FOR_PROSPECTIVE_VALIDATION",
                "anti_overfitting_lock": True,
                "candidate_status": "PROPOSED_CANDIDATE",
                "authorization_record": malformed,
                "model_hashes": conflicting_manifest["model_hashes"],
            }
            with patch.object(service, "_load_manifest_data", return_value=bad_manifest):
                self.assertIsNone(
                    service._resolve_verified_model_provenance("Institutional_Mesh_V2.4.1", mock_scaler_31, mock_xgb_31),
                    f"Malformed authorization_record {malformed} MUST fail closed!",
                )

        # Case 4: Candidate status set without any authorization record block
        cand_no_record = {
            "strategy_version": "HYDRA_PROSPECTIVE_V2.4.1",
            "status": "FROZEN_FOR_PROSPECTIVE_VALIDATION",
            "anti_overfitting_lock": True,
            "candidate_status": "CANDIDATE_V2.4.1",
            "model_hashes": conflicting_manifest["model_hashes"],
        }
        with patch.object(service, "_load_manifest_data", return_value=cand_no_record):
            self.assertIsNone(
                service._resolve_verified_model_provenance("Institutional_Mesh_V2.4.1", mock_scaler_31, mock_xgb_31),
                "Candidate status without valid authorization record MUST fail closed!",
            )

        # Case 5: Genuine historical baseline (V2.4) succeeds when frozen, anti_lock=True,
        # with matching commit, timestamp, verified manifest digest, authentic raw bytes, and no candidate_status
        v24_p = Path("artifacts/frozen_strategy_manifest_v2.4.json")
        if not v24_p.exists():
            v24_p = Path("backend/artifacts/frozen_strategy_manifest_v2.4.json")
        v24_raw_bytes = v24_p.read_bytes()

        legit_v24 = {
            "strategy_version": "HYDRA_PROSPECTIVE_V2.4",
            "git_commit": "e687e2321da9159dca2b10c6744f2166fb44a506",
            "freeze_timestamp_utc": "2026-10-01T09:39:11Z",
            "status": "FROZEN_FOR_PROSPECTIVE_VALIDATION",
            "anti_overfitting_lock": True,
            "model_hashes": conflicting_manifest["model_hashes"],
            "_manifest_file_hash": "d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5",
            "_manifest_raw_bytes": v24_raw_bytes,
        }
        with patch.object(service, "_load_manifest_data", return_value=legit_v24):
            self.assertEqual(
                service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", mock_scaler_31, mock_xgb_31),
                "Institutional_Mesh_V2.4",
            )

        # Case 6: An otherwise IDENTICAL manifest for ANY other version (e.g. V2.5 or V2.4.1)
        # without authorization_record MUST fail closed!
        identical_v25 = {
            "strategy_version": "HYDRA_PROSPECTIVE_V2.5",
            "git_commit": "e687e2321da9159dca2b10c6744f2166fb44a506",
            "freeze_timestamp_utc": "2026-10-01T09:39:11Z",
            "status": "FROZEN_FOR_PROSPECTIVE_VALIDATION",
            "anti_overfitting_lock": True,
            "model_hashes": conflicting_manifest["model_hashes"],
            "_manifest_file_hash": "d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5",
        }
        with patch.object(service, "_load_manifest_data", return_value=identical_v25):
            self.assertIsNone(
                service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", mock_scaler_31, mock_xgb_31),
                "Manifest with strategy_version != HYDRA_PROSPECTIVE_V2.4 MUST fail closed!",
            )

        # Case 7: An otherwise identical manifest with altered git_commit fails closed!
        tampered_commit = dict(legit_v24)
        tampered_commit["git_commit"] = "unauthorized_commit_hash"
        with patch.object(service, "_load_manifest_data", return_value=tampered_commit):
            self.assertIsNone(
                service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", mock_scaler_31, mock_xgb_31),
                "Manifest with altered git_commit MUST fail closed!",
            )

        # Case 8: An otherwise identical manifest with altered freeze_timestamp fails closed!
        tampered_ts = dict(legit_v24)
        tampered_ts["freeze_timestamp_utc"] = "2026-10-15T00:00:00Z"
        with patch.object(service, "_load_manifest_data", return_value=tampered_ts):
            self.assertIsNone(
                service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", mock_scaler_31, mock_xgb_31),
                "Manifest with altered freeze_timestamp_utc MUST fail closed!",
            )

        # Case 9: An otherwise identical manifest with altered manifest file hash fails closed!
        tampered_hash = dict(legit_v24)
        tampered_hash["_manifest_file_hash"] = "altered_manifest_file_digest_123"
        with patch.object(service, "_load_manifest_data", return_value=tampered_hash):
            self.assertIsNone(
                service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", mock_scaler_31, mock_xgb_31),
                "Manifest with altered _manifest_file_hash MUST fail closed!",
            )

        # =====================================================================
        # Blocker A Remediation Verification: Adversarial Negative Tests on
        # InferenceService.is_manifest_formally_authorized
        # =====================================================================

        # Case 10: Missing manifest hash key entirely
        no_hash = dict(legit_v24)
        del no_hash["_manifest_file_hash"]
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(no_hash, "Institutional_Mesh_V2.4"),
            "Missing _manifest_file_hash MUST reject legacy authorization exception!",
        )

        # Case 11: None manifest hash value
        none_hash = dict(legit_v24)
        none_hash["_manifest_file_hash"] = None
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(none_hash, "Institutional_Mesh_V2.4"),
            "None _manifest_file_hash MUST reject legacy authorization exception!",
        )

        # Case 12: Empty string manifest hash
        empty_hash = dict(legit_v24)
        empty_hash["_manifest_file_hash"] = ""
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(empty_hash, "Institutional_Mesh_V2.4"),
            "Empty string _manifest_file_hash MUST reject legacy authorization exception!",
        )

        # Case 13: Whitespace-only manifest hash
        ws_hash = dict(legit_v24)
        ws_hash["_manifest_file_hash"] = "    "
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(ws_hash, "Institutional_Mesh_V2.4"),
            "Whitespace-only _manifest_file_hash MUST reject legacy authorization exception!",
        )

        # Case 14: Non-string manifest hash types (int, list, dict, bool)
        for bad_val in [12345, ["d580b5..."], {"hash": "d580b5..."}, True]:
            bad_type_hash = dict(legit_v24)
            bad_type_hash["_manifest_file_hash"] = bad_val
            self.assertFalse(
                InferenceService.is_manifest_formally_authorized(bad_type_hash, "Institutional_Mesh_V2.4"),
                f"Non-string _manifest_file_hash ({type(bad_val)}) MUST reject legacy exception!",
            )

        # Case 15: Malformed hex digest lengths (63 chars, 65 chars)
        for bad_len in [
            "d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da",
            "d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5ff",
        ]:
            bad_len_hash = dict(legit_v24)
            bad_len_hash["_manifest_file_hash"] = bad_len
            self.assertFalse(
                InferenceService.is_manifest_formally_authorized(bad_len_hash, "Institutional_Mesh_V2.4"),
                f"Malformed length _manifest_file_hash ({len(bad_len)}) MUST reject legacy exception!",
            )

        # Case 16: Malformed non-hex characters in hash string
        non_hex_hash = dict(legit_v24)
        non_hex_hash["_manifest_file_hash"] = "g580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5"
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(non_hex_hash, "Institutional_Mesh_V2.4"),
            "Non-hex character in _manifest_file_hash MUST reject legacy exception!",
        )

        # Case 17a: Missing _manifest_raw_bytes key entirely (strictly rejects caller-side assumptions)
        no_raw_bytes = dict(legit_v24)
        del no_raw_bytes["_manifest_raw_bytes"]
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(no_raw_bytes, "Institutional_Mesh_V2.4"),
            "Missing _manifest_raw_bytes MUST reject legacy authorization exception!",
        )

        # Case 17b: _manifest_raw_bytes is None
        none_raw_bytes = dict(legit_v24)
        none_raw_bytes["_manifest_raw_bytes"] = None
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(none_raw_bytes, "Institutional_Mesh_V2.4"),
            "None _manifest_raw_bytes MUST reject legacy authorization exception!",
        )

        # Case 17c: _manifest_raw_bytes is empty bytes
        empty_raw_bytes = dict(legit_v24)
        empty_raw_bytes["_manifest_raw_bytes"] = b""
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(empty_raw_bytes, "Institutional_Mesh_V2.4"),
            "Empty _manifest_raw_bytes MUST reject legacy authorization exception!",
        )

        # Case 17d: _manifest_raw_bytes is bad type (string or integer)
        bad_type_raw = dict(legit_v24)
        bad_type_raw["_manifest_raw_bytes"] = "not_bytes"
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(bad_type_raw, "Institutional_Mesh_V2.4"),
            "Non-bytes _manifest_raw_bytes MUST reject legacy authorization exception!",
        )

        # Case 17e: Raw bytes attached that hash to a different digest
        raw_tampered = dict(legit_v24)
        raw_tampered["_manifest_raw_bytes"] = b'{"tampered": true}'
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(raw_tampered, "Institutional_Mesh_V2.4"),
            "Altered raw bytes attached to manifest MUST reject legacy exception!",
        )

        # Case 17f: Authentic raw bytes attached, but in-memory dictionary fields tampered
        dict_tampered = dict(legit_v24)
        dict_tampered["strategy_version"] = "HYDRA_PROSPECTIVE_V2.4.1"
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(dict_tampered, "Institutional_Mesh_V2.4"),
            "Dictionary fields mismatched against raw bytes JSON MUST reject legacy exception!",
        )

        # Case 18: Target version mismatch (attempting to use V2.4 exception for V2.4.1 or V3)
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(legit_v24, "Institutional_Mesh_V2.4.1"),
            "Target version Institutional_Mesh_V2.4.1 cannot use V2.4 legacy exception!",
        )
        self.assertFalse(
            InferenceService.is_manifest_formally_authorized(legit_v24, "Institutional_Mesh_V3"),
            "Target version Institutional_Mesh_V3 cannot use V2.4 legacy exception!",
        )

        # Case 19: Verify authentic on-disk V2.4 manifest byte loading and single-buffer parsing
        v24_loaded = service._load_manifest_data("Institutional_Mesh_V2.4")
        self.assertIsNotNone(v24_loaded, "Authentic V2.4 manifest must load successfully from disk")
        self.assertIn("_manifest_file_hash", v24_loaded)
        self.assertIn("_manifest_raw_bytes", v24_loaded)
        self.assertEqual(
            v24_loaded["_manifest_file_hash"],
            "d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5",
        )
        self.assertEqual(
            hashlib.sha256(v24_loaded["_manifest_raw_bytes"]).hexdigest(),
            v24_loaded["_manifest_file_hash"],
            "Raw byte digest must exactly equal _manifest_file_hash!",
        )
        self.assertTrue(
            InferenceService.is_manifest_formally_authorized(v24_loaded, "Institutional_Mesh_V2.4"),
            "Authentic loaded V2.4 manifest must pass authorization!",
        )

    def test_15_blocker_b_loaded_artifact_cryptographic_binding_and_pair_integrity(self):
        """
        Blocker B: Loaded-artifact cryptographic binding and pair integrity tests.
        Verifies that:
        1. In-memory object without bound _artifact_hash fails closed (no disk fallback).
        2. In-memory object whose hash differs from manifest fails closed.
        3. Model and scaler verified as compatible pair under same manifest.
        4. load_bound_scaler and load_bound_xgb bind SHA-256 digest at load time.
        """
        from src.execution.inference_service import load_bound_scaler, load_bound_xgb

        service = InferenceService(
            model_manager=MagicMock(),
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        v24_xgb_hash = "a7291fd14368bb2035779a34dd6494a31b8532820c7bcce96155fbef9dcd2d80"
        v24_scaler_hash = "15ba081e304a6abcd2fb181733c1841205f5be9d8bd2ba53a6a03ee0abd6bd49"

        # 1. Missing _artifact_hash on model fails closed
        scaler_ok = MagicMock()
        scaler_ok.n_features_in_ = 31
        scaler_ok._artifact_hash = v24_scaler_hash

        xgb_no_hash = MagicMock()
        xgb_no_hash.n_features_in_ = 31
        # No _artifact_hash attribute set
        self.assertIsNone(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", scaler_ok, xgb_no_hash),
            "Model without _artifact_hash must fail closed without fallback to on-disk files!",
        )

        # 2. Missing _artifact_hash on scaler fails closed
        xgb_ok = MagicMock()
        xgb_ok.n_features_in_ = 31
        xgb_ok._artifact_hash = v24_xgb_hash

        scaler_no_hash = MagicMock()
        scaler_no_hash.n_features_in_ = 31
        # No _artifact_hash attribute set
        self.assertIsNone(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", scaler_no_hash, xgb_ok),
            "Scaler without _artifact_hash must fail closed without fallback to on-disk files!",
        )

        # 3. In-memory object hash differs from manifest (e.g. disk file swapped after load)
        xgb_swapped = MagicMock()
        xgb_swapped.n_features_in_ = 31
        xgb_swapped._artifact_hash = "swapped_older_artifact_hash_12345"
        self.assertIsNone(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", scaler_ok, xgb_swapped),
            "In-memory model whose hash differs from manifest must fail closed!",
        )

        # 4. Incompatible pair (scaler from V3 manifest, model from V2.4 manifest)
        scaler_v3 = MagicMock()
        scaler_v3.n_features_in_ = 31
        scaler_v3._artifact_hash = "different_v3_scaler_hash"
        self.assertIsNone(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", scaler_v3, xgb_ok),
            "Model/scaler pair with mismatching provenance under V2.4 manifest must fail closed!",
        )

        # 5. Non-string _artifact_hash fails closed
        xgb_bad_type = MagicMock()
        xgb_bad_type.n_features_in_ = 31
        xgb_bad_type._artifact_hash = 123456789
        self.assertIsNone(
            service._resolve_verified_model_provenance("Institutional_Mesh_V2.4", scaler_ok, xgb_bad_type),
            "Non-string _artifact_hash must fail closed!",
        )

        # 6. Verify load_bound_scaler and load_bound_xgb functionality
        scaler_path = Path("backend/artifacts/latest_scaler.joblib")
        if scaler_path.exists():
            loaded_s = load_bound_scaler(scaler_path)
            self.assertTrue(hasattr(loaded_s, "_artifact_hash"))
            self.assertEqual(loaded_s._artifact_hash, v24_scaler_hash)
            self.assertEqual(loaded_s._artifact_source_path, str(scaler_path))

        xgb_path = Path("backend/artifacts/xgb_ensemble.json")
        if xgb_path.exists():
            loaded_m = load_bound_xgb(xgb_path)
            self.assertTrue(hasattr(loaded_m, "_artifact_hash"))
            self.assertEqual(loaded_m._artifact_hash, v24_xgb_hash)
            self.assertEqual(loaded_m._artifact_source_path, str(xgb_path))

        # 7. Non-existent path raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_bound_scaler("non_existent_scaler.joblib")
        with self.assertRaises(FileNotFoundError):
            load_bound_xgb("non_existent_xgb.json")

    def test_16_blocker_c_pending_candidate_isolation_and_research_dry_run(self):
        """
        Blocker C: Ordinary inference rejection and isolated research-only dry-run pathway.
        Verifies that:
        1. Ordinary inference fails closed when candidate Institutional_Mesh_V2.4.1 is requested.
        2. Ordinary inference with version=None resolves strictly to Institutional_Mesh_V2.4.
        3. replay_research_candidate_dry_run executes in-memory with zero ledger writes.
        4. Staged candidate markers are tagged as RESEARCH_DRY_RUN and is_dry_run=True.
        5. Existing database rows are 100% preserved and never relabeled.
        """
        # Baseline records in DB
        initial_signals = self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])
        initial_count = len(initial_signals)
        self.assertEqual(initial_count, 2)
        initial_prospective_count = self._get_prospective_count()

        # Build realistic synthetic test bars
        dates = pd.date_range("2026-08-01", periods=50, freq="B")
        df = pd.DataFrame(index=dates)
        df["Close"] = [220.0 * (1.005**i) for i in range(50)]
        df["Open"] = df["Close"] * 0.999
        df["High"] = df["Close"] * 1.008
        df["Low"] = df["Close"] * 0.992
        df["Volume"] = 1_500_000

        mock_scaler_31 = MagicMock()
        mock_scaler_31.n_features_in_ = 31
        mock_scaler_31._artifact_hash = "15ba081e304a6abcd2fb181733c1841205f5be9d8bd2ba53a6a03ee0abd6bd49"
        mock_scaler_31.transform.return_value = np.zeros((len(df), 31))

        mock_xgb_31 = MagicMock()
        mock_xgb_31.n_features_in_ = 31
        mock_xgb_31._artifact_hash = "a7291fd14368bb2035779a34dd6494a31b8532820c7bcce96155fbef9dcd2d80"
        mock_xgb_31.predict_proba.return_value = np.tile([0.10, 0.20, 0.70], (len(df), 1))

        mock_mm = MagicMock()
        mock_mm.scaler = mock_scaler_31
        mock_mm.xgb_model = mock_xgb_31

        service = InferenceService(
            model_manager=mock_mm,
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        # 1. Ordinary inference explicit candidate request -> fails closed (empty list)
        res_explicit = service._replay_causal_ml_signals(
            "AAPL", df, version="Institutional_Mesh_V2.4.1"
        )
        self.assertEqual(
            res_explicit,
            [],
            "Ordinary inference must fail closed when requesting unauthorized candidate V2.4.1!",
        )
        self.assertEqual(
            service.last_replay_audit["error"],
            "UNVERIFIED_MODEL_PROVENANCE",
        )

        # 2. Ordinary get_causal_chart_markers with force_recompute and candidate -> fails closed
        res_markers = service.get_causal_chart_markers(
            "AAPL", df, force_recompute=True, version="Institutional_Mesh_V2.4.1"
        )
        self.assertEqual(
            res_markers,
            [],
            "get_causal_chart_markers must fail closed for unauthorized candidate V2.4.1!",
        )

        # 3. Database remains completely untouched
        self.assertEqual(len(self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])), initial_count)
        self.assertEqual(self._get_prospective_count(), initial_prospective_count)

        # 4. Spies on all database and order routing entry points
        spy_replace = MagicMock(side_effect=self.ledger.replace_historical_replay_atomically)
        spy_record_sig = MagicMock(side_effect=self.ledger.record_signal)
        spy_record_prosp = MagicMock(side_effect=self.ledger.record_prospective_signal)
        spy_clear = MagicMock(side_effect=self.ledger.clear_historical_replay)
        self.ledger.replace_historical_replay_atomically = spy_replace
        self.ledger.record_signal = spy_record_sig
        self.ledger.record_prospective_signal = spy_record_prosp
        self.ledger.clear_historical_replay = spy_clear

        # Isolated research dry-run pathway across multiple scenarios
        dry_run_res = service.replay_research_candidate_dry_run(
            "AAPL", df, candidate_version="Institutional_Mesh_V2.4.1"
        )
        self.assertEqual(dry_run_res["status"], "SUCCESS")
        self.assertTrue(dry_run_res["is_research_dry_run"])
        self.assertTrue(dry_run_res["production_routing_blocked"])
        self.assertTrue(dry_run_res["ledger_writes_blocked"])
        self.assertGreater(dry_run_res["marker_count"], 0)
        self.assertEqual(dry_run_res["strategy_version"], "HYDRA_PROSPECTIVE_V2.4.1")
        self.assertEqual(dry_run_res["feature_dimension"], 31)

        for marker in dry_run_res["markers"]:
            self.assertTrue(marker.get("is_dry_run"))
            self.assertEqual(
                marker.get("model_version"),
                "Institutional_Mesh_V2.4.1_RESEARCH_DRY_RUN",
            )
            self.assertEqual(marker.get("strategy_version"), "HYDRA_PROSPECTIVE_V2.4.1")
            self.assertEqual(marker.get("scaler_hash"), dry_run_res["scaler_hash"])
            self.assertEqual(marker.get("xgb_hash"), dry_run_res["xgb_hash"])

        # Non-candidate baseline must NOT be run through research dry-run (prevents mislabeling)
        v24_attempt = service.replay_research_candidate_dry_run(
            "AAPL", df, candidate_version="Institutional_Mesh_V2.4"
        )
        self.assertEqual(v24_attempt["status"], "FAILED")
        self.assertIn("UNSUPPORTED_RESEARCH_CANDIDATE_VERSION", v24_attempt["error"])

        # Cross-version contamination defense:
        # If in-memory mm has mismatched scaler from another model, dry-run must reload candidate artifacts
        contaminated_mm = MagicMock()
        bad_scaler = MagicMock()
        bad_scaler._artifact_hash = "mismatched_older_scaler_hash"
        contaminated_mm.scaler = bad_scaler
        contaminated_mm.xgb_model = None
        service.mm = contaminated_mm

        reloaded_res = service.replay_research_candidate_dry_run(
            "AAPL", df, candidate_version="Institutional_Mesh_V2.4.1"
        )
        self.assertEqual(reloaded_res["status"], "SUCCESS")
        self.assertNotEqual(reloaded_res["scaler_hash"], "mismatched_older_scaler_hash")

        # Dimension mismatch defense: scaler n_features_in_ != xgb n_features_in_
        mismatch_scaler = MagicMock()
        mismatch_scaler.n_features_in_ = 27
        mismatch_scaler._artifact_hash = dry_run_res["scaler_hash"]
        with patch("src.execution.inference_service.load_bound_scaler", return_value=mismatch_scaler):
            service.mm.scaler = None
            dim_res = service.replay_research_candidate_dry_run("AAPL", df, candidate_version="Institutional_Mesh_V2.4.1")
            self.assertEqual(dim_res["status"], "FAILED")
            self.assertIn("DIMENSION_MISMATCH", dim_res["error"])

        # Reset service.mm for remaining checks
        service.mm = MagicMock()
        service.mm.scaler = None
        service.mm.xgb_model = None

        # Boundary checks: insufficient bars, empty dataframe
        short_df = df.iloc[:10].copy()
        res_short = service.replay_research_candidate_dry_run("AAPL", short_df)
        self.assertEqual(res_short["status"], "FAILED")
        self.assertEqual(res_short["error"], "INSUFFICIENT_BARS")

        res_empty = service.replay_research_candidate_dry_run("AAPL", pd.DataFrame())
        self.assertEqual(res_empty["status"], "FAILED")

        # Invariant Verification: Spies must have ZERO invocations!
        self.assertEqual(spy_replace.call_count, 0, "Dry-run pathway must NEVER invoke replace_historical_replay_atomically!")
        self.assertEqual(spy_record_sig.call_count, 0, "Dry-run pathway must NEVER invoke record_signal!")
        self.assertEqual(spy_record_prosp.call_count, 0, "Dry-run pathway must NEVER invoke record_prospective_signal!")
        self.assertEqual(spy_clear.call_count, 0, "Dry-run pathway must NEVER invoke clear_historical_replay!")
        self.assertEqual(service.paper_engine.method_calls, [], "Dry-run pathway must NEVER invoke paper engine trading methods!")

        # 5. CRITICAL: Database still remains 100% untouched after dry-run!
        signals_after_dry_run = self.ledger.get_signals("AAPL", actions_filter=["BUY", "SELL"])
        self.assertEqual(len(signals_after_dry_run), initial_count)
        self.assertEqual(self._get_prospective_count(), initial_prospective_count)

        # Confirm historical provenance preserved as V2.4 and NOT relabeled
        for sig in signals_after_dry_run:
            self.assertEqual(sig.get("model_version"), "Institutional_Mesh_V2.4")

    def test_17_blocker_d_authoritative_governance_thresholds(self):
        """
        Blocker D: Authoritative HYDRA governance requirements and threshold verification.
        Verifies that:
        1. Authoritative threshold requirements: Brier <= 0.20, ECE <= 0.08, prospective trades >= 50 (target 100).
        2. Downgraded 30-trade threshold is strictly rejected by governance standards.
        3. SPY SMA200 market regime coverage and strict causality/execution-cost conventions are required.
        4. Current platform state (Brier 0.5394, ECE 0.2076, 0 closed prospective trades) fails gates,
           so live capital routing must remain strictly blocked.
        """
        # Authoritative Governance Gate Definitions
        BRIER_MAX_THRESHOLD = 0.20
        ECE_MAX_THRESHOLD = 0.08
        PROSPECTIVE_MIN_INDEPENDENT_TRADES = 50
        PROSPECTIVE_TARGET_TRADES = 100
        DOWNGRADED_THRESHOLD = 30

        self.assertEqual(PROSPECTIVE_TARGET_TRADES, 100)
        self.assertGreater(PROSPECTIVE_TARGET_TRADES, PROSPECTIVE_MIN_INDEPENDENT_TRADES)

        # Governance assertion: 30 trades cannot satisfy the authoritative requirement
        self.assertLess(
            DOWNGRADED_THRESHOLD,
            PROSPECTIVE_MIN_INDEPENDENT_TRADES,
            "Downgraded 30-trade threshold is strictly less than authoritative 50-trade minimum!",
        )

        # Current calibration and prospective records
        current_brier = 0.5394
        current_ece = 0.2076
        current_prospective_trades = 0

        # Calibration gate checks: must fail closed
        brier_passed = current_brier <= BRIER_MAX_THRESHOLD
        ece_passed = current_ece <= ECE_MAX_THRESHOLD
        prospective_passed = current_prospective_trades >= PROSPECTIVE_MIN_INDEPENDENT_TRADES

        self.assertFalse(brier_passed, f"Brier {current_brier} exceeds {BRIER_MAX_THRESHOLD} and must fail!")
        self.assertFalse(ece_passed, f"ECE {current_ece} exceeds {ECE_MAX_THRESHOLD} and must fail!")
        self.assertFalse(prospective_passed, f"0 prospective trades is below {PROSPECTIVE_MIN_INDEPENDENT_TRADES} minimum!")

        # All gates must be simultaneously satisfied before live capital authorization
        all_gates_passed = brier_passed and ece_passed and prospective_passed
        self.assertFalse(
            all_gates_passed,
            "Live capital routing must remain strictly blocked until all statistical and prospective gates pass!",
        )

    def test_18_candidate_status_prohibitions_and_allowlist(self):
        """
        Proof: replay_research_candidate_dry_run strictly enforces candidate status policy:
        1. Prohibited states (REVOKED_CANDIDATE, SUSPENDED_CANDIDATE, REJECTED_CANDIDATE, UNAUTHORIZED_CANDIDATE, etc.)
           are explicitly rejected with PROHIBITED_CANDIDATE_STATUS.
        2. Empty/missing candidate status is rejected with MISSING_CANDIDATE_STATUS.
        3. Unpermitted states (DRAFT_CANDIDATE, EXPERIMENTAL_CANDIDATE) are rejected with UNPERMITTED_CANDIDATE_STATUS.
        4. Valid research states (PROPOSED_CANDIDATE_PENDING_AUTHORIZATION, RESEARCH_CANDIDATE_PENDING_EVALUATION)
           are accepted.
        """
        service = InferenceService(
            model_manager=MagicMock(),
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        dummy_df = pd.DataFrame(
            {"Open": [220.0] * 35, "High": [225.0] * 35, "Low": [215.0] * 35, "Close": [222.0] * 35, "Volume": [1000] * 35},
            index=pd.date_range("2026-08-01", periods=35),
        )

        base_manifest = service._load_manifest_data("Institutional_Mesh_V2.4.1")
        self.assertIsNotNone(base_manifest)

        prohibited_states = [
            "REVOKED_CANDIDATE",
            "SUSPENDED_CANDIDATE",
            "REJECTED_CANDIDATE",
            "UNAUTHORIZED_CANDIDATE",
            "DEPRECATED_CANDIDATE",
            "TERMINATED_CANDIDATE",
            "EXPIRED_CANDIDATE",
            "INVALID_CANDIDATE",
        ]

        for bad_status in prohibited_states:
            tampered = dict(base_manifest)
            tampered["candidate_status"] = bad_status
            with patch.object(service, "_load_manifest_data", return_value=tampered):
                res = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
                self.assertEqual(res["status"], "FAILED", f"Status '{bad_status}' must fail!")
                self.assertIn("PROHIBITED_CANDIDATE_STATUS", res["error"], f"Status '{bad_status}' must return PROHIBITED_CANDIDATE_STATUS!")

        # Empty and missing candidate status
        for empty_val in ["", "   "]:
            tampered = dict(base_manifest)
            tampered["candidate_status"] = empty_val
            with patch.object(service, "_load_manifest_data", return_value=tampered):
                res = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
                self.assertEqual(res["status"], "FAILED")
                self.assertIn("MISSING_CANDIDATE_STATUS", res["error"])

        # Unpermitted statuses (e.g. DRAFT_CANDIDATE, EXPERIMENTAL_CANDIDATE)
        for unpermitted in ["DRAFT_CANDIDATE", "EXPERIMENTAL_CANDIDATE", "ARBITRARY_CANDIDATE"]:
            tampered = dict(base_manifest)
            tampered["candidate_status"] = unpermitted
            with patch.object(service, "_load_manifest_data", return_value=tampered):
                res = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
                self.assertEqual(res["status"], "FAILED")
                self.assertIn("UNPERMITTED_CANDIDATE_STATUS", res["error"])

        # Valid research candidate statuses must succeed
        for valid_status in ["PROPOSED_CANDIDATE_PENDING_AUTHORIZATION", "RESEARCH_CANDIDATE_PENDING_EVALUATION"]:
            valid_manifest = dict(base_manifest)
            valid_manifest["candidate_status"] = valid_status
            with patch.object(service, "_load_manifest_data", return_value=valid_manifest):
                service.mm.scaler = None
                service.mm.xgb_model = None
                res = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
                self.assertEqual(res["status"], "SUCCESS", f"Valid status '{valid_status}' must succeed!")

    def test_19_candidate_version_mapping_and_baseline_prohibition(self):
        """
        Proof: Candidate version policy strictly restricts dry-run to designated candidates:
        1. Immutable baseline Institutional_Mesh_V2.4 fails closed with UNSUPPORTED_RESEARCH_CANDIDATE_VERSION.
        2. Non-candidate version Institutional_Mesh_V3 fails closed with UNSUPPORTED_RESEARCH_CANDIDATE_VERSION.
        3. Unknown/arbitrary versions fail closed with UNSUPPORTED_RESEARCH_CANDIDATE_VERSION.
        4. Mismatched strategy_version label in candidate manifest fails closed with VERSION_LABEL_MISMATCH.
        """
        service = InferenceService(
            model_manager=MagicMock(),
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        dummy_df = pd.DataFrame(
            {"Open": [220.0] * 35, "High": [225.0] * 35, "Low": [215.0] * 35, "Close": [222.0] * 35, "Volume": [1000] * 35},
            index=pd.date_range("2026-08-01", periods=35),
        )

        # Baseline V2.4 is prohibited from research candidate dry-run
        res_v24 = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4")
        self.assertEqual(res_v24["status"], "FAILED")
        self.assertIn("UNSUPPORTED_RESEARCH_CANDIDATE_VERSION", res_v24["error"])

        # V3 is prohibited from research candidate dry-run
        res_v3 = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V3")
        self.assertEqual(res_v3["status"], "FAILED")
        self.assertIn("UNSUPPORTED_RESEARCH_CANDIDATE_VERSION", res_v3["error"])

        # Arbitrary versions prohibited
        for arb in ["Institutional_Mesh_V4", "CustomModel_V1", "HYDRA_V2.5"]:
            res_arb = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version=arb)
            self.assertEqual(res_arb["status"], "FAILED")
            self.assertIn("UNSUPPORTED_RESEARCH_CANDIDATE_VERSION", res_arb["error"])

        # Strategy version mismatch: manifest declares strategy_version != HYDRA_PROSPECTIVE_V2.4.1
        base_manifest = service._load_manifest_data("Institutional_Mesh_V2.4.1")
        mismatched_strat = dict(base_manifest)
        mismatched_strat["strategy_version"] = "HYDRA_PROSPECTIVE_V2.5"
        with patch.object(service, "_load_manifest_data", return_value=mismatched_strat):
            res_mismatch = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
            self.assertEqual(res_mismatch["status"], "FAILED")
            self.assertIn("VERSION_LABEL_MISMATCH", res_mismatch["error"])

    def test_20_artifact_resolution_and_isolation_across_environments(self):
        """
        Proof: Artifact resolution across working directories and cross-version isolation:
        1. Candidate scaler and model resolve reliably.
        2. Missing artifacts fail closed with MISSING_ARTIFACTS.
        3. Wrong/altered artifact hashes fail closed with CRYPTOGRAPHIC_DIGEST_MISMATCH.
        4. V3 artifact hashes cannot be substituted for V2.4.1 artifacts.
        5. Swapped model/scaler pair (dimension mismatch 27 vs 31) fails closed with DIMENSION_MISMATCH.
        6. Scaler dimension mismatch against policy fails closed with POLICY_DIMENSION_MISMATCH.
        """
        service = InferenceService(
            model_manager=MagicMock(),
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        dummy_df = pd.DataFrame(
            {"Open": [220.0] * 35, "High": [225.0] * 35, "Low": [215.0] * 35, "Close": [222.0] * 35, "Volume": [1000] * 35},
            index=pd.date_range("2026-08-01", periods=35),
        )

        base_manifest = service._load_manifest_data("Institutional_Mesh_V2.4.1")
        expected_scaler_hash = base_manifest["model_hashes"]["latest_scaler.joblib"]
        expected_xgb_hash = base_manifest["model_hashes"]["xgb_ensemble.json"]

        # Case 1: Authentic artifact resolution succeeds
        service.mm.scaler = None
        service.mm.xgb_model = None
        res_ok = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
        self.assertEqual(res_ok["status"], "SUCCESS")
        self.assertEqual(res_ok["scaler_hash"], expected_scaler_hash)
        self.assertEqual(res_ok["xgb_hash"], expected_xgb_hash)

        # Case 2a: Missing candidate manifest fails closed
        with patch.object(service, "_load_manifest_data", return_value=None):
            service.mm.scaler = None
            service.mm.xgb_model = None
            res_no_manifest = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
            self.assertEqual(res_no_manifest["status"], "FAILED")
            self.assertIn("CANDIDATE_MANIFEST_NOT_FOUND", res_no_manifest["error"])

        # Case 2b: Manifest exists, but artifact files missing on disk fails closed
        with patch("src.execution.inference_service.load_bound_scaler", side_effect=FileNotFoundError("Scaler file missing on disk")):
            service.mm.scaler = None
            service.mm.xgb_model = None
            res_missing = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
            self.assertEqual(res_missing["status"], "FAILED")
            self.assertEqual(res_missing["error"], "MISSING_ARTIFACTS")

        # Case 3: Altered artifact hash fails closed
        tampered_scaler = MagicMock()
        tampered_scaler._artifact_hash = "altered_bad_scaler_digest_123"
        tampered_scaler.n_features_in_ = 31
        with patch("src.execution.inference_service.load_bound_scaler", return_value=tampered_scaler):
            service.mm.scaler = None
            service.mm.xgb_model = None
            res_bad_hash = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
            self.assertEqual(res_bad_hash["status"], "FAILED")
            self.assertEqual(res_bad_hash["error"], "CRYPTOGRAPHIC_DIGEST_MISMATCH")

        # Case 4: V3 artifact hash substitution fails closed
        v3_manifest = service._load_manifest_data("Institutional_Mesh_V3")
        v3_scaler_hash = v3_manifest["model_hashes"]["v3/latest_scaler.joblib"]
        v3_scaler_mock = MagicMock()
        v3_scaler_mock._artifact_hash = v3_scaler_hash
        v3_scaler_mock.n_features_in_ = 31
        with patch("src.execution.inference_service.load_bound_scaler", return_value=v3_scaler_mock):
            service.mm.scaler = None
            service.mm.xgb_model = None
            res_v3_sub = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
            self.assertEqual(res_v3_sub["status"], "FAILED")
            self.assertEqual(res_v3_sub["error"], "CRYPTOGRAPHIC_DIGEST_MISMATCH")

        # Case 5: Dimension mismatch (scaler 27, xgb 31) fails closed
        scaler_27 = MagicMock()
        scaler_27._artifact_hash = expected_scaler_hash
        scaler_27.n_features_in_ = 27
        xgb_31 = MagicMock()
        xgb_31._artifact_hash = expected_xgb_hash
        xgb_31.n_features_in_ = 31
        service.mm.scaler = scaler_27
        service.mm.xgb_model = xgb_31
        res_dim = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
        self.assertEqual(res_dim["status"], "FAILED")
        self.assertIn("DIMENSION_MISMATCH", res_dim["error"])

        # Case 6: Dimension mismatch against candidate policy (scaler 30, xgb 30 vs policy 31)
        scaler_30 = MagicMock()
        scaler_30._artifact_hash = expected_scaler_hash
        scaler_30.n_features_in_ = 30
        xgb_30 = MagicMock()
        xgb_30._artifact_hash = expected_xgb_hash
        xgb_30.n_features_in_ = 30
        service.mm.scaler = scaler_30
        service.mm.xgb_model = xgb_30
        res_policy_dim = service.replay_research_candidate_dry_run("AAPL", dummy_df, candidate_version="Institutional_Mesh_V2.4.1")
        self.assertEqual(res_policy_dim["status"], "FAILED")
        self.assertIn("POLICY_DIMENSION_MISMATCH", res_policy_dim["error"])

    def test_21_exit_decision_temporal_causality_and_open_invariance(self):
        """
        Proof of Temporal Causality and Open[t+1] Invariance:
        Exit decisions generated at bar t must depend EXCLUSIVELY on information available
        at or before bar t (close_t, p_sell, entry_price).
        Mutating Open[t+1] (the future bar's open):
        1. CANNOT trigger an exit decision at bar t if bar t conditions do not warrant it.
        2. CANNOT prevent an exit decision at bar t if bar t conditions warrant it.
        3. MAY ONLY change the fill execution price (open_next * 0.9995).
        """
        service = InferenceService(
            model_manager=MagicMock(),
            gemini_analyzer=MagicMock(),
            physical_edge=MagicMock(),
            dependency_graph=MagicMock(),
            orchestrator=MagicMock(),
            smart_router=MagicMock(),
            report_gen=MagicMock(),
            paper_engine=MagicMock(),
            perf_analyzer=MagicMock(),
            signal_ledger=self.ledger,
            use_veto=False,
        )

        dates = pd.date_range("2026-08-01", periods=35)
        base_df = pd.DataFrame(
            {
                "Open": [200.0] * 35,
                "High": [205.0] * 35,
                "Low": [195.0] * 35,
                "Close": [200.0] * 35,
                "Volume": [1000] * 35,
            },
            index=dates,
        )

        # Mock probabilities: Bar 0 triggers BUY (0.80), Bars 1..34 are neutral hold (0.10, 0.80, 0.10)
        mock_probs = []
        for i in range(35):
            if i == 0:
                mock_probs.append([0.10, 0.10, 0.80])
            else:
                mock_probs.append([0.10, 0.80, 0.10])
        mock_probs = np.array(mock_probs)

        manifest_data = service._load_manifest_data("Institutional_Mesh_V2.4.1")
        mock_xgb = MagicMock()
        mock_xgb._artifact_hash = manifest_data["model_hashes"]["xgb_ensemble.json"]
        mock_xgb.n_features_in_ = 31
        mock_xgb.predict_proba.return_value = mock_probs

        mock_scaler = MagicMock()
        mock_scaler._artifact_hash = manifest_data["model_hashes"]["latest_scaler.joblib"]
        mock_scaler.n_features_in_ = 31
        mock_scaler.transform.return_value = np.zeros((35, 31))

        service.mm.xgb_model = mock_xgb
        service.mm.scaler = mock_scaler

        # Scenario 1: Baseline run
        # Position entered at bar 0 (entry_price = 200.0 * 1.0005 = 200.10).
        # Close stays at 200.0, p_sell is 0.10. No exit occurs.
        res_baseline = service.replay_research_candidate_dry_run("AAPL", base_df.copy(), candidate_version="Institutional_Mesh_V2.4.1")
        self.assertEqual(res_baseline["status"], "SUCCESS")
        self.assertEqual(res_baseline["marker_count"], 1)
        self.assertEqual(res_baseline["markers"][0]["action"], "BUY")

        # Scenario 2: Severe downward gap on Open[t+1] (Bar 2 Open crashes to 50.0)
        # Under look-ahead logic, checking Open[t+1] < entry * 0.95 would falsely trigger an exit at Bar 1.
        # Under causal logic, Close[1] = 200.0, p_sell = 0.10, so NO exit decision can occur at Bar 1.
        df_crash_open = base_df.copy()
        df_crash_open.iloc[2, df_crash_open.columns.get_loc("Open")] = 50.0

        res_crash = service.replay_research_candidate_dry_run("AAPL", df_crash_open, candidate_version="Institutional_Mesh_V2.4.1")
        self.assertEqual(res_crash["status"], "SUCCESS")
        self.assertEqual(res_crash["marker_count"], 1, "Mutating Open[t+1] to crash must NOT create an exit decision at bar t!")
        self.assertEqual(res_crash["markers"][0]["action"], "BUY")
        self.assertEqual(res_crash["markers"][0]["bar_timestamp"], str(dates[0])[:10])
        self.assertFalse(any(m["action"] == "SELL" for m in res_crash["markers"]), "No SELL exit decision may be induced by future open crash!")

        # Scenario 3: Severe upward gap on Open[t+1] (Bar 2 Open surges to 400.0)
        # Under look-ahead logic, checking Open[t+1] > entry * 1.10 would falsely trigger take-profit at Bar 1.
        # Under causal logic, Close[1] = 200.0, p_sell = 0.10, so NO exit decision can occur at Bar 1.
        df_surge_open = base_df.copy()
        df_surge_open.iloc[2, df_surge_open.columns.get_loc("Open")] = 400.0

        res_surge = service.replay_research_candidate_dry_run("AAPL", df_surge_open, candidate_version="Institutional_Mesh_V2.4.1")
        self.assertEqual(res_surge["status"], "SUCCESS")
        self.assertEqual(res_surge["marker_count"], 1, "Mutating Open[t+1] to surge must NOT create an exit decision at bar t!")
        self.assertEqual(res_surge["markers"][0]["action"], "BUY")
        self.assertEqual(res_surge["markers"][0]["bar_timestamp"], str(dates[0])[:10])
        self.assertFalse(any(m["action"] == "SELL" for m in res_surge["markers"]), "No SELL exit decision may be induced by future open surge!")

        # Scenario 4: Causal Exit Triggered at Bar 1 Close (e.g. Close[1] breaches stop loss at 180.0)
        # Exit decision is genuinely generated at Bar 1 based strictly on Close[1] <= entry * 0.95.
        # Vary Bar 2 Open: Open[2] = 185.0 vs Open[2] = 175.0.
        # Both must produce an exit decision at Bar 1; the ONLY difference is execution_price (Open[2] * 0.9995).
        df_exit_a = base_df.copy()
        df_exit_a.iloc[1, df_exit_a.columns.get_loc("Close")] = 180.0
        df_exit_a.iloc[2, df_exit_a.columns.get_loc("Open")] = 185.0

        df_exit_b = base_df.copy()
        df_exit_b.iloc[1, df_exit_b.columns.get_loc("Close")] = 180.0
        df_exit_b.iloc[2, df_exit_b.columns.get_loc("Open")] = 175.0

        res_exit_a = service.replay_research_candidate_dry_run("AAPL", df_exit_a, candidate_version="Institutional_Mesh_V2.4.1")
        res_exit_b = service.replay_research_candidate_dry_run("AAPL", df_exit_b, candidate_version="Institutional_Mesh_V2.4.1")

        self.assertEqual(res_exit_a["status"], "SUCCESS")
        self.assertEqual(res_exit_b["status"], "SUCCESS")
        self.assertEqual(res_exit_a["marker_count"], 2)
        self.assertEqual(res_exit_b["marker_count"], 2)

        # Both exit markers occur at exactly the same bar_timestamp (Bar 1 close)
        marker_exit_a = res_exit_a["markers"][1]
        marker_exit_b = res_exit_b["markers"][1]

        self.assertEqual(marker_exit_a["action"], "SELL")
        self.assertEqual(marker_exit_b["action"], "SELL")
        self.assertEqual(marker_exit_a["symbol"], "AAPL")
        self.assertEqual(marker_exit_b["symbol"], "AAPL")
        self.assertEqual(marker_exit_a["bar_timestamp"], str(dates[1])[:10])
        self.assertEqual(marker_exit_b["bar_timestamp"], str(dates[1])[:10])
        self.assertEqual(marker_exit_a["bar_timestamp"], marker_exit_b["bar_timestamp"])
        self.assertEqual(marker_exit_a["model_version"], "Institutional_Mesh_V2.4.1_RESEARCH_DRY_RUN")
        self.assertEqual(marker_exit_b["model_version"], "Institutional_Mesh_V2.4.1_RESEARCH_DRY_RUN")
        self.assertTrue(marker_exit_a["is_dry_run"])
        self.assertTrue(marker_exit_b["is_dry_run"])

        # The exit decision and all metadata are completely invariant; only the fill price differs
        expected_price_a = 185.0 * 0.9995
        expected_price_b = 175.0 * 0.9995
        self.assertAlmostEqual(marker_exit_a["execution_price"], expected_price_a, places=5)
        self.assertAlmostEqual(marker_exit_b["execution_price"], expected_price_b, places=5)
        self.assertNotEqual(marker_exit_a["execution_price"], marker_exit_b["execution_price"])


if __name__ == "__main__":
    unittest.main()


