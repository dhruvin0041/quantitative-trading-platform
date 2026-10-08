# tests/test_ledger_isolation_and_recovery.py
"""
Comprehensive Unit and Integration Test Suite for HYDRA V2.2:
Legacy Ledger Isolation, Hash-Chain Recovery, Online Backup/Restoration,
Concurrent Ingestion, and Deprecation Safeguards.

Covers all tasks defined in HYDRA V2.2 Recovery Assurance Audit:
- Task B: Verify legacy-table isolation (updates to prospective_signals cannot alter returns, metrics, win rate).
- Task C: Hash-chain recovery audit (restart, missing, modified, deleted, reordered, invalid genesis/predecessor).
- Task D: Online backup and restoration to isolated target (chain continuation test).
- Task E: Concurrent ingestion race-condition handling (atomic deduplication, zero partial events).
- Task F: Reconciliation safeguard between legacy and immutable ledgers.
"""

import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.ops.run_prospective_validation import (
    BACKEND_DIR,
    ProspectiveValidationManager,
    get_high_precision_utc_now,
)


class TestLedgerIsolationAndRecovery(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_ledger.db")
        self.manager = ProspectiveValidationManager(
            backend_dir=BACKEND_DIR,
            db_path=self.db_path,
            reports_dir=Path(self.temp_dir) / "reports",
        )
        self.manifest_hash = self.manager.verify_frozen_configuration()["manifest_sha256"]

    def tearDown(self):
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def _insert_observation(
        self,
        signal_id: str,
        date_str: str,
        decision: str,
        ref_price: float,
        prev_hash: str,
        entry_price: float = None,
        exit_price: float = None,
        commission: float = 0.50,
        position_size: float = 100.0,
    ) -> str:
        temp_dict = {
            "signal_id": signal_id,
            "strategy_version": "HYDRA_PROSPECTIVE_V2.2",
            "source_asset": "AAPL",
            "source_candle_date": date_str,
            "candle_finalization_timestamp": f"{date_str}T20:00:00.000000Z",
            "signal_generation_timestamp": f"{date_str}T20:00:05.123456Z",
            "primary_model_prediction": "SELL" if decision == "HOLD" else decision,
            "final_trading_decision": decision,
            "signal_reference_price": round(ref_price, 4),
            "manifest_hash": self.manifest_hash,
        }
        canon_hash = self.manager.compute_canonical_signal_hash(temp_dict)
        obs_hash = hashlib.sha256(f"{prev_hash}:{canon_hash}".encode("utf-8")).hexdigest()

        with self.manager._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO prospective_observations (
                    signal_id, strategy_version, source_asset, source_candle_date,
                    signal_date, execution_date, candle_finalization_timestamp,
                    data_ingestion_timestamp, feature_computation_timestamp,
                    signal_generation_timestamp, order_submission_timestamp,
                    execution_timestamp, model_prediction_probabilities,
                    individual_model_predictions, primary_model_prediction,
                    veto_result, macro_filter_result, final_trading_decision,
                    signal_reference_price, modeled_execution_price,
                    modeled_entry_price, modeled_exit_price, slippage_amount,
                    commission_amount, position_size, execution_status,
                    manifest_hash, canonical_signal_hash, prev_observation_hash,
                    observation_hash, market_data_vendor, source_timestamp,
                    feature_computation_start, feature_computation_end,
                    inference_start, inference_end, signal_finalization_timestamp,
                    veto_audit_json, unambiguous_model_outputs, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    signal_id,
                    "HYDRA_PROSPECTIVE_V2.2",
                    "AAPL",
                    date_str,
                    date_str,
                    "N/A (HOLD)" if decision == "HOLD" else "2026-10-02",
                    f"{date_str}T20:00:00.000000Z",
                    f"{date_str}T20:00:01.000000Z",
                    f"{date_str}T20:00:02.000000Z",
                    f"{date_str}T20:00:05.123456Z",
                    f"{date_str}T20:00:05.123456Z",
                    None,
                    json.dumps({"XGBoost": [0.48, 0.13, 0.39]}),
                    json.dumps({"XGBoost": "SELL"}),
                    "SELL" if decision == "HOLD" else decision,
                    "PASSED",
                    "BULL",
                    decision,
                    ref_price,
                    entry_price or exit_price,
                    entry_price,
                    exit_price,
                    0.05,
                    commission,
                    position_size,
                    "SIMULATED_FILLED" if decision in ("BUY", "SELL") else "NOT_APPLICABLE_HOLD",
                    self.manifest_hash,
                    canon_hash,
                    prev_hash,
                    obs_hash,
                    "yfinance (Yahoo Finance Market Data API)",
                    "NOT_PROVIDED_BY_VENDOR",
                    f"{date_str}T20:00:01.000000Z",
                    f"{date_str}T20:00:02.000000Z",
                    f"{date_str}T20:00:03.000000Z",
                    f"{date_str}T20:00:04.000000Z",
                    f"{date_str}T20:00:05.123456Z",
                    json.dumps({"relative_conviction_veto": False}),
                    json.dumps({"primary_model": "XGBoost"}),
                    get_high_precision_utc_now(),
                ),
            )
            conn.commit()
        return obs_hash

    # =========================================================================
    # TASK B: VERIFY LEGACY TABLE ISOLATION
    # =========================================================================
    def test_legacy_table_mutation_isolation(self):
        """
        Verifies that mutations, in-place updates, or deletions in the legacy
        prospective_signals table cannot alter operational trade returns, win rate,
        profit factor, max drawdown, completed trade count, or performance metrics.
        """
        # Step 1: Insert completed BUY -> SELL trade in authoritative prospective_observations
        genesis_prev = f"GENESIS_V2_2_PROSPECTIVE_{self.manifest_hash[:16]}"
        hash1 = self._insert_observation(
            signal_id="PROP-AAPL-20261001-BUY1",
            date_str="2026-10-01",
            decision="BUY",
            ref_price=330.00,
            prev_hash=genesis_prev,
            entry_price=330.165,
        )
        self._insert_observation(
            signal_id="PROP-AAPL-20261002-SELL1",
            date_str="2026-10-02",
            decision="SELL",
            ref_price=340.00,
            prev_hash=hash1,
            exit_price=339.83,
        )

        # Step 2: Cross-log to legacy prospective_signals
        self.manager.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-10-01 16:00:00 EDT",
            signal_generation_timestamp="2026-10-01T20:00:05.123456Z",
            signal="BUY",
            probability=0.80,
            confidence=0.80,
            feature_hash="feat_hash_1",
            model_hash="mod_hash_1",
            execution_target_timestamp="2026-10-02 09:30:00 EDT",
            signal_reference_price=330.00,
            manifest_hash=self.manifest_hash,
        )

        # Step 3: Compute authoritative baseline performance
        baseline_perf = self.manager.compute_prospective_performance_and_checkpoints("AAPL")
        self.assertEqual(baseline_perf["completed_trades_count"], 1)
        self.assertEqual(baseline_perf["signal_counts"]["BUY"], 1)
        self.assertEqual(baseline_perf["signal_counts"]["SELL"], 1)
        self.assertEqual(baseline_perf["win_rate_pct"], 100.0)
        self.assertGreater(baseline_perf["net_compounded_return_pct"], 0.0)

        # Step 4: Maliciously mutate legacy prospective_signals
        with self.manager._get_connection() as conn:
            # Overwrite signal from BUY to SELL, set fake loss PnL
            conn.execute(
                """
                UPDATE prospective_signals
                SET signal = 'SELL', return_1d = -0.50, outcome = 'LOSS'
                WHERE symbol = 'AAPL';
                """
            )
            conn.commit()

        # Step 5: Recompute authoritative performance
        mutated_perf = self.manager.compute_prospective_performance_and_checkpoints("AAPL")

        # Step 6: Verify ZERO mutation leaked into authoritative metrics
        self.assertEqual(mutated_perf["completed_trades_count"], baseline_perf["completed_trades_count"])
        self.assertEqual(mutated_perf["signal_counts"], baseline_perf["signal_counts"])
        self.assertEqual(mutated_perf["win_rate_pct"], baseline_perf["win_rate_pct"])
        self.assertEqual(mutated_perf["profit_factor"], baseline_perf["profit_factor"])
        self.assertEqual(mutated_perf["net_compounded_return_pct"], baseline_perf["net_compounded_return_pct"])
        self.assertEqual(mutated_perf["max_drawdown_pct"], baseline_perf["max_drawdown_pct"])

        # Step 7: Delete entire legacy table contents
        with self.manager._get_connection() as conn:
            conn.execute("DELETE FROM prospective_signals;")
            conn.commit()

        deleted_perf = self.manager.compute_prospective_performance_and_checkpoints("AAPL")
        self.assertEqual(deleted_perf["completed_trades_count"], baseline_perf["completed_trades_count"])
        self.assertEqual(deleted_perf["win_rate_pct"], baseline_perf["win_rate_pct"])

    # =========================================================================
    # TASK C: HASH CHAIN RECOVERY AUDIT
    # =========================================================================
    def test_hash_chain_database_close_and_reopen(self):
        """
        Verifies database close, process recreation, and seamless hash-chain verification.
        """
        genesis_prev = f"GENESIS_V2_2_PROSPECTIVE_{self.manifest_hash[:16]}"
        hash1 = self._insert_observation("PROP-AAPL-20261001-A", "2026-10-01", "HOLD", 330.00, genesis_prev)
        self._insert_observation("PROP-AAPL-20261002-B", "2026-10-02", "HOLD", 332.00, hash1)

        # Close manager connection
        del self.manager

        # Reopen with completely new instance
        reopened_mgr = ProspectiveValidationManager(backend_dir=BACKEND_DIR, db_path=self.db_path)
        audit = reopened_mgr.verify_hash_chain("AAPL")
        self.assertTrue(audit["verified"])
        self.assertEqual(audit["chain_length"], 2)
        self.assertEqual(len(audit["violations"]), 0)

    def test_hash_chain_missing_observation_detection(self):
        """
        Verifies that deleting an intermediate observation in the chain triggers PREDECESSOR_HASH_MISMATCH.
        """
        genesis_prev = f"GENESIS_V2_2_PROSPECTIVE_{self.manifest_hash[:16]}"
        hash1 = self._insert_observation("PROP-AAPL-20261001-A", "2026-10-01", "HOLD", 330.00, genesis_prev)
        hash2 = self._insert_observation("PROP-AAPL-20261002-B", "2026-10-02", "HOLD", 332.00, hash1)
        self._insert_observation("PROP-AAPL-20261003-C", "2026-10-03", "HOLD", 335.00, hash2)

        # Disable trigger temporarily to simulate low-level filesystem corruption / unauthorized deletion
        with self.manager._get_connection() as conn:
            conn.execute("DROP TRIGGER prevent_delete_prospective_obs;")
            conn.execute("DELETE FROM prospective_observations WHERE signal_id = 'PROP-AAPL-20261002-B';")
            conn.commit()

        audit = self.manager.verify_hash_chain("AAPL")
        self.assertFalse(audit["verified"])
        self.assertTrue(
            any("PREDECESSOR_HASH_MISMATCH" in v.get("error", "") or "PREV_HASH_MISMATCH" in v.get("error", "")
                for v in audit["violations"])
        )

    def test_hash_chain_modified_observation_detection(self):
        """
        Verifies that modifying canonical signal attributes (e.g. price or decision)
        causes immediate CANONICAL_SIGNAL_HASH_MUTATION failure.
        """
        genesis_prev = f"GENESIS_V2_2_PROSPECTIVE_{self.manifest_hash[:16]}"
        hash1 = self._insert_observation("PROP-AAPL-20261001-A", "2026-10-01", "HOLD", 330.00, genesis_prev)
        self._insert_observation("PROP-AAPL-20261002-B", "2026-10-02", "HOLD", 332.00, hash1)

        # Drop trigger to simulate unauthorized direct byte/db mutation
        with self.manager._get_connection() as conn:
            conn.execute("DROP TRIGGER prevent_update_prospective_obs;")
            conn.execute(
                "UPDATE prospective_observations SET signal_reference_price = 999.99 WHERE signal_id = 'PROP-AAPL-20261001-A';"
            )
            conn.commit()

        audit = self.manager.verify_hash_chain("AAPL")
        self.assertFalse(audit["verified"])
        self.assertTrue(
            any("CANONICAL_SIGNAL_HASH_MUTATION" in v.get("error", "") for v in audit["violations"])
        )

    def test_hash_chain_reordered_observation_detection(self):
        """
        Verifies that out-of-order date sequencing is detected as CHRONOLOGICAL_SEQUENCE_VIOLATION.
        """
        genesis_prev = f"GENESIS_V2_2_PROSPECTIVE_{self.manifest_hash[:16]}"
        # Insert 2026-10-05 first
        hash1 = self._insert_observation("PROP-AAPL-20261005-A", "2026-10-05", "HOLD", 330.00, genesis_prev)
        # Attempt to link 2026-10-02 backwards
        self._insert_observation("PROP-AAPL-20261002-B", "2026-10-02", "HOLD", 332.00, hash1)

        audit = self.manager.verify_hash_chain("AAPL")
        self.assertFalse(audit["verified"])
        self.assertTrue(
            any("CHRONOLOGICAL_SEQUENCE_VIOLATION" in v.get("error", "") for v in audit["violations"])
        )

    def test_hash_chain_incorrect_genesis_detection(self):
        """
        Verifies that an incorrect genesis predecessor hash is caught as GENESIS_HASH_MISMATCH.
        """
        bogus_genesis = "GENESIS_V2_2_PROSPECTIVE_0000000000000000"
        self._insert_observation("PROP-AAPL-20261001-A", "2026-10-01", "HOLD", 330.00, bogus_genesis)

        audit = self.manager.verify_hash_chain("AAPL")
        self.assertFalse(audit["verified"])
        self.assertTrue(
            any("GENESIS_HASH_MISMATCH" in v.get("error", "") for v in audit["violations"])
        )

    def test_hash_chain_invalid_predecessor_hash_detection(self):
        """
        Verifies that an altered predecessor link is caught as PREDECESSOR_HASH_MISMATCH.
        """
        genesis_prev = f"GENESIS_V2_2_PROSPECTIVE_{self.manifest_hash[:16]}"
        self._insert_observation("PROP-AAPL-20261001-A", "2026-10-01", "HOLD", 330.00, genesis_prev)
        # Point to arbitrary corrupted predecessor hash
        self._insert_observation("PROP-AAPL-20261002-B", "2026-10-02", "HOLD", 332.00, "corrupted_prev_hash_1234")

        audit = self.manager.verify_hash_chain("AAPL")
        self.assertFalse(audit["verified"])
        self.assertTrue(
            any("PREDECESSOR_HASH_MISMATCH" in v.get("error", "") for v in audit["violations"])
        )

    # =========================================================================
    # TASK D: BACKUP AND RESTORATION
    # =========================================================================
    def test_backup_and_restoration(self):
        """
        Performs an online crash-consistent backup of the live ledger, restores it
        into a separate isolated database, verifies observations, events, manifest hash,
        and confirms the restored database accepts new observations seamlessly.
        """
        # 1. Populate source ledger
        genesis_prev = f"GENESIS_V2_2_PROSPECTIVE_{self.manifest_hash[:16]}"
        hash1 = self._insert_observation("PROP-AAPL-20261001-A", "2026-10-01", "HOLD", 330.00, genesis_prev)
        hash2 = self._insert_observation("PROP-AAPL-20261002-B", "2026-10-02", "BUY", 332.00, hash1, entry_price=332.16)

        evt_payload = self.manager.record_execution_event(
            signal_id="PROP-AAPL-20261002-B",
            source_asset="AAPL",
            event_type="ENTRY_FILL",
            target_execution_date="2026-10-03",
            modeled_fill_price=332.16,
            slippage_bps=5.0,
            slippage_amount=0.166,
            commission_total=0.50,
            shares_allocated=100.0,
        )
        self.assertEqual(evt_payload["execution_status"], "SIMULATED_FILLED")

        # 2. Take backup
        backup_file = os.path.join(self.temp_dir, "backup_vault", "ledger_backup.sqlite3")
        backup_meta = self.manager.backup_ledger(backup_file)
        self.assertEqual(backup_meta["status"], "BACKUP_COMPLETED")
        self.assertTrue(Path(backup_file).exists())
        self.assertGreater(backup_meta["backup_size_bytes"], 0)

        # 3. Restore to isolated target path (NEVER LIVE DB)
        restore_db_file = os.path.join(self.temp_dir, "restored_env", "restored_signal_ledger.db")
        restored_mgr = ProspectiveValidationManager.restore_ledger(
            backup_path=backup_file,
            restore_db_path=restore_db_file,
            backend_dir=BACKEND_DIR,
        )

        # 4. Verify all observations and events are restored
        with restored_mgr._get_connection() as r_conn:
            obs_count = r_conn.execute("SELECT COUNT(*) FROM prospective_observations;").fetchone()[0]
            evt_count = r_conn.execute("SELECT COUNT(*) FROM prospective_execution_events;").fetchone()[0]
            restored_manifest = r_conn.execute("SELECT manifest_hash FROM prospective_observations LIMIT 1;").fetchone()[0]

        self.assertEqual(obs_count, 2)
        self.assertEqual(evt_count, 1)
        self.assertEqual(restored_manifest, self.manifest_hash)

        # 5. Verify hash chain on restored db
        restored_chain = restored_mgr.verify_hash_chain("AAPL")
        self.assertTrue(restored_chain["verified"])
        self.assertEqual(restored_chain["chain_length"], 2)

        # 6. Verify restored database can continue accepting new observations without breaking chain
        # Re-use helper logic on restored_mgr
        temp_dict = {
            "signal_id": "PROP-AAPL-20261003-C",
            "strategy_version": "HYDRA_PROSPECTIVE_V2.2",
            "source_asset": "AAPL",
            "source_candle_date": "2026-10-03",
            "candle_finalization_timestamp": "2026-10-03T20:00:00.000000Z",
            "signal_generation_timestamp": "2026-10-03T20:00:05.123456Z",
            "primary_model_prediction": "SELL",
            "final_trading_decision": "SELL",
            "signal_reference_price": 340.00,
            "manifest_hash": self.manifest_hash,
        }
        canon_hash = restored_mgr.compute_canonical_signal_hash(temp_dict)
        obs_hash_3 = hashlib.sha256(f"{hash2}:{canon_hash}".encode("utf-8")).hexdigest()

        with restored_mgr._get_connection() as r_conn:
            r_conn.execute(
                """
                INSERT INTO prospective_observations (
                    signal_id, strategy_version, source_asset, source_candle_date,
                    signal_date, execution_date, candle_finalization_timestamp,
                    data_ingestion_timestamp, feature_computation_timestamp,
                    signal_generation_timestamp, order_submission_timestamp,
                    execution_timestamp, model_prediction_probabilities,
                    individual_model_predictions, primary_model_prediction,
                    veto_result, macro_filter_result, final_trading_decision,
                    signal_reference_price, modeled_execution_price,
                    modeled_entry_price, modeled_exit_price, slippage_amount,
                    commission_amount, position_size, execution_status,
                    manifest_hash, canonical_signal_hash, prev_observation_hash,
                    observation_hash, market_data_vendor, source_timestamp,
                    feature_computation_start, feature_computation_end,
                    inference_start, inference_end, signal_finalization_timestamp,
                    veto_audit_json, unambiguous_model_outputs, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "PROP-AAPL-20261003-C",
                    "HYDRA_PROSPECTIVE_V2.2",
                    "AAPL",
                    "2026-10-03",
                    "2026-10-03",
                    "2026-10-04",
                    "2026-10-03T20:00:00.000000Z",
                    "2026-10-03T20:00:01.000000Z",
                    "2026-10-03T20:00:02.000000Z",
                    "2026-10-03T20:00:05.123456Z",
                    "2026-10-03T20:00:05.123456Z",
                    None,
                    json.dumps({"XGBoost": [0.3, 0.2, 0.5]}),
                    json.dumps({"XGBoost": "SELL"}),
                    "SELL",
                    "PASSED",
                    "BULL",
                    "SELL",
                    340.00,
                    339.83,
                    None,
                    339.83,
                    0.05,
                    0.50,
                    100.0,
                    "SIMULATED_FILLED",
                    self.manifest_hash,
                    canon_hash,
                    hash2,
                    obs_hash_3,
                    "yfinance (Yahoo Finance Market Data API)",
                    "NOT_PROVIDED_BY_VENDOR",
                    "2026-10-03T20:00:01.000000Z",
                    "2026-10-03T20:00:02.000000Z",
                    "2026-10-03T20:00:03.000000Z",
                    "2026-10-03T20:00:04.000000Z",
                    "2026-10-03T20:00:05.123456Z",
                    json.dumps({}),
                    json.dumps({}),
                    get_high_precision_utc_now(),
                ),
            )
            r_conn.commit()

        # Audit extended chain on restored manager
        chain_extended = restored_mgr.verify_hash_chain("AAPL")
        self.assertTrue(chain_extended["verified"])
        self.assertEqual(chain_extended["chain_length"], 3)
        self.assertEqual(len(chain_extended["violations"]), 0)

    # =========================================================================
    # TASK E: CONCURRENT INGESTION RACE CONDITION
    # =========================================================================
    def test_concurrent_ingestion_race_condition(self):
        """
        Simulates two runner instances attempting to finalize and commit the same signal
        for the same candle date simultaneously.
        Verifies:
        - Exactly one observation is committed.
        - Duplicate insertion is rejected safely via UNIQUE constraint.
        - No partial execution event is created.
        - No hash-chain corruption occurs.
        - The losing runner reports is_new=False.
        """
        genesis_prev = f"GENESIS_V2_2_PROSPECTIVE_{self.manifest_hash[:16]}"
        candle_date = "2026-10-02"

        def attempt_ingestion(runner_id: int) -> Dict[str, Any]:
            # Each worker uses its own manager instance accessing the same shared SQLite file
            mgr = ProspectiveValidationManager(backend_dir=BACKEND_DIR, db_path=self.db_path)
            sig_id = f"PROP-AAPL-20261002-RUNNER{runner_id}"
            temp_record = {
                "signal_id": sig_id,
                "strategy_version": "HYDRA_PROSPECTIVE_V2.2",
                "source_asset": "AAPL",
                "source_candle_date": candle_date,
                "candle_finalization_timestamp": f"{candle_date}T20:00:00.000000Z",
                "signal_generation_timestamp": f"{candle_date}T20:00:05.123456Z",
                "primary_model_prediction": "HOLD",
                "final_trading_decision": "HOLD",
                "signal_reference_price": 332.50,
                "manifest_hash": self.manifest_hash,
            }
            canon_hash = mgr.compute_canonical_signal_hash(temp_record)
            obs_hash = hashlib.sha256(f"{genesis_prev}:{canon_hash}".encode("utf-8")).hexdigest()

            with mgr._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO prospective_observations (
                        signal_id, strategy_version, source_asset, source_candle_date,
                        signal_date, execution_date, candle_finalization_timestamp,
                        data_ingestion_timestamp, feature_computation_timestamp,
                        signal_generation_timestamp, order_submission_timestamp,
                        execution_timestamp, model_prediction_probabilities,
                        individual_model_predictions, primary_model_prediction,
                        veto_result, macro_filter_result, final_trading_decision,
                        signal_reference_price, modeled_execution_price,
                        modeled_entry_price, modeled_exit_price, slippage_amount,
                        commission_amount, position_size, execution_status,
                        manifest_hash, canonical_signal_hash, prev_observation_hash,
                        observation_hash, market_data_vendor, source_timestamp,
                        feature_computation_start, feature_computation_end,
                        inference_start, inference_end, signal_finalization_timestamp,
                        veto_audit_json, unambiguous_model_outputs, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        sig_id,
                        "HYDRA_PROSPECTIVE_V2.2",
                        "AAPL",
                        candle_date,
                        candle_date,
                        "N/A (HOLD)",
                        f"{candle_date}T20:00:00.000000Z",
                        f"{candle_date}T20:00:01.000000Z",
                        f"{candle_date}T20:00:02.000000Z",
                        f"{candle_date}T20:00:05.123456Z",
                        f"{candle_date}T20:00:05.123456Z",
                        None,
                        json.dumps({"XGBoost": [0.1, 0.8, 0.1]}),
                        json.dumps({"XGBoost": "HOLD"}),
                        "HOLD",
                        "PASSED",
                        "BULL",
                        "HOLD",
                        332.50,
                        None,
                        None,
                        None,
                        0.0,
                        0.0,
                        0.0,
                        "NOT_APPLICABLE_HOLD",
                        self.manifest_hash,
                        canon_hash,
                        genesis_prev,
                        obs_hash,
                        "yfinance",
                        "NOT_PROVIDED",
                        f"{candle_date}T20:00:01.000000Z",
                        f"{candle_date}T20:00:02.000000Z",
                        f"{candle_date}T20:00:03.000000Z",
                        f"{candle_date}T20:00:04.000000Z",
                        f"{candle_date}T20:00:05.123456Z",
                        json.dumps({}),
                        json.dumps({}),
                        get_high_precision_utc_now(),
                    ),
                )
                conn.commit()
                is_new = cursor.rowcount > 0

            # Only winner logs execution event
            if is_new:
                mgr.record_execution_event(
                    signal_id=sig_id,
                    source_asset="AAPL",
                    event_type="HOLD_DECISION_NO_EXECUTION",
                    execution_status="NOT_APPLICABLE_HOLD",
                    trade_completion_status="NOT_APPLICABLE",
                )

            return {"runner_id": runner_id, "is_new": is_new, "signal_id": sig_id}

        # Run both runners concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(attempt_ingestion, 1)
            fut2 = executor.submit(attempt_ingestion, 2)
            res1 = fut1.result()
            res2 = fut2.result()

        results = [res1, res2]
        winners = [r for r in results if r["is_new"]]
        losers = [r for r in results if not r["is_new"]]

        # Exactly 1 winner and 1 loser
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(losers), 1)

        # Check database records
        with self.manager._get_connection() as conn:
            obs_rows = conn.execute("SELECT * FROM prospective_observations;").fetchall()
            evt_rows = conn.execute("SELECT * FROM prospective_execution_events;").fetchall()

        self.assertEqual(len(obs_rows), 1)
        self.assertEqual(len(evt_rows), 1)
        self.assertEqual(obs_rows[0]["signal_id"], winners[0]["signal_id"])
        self.assertEqual(evt_rows[0]["signal_id"], winners[0]["signal_id"])

        # Chain remains pristine
        chain_audit = self.manager.verify_hash_chain("AAPL")
        self.assertTrue(chain_audit["verified"])
        self.assertEqual(chain_audit["chain_length"], 1)

    # =========================================================================
    # TASK F: DEPRECATION SAFEGUARDS & RECONCILIATION
    # =========================================================================
    def test_reconciliation_safeguard(self):
        """
        Verifies reconciliation checks between legacy and immutable records,
        confirming that discrepancies in decisions, prices, or manifests trigger alerts.
        """
        genesis_prev = f"GENESIS_V2_2_PROSPECTIVE_{self.manifest_hash[:16]}"
        self._insert_observation("PROP-AAPL-20261001-TEST", "2026-10-01", "HOLD", 330.32, genesis_prev)

        # Insert matching record into legacy table
        self.manager.ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-10-01 16:00:00 EDT",
            signal_generation_timestamp="2026-10-01T20:00:05.123456Z",
            signal="HOLD",
            probability=0.50,
            confidence=0.50,
            feature_hash="feat_hash",
            model_hash="model_hash",
            execution_target_timestamp="2026-10-02 09:30:00 EDT",
            signal_reference_price=330.32,
            manifest_hash=self.manifest_hash,
        )

        # Step 1: Matching test
        recon = self.manager.reconcile_legacy_and_immutable_ledgers("AAPL")
        self.assertTrue(recon["reconciled"])
        self.assertEqual(recon["discrepancies_count"], 0)
        self.assertEqual(recon["legacy_table"], "prospective_signals (DEPRECATED / NON-AUTHORITATIVE)")

        # Step 2: Corrupt legacy row (decision divergence)
        with self.manager._get_connection() as conn:
            conn.execute(
                "UPDATE prospective_signals SET signal = 'BUY', signal_reference_price = 350.00 WHERE symbol = 'AAPL';"
            )
            conn.commit()

        recon_diverged = self.manager.reconcile_legacy_and_immutable_ledgers("AAPL")
        self.assertFalse(recon_diverged["reconciled"])
        self.assertGreaterEqual(recon_diverged["discrepancies_count"], 2)
        disc_types = [d["type"] for d in recon_diverged["discrepancies"]]
        self.assertIn("DECISION_MISMATCH", disc_types)
        self.assertIn("REFERENCE_PRICE_MISMATCH", disc_types)


if __name__ == "__main__":
    unittest.main()
