"""
HYDRA V2.2 — Prospective Operations Unit & Integration Test Suite.
Validates prospective operational integrity, ledger immutability,
execution cost modeling, and checkpoint tracking under frozen conditions.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.ops.run_prospective_validation import ProspectiveValidationManager


class TestProspectiveOperations(unittest.TestCase):
    def setUp(self):
        self.backend_dir = Path(__file__).resolve().parent.parent
        self.temp_dir = tempfile.mkdtemp()
        self.mgr = ProspectiveValidationManager(backend_dir=self.backend_dir)

    def tearDown(self):
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_1_frozen_configuration_verification(self):
        """Verifies that all model and config hashes match the frozen V2.2 manifest."""
        cfg = self.mgr.verify_frozen_configuration()
        self.assertEqual(cfg["strategy_version"], "HYDRA_PROSPECTIVE_V2.2")
        self.assertTrue(cfg["models_valid"], f"Model hash mismatch: {cfg['model_hashes']}")
        self.assertTrue(cfg["configs_valid"], f"Config hash mismatch: {cfg['config_hashes']}")
        self.assertEqual(
            cfg["manifest_sha256"],
            "e34d9d472505d6b2d7a79924cbdf1fc6c0aa2346f67c72f9c0afd94b5352b195",
        )
        self.assertEqual(cfg["active_primary_model"], "XGB_AGENT (XGBoost Classifier)")

    def test_2_prospective_ledger_field_completeness(self):
        """Verifies that the prospective ledger table contains all required operational fields."""
        with self.mgr._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(prospective_observations);")
            cols = [row["name"] for row in cursor.fetchall()]

        required_cols = [
            "signal_id",
            "source_asset",
            "source_candle_date",
            "signal_date",
            "execution_date",
            "candle_finalization_timestamp",
            "data_ingestion_timestamp",
            "feature_computation_timestamp",
            "signal_generation_timestamp",
            "order_submission_timestamp",
            "execution_timestamp",
            "model_prediction_probabilities",
            "individual_model_predictions",
            "primary_model_prediction",
            "veto_result",
            "macro_filter_result",
            "final_trading_decision",
            "signal_reference_price",
            "modeled_execution_price",
            "modeled_entry_price",
            "modeled_exit_price",
            "slippage_amount",
            "commission_amount",
            "position_size",
            "execution_status",
            "manifest_hash",
        ]
        for col in required_cols:
            self.assertIn(col, cols, f"Mandated column {col} missing from prospective_observations table.")

    def test_3_hold_decision_recorded_with_date_separation(self):
        """Verifies that HOLD decisions are preserved in the ledger with distinct dates."""
        with self.mgr._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM prospective_observations
                WHERE source_asset = 'AAPL' AND source_candle_date = '2026-10-01';
                """
            ).fetchall()

        self.assertGreaterEqual(len(rows), 1, "Prospective observation for 2026-10-01 must exist.")
        r = dict(rows[0])
        self.assertEqual(r["final_trading_decision"], "HOLD")
        self.assertEqual(r["signal_date"], "2026-10-01")
        self.assertEqual(r["execution_date"], "N/A (HOLD)")
        self.assertEqual(r["execution_status"], "NOT_APPLICABLE_HOLD")
        self.assertEqual(r["manifest_hash"], "e34d9d472505d6b2d7a79924cbdf1fc6c0aa2346f67c72f9c0afd94b5352b195")

        # Verify model prediction probabilities are recorded as valid JSON
        probs = json.loads(r["model_prediction_probabilities"])
        self.assertIn("XGBoost", probs)
        self.assertIn("LightGBM", probs)
        self.assertIn("DL_Fusion_Raw", probs)
        self.assertIn("DL_Fusion_Calibrated", probs)
        self.assertIn("DQN_Q_Values", probs)
        self.assertIn("Meta_Ensemble", probs)

    def test_4_checkpoint_tracking_readiness(self):
        """Verifies checkpoint tracking and ensures incomplete trades are excluded from metrics."""
        perf = self.mgr.compute_prospective_performance_and_checkpoints(ticker="AAPL")
        self.assertIn("checkpoints", perf)
        checkpoints = perf["checkpoints"]

        for cp_key in [
            "checkpoint_1_10_trades",
            "checkpoint_2_30_trades",
            "checkpoint_3_50_trades",
            "checkpoint_4_100_trades",
        ]:
            self.assertIn(cp_key, checkpoints)
            self.assertFalse(checkpoints[cp_key]["reached"])
            self.assertEqual(checkpoints[cp_key]["completed"], 0)

        # Incomplete trades must not be counted
        self.assertEqual(perf["completed_trades_count"], 0)
        self.assertIsNone(perf["win_rate_pct"])
        self.assertIsNone(perf["profit_factor"])


if __name__ == "__main__":
    unittest.main()
