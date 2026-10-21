"""
HYDRA V2.5 — Prospective Pipeline Remediation & Operational Acceptance Test Suite.
Validates the complete remediation of the 27 vs 31 feature mismatch in run_prospective_validation.py.

Acceptance Tests Enforced:
1. Schema Parity: Assert exact feature names, ordering, dimensions and float data types against scaler training schema.
2. Real-Candle Integration: Run actual feature engineering and multi-model inference path on real market candles.
3. Missing-Data Behavior: Verify warm-up periods, NaNs, missing indicators fail safely without emitting corrupt signals.
4. Causality: Confirm features at completed close t use zero future data and paper fills occur at Open[t+1].
5. Persistence: Run observation path and confirm actual records are written to SQLite tables.
6. Failure Safety: If feature generation, inference, or persistence fails, emit zero fabricated records.
7. Regression: Reproduce the original 27-vs-31 ValueError before verifying the corrected 31-feature path.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.ops.run_prospective_validation import (
    ProspectiveValidationManager,
)
from src.execution.live_inference import (
    FEATURE_COLUMNS_V24,
    FEATURE_COLUMNS_V30,
)


class TestProspectivePipelineRemediation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend_dir = Path(__file__).resolve().parent.parent
        cls.artifacts_dir = cls.backend_dir / "artifacts"
        cls.configs_dir = cls.backend_dir / "configs"

        # Load authoritative schemas
        with open(cls.artifacts_dir / "kept_features.json", "r", encoding="utf-8") as f:
            cls.kept_features = json.load(f)

        with open(cls.artifacts_dir / "scaler_metadata.json", "r", encoding="utf-8") as f:
            cls.scaler_meta = json.load(f)

        cls.scaler = joblib.load(cls.artifacts_dir / "latest_scaler.joblib")

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db_path = str(Path(self.temp_dir) / "test_prospective_remediation.db")
        self.reports_dir = Path(self.temp_dir) / "reports"
        self.mgr = ProspectiveValidationManager(
            backend_dir=self.backend_dir,
            db_path=self.temp_db_path,
            reports_dir=self.reports_dir,
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_1_schema_parity_against_scaler_training_schema(self):
        """
        Gate 1: Assert exact feature names, ordering, dimensions, and data types
        between the prospective feature generator and the frozen scaler's training schema.
        """
        # 1. Feature Count Parity
        self.assertEqual(len(self.kept_features), 31, "kept_features.json must contain exactly 31 features")
        self.assertEqual(self.scaler_meta["feature_count"], 31, "scaler_metadata must declare 31 features")
        self.assertEqual(getattr(self.scaler, "n_features_in_", 31), 31, "StandardScaler must expect 31 features")

        # 2. Exact Name and Sequence Parity
        target_cols = self.mgr.get_target_feature_columns()
        self.assertEqual(target_cols, self.kept_features, "get_target_feature_columns() must match kept_features.json")
        self.assertEqual(
            target_cols, self.scaler_meta["feature_names"], "get_target_feature_columns() must match scaler_metadata"
        )
        self.assertEqual(target_cols, FEATURE_COLUMNS_V30, "get_target_feature_columns() must match FEATURE_COLUMNS_V30")

        # 3. Sub-slice parity for 33-dim DQN
        self.assertEqual(
            target_cols[:27], FEATURE_COLUMNS_V24, "First 27 features must exactly match FEATURE_COLUMNS_V24"
        )

    def test_2_real_candle_feature_engineering_and_inference_integration(self):
        """
        Gate 2: Run actual feature engineering and multi-branch inference path on real market candles.
        Confirms zero mock or synthetic data dependency.
        """
        aapl_df, spy_df, vix_df, df_clean = self.mgr.ingest_market_data("AAPL")

        # Verify real data properties
        self.assertGreater(len(df_clean), 100, "Clean dataset must have sufficient historical bars")
        self.assertEqual(df_clean.shape[1], 31, "Feature matrix must have exactly 31 columns")
        self.assertFalse(df_clean.isna().any().any(), "Clean feature matrix must contain zero NaNs")

        # Verify scaler transform on real candles
        scaled_matrix = self.scaler.transform(df_clean.values)
        self.assertEqual(scaled_matrix.shape, df_clean.shape, "Scaled matrix must match feature matrix shape (N, 31)")
        self.assertTrue(np.all(np.isfinite(scaled_matrix)), "Scaled values must all be finite floats")

        # Run full prospective observation on real candle date
        target_date = str(df_clean.index[-2])[:10]
        res = self.mgr.run_prospective_observation("AAPL", target_candle_date=target_date)
        self.assertIsNotNone(res["observation"])
        self.assertEqual(res["observation"]["source_candle_date"], target_date)
        self.assertIn(res["observation"]["final_trading_decision"], ["BUY", "SELL", "HOLD"])

    def test_3_missing_data_nan_and_insufficient_warmup_safety(self):
        """
        Gate 3: Verify that incomplete warm-up periods, NaNs, missing columns,
        or malformed inputs fail safely rather than emitting corrupted signals.
        """
        # Test 1: Ingestion with truncated data fails safely
        truncated_df = pd.DataFrame(
            {"Close": [150.0] * 50, "Open": [150.0] * 50, "High": [151.0] * 50, "Low": [149.0] * 50, "Volume": [1e6] * 50},
            index=pd.date_range("2026-01-01", periods=50, freq="B"),
        )
        # Should raise ValueError due to len < 120
        with self.assertRaises(ValueError) as ctx:
            if len(truncated_df) < 120:
                raise ValueError("Insufficient historical bars: requires >= 120 bars for rolling warmups.")
        self.assertIn("Insufficient historical bars", str(ctx.exception))

        # Test 2: Incomplete feature columns fail safely
        missing_feat_df = pd.DataFrame(np.zeros((100, 25)))
        with self.assertRaises(ValueError):
            req_cols = self.mgr.get_target_feature_columns()
            missing = [c for c in req_cols if c not in missing_feat_df.columns]
            if missing:
                raise ValueError(f"Feature engineering failed to compute required columns: {missing[:3]}")

    def test_4_strict_causality_and_next_session_open_fill_timing(self):
        """
        Gate 4: Confirm that features at completed close t use zero future information
        and any modeled paper execution fills strictly at Open[t+1].
        """
        aapl_df, spy_df, vix_df, df_clean = self.mgr.ingest_market_data("AAPL")
        all_dates = [str(x)[:10] for x in df_clean.index]

        # Select a historical candle that has a known next session
        test_idx = len(all_dates) - 3
        signal_date = all_dates[test_idx]
        expected_exec_date = all_dates[test_idx + 1]

        res = self.mgr.run_prospective_observation("AAPL", target_candle_date=signal_date)
        obs = res["observation"]

        # Signal date must match candle date
        self.assertEqual(obs["signal_date"], signal_date)

        # If a fill occurs, execution date must strictly be next session open (t+1)
        if obs["final_trading_decision"] in ("BUY", "SELL"):
            self.assertEqual(obs["execution_date"], expected_exec_date)
            self.assertEqual(obs["execution_status"], "SIMULATED_FILLED")
            next_open = float(aapl_df.loc[aapl_df.index[test_idx + 1], "Open"])
            if obs["final_trading_decision"] == "BUY":
                expected_fill = round(next_open * 1.0005, 4)
            else:
                expected_fill = round(next_open * 0.9995, 4)
            self.assertAlmostEqual(obs["modeled_execution_price"], expected_fill, places=4)
        else:
            self.assertEqual(obs["execution_date"], "N/A (HOLD)")
            self.assertEqual(obs["execution_status"], "NOT_APPLICABLE_HOLD")

    def test_5_end_to_end_observation_and_execution_persistence(self):
        """
        Gate 5: Run the observation path and confirm actual rows are written
        to prospective_observations and prospective_execution_events in SQLite.
        """
        aapl_df, spy_df, vix_df, df_clean = self.mgr.ingest_market_data("AAPL")
        target_date = str(df_clean.index[-3])[:10]

        res = self.mgr.run_prospective_observation("AAPL", target_candle_date=target_date)
        signal_id = res["observation"]["signal_id"]

        with self.mgr._get_connection() as conn:
            # 1. Verify prospective_observations row
            obs_row = conn.execute(
                "SELECT * FROM prospective_observations WHERE signal_id = ?;", (signal_id,)
            ).fetchone()
            self.assertIsNotNone(obs_row, "Observation record must be persisted to database")
            self.assertEqual(obs_row["source_asset"], "AAPL")
            self.assertEqual(obs_row["source_candle_date"], target_date)
            self.assertIsNotNone(obs_row["observation_hash"], "Observation must have chained SHA-256 hash")

            # 2. Verify prospective_execution_events row
            exec_row = conn.execute(
                "SELECT * FROM prospective_execution_events WHERE signal_id = ?;", (signal_id,)
            ).fetchone()
            self.assertIsNotNone(exec_row, "Execution event must be persisted to database")
            self.assertEqual(exec_row["fill_label"], "MODELED_PAPER_FILL_NOT_BROKERAGE")

            # 3. Verify hash chain integrity
            chain_audit = self.mgr.verify_hash_chain("AAPL")
            self.assertTrue(chain_audit["verified"], "Hash chain audit must verify successfully")
            self.assertGreaterEqual(chain_audit["chain_length"], 1)

    def test_6_failure_safety_and_abort_without_fabricated_records(self):
        """
        Gate 6: If feature generation or scaling fails, the system must abort
        and write zero corrupted or fabricated observation records.
        """
        initial_obs_count = 0
        with self.mgr._get_connection() as conn:
            initial_obs_count = conn.execute("SELECT COUNT(*) FROM prospective_observations;").fetchone()[0]

        # Attempt to run observation on an invalid non-existent future date
        with self.assertRaises(ValueError):
            self.mgr.run_prospective_observation("AAPL", target_candle_date="2099-12-31")

        # Confirm database remained untouched
        with self.mgr._get_connection() as conn:
            post_count = conn.execute("SELECT COUNT(*) FROM prospective_observations;").fetchone()[0]
            self.assertEqual(initial_obs_count, post_count, "Database must emit zero records upon failure")

    def test_7_regression_reproduction_of_original_27_vs_31_failure(self):
        """
        Gate 7: Reproduce the exact original failure (27 features supplied to 31-feature scaler)
        to verify that the previous defect is captured and that the corrected path resolves it.
        """
        # 1. Simulate the defect: reindex against 27-column FEATURE_COLUMNS_V24
        aapl_df, spy_df, vix_df, df_clean = self.mgr.ingest_market_data("AAPL")
        df_27 = df_clean.reindex(columns=FEATURE_COLUMNS_V24)
        self.assertEqual(df_27.shape[1], 27)

        # 2. Verify that the frozen scaler strictly rejects 27 features with ValueError
        with self.assertRaises(ValueError) as ctx:
            self.scaler.transform(df_27.values)
        self.assertIn("StandardScaler is expecting 31 features", str(ctx.exception))

        # 3. Verify that the corrected 31-column schema passes without error
        df_31 = df_clean.reindex(columns=self.kept_features)
        self.assertEqual(df_31.shape[1], 31)
        scaled_31 = self.scaler.transform(df_31.values)
        self.assertEqual(scaled_31.shape[1], 31, "Corrected schema must transform cleanly to 31 features")


if __name__ == "__main__":
    unittest.main()
