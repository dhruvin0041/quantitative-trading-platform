import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_ingestion.market_data import apply_dynamic_triple_barrier
from src.execution.data_firewall import DataContaminationError, TemporalFirewall
from src.execution.signal_ledger import SignalLedger


class TestV21MethodologyAndLeakage(unittest.TestCase):
    """
    Unit and integration test suite verifying HYDRA_PROSPECTIVE_V2.1 methodology corrections:
    1. Temporal firewall & boundaries
    2. Training leakage isolation
    3. Forward-label leakage prevention
    4. DQN training-period provenance (2016-2024 exclusively)
    5. Meta-ensemble walk-forward OOF verification (2016-2024 exclusively)
    6. Calibration isolation (H1 2025 fit vs H2 2025 independent eval)
    7. Scaler provenance (Asset-Specific AAPL scaler, 2,130 bars)
    8. Feature timestamp & VIX/SPY alignment
    9. Immutable ledger & prospective execution semantics
    10. Strategy versioning & preservation (V1.0, V2.0, V2.1)
    """

    def setUp(self):
        self.backend_dir = Path(__file__).resolve().parent.parent
        self.artifacts_dir = self.backend_dir / "artifacts"
        self.reports_dir = self.backend_dir / "reports"
        self.configs_dir = self.backend_dir / "configs"

    def test_temporal_firewall_boundaries(self):
        """Enforces strictly: 2016-2024 Dev, 2025 Val, 2026+ OOS."""
        # 2016-2024 passes dev
        dev_dates = pd.date_range("2016-01-04", "2024-12-30", freq="B")
        df_dev = pd.DataFrame({"Close": np.ones(len(dev_dates))}, index=dev_dates)
        TemporalFirewall.validate_development_data(df_dev, "valid_dev")

        # 2025 fails dev
        val_dates = pd.date_range("2025-01-02", "2025-12-30", freq="B")
        df_val = pd.DataFrame({"Close": np.ones(len(val_dates))}, index=val_dates)
        with self.assertRaises(DataContaminationError):
            TemporalFirewall.validate_development_data(df_val, "invalid_dev")

        # 2025 passes val
        TemporalFirewall.validate_validation_data(df_val, "valid_val")

        # 2026 fails val
        oos_dates = pd.date_range("2026-01-02", "2026-06-30", freq="B")
        df_oos = pd.DataFrame({"Close": np.ones(len(oos_dates))}, index=oos_dates)
        with self.assertRaises(DataContaminationError):
            TemporalFirewall.validate_validation_data(df_oos, "invalid_val")

        # 2026 fails no_2026_leakage
        with self.assertRaises(DataContaminationError):
            TemporalFirewall.validate_no_2026_leakage(df_oos, "hard_breach")

    def test_forward_label_leakage_barrier_drop(self):
        """Proves that triple barrier dropping ensures zero outcome leakage across boundaries."""
        horizon = 15
        dates = pd.date_range("2024-01-02", "2024-12-30", freq="B")
        n = len(dates)
        np.random.seed(42)
        close = 150.0 + np.cumsum(np.random.randn(n) * 1.5)
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close + 2.0,
                "Low": close - 2.0,
                "Close": close,
                "ATR": pd.Series(1.5, index=dates),
            },
            index=dates,
        )

        df_labeled = apply_dynamic_triple_barrier(
            df.copy(), tp_atr_multiplier=1.5, sl_atr_multiplier=2.0, horizon=horizon
        )

        self.assertEqual(len(df) - len(df_labeled), horizon)
        self.assertEqual(df_labeled.index[-1], dates[-(horizon + 1)])
        self.assertTrue(df_labeled.index[-1] <= pd.Timestamp("2024-12-30"))

    def test_scaler_provenance_and_architecture(self):
        """Verifies Asset-Specific AAPL StandardScaler with 2,130 bars."""
        import joblib

        scaler_path = self.artifacts_dir / "latest_scaler.joblib"
        self.assertTrue(scaler_path.exists())
        scaler = joblib.load(scaler_path)

        # Must have exactly 2,130 samples seen
        self.assertEqual(int(scaler.n_samples_seen_), 2130)
        # Must have exactly 27 features
        self.assertEqual(scaler.mean_.shape[0], 27)

        # Check metadata
        meta_path = self.artifacts_dir / "scaler_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["architecture"], "asset_specific_standard_scaler")
            self.assertEqual(meta["sample_count"], 2130)
            self.assertEqual(meta["feature_count"], 27)
            self.assertTrue(meta["zero_2025_leakage"])
            self.assertTrue(meta["zero_2026_leakage"])

    def test_dqn_training_period_and_metadata(self):
        """Verifies DQN was trained exclusively on 2016-2024 development transitions."""
        dqn_path = self.artifacts_dir / "dqn_model.pth"
        self.assertTrue(dqn_path.exists())

        meta_path = self.artifacts_dir / "dqn_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                dqn_meta = json.load(f)
            self.assertTrue(dqn_meta["zero_2025_data_used"])
            self.assertTrue(dqn_meta["zero_2026_data_used"])
            self.assertTrue(dqn_meta["training_period"]["end"] <= "2024-12-31")
            self.assertGreaterEqual(dqn_meta["unique_transitions"], 2000)

            # Check SHA-256 matches actual file
            with open(dqn_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(dqn_meta["sha256"], actual_hash)

    def test_meta_ensemble_walk_forward_oof_stacking(self):
        """Verifies Meta-Ensemble was fitted on chronological OOF predictions inside 2016-2024."""
        meta_path = self.artifacts_dir / "meta_ensemble.joblib"
        self.assertTrue(meta_path.exists())

        meta_file = self.artifacts_dir / "meta_ensemble_metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            self.assertTrue(meta["validation_2025_excluded"])
            self.assertEqual(
                meta["methodology"],
                "Expanding-Window Walk-Forward Out-of-Fold (OOF) Stacking",
            )
            self.assertGreaterEqual(meta["total_oof_samples"], 1000)

            # Check SHA-256 matches actual file
            with open(meta_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(meta["sha256"], actual_hash)

    def test_calibration_h1_h2_isolation(self):
        """Verifies probability calibrator was fitted on H1 2025 and independently evaluated on H2 2025."""
        cal_path = self.artifacts_dir / "model_calibrator.joblib"
        self.assertTrue(cal_path.exists())

        rep_path = self.reports_dir / "calibration_evaluation_report.json"
        if rep_path.exists():
            with open(rep_path) as f:
                rep = json.load(f)
            self.assertEqual(rep["calibration_period"]["split"], "H1_2025")
            self.assertTrue(rep["calibration_period"]["end_date"] <= "2025-06-30")
            self.assertEqual(
                rep["evaluation_period"]["split"], "H2_2025_INDEPENDENT"
            )
            self.assertTrue(rep["evaluation_period"]["start_date"] >= "2025-07-01")
            self.assertFalse(rep["evaluation_period"]["refitted"])
            self.assertIn("models", rep)
            self.assertIn("DL_FUSION", rep["models"])
            self.assertIn("XGB", rep["models"])
            self.assertIn("LGBM", rep["models"])

    def test_strategy_versioning_and_preservation(self):
        """Verifies that V1.0 and V2.0 manifests are preserved immutably and V2.1 exists."""
        v1_path = self.artifacts_dir / "frozen_strategy_manifest_v1.0.json"
        v2_path = self.artifacts_dir / "frozen_strategy_manifest_v2.0.json"
        v2_1_path = self.artifacts_dir / "frozen_strategy_manifest_v2.1.json"
        active_path = self.artifacts_dir / "frozen_strategy_manifest.json"

        self.assertTrue(v1_path.exists(), "V1.0 manifest must be preserved")
        self.assertTrue(v2_path.exists(), "V2.0 manifest must be preserved")
        self.assertTrue(v2_1_path.exists(), "V2.1 manifest must exist")
        self.assertTrue(active_path.exists(), "Active manifest must exist")

        with open(v1_path) as f:
            v1 = json.load(f)
        with open(v2_path) as f:
            v2 = json.load(f)
        with open(v2_1_path) as f:
            v2_1 = json.load(f)
        with open(active_path) as f:
            active = json.load(f)

        self.assertEqual(v1["strategy_version"], "HYDRA_PROSPECTIVE_V1.0")
        self.assertEqual(v2["strategy_version"], "HYDRA_PROSPECTIVE_V2.0")
        self.assertEqual(v2_1["strategy_version"], "HYDRA_PROSPECTIVE_V2.1")
        self.assertEqual(active["strategy_version"], "HYDRA_PROSPECTIVE_V2.1")

    def test_signal_ledger_v2_1_semantics(self):
        """Verifies prospective ledger semantics: reference price, null fills at gen time, 5bps slippage."""
        test_db = self.backend_dir / "artifacts" / "test_prospective_ledger.db"
        if test_db.exists():
            test_db.unlink()
        try:
            ledger = SignalLedger(db_path=str(test_db))

            # Record a prospective signal
            sig_id = ledger.record_prospective_signal(
            symbol="AAPL",
            source_candle_timestamp="2026-10-01 16:00:00 EDT",
            signal_generation_timestamp="2026-10-01 16:00:02 EDT",
            signal="BUY",
            probability=0.75,
            confidence=0.85,
            feature_hash="test_feat_hash",
            model_hash="test_model_hash",
            execution_target_timestamp="2026-10-02 09:30:00 EDT",
            signal_reference_price=150.0,
            strategy_version="HYDRA_PROSPECTIVE_V2.1",
        )

            self.assertIsNotNone(sig_id)

            with ledger._get_connection() as conn:
                sig = conn.execute("SELECT * FROM prospective_signals WHERE signal_id = ?", (sig_id,)).fetchone()
                self.assertIsNotNone(sig)
                self.assertEqual(sig["strategy_version"], "HYDRA_PROSPECTIVE_V2.1")
                self.assertEqual(sig["signal_reference_price"], 150.0)
                self.assertIsNone(sig["market_open_price"])
                self.assertIsNone(sig["modeled_fill_price"])

            # Simulate outcome evaluation at next day open
            price_df = pd.DataFrame(
                {"Open": [150.0, 151.0], "Close": [150.5, 152.0]},
                index=pd.to_datetime(["2026-10-01", "2026-10-02"]),
            )
            ledger.evaluate_prospective_outcomes("AAPL", price_df)

            with ledger._get_connection() as conn:
                sig_updated = conn.execute("SELECT * FROM prospective_signals WHERE signal_id = ?", (sig_id,)).fetchone()
                self.assertEqual(sig_updated["market_open_price"], 151.0)
                # BUY fill = Open * (1 + 0.0005) = 151.0 * 1.0005 = 151.0755
                self.assertAlmostEqual(sig_updated["modeled_fill_price"], round(151.0 * 1.0005, 4), places=4)
                self.assertEqual(sig_updated["slippage_assumption_bps"], 5.0)
                self.assertEqual(sig_updated["commission_assumption"], 0.005)
        finally:
            if test_db.exists():
                try:
                    test_db.unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
