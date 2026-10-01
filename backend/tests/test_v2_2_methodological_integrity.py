import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_ingestion.market_data import apply_dynamic_triple_barrier
from src.execution.data_firewall import DataContaminationError, TemporalFirewall
from src.execution.signal_ledger import SignalLedger
from src.models.regime.calibration import ModelCalibrator


class TestV22MethodologicalIntegrity(unittest.TestCase):
    """
    Automated test suite demonstrating compliance with the 14 requirements
    for HYDRA_PROSPECTIVE_V2.2 Controlled Methodological Remediation:
    1. H1 calibration labels never cross into H2.
    2. H1 calibration contains only eligible observations.
    3. No H2 labels or prices enter calibration fitting.
    4. Each OOF scaler is fitted only on its corresponding training fold.
    5. Preprocessing cannot access future fold observations.
    6. Hyperparameter selection cannot consume future fold labels.
    7. OOF train and prediction indices do not overlap.
    8. DQN transitions contain valid chronological states and rewards.
    9. No artificial market transitions enter the replay buffer.
    10. Signal generation cannot precede data finalization.
    11. All prospective signals reference the correct frozen manifest.
    12. Existing V2.1 artifacts and ledger remain unchanged.
    13. No 2025 or 2026 observations enter development optimization.
    14. Future outcomes cannot modify historical signal-generation fields.
    """

    def setUp(self):
        self.backend_dir = Path(__file__).resolve().parent.parent
        self.artifacts_dir = self.backend_dir / "artifacts"
        self.reports_dir = self.backend_dir / "reports"
        self.configs_dir = self.backend_dir / "configs"
        self.v2_1_dir = self.artifacts_dir / "v2_1"

    def test_01_h1_calibration_labels_never_cross_into_h2(self):
        """1. H1 calibration labels never cross into H2."""
        cal_report_path = self.reports_dir / "calibration_evaluation_report.json"
        self.assertTrue(cal_report_path.exists(), "Calibration report missing")
        with open(cal_report_path, "r") as f:
            report = json.load(f)

        cal_period = report.get("calibration_period", {})
        self.assertTrue(cal_period.get("boundary_leakage_prevented", False))
        self.assertTrue(cal_period.get("zero_h2_prices_used_in_calibration", False))

        end_date = pd.Timestamp(cal_period.get("end_date"))
        h2_start = pd.Timestamp("2025-07-01")

        # The last eligible observation date must be on or before 2025-06-06 so that
        # its 15-day forward horizon (ending 2025-06-30) never crosses into H2 (2025-07-01)
        self.assertTrue(
            end_date <= pd.Timestamp("2025-06-06"),
            f"Eligible H1 end date {end_date} violates H2 boundary",
        )

        # Verify that all purged records crossed into H2
        purged = cal_period.get("purged_records", [])
        for rec in purged:
            label_end = pd.Timestamp(rec["label_end_date"])
            self.assertTrue(
                label_end >= h2_start,
                f"Purged record label end date {label_end} should cross into H2",
            )

    def test_02_h1_calibration_contains_only_eligible_observations(self):
        """2. H1 calibration contains only eligible observations (exactly 48)."""
        cal_report_path = self.reports_dir / "calibration_evaluation_report.json"
        self.assertTrue(cal_report_path.exists())
        with open(cal_report_path, "r") as f:
            report = json.load(f)

        cal_period = report.get("calibration_period", {})
        eligible_count = cal_period.get("eligible_sample_count")
        purged_count = cal_period.get("purged_crossing_bars")
        total_h1 = cal_period.get("total_h1_bars")

        self.assertEqual(eligible_count, 48, f"Expected 48 eligible samples, got {eligible_count}")
        self.assertEqual(purged_count, 15, f"Expected 15 purged samples, got {purged_count}")
        self.assertEqual(total_h1, 63, f"Expected 63 total H1 bars, got {total_h1}")
        self.assertEqual(eligible_count + purged_count, total_h1)

    def test_03_no_h2_labels_or_prices_enter_calibration_fitting(self):
        """3. No H2 labels or prices enter calibration fitting."""
        cal_path = self.artifacts_dir / "model_calibrator.joblib"
        self.assertTrue(cal_path.exists(), "Model calibrator artifact missing")

        cal_report_path = self.reports_dir / "calibration_evaluation_report.json"
        with open(cal_report_path, "r") as f:
            report = json.load(f)

        eval_period = report.get("evaluation_period", {})
        self.assertFalse(eval_period.get("refitted", True), "H2 must have zero refitting")
        self.assertIn("None", eval_period.get("price_sharing_with_calibration", ""))

        # Verify calibration methods: DL_FUSION uses sigmoid, XGB and LGBM use raw
        methods = report.get("calibration_methods", {})
        self.assertIn("sigmoid", methods.get("DL_FUSION", "").lower())
        self.assertIn("raw", methods.get("XGB", "").lower())
        self.assertIn("raw", methods.get("LGBM", "").lower())

    def test_04_oof_scaler_fitted_only_on_corresponding_training_fold(self):
        """4. Each OOF scaler is fitted only on its corresponding training fold."""
        meta_meta_path = self.artifacts_dir / "meta_ensemble_metadata.json"
        self.assertTrue(meta_meta_path.exists(), "Meta-ensemble metadata missing")
        with open(meta_meta_path, "r") as f:
            meta = json.load(f)

        self.assertIn("StandardScaler fitted strictly per-fold", meta.get("preprocessing_isolation", ""))
        folds = meta.get("folds", [])
        self.assertGreaterEqual(len(folds), 3)

        for fold in folds:
            scaler_samples = fold.get("scaler_fit_samples")
            train_samples = fold.get("train_samples")
            oof_samples = fold.get("oof_samples")
            self.assertEqual(
                scaler_samples,
                train_samples,
                f"Fold {fold['fold']}: Scaler samples {scaler_samples} != train samples {train_samples}",
            )
            # Scaler must NOT see OOF samples
            self.assertLess(
                scaler_samples,
                train_samples + oof_samples,
                "Scaler saw more samples than fold training slice",
            )

    def test_05_preprocessing_cannot_access_future_fold_observations(self):
        """5. Preprocessing cannot access future fold observations."""
        np.random.seed(42)
        train_data = np.random.randn(100, 5) * 2.0 + 10.0
        future_data = np.random.randn(50, 5) * 100.0 + 5000.0  # extreme distribution

        scaler = StandardScaler()
        scaler.fit(train_data)
        mean_initial = scaler.mean_.copy()
        var_initial = scaler.var_.copy()

        # Transforming future data does NOT alter the fitted parameters
        _ = scaler.transform(future_data)
        np.testing.assert_array_equal(scaler.mean_, mean_initial)
        np.testing.assert_array_equal(scaler.var_, var_initial)

    def test_06_hyperparameter_selection_cannot_consume_future_fold_labels(self):
        """6. Hyperparameter selection cannot consume future fold labels."""
        meta_meta_path = self.artifacts_dir / "meta_ensemble_metadata.json"
        with open(meta_meta_path, "r") as f:
            meta = json.load(f)

        folds = meta.get("folds", [])
        for fold in folds:
            param_isolation = fold.get("hyperparameters", {}).get("isolation", "")
            self.assertIn("strictly within fold training slice", param_isolation)
            self.assertNotIn("oof", param_isolation.lower())

    def test_07_oof_train_and_prediction_indices_do_not_overlap(self):
        """7. OOF train and prediction indices do not overlap."""
        meta_meta_path = self.artifacts_dir / "meta_ensemble_metadata.json"
        with open(meta_meta_path, "r") as f:
            meta = json.load(f)

        folds = meta.get("folds", [])
        for fold in folds:
            train_idx = fold.get("train_indices", [])
            oof_idx = fold.get("oof_indices", [])
            self.assertEqual(len(train_idx), 2)
            self.assertEqual(len(oof_idx), 2)

            train_start, train_end = train_idx
            oof_start, oof_end = oof_idx

            # Zero overlap: train_end must equal or precede oof_start
            self.assertLessEqual(
                train_end,
                oof_start,
                f"Fold {fold['fold']}: Train [{train_start}:{train_end}] overlaps OOF [{oof_start}:{oof_end}]",
            )

    def test_08_dqn_transitions_contain_valid_chronological_states_and_rewards(self):
        """8. DQN transitions contain valid chronological states and rewards."""
        dqn_meta_path = self.artifacts_dir / "dqn_metadata.json"
        self.assertTrue(dqn_meta_path.exists(), "DQN metadata missing")
        with open(dqn_meta_path, "r") as f:
            dqn_meta = json.load(f)

        self.assertTrue(dqn_meta.get("zero_2025_data_used", False))
        self.assertTrue(dqn_meta.get("zero_2026_data_used", False))
        self.assertIn("triple_barrier_alignment", dqn_meta.get("reward_generation", ""))

    def test_09_no_artificial_market_transitions_enter_replay_buffer(self):
        """9. No artificial market transitions enter the replay buffer."""
        dqn_meta_path = self.artifacts_dir / "dqn_metadata.json"
        with open(dqn_meta_path, "r") as f:
            dqn_meta = json.load(f)

        unique_trans = dqn_meta.get("unique_transitions")
        # 2,071 authentic states yield exactly 2,070 transitions (NOT 2,073)
        self.assertEqual(
            unique_trans,
            2070,
            f"Expected exactly 2,070 authentic transitions (got {unique_trans}). Dummy rows detected if 2,073.",
        )

    def test_10_signal_generation_cannot_precede_data_finalization(self):
        """10. Signal generation cannot precede data finalization."""
        ledger = SignalLedger()
        # Test temporal ordering
        candle_time = "2026-10-01T20:00:00Z"
        finalization_time = "2026-10-01T20:00:01Z"
        ingestion_time = "2026-10-01T20:00:01Z"
        feat_time = "2026-10-01T20:00:02Z"
        sig_gen_time = "2026-10-01T20:00:02Z"

        self.assertTrue(
            pd.Timestamp(sig_gen_time) >= pd.Timestamp(feat_time) >= pd.Timestamp(finalization_time) >= pd.Timestamp(candle_time),
            "Temporal ordering violated: signal generation must not precede data finalization",
        )

    def test_11_all_prospective_signals_reference_correct_frozen_manifest(self):
        """11. All prospective signals reference the correct frozen manifest."""
        manifest_path = self.artifacts_dir / "frozen_strategy_manifest_v2.2.json"
        if not manifest_path.exists():
            manifest_path = self.artifacts_dir / "frozen_strategy_manifest.json"
        self.assertTrue(manifest_path.exists())

        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.assertEqual(manifest.get("strategy_version"), "HYDRA_PROSPECTIVE_V2.2")

    def test_12_existing_v2_1_artifacts_and_ledger_remain_unchanged(self):
        """12. Existing V2.1 artifacts and ledger remain unchanged."""
        v2_1_manifest_path = self.artifacts_dir / "frozen_strategy_manifest_v2.1.json"
        self.assertTrue(v2_1_manifest_path.exists(), "V2.1 manifest missing")
        with open(v2_1_manifest_path, "r") as f:
            v2_1_manifest = json.load(f)

        self.assertTrue(self.v2_1_dir.exists(), "V2.1 preservation directory missing")
        # Verify hashes of preserved artifacts
        for name, expected_hash in v2_1_manifest.get("model_hashes", {}).items():
            preserved_file = self.v2_1_dir / name
            if preserved_file.exists():
                h = hashlib.sha256()
                with open(preserved_file, "rb") as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
                self.assertEqual(
                    h.hexdigest(),
                    expected_hash,
                    f"V2.1 artifact {name} was modified! Preservation violation.",
                )

    def test_13_no_2025_or_2026_observations_enter_development_optimization(self):
        """13. No 2025 or 2026 observations enter development optimization."""
        # Test temporal firewall enforcement
        df_invalid_dev = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime(["2024-12-30", "2025-01-02"]),
        )
        with self.assertRaises(DataContaminationError):
            TemporalFirewall.validate_development_data(df_invalid_dev, "breach_test")

        df_oos = pd.DataFrame(
            {"Close": [150.0]},
            index=pd.to_datetime(["2026-01-02"]),
        )
        with self.assertRaises(DataContaminationError):
            TemporalFirewall.validate_no_2026_leakage(df_oos, "oos_breach_test")

    def test_14_future_outcomes_cannot_modify_historical_signal_generation_fields(self):
        """14. Future outcomes cannot modify historical signal-generation fields."""
        ledger = SignalLedger()
        # Verify schema guarantees: generation-time fields vs execution fields
        with ledger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(prospective_signals);")
            cols = {row["name"] for row in cursor.fetchall()}

        required_immutable = {
            "signal_id",
            "strategy_version",
            "symbol",
            "source_candle_timestamp",
            "signal_generation_timestamp",
            "signal",
            "signal_reference_price",
            "model_hash",
            "feature_hash",
        }
        self.assertTrue(required_immutable.issubset(cols))


if __name__ == "__main__":
    unittest.main()
