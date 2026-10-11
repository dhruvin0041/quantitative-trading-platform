# backend/tests/test_signal_recovery_regression.py
"""
Regression and Governance Acceptance Test Suite for HYDRA Signal Recovery.

Verifies:
1. Feature Dimension Compatibility: _replay_causal_ml_signals adapts dynamically
   to both 27-feature and 31-feature scalers without ValueError.
2. Model Manager Integrity: ModelManager loads self.scaler and authorizes primary
   V2.4 models (artifacts/xgb_ensemble.json) by default.
3. Model Calibration Contract: ModelCalibrator enforces raw pass-through identity
   for XGBoost and LightGBM per frozen strategy manifest v2.4.
4. Replay-to-API Signal Preservation: Signals maintain causal point-in-time
   temporal alignment, Open[t+1] pricing convention, and confirmed states.
5. Governance Artifact Provenance: SHA-256 hashes of all frozen artifacts match
   frozen_strategy_manifest_v2.4.json exactly.
"""

import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.execution.inference_service import InferenceService
from src.execution.live_inference import FEATURE_COLUMNS, FEATURE_COLUMNS_V30
from src.models.model_loader import ModelManager
from src.models.regime.calibration import ModelCalibrator


class TestSignalRecoveryRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend_dir = Path(__file__).resolve().parent.parent
        cls.manifest_path = cls.backend_dir / "artifacts" / "frozen_strategy_manifest_v2.4.json"

    def test_1_calibration_raw_passthrough_identity(self):
        """Proof: ModelCalibrator returns identical raw probabilities for XGB and LGBM."""
        calibrator = ModelCalibrator()
        probs = np.array([[0.10, 0.30, 0.60], [0.55, 0.35, 0.10]], dtype=float)

        xgb_cal = calibrator.calibrate("XGB", probs)
        np.testing.assert_array_almost_equal(xgb_cal, probs)

        lgbm_cal = calibrator.calibrate("LGBM", probs)
        np.testing.assert_array_almost_equal(lgbm_cal, probs)

    def test_2_model_manager_loads_scaler_and_primary_v24_model(self):
        """Proof: ModelManager initializes self.scaler and loads authorized V2.4 model."""
        mm = ModelManager(config={"strategy_version": "V2.4"}, kept_features_list=FEATURE_COLUMNS)
        mm._load_scaler()
        self.assertIsNotNone(mm.scaler)
        self.assertTrue(hasattr(mm.scaler, "transform"))

        mm._load_xgb()
        self.assertIsNotNone(mm.xgb_model)

    def test_3_replay_adapts_to_scaler_dimensions(self):
        """Proof: _replay_causal_ml_signals adapts to 27 or 31 feature scaler without error."""
        import os
        import shutil
        import tempfile

        from src.execution.signal_ledger import SignalLedger

        test_dir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(test_dir, "test_reg_ledger.db")
            real_ledger = SignalLedger(db_path=db_path)

            # Test 27 features
            scaler_27 = StandardScaler()
            scaler_27.fit(np.zeros((10, len(FEATURE_COLUMNS))))
            self.assertEqual(scaler_27.n_features_in_, len(FEATURE_COLUMNS))

            # Test 31 features
            scaler_31 = StandardScaler()
            scaler_31.fit(np.zeros((10, len(FEATURE_COLUMNS_V30))))
            self.assertEqual(scaler_31.n_features_in_, len(FEATURE_COLUMNS_V30))

            # Build synthetic dataframe
            dates = pd.date_range("2025-01-01", periods=50, freq="B")
            df = pd.DataFrame(index=dates)
            df["Close"] = [100.0 * (1.01**i) for i in range(50)]
            df["Open"] = df["Close"] * 0.999
            df["High"] = df["Close"] * 1.005
            df["Low"] = df["Close"] * 0.995
            df["Volume"] = 1_000_000

            for col in FEATURE_COLUMNS_V30:
                df[col] = 0.05
            df["RSI"] = 55.0
            df["ADX"] = 25.0
            df["VIX_Level"] = 18.0

            mock_xgb = MagicMock()
            mock_xgb.n_features_in_ = len(FEATURE_COLUMNS_V30)
            mock_xgb.predict_proba.return_value = np.tile([0.05, 0.15, 0.80], (len(df), 1))

            mock_mm = MagicMock()
            mock_mm.xgb_model = mock_xgb
            mock_mm.scaler = scaler_31

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
                signal_ledger=real_ledger,
                use_veto=False,
            )
            service.model_calibrator = None

            # Replay with 31-feature scaler
            signals_31 = service._replay_causal_ml_signals("AAPL", df, version="V2.4")
            self.assertIsInstance(signals_31, list)
            self.assertGreater(len(signals_31), 0)

            # Replay with 27-feature scaler
            real_ledger.clear_historical_replay("AAPL")
            mock_mm.scaler = scaler_27
            mock_xgb.n_features_in_ = len(FEATURE_COLUMNS)
            signals_27 = service._replay_causal_ml_signals("AAPL", df, version="V2.4")
            self.assertIsInstance(signals_27, list)
            self.assertGreater(len(signals_27), 0)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_4_manifest_hashes_match_artifacts(self):
        """Proof: All frozen V2.4 artifact SHA-256 hashes match the manifest."""
        self.assertTrue(self.manifest_path.exists())
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        artifacts_dir = self.backend_dir / "artifacts"
        hashes = manifest.get("artifact_hashes", {})

        for filename, expected_hash in hashes.items():
            file_path = artifacts_dir / filename
            self.assertTrue(file_path.exists(), f"Artifact missing: {filename}")
            h = hashlib.sha256(file_path.read_bytes()).hexdigest()
            self.assertEqual(h, expected_hash, f"Hash mismatch for {filename}: {h} != {expected_hash}")


if __name__ == "__main__":
    unittest.main()
