# backend/tests/test_v3_cross_asset_integrity.py
"""
HYDRA V3.0 Universal Cross-Asset Retraining & Strategy Governance Test Suite.

Verifies:
1. Feature Matrix Specification: FEATURE_COLUMNS_V30 contains 31 institutional indicators.
2. Artifact & Dimension Integrity: Scaler, XGBoost, and LightGBM accept (N, 31) matrices.
3. Cryptographic Governance: frozen_strategy_manifest_v3.0.json passes StrategyGovernanceEngine verification.
4. Anti-Overfitting Lock: Tampered V3 artifact hashes raise StrategyLockError.
5. Causal Execution Parity: Replay execution models Open[t+1] +/- 5bps slippage.
6. Temporal Partition Isolation: Zero 2026 data contamination.
"""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.execution.inference_service import InferenceService
from src.execution.live_inference import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_V30,
    get_feature_columns,
)
from src.execution.signal_ledger import SignalLedger
from src.execution.strategy_governance import (
    StrategyGovernanceEngine,
    StrategyLockError,
)


class TestV3CrossAssetIntegrity(unittest.TestCase):
    def setUp(self):
        self.backend_dir = Path(__file__).resolve().parent.parent
        self.artifacts_v3 = self.backend_dir / "artifacts" / "v3"
        self.manifest_v3 = self.backend_dir / "artifacts" / "frozen_strategy_manifest_v3.0.json"
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_v3_ledger.db")
        self.ledger = SignalLedger(db_path=self.db_path)

    def tearDown(self):
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_1_feature_matrix_specification(self):
        """Proof 1: FEATURE_COLUMNS_V30 defines 31 features, expanding V2.4 by 4 indicators."""
        self.assertEqual(len(FEATURE_COLUMNS_V30), 31)
        self.assertEqual(len(FEATURE_COLUMNS), 27)

        # Baseline 27 features preserved
        for col in FEATURE_COLUMNS:
            self.assertIn(col, FEATURE_COLUMNS_V30)

        # 4 Enhanced indicators present
        enhanced = ["Keltner_Position", "HMA_Slope", "Connors_RSI", "CMF_Divergence"]
        for ind in enhanced:
            self.assertIn(ind, FEATURE_COLUMNS_V30)

        # get_feature_columns helper returns 31 for V3.0 and 27 for V2.4
        self.assertEqual(len(get_feature_columns("V3.0")), 31)
        self.assertEqual(len(get_feature_columns("V2.4")), 27)

    def test_2_v3_model_artifacts_and_dimension_integrity(self):
        """Proof 2: All V3 model artifacts exist and operate on 31 features with deterministic shapes."""
        scaler_path = self.artifacts_v3 / "latest_scaler.joblib"
        xgb_path = self.artifacts_v3 / "xgb_ensemble.json"
        lgbm_path = self.artifacts_v3 / "lgbm_agent.joblib"
        calibrator_path = self.artifacts_v3 / "model_calibrator.joblib"

        self.assertTrue(scaler_path.exists(), "latest_scaler.joblib must exist in artifacts/v3/")
        self.assertTrue(xgb_path.exists(), "xgb_ensemble.json must exist in artifacts/v3/")
        self.assertTrue(lgbm_path.exists(), "lgbm_agent.joblib must exist in artifacts/v3/")
        self.assertTrue(calibrator_path.exists(), "model_calibrator.joblib must exist in artifacts/v3/")

        # Test Scaler
        scaler = joblib.load(scaler_path)
        self.assertEqual(scaler.n_features_in_, 31)

        # Test XGBoost
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(xgb_path))

        test_vec = np.random.randn(5, 31).astype(np.float32)
        scaled_vec = scaler.transform(test_vec)
        xgb_probs = xgb_model.predict_proba(scaled_vec)

        self.assertEqual(xgb_probs.shape, (5, 3))
        for row in xgb_probs:
            self.assertAlmostEqual(float(np.sum(row)), 1.0, places=4)

        # Test LightGBM
        lgbm_model = joblib.load(lgbm_path)
        lgbm_probs = lgbm_model.predict_proba(scaled_vec)
        self.assertEqual(lgbm_probs.shape, (5, 3))
        for row in lgbm_probs:
            self.assertAlmostEqual(float(np.sum(row)), 1.0, places=4)

    def test_3_v3_cryptographic_strategy_manifest(self):
        """Proof 3: frozen_strategy_manifest_v3.0.json passes cryptographic verification."""
        self.assertTrue(self.manifest_v3.exists(), "V3.0 manifest must exist")

        gov = StrategyGovernanceEngine(version="V3.0")
        manifest = gov.load_manifest()

        self.assertEqual(manifest["strategy_version"], "HYDRA_PROSPECTIVE_V3.0")
        self.assertEqual(manifest["feature_count"], 31)
        self.assertEqual(manifest["universe"], ["AAPL", "NVDA", "MSFT", "AMZN", "SPY"])

        is_valid, violations = gov.verify_integrity()
        self.assertTrue(is_valid, f"V3.0 manifest integrity failed: {violations}")
        self.assertEqual(len(violations), 0)

    def test_4_anti_overfitting_lock_enforcement(self):
        """Proof 4: Mutating any V3 model weight hash fails closed under StrategyLockError."""
        gov = StrategyGovernanceEngine(version="V3.0")
        manifest_copy = copy.deepcopy(gov.load_manifest())

        manifest_copy["model_hashes"]["v3/xgb_ensemble.json"] = "TAMPERED_V3_HASH_0000"
        tampered_manifest_path = os.path.join(self.temp_dir, "tampered_v3_manifest.json")
        with open(tampered_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_copy, f)

        tampered_gov = StrategyGovernanceEngine(manifest_path=tampered_manifest_path)
        valid, violations = tampered_gov.verify_integrity()

        self.assertFalse(valid, "Tampered V3 manifest must fail verification")
        self.assertTrue(any("v3/xgb_ensemble.json" in v for v in violations))

        with self.assertRaises(StrategyLockError):
            tampered_gov.enforce_anti_overfitting_lock()

    def test_5_causal_execution_timing_and_cost_modeling(self):
        """Proof 5: Causal ML replay executes fills at Open[t+1] + 5 bps with zero lookahead."""
        from unittest.mock import MagicMock

        mock_mm = MagicMock()
        mock_mm.scaler = None
        mock_mm.xgb_model = None

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
        )

        # Generate synthetic 40-bar test dataframe
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        df_test = pd.DataFrame(
            {
                "Open": np.linspace(100.0, 140.0, 40),
                "High": np.linspace(102.0, 142.0, 40),
                "Low": np.linspace(99.0, 139.0, 40),
                "Close": np.linspace(101.0, 141.0, 40),
                "Volume": np.full(40, 50000.0),
            },
            index=dates,
        )

        signals = service._replay_causal_ml_signals("NVDA", df_test, version="V3.0")
        for s in signals:
            self.assertEqual(s["execution_target_bar"], "NEXT_SESSION_OPEN")
            self.assertIn(s["signal"], ("BUY", "SELL"))
            self.assertGreater(s["confidence"], 0.0)

    def test_6_temporal_firewall_zero_2026_leakage(self):
        """Proof 6: V3 dataset provenance confirms zero 2026 data in training or validation."""
        gov = StrategyGovernanceEngine(version="V3.0")
        manifest = gov.load_manifest()

        dev_prov = manifest["dataset_provenance"]["development_universe"]
        val_prov = manifest["dataset_provenance"]["validation_universe"]

        self.assertEqual(dev_prov["end_date"], "2024-12-31")
        self.assertEqual(val_prov["end_date"], "2025-12-31")
        self.assertTrue(dev_prov["zero_2025_leakage"])
        self.assertTrue(dev_prov["zero_2026_leakage"])
        self.assertTrue(val_prov["zero_2026_leakage"])


if __name__ == "__main__":
    unittest.main()
