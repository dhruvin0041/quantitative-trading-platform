from pathlib import Path

import numpy as np
import pytest

from src.execution.consensus_engine import WeightedConsensusEngine
from src.execution.live_inference import FEATURE_COLUMNS, FEATURE_COLUMNS_V30, load_config
from src.models.regime.calibration import ModelCalibrator
from src.models.rl.dqn_agent import DQNAgent


class TestAllModelsVerification:
    """Automated verification suite testing every model artifact for the Core + Tactical strategy."""

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        self.artifacts_dir = Path("backend/artifacts")
        if not self.artifacts_dir.exists():
            self.artifacts_dir = Path("artifacts")
        self.config = load_config()

    def test_scaler_artifact(self):
        """Verify scaler artifact integrity and feature dimensionality."""
        import joblib
        scaler_path = self.artifacts_dir / "latest_scaler.joblib"
        assert scaler_path.exists(), "latest_scaler.joblib must exist"
        scaler = joblib.load(scaler_path)
        n_features = getattr(scaler, "n_features_in_", 0)
        assert n_features in [len(FEATURE_COLUMNS), len(FEATURE_COLUMNS_V30)], (
            f"Scaler features ({n_features}) must match V2.4 (27) or V3.0 (31)"
        )
        assert len(scaler.mean_) == n_features
        assert len(scaler.scale_) == n_features

    def test_xgboost_alpha_agent(self):
        """Verify XGBoost alpha agent artifact, classes, and prediction outputs."""
        import xgboost as xgb
        xgb_path = self.artifacts_dir / "xgb_ensemble.json"
        assert xgb_path.exists(), "xgb_ensemble.json must exist"
        model = xgb.XGBClassifier()
        model.load_model(str(xgb_path))
        n_features = getattr(model, "n_features_in_", 0)
        assert n_features in [len(FEATURE_COLUMNS), len(FEATURE_COLUMNS_V30)]

        dummy_x = np.zeros((1, n_features))
        probs = model.predict_proba(dummy_x)[0]
        assert len(probs) == 3, "XGBoost must output [Sell, Hold, Buy] probabilities"
        assert abs(np.sum(probs) - 1.0) < 1e-4

    def test_lightgbm_risk_agent(self):
        """Verify LightGBM risk veto agent artifact and probability outputs."""
        import joblib
        lgbm_path = self.artifacts_dir / "lgbm_agent.joblib"
        assert lgbm_path.exists(), "lgbm_agent.joblib must exist"
        model = joblib.load(lgbm_path)
        n_features = getattr(model, "n_features_in_", 0)
        assert n_features in [len(FEATURE_COLUMNS), len(FEATURE_COLUMNS_V30)]

        dummy_x = np.zeros((1, n_features))
        probs = model.predict_proba(dummy_x)[0]
        assert len(probs) == 3, "LightGBM must output [Sell, Hold, Buy] probabilities"
        assert abs(np.sum(probs) - 1.0) < 1e-4

    def test_model_calibrator(self):
        """Verify isotonic ModelCalibrator behavior and probability preservation."""
        cal_path = self.artifacts_dir / "model_calibrator.joblib"
        assert cal_path.exists(), "model_calibrator.joblib must exist"
        calibrator = ModelCalibrator.load(str(cal_path))
        raw_p = np.array([0.15, 0.25, 0.60])
        cal_xgb = calibrator.calibrate("XGB", raw_p)
        cal_lgbm = calibrator.calibrate("LGBM", raw_p)

        assert len(cal_xgb) == 3
        assert len(cal_lgbm) == 3
        assert abs(np.sum(cal_xgb) - 1.0) < 1e-4
        assert abs(np.sum(cal_lgbm) - 1.0) < 1e-4

    def test_dqn_policy_agent(self):
        """Verify DQN reinforcement learning agent dimensions and policy outputs."""
        dqn_path = self.artifacts_dir / "dqn_model.pth"
        assert dqn_path.exists(), "dqn_model.pth must exist"
        agent = DQNAgent(state_size=33, action_size=3)
        agent.load(str(dqn_path))

        dummy_state = np.zeros(33)
        action = agent.act(dummy_state)
        probs = agent.predict_proba(dummy_state, temperature=1.5)

        assert action in [0, 1, 2]
        assert len(probs) == 3
        assert abs(np.sum(probs) - 1.0) < 1e-4

    def test_consensus_engine_veto_logic(self):
        """Verify consensus engine behavior under alpha buy and secondary risk veto scenarios."""
        engine = WeightedConsensusEngine()

        # Normal Alpha Buy
        base_probs = {
            "XGB_AGENT": np.array([0.05, 0.20, 0.75]),
            "LGBM_AGENT": np.array([0.10, 0.35, 0.55]),
        }
        res = engine.compute_asymmetric_veto(
            base_probs,
            primary_key="XGB_AGENT",
            primary_threshold=0.58,
            veto_threshold=0.65,
            veto_short=True,
        )
        assert res["dominant_direction"] == "BUY"
        assert not res["is_vetoed"]

        # Risk Agent Veto
        base_probs_veto = {
            "XGB_AGENT": np.array([0.05, 0.20, 0.75]),
            "LGBM_AGENT": np.array([0.70, 0.20, 0.10]),
        }
        res_veto = engine.compute_asymmetric_veto(
            base_probs_veto,
            primary_key="XGB_AGENT",
            primary_threshold=0.58,
            veto_threshold=0.65,
            veto_short=True,
        )
        assert res_veto["is_vetoed"] is True
        assert res_veto["dominant_direction"] == "HOLD"
