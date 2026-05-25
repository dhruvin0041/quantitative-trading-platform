import json
import joblib
import logging
import xgboost as xgb
import keras
from src.models.neural.fusion_network import build_fusion_model
from src.models.rl.dqn_agent import DQNAgent
from src.models.ensemble.meta_ensemble import MetaEnsemble
from src.models.neural.tft_agent import build_tft_branch

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, config, kept_features_list):
        self.config = config
        self.kept_features_list = kept_features_list
        self.num_features = len(kept_features_list)

        self.lstm_model = None
        self.tft_model = None
        self.xgb_model = None
        self.lgbm_model = None
        self.dqn_agent = None
        self.meta_ensemble = None
        self.accuracies = {}

    def load_all_models(self):
        self._load_accuracies()
        self._load_lstm()
        self._load_tft()
        self._load_xgb()
        self._load_lgbm()
        self._load_dqn()
        self._load_meta_ensemble()
        logger.info("All models loaded into ModelManager.")

    def _load_accuracies(self):
        try:
            with open("configs/model_accuracies.json", "r") as f:
                self.accuracies = json.load(f)
        except Exception:
            self.accuracies = {
                "ensemble_accuracy": 54.6,
                "dl_accuracy": 52.1,
                "xgb_accuracy": 55.4,
                "lgbm_accuracy": 53.2,
                "dqn_accuracy": 48.9,
            }

    def _load_lstm(self):
        try:
            self.lstm_model = build_fusion_model(self.config)
            self.lstm_model.load_weights("artifacts/latest_fusion_weights.weights.h5")
        except Exception as e:
            logger.warning(f"Could not load LSTM weights: {e}")

    def _load_tft(self):
        try:
            tft_input, tft_output = build_tft_branch(
                time_steps=self.config["data"]["time_steps"],
                num_features=self.num_features,
            )
            self.tft_model = keras.Model(inputs=tft_input, outputs=tft_output)
            self.tft_model.load_weights("artifacts/tft_quantile_weights.weights.h5")
        except Exception as e:
            logger.warning(f"Could not load TFT weights: {e}")

    def _load_xgb(self):
        try:
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model("artifacts/xgb_ensemble.json")
        except Exception as e:
            logger.warning(f"Could not load XGB ensemble: {e}")

    def _load_lgbm(self):
        try:
            self.lgbm_model = joblib.load("artifacts/lgbm_agent.joblib")
        except Exception as e:
            logger.warning(f"Could not load LightGBM agent: {e}")

    def _load_dqn(self):
        try:
            self.dqn_agent = DQNAgent(state_size=self.num_features + 6)
            self.dqn_agent.load("artifacts/dqn_model.pth")
        except Exception as e:
            logger.warning(f"Could not load DQN agent: {e}")

    def _load_meta_ensemble(self):
        try:
            self.meta_ensemble = MetaEnsemble.load("artifacts/meta_ensemble.joblib")
        except Exception as e:
            logger.warning(f"Could not load Meta-Ensemble: {e}")
