import json
import logging
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.models.classical.rf_agent import RandomForestAgent
from src.models.ensemble.meta_ensemble import MetaEnsemble
from src.models.neural.fusion_network import build_fusion_model
from src.models.neural.tft_agent import build_tft_branch
from src.models.regime.svm_regime_classifier import SVMRegimeClassifier
from src.models.rl.dqn_agent import DQNAgent
from src.models.rl.ppo_agent import PPOAgent
from src.utils.gpu_utils import (
    configure_tensorflow_gpu,
    get_device,
    get_xgboost_gpu_params,
)

logger = logging.getLogger(__name__)



class PurgedGroupTimeSeriesSplit:
    """
    Time Series cross-validator for panel datasets (multi-ticker).
    Purges overlapping observations (embargo) to prevent look-ahead bias and data leakage
    when using horizon-based targets like the Dynamic Triple Barrier.
    """
    def __init__(self, n_splits=5, embargo=10):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, df):
        # Assumes df has a DatetimeIndex or is sorted chronologically
        dates = df.index.get_level_values(0) if isinstance(df.index, pd.MultiIndex) else df.index
        unique_dates = np.unique(dates)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for train_idx, test_idx in tscv.split(unique_dates):
            train_dates = unique_dates[train_idx]
            test_dates = unique_dates[test_idx]

            # Apply embargo: drop the last 'embargo' dates from the training set
            # This ensures the forward-looking Triple Barrier horizon doesn't bleed into the test set
            if self.embargo > 0 and len(train_dates) > self.embargo:
                train_dates = train_dates[:-self.embargo]

            train_mask = np.isin(dates, train_dates)
            test_mask = np.isin(dates, test_dates)

            yield np.where(train_mask)[0], np.where(test_mask)[0]


class ModelManager:
    def __init__(self, config, kept_features_list):
        self.config = config
        self.kept_features_list = kept_features_list
        self.num_features = len(kept_features_list)

        self.lstm_model = None
        self.tft_model = None
        self.xgb_model = None
        self.svm_model = None
        self.knn_model = None
        self.lgbm_model = None
        self.rf_agent = None
        self.svm_regime_model = None
        self.dqn_agent = None
        self.ppo_agent = None
        self.meta_ensemble = None
        self.accuracies = {}

    def load_all_models(self):
        configure_tensorflow_gpu()
        logger.info("GPU device for PyTorch: %s", get_device())
        self._load_accuracies()
        self._load_lstm()
        self._load_tft()

        # Load Core Classifiers & Regime Models
        self._load_xgb()
        self._load_svm()
        self._load_knn()
        self._load_rf()
        self._load_svm_regime()

        self._load_lgbm()
        self._load_dqn()
        self._load_ppo()
        self._load_meta_ensemble()
        logger.info("All models loaded into ModelManager.")

    def train_core_classifiers(self, df: pd.DataFrame, target_col="target_signal"):
        """
        Trains XGBoost, SVM, and KNN on the vectorized panel dataset safely using Purged Split.
        """
        X = df[self.kept_features_list].values
        y = df[target_col].values

        # 1. Temporal & Group-Aware Split
        pg_tscv = PurgedGroupTimeSeriesSplit(n_splits=3, embargo=10)

        # We'll use the last split for final training/validation representation
        train_idx, val_idx = list(pg_tscv.split(df))[-1]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        logger.info(f"Training Core Classifiers. Train Size: {len(X_train)}, Val Size: {len(X_val)}")

        # 2. Train XGBoost
        self.xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=100,
            **get_xgboost_gpu_params(),
        )
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # 3. Train SVM (using Probability=True for ensemble consensus)
        self.svm_model = SVC(probability=True, kernel="rbf", C=1.0)
        self.svm_model.fit(X_train, y_train)

        # 4. Train KNN
        self.knn_model = KNeighborsClassifier(n_neighbors=5, weights="distance")
        self.knn_model.fit(X_train, y_train)

        # Save artifacts
        self.xgb_model.save_model("artifacts/xgb_ensemble.json")
        joblib.dump(self.svm_model, "artifacts/svm_ensemble.joblib")
        joblib.dump(self.knn_model, "artifacts/knn_ensemble.joblib")
        logger.info("Core Classifiers trained and saved.")

    def get_bundled_predictions(self, X: np.ndarray) -> dict:
        """
        Runs inference across the core classifiers and bundles the probabilistic
        consensus payload for the AlphaAgent handoff.

        Triple Barrier Classes: 0 (Sell/Stop Loss), 1 (Hold/Time), 2 (Buy/Take Profit)
        """
        if self.xgb_model is None or self.svm_model is None or self.knn_model is None:
            raise ValueError("Core classifiers are not fully loaded.")

        # Extract probability arrays [P(Sell), P(Hold), P(Buy)]
        xgb_probs = self.xgb_model.predict_proba(X)
        svm_probs = self.svm_model.predict_proba(X)
        knn_probs = self.knn_model.predict_proba(X)

        # Equal-weighted ensemble probability matrix
        ensemble_probs = (xgb_probs + svm_probs + knn_probs) / 3.0

        # Generate predictions for the batch
        dominant_indices = np.argmax(ensemble_probs, axis=1)
        confidence_scores = np.max(ensemble_probs, axis=1) * 100.0

        # We return the payload mapped for the AlphaAgent (assuming batch size of 1 for live routing,
        # or returning a list of dicts for backtesting panels).
        # Returning the latest/last entry for real-time inference routing:
        return {
            "dominant_idx": int(dominant_indices[-1]),
            "agreement_score": float(confidence_scores[-1]),
            "raw_probabilities": ensemble_probs[-1].tolist()
        }

    # --- Loading Methods ---

    def _load_accuracies(self):
        try:
            from pathlib import Path
            path = Path("configs/model_accuracies.json")
            if not path.exists():
                path = Path(__file__).resolve().parent.parent.parent / "configs" / "model_accuracies.json"
            with open(path, "r") as f:
                self.accuracies = json.load(f)
        except Exception:
            self.accuracies = {
                "xgb_accuracy": 0.4968,
                "lgbm_accuracy": 0.4559,
                "dl_accuracy": 0.3547,
                "dqn_accuracy": 0.3800,
                "xgb": 0.4968,
                "lgbm": 0.4559,
                "dl_fusion": 0.3547,
                "dqn": 0.3800,
            }

    def _load_lstm(self):
        from src.execution.asset_intelligence import MODEL_REGISTRY, ModelRole

        dl_entry = MODEL_REGISTRY.get("DL_FUSION", {})
        if dl_entry.get("role") == ModelRole.QUARANTINED or dl_entry.get("status") == "QUARANTINED":
            logger.info("DL_FUSION is permanently QUARANTINED in MODEL_REGISTRY; bypassing neural weight loading and tensor allocations.")
            self.lstm_model = None
            return

        try:
            from src.execution.live_inference import apply_optimized_model_params

            self.config = apply_optimized_model_params(self.config)
            self.lstm_model = build_fusion_model(self.config)
            self.lstm_model.load_weights(
                "artifacts/latest_fusion_weights.weights.h5", skip_mismatch=False
            )
            assert len(self.lstm_model.weights) > 0, "No weights found in LSTM model"
            logger.info("Successfully loaded 100% of LSTM fusion weights (skip_mismatch=False)")
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
            v3_path = Path("artifacts/v3/xgb_ensemble.json")
            if not v3_path.exists():
                v3_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "v3" / "xgb_ensemble.json"

            if self.num_features == 31 and v3_path.exists():
                self.xgb_model.load_model(str(v3_path))
                logger.info("Loaded V3.0 Universal XGBoost ensemble (31 features).")
            else:
                self.xgb_model.load_model("artifacts/xgb_ensemble.json")
                logger.info("Loaded V2.4 XGBoost ensemble (27 features).")
        except Exception as e:
            logger.warning(f"Could not load XGB ensemble: {e}")

    def _load_svm(self):
        try:
            self.svm_model = joblib.load("artifacts/svm_ensemble.joblib")
        except Exception as e:
            logger.warning(f"Could not load SVM model: {e}")

    def _load_knn(self):
        try:
            self.knn_model = joblib.load("artifacts/knn_ensemble.joblib")
        except Exception as e:
            logger.warning(f"Could not load KNN model: {e}")

    def _load_lgbm(self):
        try:
            v3_path = Path("artifacts/v3/lgbm_agent.joblib")
            if not v3_path.exists():
                v3_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "v3" / "lgbm_agent.joblib"

            if self.num_features == 31 and v3_path.exists():
                self.lgbm_model = joblib.load(v3_path)
                logger.info("Loaded V3.0 Universal LightGBM agent (31 features).")
            else:
                self.lgbm_model = joblib.load("artifacts/lgbm_agent.joblib")
                logger.info("Loaded V2.4 LightGBM agent (27 features).")
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

    def _load_rf(self):
        try:
            rf_path = Path("artifacts/rf_agent.joblib")
            if not rf_path.exists():
                rf_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "rf_agent.joblib"
            if rf_path.exists():
                self.rf_agent = RandomForestAgent.load(rf_path)
                logger.info("Loaded RandomForestAgent artifact.")
        except Exception as e:
            logger.warning(f"Could not load RandomForestAgent: {e}")

    def _load_svm_regime(self):
        try:
            regime_path = Path("artifacts/svm_regime_classifier.joblib")
            if not regime_path.exists():
                regime_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "svm_regime_classifier.joblib"
            if regime_path.exists():
                self.svm_regime_model = SVMRegimeClassifier.load(regime_path)
                logger.info("Loaded SVMRegimeClassifier artifact.")
        except Exception as e:
            logger.warning(f"Could not load SVMRegimeClassifier: {e}")

    def _load_ppo(self):
        try:
            ppo_path = Path("artifacts/ppo_agent.pth")
            if not ppo_path.exists():
                ppo_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "ppo_agent.pth"
            if ppo_path.exists():
                self.ppo_agent = PPOAgent(state_size=self.num_features + 6)
                self.ppo_agent.load(ppo_path)
                logger.info("Loaded PPOAgent artifact.")
        except Exception as e:
            logger.warning(f"Could not load PPOAgent: {e}")

