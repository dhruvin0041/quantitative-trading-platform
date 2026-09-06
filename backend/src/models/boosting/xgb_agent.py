import logging

import joblib
from xgboost import XGBClassifier

from src.utils.gpu_utils import benchmark_context, get_xgboost_gpu_params

logger = logging.getLogger(__name__)


def train_xgb_agent(X_train, y_train, save_path="artifacts/xgb_agent.joblib"):
    """
    Trains an XGBoost Classifier on provided scaled data and saves the artifact.
    Automatically uses GPU acceleration if CUDA is available.
    """
    print(f"--- Training XGBoost Agent ({len(X_train)} samples) ---")
    gpu_params = get_xgboost_gpu_params()
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=-1,
        **gpu_params,
    )
    with benchmark_context("XGBoost Training"):
        model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    logger.info("XGBoost model saved to %s", save_path)
    print(f"XGBoost model saved to {save_path}")
    return model
