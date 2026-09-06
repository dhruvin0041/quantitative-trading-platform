import logging

import joblib
from lightgbm import LGBMClassifier

from src.utils.gpu_utils import benchmark_context, get_lightgbm_gpu_params

logger = logging.getLogger(__name__)


def train_lgbm_agent(X_train, y_train, save_path="artifacts/lgbm_agent.joblib"):
    """
    Trains a LightGBM Classifier on provided scaled data and saves the artifact.
    Automatically uses GPU acceleration if available.
    """
    print(f"--- Training LightGBM Agent ({len(X_train)} samples) ---")
    gpu_params = get_lightgbm_gpu_params()
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        objective="multiclass",
        num_class=3,
        random_state=42,
        verbose=-1,
        **gpu_params,
    )
    with benchmark_context("LightGBM Training"):
        model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    logger.info("LightGBM model saved to %s", save_path)
    print(f"LightGBM model saved to {save_path}")
    return model
