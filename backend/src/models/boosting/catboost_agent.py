import logging

import joblib
from catboost import CatBoostClassifier

from src.utils.gpu_utils import benchmark_context, get_catboost_gpu_params

logger = logging.getLogger(__name__)


def train_catboost_agent(X_train, y_train, save_path="artifacts/catboost_agent.joblib"):
    """
    Trains a CatBoost Classifier on provided scaled data and saves the artifact.
    Highly resistant to overfitting. Uses GPU acceleration when available.
    """
    print(f"--- Training CatBoost Agent ({len(X_train)} samples) ---")
    gpu_params = get_catboost_gpu_params()
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=6,
        loss_function="MultiClass",
        random_seed=42,
        verbose=0,
        **gpu_params,
    )
    # CatBoost can natively handle categorical features,
    # but we assume X_train is fully numerical and preprocessed for now.
    with benchmark_context("CatBoost Training"):
        model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    logger.info("CatBoost model saved to %s", save_path)
    print(f"CatBoost model saved to {save_path}")
    return model
