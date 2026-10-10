import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils.gpu_utils import benchmark_context

logger = logging.getLogger(__name__)


class RandomForestAgent:
    """
    Institutional Random Forest agent configured with strict depth constraints.

    Roles:
    1. Feature Selection: Evaluates non-linear feature interactions (e.g., RSI < 25
       conditioned on Volume > 2x average) via Mean Decrease in Impurity (MDI).
    2. Ensemble Voting: Generates smooth, low-variance probability distributions
       for multi-class trade signals [Sell, Hold, Buy] without overfitting.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        max_features: Union[str, float] = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
    ) -> "RandomForestAgent":
        """Fits the constrained Random Forest on stationary features."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_mat = X.values
        else:
            self.feature_names = feature_names
            X_mat = np.asarray(X)

        y_vec = np.asarray(y)

        logger.info(
            "Fitting RandomForestAgent (n_estimators=%d, max_depth=%d, min_samples_leaf=%d)",
            self.n_estimators,
            self.max_depth,
            self.min_samples_leaf,
        )
        with benchmark_context("RandomForest Fit"):
            self.model.fit(X_mat, y_vec)

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predicts discrete signal classes [0: Sell, 1: Hold, 2: Buy]."""
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict(X_mat)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns ensemble probability distribution across classes."""
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict_proba(X_mat)

    def get_feature_importances(
        self, feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Extracts Gini feature importances sorted descending.
        Essential for institutional feature selection and pruning.
        """
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model is not fitted yet.")

        names = feature_names or self.feature_names
        if names is None:
            names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]

        importances = dict(zip(names, self.model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    def select_top_features(
        self, top_k: int = 20, feature_names: Optional[List[str]] = None
    ) -> List[str]:
        """Returns the top_k most predictive features based on tree splits."""
        sorted_imp = self.get_feature_importances(feature_names=feature_names)
        return list(sorted_imp.keys())[:top_k]

    def save(self, save_path: Union[str, Path] = "artifacts/rf_agent.joblib") -> None:
        """Serializes the Random Forest model and metadata to disk."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "max_depth": self.max_depth,
                "n_estimators": self.n_estimators,
            },
            path,
        )
        logger.info("RandomForestAgent saved to %s", path)

    @classmethod
    def load(cls, load_path: Union[str, Path] = "artifacts/rf_agent.joblib") -> "RandomForestAgent":
        """Deserializes a persisted Random Forest agent."""
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        payload = joblib.load(path)
        agent = cls(
            n_estimators=payload.get("n_estimators", 300),
            max_depth=payload.get("max_depth", 5),
        )
        agent.model = payload["model"]
        agent.feature_names = payload.get("feature_names")
        return agent


def train_rf_agent(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    save_path: str = "artifacts/rf_agent.joblib",
    max_depth: int = 5,
    n_estimators: int = 300,
    min_samples_leaf: int = 20,
) -> RandomForestAgent:
    """
    Trains a constrained Random Forest agent on stationary features and saves artifact.
    """
    logger.info("--- Training Constrained Random Forest Agent (%d samples) ---", len(X_train))
    agent = RandomForestAgent(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    agent.fit(X_train, y_train)
    agent.save(save_path)
    return agent
