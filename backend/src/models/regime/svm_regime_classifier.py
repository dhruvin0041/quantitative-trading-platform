import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.gpu_utils import benchmark_context

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    MEAN_REVERTING = "Mean-Reverting"
    STRONG_TREND = "Strong Trend"
    HIGH_VOLATILITY_NOISE = "High-Volatility Noise"


REGIME_INDEX_TO_NAME = {
    0: MarketRegime.MEAN_REVERTING.value,
    1: MarketRegime.STRONG_TREND.value,
    2: MarketRegime.HIGH_VOLATILITY_NOISE.value,
}

REGIME_NAME_TO_INDEX = {
    MarketRegime.MEAN_REVERTING.value: 0,
    MarketRegime.STRONG_TREND.value: 1,
    MarketRegime.HIGH_VOLATILITY_NOISE.value: 2,
}


class SVMRegimeClassifier:
    """
    RBF-Kernel Support Vector Machine for Deterministic Regime Classification.

    Roles:
    - Discovers non-linear, deterministic decision boundaries separating:
      0: Mean-Reverting (oscillator-dominated, range-bound)
      1: Strong Trend (directional momentum, high ADX/HMA slope)
      2: High-Volatility Noise (erratic expansion, high ATR percentile/BB width)
    - Deterministic live inference prevents probabilistic drift during real-time streaming.
    """

    DEFAULT_FEATURES = [
        "ADX",
        "ATR_Percentile",
        "BB_Width",
        "HMA_Slope",
    ]

    def __init__(
        self,
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.C = C
        self.gamma = gamma
        self.feature_names = feature_names or list(self.DEFAULT_FEATURES)
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = SVC(
            kernel="rbf",
            C=self.C,
            gamma=self.gamma,
            probability=True,
            decision_function_shape="ovr",
            random_state=self.random_state,
        )
        self.is_fitted = False

    @staticmethod
    def synthesize_regime_labels(df: pd.DataFrame) -> np.ndarray:
        """
        Derives objective, stationary ground-truth regime targets:
        - High-Volatility Noise (2): ATR_Percentile > 0.80 or extreme BB_Width expansion.
        - Strong Trend (1): ADX >= 25.0 or prominent directional HMA_Slope.
        - Mean-Reverting (0): Compressed volatility, bounded ADX (< 20).
        """
        adx = df["ADX"].values if "ADX" in df.columns else np.zeros(len(df))
        atr_pct = (
            df["ATR_Percentile"].values
            if "ATR_Percentile" in df.columns
            else np.full(len(df), 0.5)
        )
        bb_width = (
            df["BB_Width"].values
            if "BB_Width" in df.columns
            else np.full(len(df), 0.05)
        )
        hma_slope = (
            df["HMA_Slope"].values
            if "HMA_Slope" in df.columns
            else np.zeros(len(df))
        )

        labels = np.zeros(len(df), dtype=int)  # Default: 0 = Mean-Reverting

        # Rule 1: High-Volatility Noise takes precedence (risk off / chop)
        high_vol_mask = (atr_pct >= 0.80) | (bb_width > np.nanpercentile(bb_width, 85))
        labels[high_vol_mask] = 2

        # Rule 2: Strong Trend
        trend_mask = (~high_vol_mask) & ((adx >= 25.0) | (np.abs(hma_slope) >= 0.004))
        labels[trend_mask] = 1

        return labels

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "SVMRegimeClassifier":
        """Fits the StandardScaler and RBF-kernel SVM on stationary regime features."""
        if isinstance(X, pd.DataFrame):
            missing = [col for col in self.feature_names if col not in X.columns]
            if not missing:
                X_mat = X[self.feature_names].values
            else:
                X_mat = X.values
        else:
            X_mat = np.asarray(X)

        y_vec = np.asarray(y)

        # Scale inputs (critical for RBF kernel distance metric)
        X_scaled = self.scaler.fit_transform(X_mat)

        logger.info(
            "Fitting SVMRegimeClassifier (C=%.2f, gamma=%s, samples=%d)",
            self.C,
            str(self.gamma),
            len(X_scaled),
        )
        with benchmark_context("SVM Regime Fit"):
            self.model.fit(X_scaled, y_vec)

        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns integer class predictions [0, 1, 2]."""
        if not self.is_fitted:
            raise ValueError("SVMRegimeClassifier must be fitted before predict.")

        if isinstance(X, pd.DataFrame):
            missing = [col for col in self.feature_names if col not in X.columns]
            X_mat = X.values if missing else X[self.feature_names].values
        else:
            X_mat = np.asarray(X)

        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)

        X_scaled = self.scaler.transform(X_mat)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns calibrated probability distribution across [Mean-Reverting, Trend, Noise]."""
        if not self.is_fitted:
            raise ValueError("SVMRegimeClassifier must be fitted before predict_proba.")

        if isinstance(X, pd.DataFrame):
            missing = [col for col in self.feature_names if col not in X.columns]
            X_mat = X.values if missing else X[self.feature_names].values
        else:
            X_mat = np.asarray(X)

        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)

        X_scaled = self.scaler.transform(X_mat)
        return self.model.predict_proba(X_scaled)

    def classify_regime(
        self, features: Union[np.ndarray, pd.Series, dict]
    ) -> Tuple[str, float]:
        """
        Inference hook for live streaming feeds.
        Returns: (regime_name: str, confidence: float)
        """
        if isinstance(features, dict):
            vec = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        elif isinstance(features, pd.Series):
            vec = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        else:
            vec = np.asarray(features)
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)

        probs = self.predict_proba(vec)[0]
        dominant_idx = int(np.argmax(probs))
        regime_name = REGIME_INDEX_TO_NAME.get(dominant_idx, MarketRegime.MEAN_REVERTING.value)
        confidence = float(probs[dominant_idx])

        return regime_name, confidence

    def save(
        self, save_path: Union[str, Path] = "artifacts/svm_regime_classifier.joblib"
    ) -> None:
        """Saves fitted SVM and scaler."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "C": self.C,
                "gamma": self.gamma,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info("SVMRegimeClassifier saved to %s", path)

    @classmethod
    def load(
        cls, load_path: Union[str, Path] = "artifacts/svm_regime_classifier.joblib"
    ) -> "SVMRegimeClassifier":
        """Loads persisted SVM and scaler."""
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        payload = joblib.load(path)
        instance = cls(
            C=payload.get("C", 1.0),
            gamma=payload.get("gamma", "scale"),
            feature_names=payload.get("feature_names"),
        )
        instance.model = payload["model"]
        instance.scaler = payload["scaler"]
        instance.is_fitted = payload.get("is_fitted", True)
        return instance
