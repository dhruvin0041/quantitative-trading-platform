import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import joblib
import os


class RegimeDetector:
    """
    Institutional-grade Market Regime Detection using Gaussian Mixture Models
    and Hidden Markov Models (HMM) on volatility and returns.
    """

    def __init__(self, n_regimes=3, method="hmm"):
        """
        n_regimes:
            0 = Bear/High Vol
            1 = Sideways/Neutral
            2 = Bull/Low Vol
        method: 'hmm' or 'gmm'
        """
        self.n_regimes = n_regimes
        self.method = method

        if method == "hmm":
            # Gaussian HMM to model hidden states
            self.model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42,
            )
        else:
            self.model = GaussianMixture(
                n_components=n_regimes,
                covariance_type="full",
                random_state=42,
                max_iter=1000,
            )

        self.is_fitted = False

    def _prepare_features(self, df):
        """
        Extract features that represent market states (Volatility and Momentum).
        """
        features = pd.DataFrame(index=df.index)

        # We need log returns and realized volatility
        if "Log_Ret" in df.columns:
            features["returns"] = df["Log_Ret"]
        else:
            features["returns"] = np.log(df["Close"] / df["Close"].shift(1))

        if "Realized_Vol_20" in df.columns:
            features["volatility"] = df["Realized_Vol_20"]
        else:
            features["volatility"] = features["returns"].rolling(20).std() * np.sqrt(
                252
            )

        # Drop NaNs
        features = features.dropna()
        return features

    def fit(self, df):
        """
        Fit the regime detection model.
        """
        features = self._prepare_features(df)
        X = features.values

        print(f"Fitting {self.method.upper()} Regime Detector...")
        self.model.fit(X)
        self.is_fitted = True

        # Identify states by their volatility and return profiles
        if self.method == "hmm":
            means = self.model.means_
        else:
            means = self.model.means_

        # means shape: (n_regimes, n_features) -> [returns, volatility]
        # Sort states primarily by volatility (index 1) to establish consistent state IDs
        # High Vol = State 0 (Crisis/Bear)
        # Low Vol = State 2 (Bull/Calm)
        self.state_map = np.argsort(means[:, 1])[::-1]  # Descending vol

    def predict(self, df):
        """
        Predict regimes and return state probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        features = self._prepare_features(df)
        X = features.values

        if self.method == "hmm":
            states = self.model.predict(X)
            probs = self.model.predict_proba(X)
        else:
            states = self.model.predict(X)
            probs = self.model.predict_proba(X)

        # Map to consistent states
        # 0: High Vol/Bear, 1: Medium Vol/Sideways, 2: Low Vol/Bull
        mapped_states = np.zeros_like(states)
        mapped_probs = np.zeros_like(probs)

        for i, original_state in enumerate(self.state_map):
            mapped_states[states == original_state] = i
            mapped_probs[:, i] = probs[:, original_state]

        # Re-align with original dataframe index
        result = pd.DataFrame(index=df.index)

        # Fill missing early rows (due to rolling windows) with the first valid state
        # Create a series with matching index to features
        state_series = pd.Series(mapped_states, index=features.index)
        prob_df = pd.DataFrame(
            mapped_probs,
            index=features.index,
            columns=[f"Regime_Prob_{i}" for i in range(self.n_regimes)],
        )

        result["Regime_ID"] = state_series
        result = result.join(prob_df)

        # Bfill for the rows lost during feature prep
        result = result.bfill()

        # Add Regime Confidence (max probability)
        result["Regime_Confidence"] = result[
            [f"Regime_Prob_{i}" for i in range(self.n_regimes)]
        ].max(axis=1)

        return result

    def save(self, filepath="artifacts/regime_detector.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath="artifacts/regime_detector.joblib"):
        return joblib.load(filepath)
