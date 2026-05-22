# src/models/regime_detector.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib


def train_macro_regime_model():
    print("Fetching VIX Macro Data (2014-Present)...")
    # Fetch 10 years of the Volatility Index
    vix_data = yf.download(
        "^VIX", start="2014-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d")
    )

    # Calculate 5-day rate of change in Volatility
    vix_data["VIX_ROC"] = vix_data["Close"].pct_change(periods=5)
    vix_data = vix_data.dropna()

    # Features for the Unsupervised Model
    X = vix_data[["Close", "VIX_ROC"]].values

    print("Training Gaussian Mixture Model (Regime Detection)...")
    # We force the AI to cluster the market into 2 regimes: 0 (Normal) and 1 (Panic)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gmm.fit(X)

    # Determine which cluster represents "Panic" (the one with the higher average VIX)
    means = gmm.means_[:, 0]
    panic_cluster_index = np.argmax(means)

    # Save the model and the panic index
    joblib.dump(
        {"model": gmm, "panic_cluster": panic_cluster_index}, "macro_kill_switch.joblib"
    )
    print(f"Macro Kill-Switch Armed. Panic Cluster ID: {panic_cluster_index}")


if __name__ == "__main__":
    train_macro_regime_model()
