# src/models/regime_detector.py
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.mixture import GaussianMixture


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

class MultiFactorRegimeDetector:
    """
    Institutional Multi-Factor Regime Detection Engine.
    Synthesizes Volatility (ATR Regime Ratio), Trend Strength (ADX),
    and Moving Average Alignment into forward probabilistic market states.
    """

    @staticmethod
    def classify_market_state(
        df: pd.DataFrame,
        vix_series: pd.Series = None,
        spy_df: pd.DataFrame = None,
    ) -> dict:
        """
        Classifies asset into:
        - BULL_EXPANSION: Strong trend, expanding momentum, safe volatility.
        - BULL_CONSOLIDATION: Uptrend with temporary pullbacks to value zones.
        - RANGE_BOUND: Compressed volatility, mean-reverting chop.
        - BEAR_DISTRIBUTION: Breakdown below key moving averages.
        - HIGH_VOLATILITY_PANIC: Elevated VIX or extreme ATR expansion.
        """
        if df.empty or len(df) < 20:
            return {
                "regime": "NEUTRAL",
                "confidence": 0.50,
                "allow_long": True,
                "allow_short": False,
                "regime_ratio": 1.0,
            }

        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        high = df["High"].squeeze() if "High" in df.columns else close
        low = df["Low"].squeeze() if "Low" in df.columns else close

        # 1. Volatility Regime Ratio (ARR = ATR_14 / ATR_60)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=5).mean().iloc[-1]
        atr60 = tr.rolling(60, min_periods=10).mean().iloc[-1]
        arr = float(atr14 / (atr60 + 1e-8)) if atr60 > 0 else 1.0

        # 2. Moving Average Alignment
        ma20 = close.rolling(20, min_periods=5).mean().iloc[-1]
        ma50 = close.rolling(50, min_periods=10).mean().iloc[-1]
        ma200 = close.rolling(200, min_periods=20).mean().iloc[-1]
        cur_c = float(close.iloc[-1])

        # 3. Macro VIX Level
        vix_val = float(vix_series.iloc[-1]) if (vix_series is not None and not vix_series.empty) else 18.0

        # Classification Hierarchy
        if vix_val >= 32.0 or arr >= 1.50:
            regime = "HIGH_VOLATILITY_PANIC"
            confidence = min(0.95, arr / 1.50 * 0.85)
            allow_long = False
            allow_short = True
        elif cur_c >= ma50 >= ma200 and cur_c >= ma20:
            regime = "BULL_EXPANSION"
            confidence = 0.88
            allow_long = True
            allow_short = False
        elif cur_c >= ma50 >= ma200 and cur_c < ma20:
            regime = "BULL_CONSOLIDATION"
            confidence = 0.82
            allow_long = True  # Prime for Pullback Re-entry
            allow_short = False
        elif cur_c < ma50 and cur_c < ma200:
            regime = "BEAR_DISTRIBUTION"
            confidence = 0.85
            allow_long = False
            allow_short = True
        else:
            regime = "RANGE_BOUND"
            confidence = 0.65
            allow_long = True
            allow_short = True

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "allow_long": allow_long,
            "allow_short": allow_short,
            "regime_ratio": round(arr, 3),
            "vix_level": round(vix_val, 2),
        }

