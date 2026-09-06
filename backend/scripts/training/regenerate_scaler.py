# regenerate_scaler.py
import json

import joblib
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from src.data_ingestion.market_data import fetch_historical_data
from src.execution.live_inference import FEATURE_COLUMNS, add_upgraded_features


def regenerate():
    print("Regenerating scaler for 30 features...")
    # Diverse tickers for a robust scaler
    tickers = [
        "MSFT",
        "AAPL",
        "TSLA",
        "NVDA",
        "JPM",
        "AMD",
        "AMZN",
        "GOOGL",
        "META",
        "BRK-B",
    ]
    all_data = []

    spy_df = yf.download("SPY", period="2y", interval="1d", progress=False)
    vix_df = yf.download("^VIX", period="2y", interval="1d", progress=False)

    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.droplevel(1)

    for t in tickers:
        try:
            print(f"Processing {t}...")
            # Fetch up to end of 2023 for scaler fitting to avoid data leakage
            df = fetch_historical_data(t, "2019-01-01", "2023-12-31")
            df = add_upgraded_features(df, spy_df, vix_df)

            # Target generation just to match preprocessing, though not needed for scaling features
            df["target_direction"] = (
                df["Close"].shift(-5) > df["Close"] * 1.02
            ).astype(int)
            df["target_min"] = (
                df["Low"].rolling(5).min().shift(-5) - df["Close"]
            ) / df["Close"]
            df["target_max"] = (
                df["High"].rolling(5).max().shift(-5) - df["Close"]
            ) / df["Close"]

            df = df.dropna()

            all_data.append(df[FEATURE_COLUMNS])
        except Exception as e:
            print(f"Error processing {t}: {e}")

    if not all_data:
        print("No data collected. Scaler regeneration failed.")
        return

    full_df = pd.concat(all_data)
    scaler = StandardScaler()
    scaler.fit(full_df)

    joblib.dump(scaler, "artifacts/latest_scaler.joblib")

    with open("configs/kept_features.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    print(
        f"Scaler fitted on {len(FEATURE_COLUMNS)} features. Saved to artifacts/latest_scaler.joblib"
    )


if __name__ == "__main__":
    regenerate()
