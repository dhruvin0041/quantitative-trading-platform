import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

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

    spy_df = yf.download(
        "SPY", start="2016-01-01", end="2024-12-31", interval="1d", progress=False
    )
    vix_df = yf.download(
        "^VIX", start="2016-01-01", end="2024-12-31", interval="1d", progress=False
    )

    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.droplevel(1)

    from src.execution.data_firewall import TemporalFirewall

    for t in tickers:
        try:
            print(f"Processing {t}...")
            # Fetch strictly 2016-2024 for development/scaler fitting (zero 2025/2026 leakage)
            df = fetch_historical_data(t, "2016-01-01", "2024-12-31")
            TemporalFirewall.validate_development_data(df, f"scaler_{t}")
            df = add_upgraded_features(df, spy_df, vix_df)

            # Drop uninitialized warmup period (first 119 bars for rolling 120-bar indicators)
            if len(df) > 120:
                df = df.iloc[119:]

            df = df.dropna(subset=FEATURE_COLUMNS)
            all_data.append(df[FEATURE_COLUMNS])
        except Exception as e:
            print(f"Error processing {t}: {e}")

    if not all_data:
        print("No data collected. Scaler regeneration failed.")
        return

    full_df = pd.concat(all_data)
    TemporalFirewall.validate_development_data(full_df, "global_scaler_full_df")
    TemporalFirewall.validate_no_2026_leakage(full_df, "global_scaler_full_df")

    scaler = StandardScaler()
    scaler.fit(full_df)

    joblib.dump(scaler, "artifacts/latest_scaler.joblib")

    with open("configs/kept_features.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    print(
        f"Scaler fitted on {len(FEATURE_COLUMNS)} features across {len(full_df)} samples strictly within 2016-2024."
    )


if __name__ == "__main__":
    regenerate()
