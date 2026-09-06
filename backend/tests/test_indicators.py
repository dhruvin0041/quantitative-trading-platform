import numpy as np
import pandas as pd

from src.data_ingestion.technical_indicators import add_advanced_features


def test_add_advanced_features():
    # Create dummy data
    dates = pd.date_range("2023-01-01", periods=100)
    data = {
        "Open": np.random.uniform(100, 150, 100),
        "High": np.random.uniform(110, 160, 100),
        "Low": np.random.uniform(90, 140, 100),
        "Close": np.random.uniform(100, 150, 100),
        "Volume": np.random.randint(1000, 10000, 100),
    }
    df = pd.DataFrame(data, index=dates)

    # Mock VIX and TNX to avoid network calls during tests
    vix_df = pd.DataFrame({"Close": np.random.uniform(10, 30, 100)}, index=dates)
    tnx_df = pd.DataFrame({"Close": np.random.uniform(1, 5, 100)}, index=dates)

    processed_df = add_advanced_features(df, vix_data=vix_df, tnx_data=tnx_df)

    assert not processed_df.empty
    assert "ATR" in processed_df.columns
    assert "MACD" in processed_df.columns
    assert "RSI_14" in processed_df.columns
    assert "VIX" in processed_df.columns

    # Check for NaN values
    assert processed_df["ATR"].isna().sum() == 0
    assert processed_df["BB_Upper"].isna().sum() == 0
