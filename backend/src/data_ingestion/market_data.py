# src/data_ingestion/market_data.py
import yfinance as yf
import pandas as pd
import numpy as np


def fetch_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches real historical daily data from Yahoo Finance.
    """
    print(f"Fetching real-world data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(f"No data found for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.dropna(inplace=True)
    return df


def get_sector_peer(ticker_symbol: str) -> str:
    """
    Dynamically identifies a high-correlation peer in the same sector.
    """
    # Hardcoded mapping for institutional speed for top sectors
    sector_map = {
        "Technology": "MSFT" if ticker_symbol != "MSFT" else "AAPL",
        "Financial Services": "JPM" if ticker_symbol != "JPM" else "BAC",
        "Healthcare": "JNJ" if ticker_symbol != "JNJ" else "PFE",
        "Consumer Cyclical": "AMZN" if ticker_symbol != "AMZN" else "TSLA",
        "Communication Services": "GOOGL" if ticker_symbol != "GOOGL" else "META",
        "Energy": "XOM" if ticker_symbol != "XOM" else "CVX",
    }

    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        sector = info.get("sector", "Unknown")
        peer = sector_map.get(sector, "SPY")  # Default to SPY if sector not found
        print(f"Detected Sector: {sector}. Assigning Lead-Lag Peer: {peer}")
        return peer
    except Exception:
        return "SPY"


def apply_dynamic_triple_barrier(
    df: pd.DataFrame, tp_atr_multiplier=2.0, sl_atr_multiplier=1.0, horizon=10
) -> pd.DataFrame:
    """
    PATH 1: Dynamic Barriers based on current volatility (ATR).
    - Upper Barrier: Take Profit based on ATR
    - Lower Barrier: Stop Loss based on ATR
    - Vertical Barrier: Time Limit (e.g., 10 days)
    """
    print(
        f"Applying Dynamic Triple Barrier Labeling (TP: {tp_atr_multiplier}x ATR, SL: {sl_atr_multiplier}x ATR, Horizon: {horizon} days)..."
    )

    # Default everything to 1 (Hold/Skip)
    signals = np.ones(len(df))
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    # Ensure ATR exists before proceeding
    if "ATR" not in df.columns:
        raise KeyError(
            "ATR column is missing. Ensure add_advanced_features is run before apply_dynamic_triple_barrier."
        )

    atrs = df["ATR"].values

    # We must stop the loop before the end of the dataset to have room to "look forward"
    for i in range(len(closes) - horizon):
        current_price = closes[i]
        current_atr = atrs[i]

        # Dynamic barrier thresholds (Absolute Price Levels)
        upper_barrier_price = current_price + (current_atr * tp_atr_multiplier)
        lower_barrier_price = current_price - (current_atr * sl_atr_multiplier)

        # Extract the future price path for the next 'horizon' days
        future_highs = highs[i + 1 : i + 1 + horizon]
        future_lows = lows[i + 1 : i + 1 + horizon]

        # Find the exact indices where the barriers are breached
        upper_hits = np.where(future_highs >= upper_barrier_price)[0]
        lower_hits = np.where(future_lows <= lower_barrier_price)[0]

        # Scenario 1: Both barriers are hit within the 10 days
        if len(upper_hits) > 0 and len(lower_hits) > 0:
            if upper_hits[0] < lower_hits[0]:
                signals[i] = 2  # Hit Take Profit first -> BUY
            elif lower_hits[0] < upper_hits[0]:
                signals[i] = 0  # Hit Stop Loss first -> SELL
            else:
                # Extreme Volatility Edge Case: Hit both on the exact same day.
                # Institutional rule: Always assume the stop-loss hit first to be conservative.
                signals[i] = 0

        # Scenario 2: Only Take Profit is hit
        elif len(upper_hits) > 0:
            signals[i] = 2

        # Scenario 3: Only Stop Loss is hit
        elif len(lower_hits) > 0:
            signals[i] = 0

        # Scenario 4: Neither is hit before the 10 days expire (Vertical Barrier)
        else:
            signals[i] = 1  # -> SKIP / HOLD

    df["target_signal"] = signals

    # We must calculate Range targets for your regression output as well
    df["future_high"] = df["High"].rolling(window=horizon).max().shift(-horizon)
    df["future_low"] = df["Low"].rolling(window=horizon).min().shift(-horizon)
    df["target_min"] = df["future_low"] - df["Close"]
    df["target_max"] = df["future_high"] - df["Close"]

    # Direction is now based on the final horizon day, just to keep the tensor shapes happy
    df["future_close"] = df["Close"].shift(-horizon)
    df["target_direction"] = (df["future_close"] > df["Close"]).astype(int)

    # Drop the last 'horizon' rows because we can't look into the future for them
    df = df.iloc[:-horizon].copy()
    df.dropna(inplace=True)

    return df
