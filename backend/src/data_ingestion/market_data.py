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

    if "-" in ticker_symbol and "USD" in ticker_symbol:
        return "BTC-USD" if ticker_symbol != "BTC-USD" else "ETH-USD"
    elif "=X" in ticker_symbol:
        return "EURUSD=X" if ticker_symbol != "EURUSD=X" else "GBPUSD=X"
    elif "=F" in ticker_symbol:
        return "GC=F" if ticker_symbol != "GC=F" else "SI=F"

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
    
    Refactored to use vectorized Pandas/NumPy operations for massive panel datasets.
    """
    print(
        f"Applying Dynamic Triple Barrier Labeling (TP: {tp_atr_multiplier}x ATR, SL: {sl_atr_multiplier}x ATR, Horizon: {horizon} days)..."
    )

    if "ATR" not in df.columns:
        raise KeyError(
            "ATR column is missing. Ensure add_advanced_features is run before apply_dynamic_triple_barrier."
        )

    # Calculate absolute barrier thresholds
    upper_barrier = df["Close"] + (df["ATR"] * tp_atr_multiplier)
    lower_barrier = df["Close"] - (df["ATR"] * sl_atr_multiplier)

    # Detect if dataset is a multi-asset panel (Ticker in index or columns)
    has_ticker_col = "Ticker" in df.columns
    has_ticker_idx = "Ticker" in df.index.names

    # Grouping logic for panel datasets
    if has_ticker_idx:
        grouped_high = df.groupby(level="Ticker")["High"]
        grouped_low = df.groupby(level="Ticker")["Low"]
    elif has_ticker_col:
        grouped_high = df.groupby("Ticker")["High"]
        grouped_low = df.groupby("Ticker")["Low"]
    else:
        grouped_high = df["High"]
        grouped_low = df["Low"]

    # Build vectorized future matrices
    future_highs, future_lows = [], []
    for h in range(1, horizon + 1):
        future_highs.append(grouped_high.shift(-h))
        future_lows.append(grouped_low.shift(-h))

    fh_df = pd.concat(future_highs, axis=1)
    fl_df = pd.concat(future_lows, axis=1)

    # Vectorized breach detection across the horizon
    upper_hits = fh_df.values >= upper_barrier.values[:, None]
    lower_hits = fl_df.values <= lower_barrier.values[:, None]

    # Find the earliest step each barrier is hit (horizon + 1 if never hit)
    upper_hit_steps = np.where(upper_hits.any(axis=1), upper_hits.argmax(axis=1), horizon + 1)
    lower_hit_steps = np.where(lower_hits.any(axis=1), lower_hits.argmax(axis=1), horizon + 1)

    # Consensus Rule: if both hit simultaneously, default to Stop-Loss (0) for safety
    neither_hit = (upper_hit_steps > horizon) & (lower_hit_steps > horizon)
    signals = np.where(
        neither_hit, 1, np.where(upper_hit_steps < lower_hit_steps, 2, 0)
    )

    df["target_signal"] = signals

    # Target calculation logic
    def get_future_max(s):
        return s.rolling(window=horizon).max().shift(-horizon)

    def get_future_min(s):
        return s.rolling(window=horizon).min().shift(-horizon)

    if has_ticker_idx or has_ticker_col:
        grouped = df.groupby(level="Ticker") if has_ticker_idx else df.groupby("Ticker")
        df["future_high"] = grouped["High"].transform(get_future_max)
        df["future_low"] = grouped["Low"].transform(get_future_min)
        df["future_close"] = grouped["Close"].shift(-horizon)
    else:
        df["future_high"] = df["High"].rolling(window=horizon).max().shift(-horizon)
        df["future_low"] = df["Low"].rolling(window=horizon).min().shift(-horizon)
        df["future_close"] = df["Close"].shift(-horizon)

    df["target_min"] = (df["future_low"] - df["Close"]) / df["Close"]
    df["target_max"] = (df["future_high"] - df["Close"]) / df["Close"]
    df["target_direction"] = (df["future_close"] > df["Close"] * 1.02).astype(int)

    # Drop NaNs that appear at the end of the horizon where future data isn't available
    df.dropna(subset=["future_close"], inplace=True)
    df.dropna(inplace=True)

    return df
