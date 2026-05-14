# src/data_ingestion/technical_indicators.py
import pandas as pd
import numpy as np
import ta
import yfinance as yf


def clean_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes yfinance MultiIndex columns to a single level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_advanced_features(
    df: pd.DataFrame, vix_data: pd.DataFrame = None, tnx_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Adds advanced, trend-strength, volatility-aware, and macro features.
    """
    # Standardize input columns
    df = clean_multiindex_columns(df)

    # --- 1. Trend Strength (ADX) ---
    adx_ins = ta.trend.ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    )
    df["ADX"] = adx_ins.adx()
    df["ADX_Pos"] = adx_ins.adx_pos()  # Positive trend strength
    df["ADX_Neg"] = adx_ins.adx_neg()  # Negative trend strength

    # --- 2. Volatility (ATR) ---
    df["ATR"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()

    # --- 3. Price Distance from Mean ---
    # How far is the price from the 50-day moving average? (Mean Reversion)
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["Price_to_SMA"] = df["Close"] / df["SMA_50"]

    # --- 4. Broad Market Context (VIX) ---
    if vix_data is None:
        vix_df = yf.download(
            "^VIX", start=df.index[0], end=df.index[-1], progress=False
        )
        vix_df = clean_multiindex_columns(vix_df)
    else:
        vix_df = vix_data

    # Assign using index alignment and fill gaps
    df["VIX"] = vix_df["Close"]
    df["VIX"] = df["VIX"].ffill().bfill()  # Fill gaps and edge cases

    # --- ALTERNATIVE DATA: Institutional Volume Metrics ---
    # VWAP (Volume Weighted Average Price) - The institutional baseline
    df["VWAP"] = ta.volume.VolumeWeightedAveragePrice(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"],
        window=14,
    ).volume_weighted_average_price()

    # OBV (On-Balance Volume) - Measures buying vs selling pressure
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"]
    ).on_balance_volume()

    # Volume Anomaly - Is today's volume unusually high? (Smart money footprint)
    df["Vol_Anomaly"] = df["Volume"] / df["Volume"].rolling(window=20).mean()

    # --- 5. ALTERNATIVE DATA: 10-Year Treasury Yield (Macro Environment) ---
    if tnx_data is None:
        tnx_df = yf.download(
            "^TNX", start=df.index[0], end=df.index[-1], progress=False
        )
        tnx_df = clean_multiindex_columns(tnx_df)
    else:
        tnx_df = tnx_data

    df["Treasury_10Y"] = tnx_df["Close"]
    df["Treasury_10Y"] = df["Treasury_10Y"].ffill().bfill()

    # --- 6. Log Returns (Better for Neural Networks than raw prices) ---
    df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))

    # ==========================================
    # MEAN REVERSION FEATURES (Bollinger Bands)
    # ==========================================
    # Calculate the 20-day moving average and standard deviation
    df["BB_Mid"] = df["Close"].rolling(window=20).mean()
    df["BB_Std"] = df["Close"].rolling(window=20).std()

    # Calculate Upper and Lower bands (2 standard deviations away)
    df["BB_Upper"] = df["BB_Mid"] + (df["BB_Std"] * 2)
    df["BB_Lower"] = df["BB_Mid"] - (df["BB_Std"] * 2)

    # The most important feature for the AI: "Where are we inside the bands?"
    # 1.0 means touching the absolute top (Sell zone). 0.0 means touching the bottom (Buy zone).
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (
        df["BB_Upper"] - df["BB_Lower"]
    )

    # Fast RSI (Makes the AI highly sensitive to sudden 3-day drops/spikes)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=3).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
    rs = gain / loss
    df["RSI_Fast_3"] = 100 - (100 / (1 + rs))

    # ==========================================
    # SWIFT ALGO / MOMENTUM FEATURES
    # ==========================================
    # 1. Standard RSI (14-day)
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    # 2. MACD (Moving Average Convergence Divergence) - Highly reactive to trend
    macd = ta.trend.MACD(
        close=df["Close"], window_slow=26, window_fast=12, window_sign=9
    )
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()  # Histogram is the "swift" momentum change

    # 3. Fast EMA Cross (9-day vs 21-day)
    df["EMA_9"] = ta.trend.EMAIndicator(close=df["Close"], window=9).ema_indicator()
    df["EMA_21"] = ta.trend.EMAIndicator(close=df["Close"], window=21).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(close=df["Close"], window=50).ema_indicator()
    df["EMA_200"] = ta.trend.EMAIndicator(close=df["Close"], window=200).ema_indicator()
    df["EMA_Cross"] = df["EMA_9"] - df["EMA_21"]

    # Volume Change %
    df["Vol_Change_Pct"] = df["Volume"].pct_change() * 100

    # 4. Commodity Channel Index (CCI) - Fast trend exhaustion identifier
    df["CCI_20"] = ta.trend.cci(
        high=df["High"], low=df["Low"], close=df["Close"], window=20
    )

    # ==========================================
    # NEW ADVANCED INDICATORS
    # ==========================================
    # 5. Money Flow Index (MFI) - Volume-weighted RSI
    df["MFI_14"] = ta.volume.MFIIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"],
        window=14,
    ).money_flow_index()

    # 6. Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # 7. Ichimoku Cloud (Selected components)
    ichimoku = ta.trend.IchimokuIndicator(
        high=df["High"], low=df["Low"], window1=9, window2=26, window3=52
    )
    df["Ichimoku_A"] = ichimoku.ichimoku_a()
    df["Ichimoku_B"] = ichimoku.ichimoku_b()
    df["Ichimoku_Base"] = ichimoku.ichimoku_base_line()

    # 8. Chaikin Money Flow (CMF)
    df["CMF_20"] = ta.volume.ChaikinMoneyFlowIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"],
        window=20,
    ).chaikin_money_flow()

    # 9. Rate of Change (ROC)
    df["ROC_12"] = ta.momentum.ROCIndicator(close=df["Close"], window=12).roc()

    # ==========================================
    # "IDEAL BB" STYLE INDICATORS (From Screenshot)
    # ==========================================
    # 1. Long-term 120-period Moving Average (The Baseline)
    df["MA_120"] = ta.trend.EMAIndicator(close=df["Close"], window=120).ema_indicator()

    # 2. The Trend Ribbon (Fast 12 EMA vs Slow 24 EMA)
    df["Ribbon_Fast"] = ta.trend.EMAIndicator(
        close=df["Close"], window=12
    ).ema_indicator()
    df["Ribbon_Slow"] = ta.trend.EMAIndicator(
        close=df["Close"], window=24
    ).ema_indicator()

    # 3. Ribbon State (Bullish/Bearish Cloud)
    # 1 = Bullish (Green Cloud), -1 = Bearish (Red Cloud)
    df["Ribbon_State"] = np.where(df["Ribbon_Fast"] > df["Ribbon_Slow"], 1, -1)

    # 4. Long-term Bollinger Bands (120, 2)
    bb_120 = ta.volatility.BollingerBands(close=df["Close"], window=120, window_dev=2)
    df["BB_120_Upper"] = bb_120.bollinger_hband()
    df["BB_120_Lower"] = bb_120.bollinger_lband()

    df.dropna(inplace=True)
    return df


def feature_deflation(df: pd.DataFrame, threshold=0.85) -> pd.DataFrame:
    """
    PATH 4: Drops highly correlated features to prevent multicollinearity (network confusion),
    while explicitly protecting critical Mean Reversion indicators.
    """
    print("Deflating redundant features...")

    cols_to_check = [
        col
        for col in df.columns
        if not col.startswith("target_")
        and not col.startswith("future_")
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Calculate correlation matrix
    corr_matrix = df[cols_to_check].corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # --- THE FIX: AI SIGHT RESTORATION ---
    # These features are legally protected. The AI must be allowed to see them
    # to understand when a stock is mathematically "oversold" or "overbought."
    critical_features = [
        "BB_Lower",
        "BB_Upper",
        "BB_Mid",
        "BB_Position",
        "RSI_14",
        "ATR",
        "MACD_Hist",
        "EMA_Cross",
        "MFI_14",
        "Stoch_K",
        "CMF_20",
    ]

    # Only drop the column if it crosses the threshold AND is not on the critical list
    to_drop = [
        column
        for column in upper.columns
        if any(upper[column] > threshold) and column not in critical_features
    ]

    if to_drop:
        print(f"Dropped {len(to_drop)} highly correlated features: {to_drop}")
    else:
        print("No redundant features found above threshold.")

    return df.drop(columns=to_drop)
