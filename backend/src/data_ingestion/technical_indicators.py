# src/data_ingestion/technical_indicators.py
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from typing import Any


def clean_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes yfinance MultiIndex columns to a single level."""
    if isinstance(df.columns, pd.MultiIndex):
        # If there's only one ticker, level 1 might be the ticker name
        if len(df.columns.levels[0]) == 1:
            df.columns = df.columns.get_level_values(0)
        else:
            # Flatten or pick first ticker if multiple present (fallback)
            df.columns = [f"{col[0]}" for col in df.columns.values]
    return df


def ensure_series(data: Any) -> pd.Series:
    """Ensures input is a pandas Series, picking first column if DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


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
    df["VIX"] = ensure_series(vix_df["Close"])
    df["VIX"] = (
        df["VIX"].ffill().fillna(0)
    )  # Fill gaps and edge cases without lookahead bias

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
    df["Vol_Anomaly"] = df["Volume"] / (df["Volume"].rolling(window=20).mean() + 1e-9)

    # --- 5. ALTERNATIVE DATA: 10-Year Treasury Yield (Macro Environment) ---
    if tnx_data is None:
        tnx_df = yf.download(
            "^TNX", start=df.index[0], end=df.index[-1], progress=False
        )
        tnx_df = clean_multiindex_columns(tnx_df)
    else:
        tnx_df = tnx_data

    df["Treasury_10Y"] = ensure_series(tnx_df["Close"])
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
        df["BB_Upper"] - df["BB_Lower"] + 1e-9
    )

    # Fast RSI (Makes the AI highly sensitive to sudden 3-day drops/spikes)
    df["RSI_Fast_3"] = ta.momentum.RSIIndicator(close=df["Close"], window=3).rsi()

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

    # ==========================================
    # PHASE 2: INSTITUTIONAL-GRADE FEATURES
    # ==========================================

    # --- Market Structure ---
    # Swing Highs and Lows (Window = 5)
    df["Swing_High"] = df["High"] == df["High"].rolling(window=5, center=True).max()
    df["Swing_Low"] = df["Low"] == df["Low"].rolling(window=5, center=True).min()
    df["Swing_High"] = df["Swing_High"].astype(int)
    df["Swing_Low"] = df["Swing_Low"].astype(int)

    # Higher Highs / Lower Lows (simplified rolling check)
    rolling_max = df["High"].rolling(20).max()
    rolling_min = df["Low"].rolling(20).min()
    df["Higher_High"] = (df["High"] >= rolling_max).astype(int)
    df["Lower_Low"] = (df["Low"] <= rolling_min).astype(int)

    # Break of Structure (BOS) / Change of Character (ChoCh) proxy
    df["BOS_Bullish"] = (
        (df["Close"] > df["High"].shift(1).rolling(10).max())
        & (df["SMA_50"] > df["MA_120"])
    ).astype(int)
    df["ChoCh_Bearish"] = (
        (df["Close"] < df["Low"].shift(1).rolling(10).min())
        & (df["SMA_50"] > df["MA_120"])
    ).astype(int)

    # --- Volatility Features ---
    # Realized Volatility (20 day rolling std of log returns * sqrt(252))
    df["Realized_Vol_20"] = df["Log_Ret"].rolling(window=20).std() * np.sqrt(252)

    # ATR Percentile (Where is current ATR relative to last 252 days)
    df["ATR_Percentile"] = (
        df["ATR"]
        .rolling(window=252)
        .apply(
            lambda x: (
                pd.Series(x).rank(pct=True).iloc[-1] if len(x.dropna()) > 0 else np.nan
            )
        )
    )

    # Volatility Expansion / Compression (Bollinger Band Width)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Mid"] + 1e-9)
    df["Vol_Expansion"] = (
        df["BB_Width"]
        > df["BB_Width"].rolling(20).mean() + df["BB_Width"].rolling(20).std()
    ).astype(int)
    df["Vol_Compression"] = (
        df["BB_Width"]
        < df["BB_Width"].rolling(20).mean() - df["BB_Width"].rolling(20).std()
    ).astype(int)
    df["Vol_Regime"] = np.where(
        df["Realized_Vol_20"] > df["Realized_Vol_20"].rolling(252).median(), 1, 0
    )

    # --- Statistical Features ---
    # Z-Score Price and Volume
    df["Z_Score_Price"] = (df["Close"] - df["Close"].rolling(20).mean()) / (
        df["Close"].rolling(20).std() + 1e-9
    )
    df["Z_Score_Volume"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (
        df["Volume"].rolling(20).std() + 1e-9
    )

    # Rolling Skewness and Kurtosis
    df["Skewness_20"] = df["Log_Ret"].rolling(window=20).skew()
    df["Kurtosis_20"] = df["Log_Ret"].rolling(window=20).kurt()

    # Hurst Exponent Proxy (Variance Ratio test simplification)
    # H < 0.5 mean reverting, H = 0.5 random walk, H > 0.5 trending
    var_5 = df["Log_Ret"].rolling(5).sum().rolling(20).var()
    var_1 = df["Log_Ret"].rolling(20).var()

    # Numerical Stability: log(ratio) can explode if denominator is tiny or numerator is <= 0
    ratio = var_5 / (5 * var_1 + 1e-9)
    df["Hurst_Proxy"] = np.log(np.maximum(ratio, 1e-9)) / np.log(5) + 0.5
    df["Hurst_Proxy"] = df["Hurst_Proxy"].fillna(0.5)

    # Fractal Dimension Proxy
    # D = (log(L) + log(2)) / log(N)
    high_low_20 = df["High"].rolling(20).max() - df["Low"].rolling(20).min()
    path_len = (df["High"] - df["Low"]).rolling(20).sum()
    df["Fractal_Dim"] = np.where(
        high_low_20 > 0, 1 + np.log(path_len / high_low_20) / np.log(20 * 2), 1.5
    )

    # Rolling Entropy (Shannon entropy of up/down days)
    up_days = (df["Log_Ret"] > 0).astype(int)
    p_up = up_days.rolling(20).mean()
    p_down = 1 - p_up
    # Clip to avoid log(0)
    p_up_c = np.clip(p_up, 1e-5, 1 - 1e-5)
    p_down_c = np.clip(p_down, 1e-5, 1 - 1e-5)
    df["Rolling_Entropy"] = -(p_up_c * np.log2(p_up_c) + p_down_c * np.log2(p_down_c))

    # Regime Persistence (days since MA_120 cross)
    cross_events = (df["Close"] > df["MA_120"]).astype(int).diff().abs()
    df["Regime_Persistence"] = cross_events.groupby(
        (cross_events == 1).cumsum()
    ).cumcount()

    # --- Liquidity Features ---
    # Dollar Volume
    df["Dollar_Volume"] = df["Close"] * df["Volume"]

    # Relative Volume (RVOL)
    df["Relative_Volume"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["Volume_Shock"] = (df["Relative_Volume"] > 3.0).astype(int)

    # Roll Spread Estimate (Roll 1984 effective spread proxy)
    # 2 * sqrt(-Cov(dP_t, dP_t-1))
    dp = df["Close"].diff()
    cov_dp = dp.rolling(20).cov(dp.shift(1))
    df["Roll_Spread"] = np.where(cov_dp < 0, 2 * np.sqrt(-cov_dp), 0)

    # Amihud Liquidity Proxy (Absolute return / Dollar Volume)
    # Add 1e-9 to prevent division by zero which causes inf/NaN and drops recent data
    df["Illiquidity_Amihud"] = df["Log_Ret"].abs() / (df["Dollar_Volume"] + 1e-9)
    df["Illiquidity_Amihud"] = (
        df["Illiquidity_Amihud"].rolling(20).mean().ffill().fillna(0)
    )

    # --- Cross Asset Features (SPY Beta & Correlation) ---
    try:
        spy_df = yf.download("SPY", start=df.index[0], end=df.index[-1], progress=False)
        spy_df = clean_multiindex_columns(spy_df)
        spy_ret = np.log(spy_df["Close"] / spy_df["Close"].shift(1))

        # Align index
        spy_ret = spy_ret.reindex(df.index).ffill().fillna(0)

        # Rolling Correlation
        df["SPY_Corr_20"] = df["Log_Ret"].rolling(20).corr(spy_ret)

        # Market Beta
        spy_var = spy_ret.rolling(20).var()
        df["Market_Beta_20"] = np.where(
            spy_var > 0, df["Log_Ret"].rolling(20).cov(spy_ret) / spy_var, 1.0
        )

        # Relative Performance
        df["Rel_Perf_SPY_20"] = df["Close"].pct_change(20) - spy_df["Close"].pct_change(
            20
        ).reindex(df.index).ffill().fillna(0)
    except Exception as e:
        print(f"Warning: Could not fetch SPY data for cross-asset features. {e}")
        df["SPY_Corr_20"] = 0.0
        df["Market_Beta_20"] = 1.0
        df["Rel_Perf_SPY_20"] = 0.0

    # Ensure no trailing NaNs from forward-fillable metrics
    df["VIX"] = df["VIX"].ffill().fillna(0)
    df["Treasury_10Y"] = df["Treasury_10Y"].ffill().fillna(0)

    # ==========================================
    # FINAL NUMERICAL STABILITY GUARD
    # ==========================================
    # Replace inf and -inf with NaN to prevent ML model crashes
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Forward fill then zero fill for consistency
    df = df.ffill().fillna(0)

    # Return full dataframe to preserve chart alignment
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
