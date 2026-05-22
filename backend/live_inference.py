# live_inference.py
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from src.models.fusion_network import build_fusion_model
from src.models.dqn_agent import DQNAgent
from src.data_ingestion.market_data import fetch_historical_data, get_sector_peer


FEATURE_COLUMNS = [
    'Return', 'Volume_Change', 'High_Low', 'MA20', 'MA50',
    'MA20_vs_MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'Stoch_K', 'Stoch_D', 'BB_Width', 'BB_Position', 'ATR',
    'ATR_Pct', 'OBV_Change', 'Volume_Ratio', 'EMA9_vs_EMA21',
    'Price_vs_EMA9', 'Price_vs_EMA21', 'ADX', 'Candle_Body',
    'Upper_Shadow', 'Lower_Shadow', 'Gap', 'SPY_Return',
    'VIX_Level', 'VIX_Change', 'Relative_Strength'
]


def load_config():
    with open("configs/model_params.yaml", "r") as file:
        return yaml.safe_load(file)


def detect_regime(spy_data):
    spy_ma50 = spy_data['Close'].rolling(50).mean().iloc[-1]
    spy_ma200 = spy_data['Close'].rolling(200).mean().iloc[-1]
    spy_current = spy_data['Close'].iloc[-1]
    
    if spy_current > spy_ma50 > spy_ma200:
        return 'BULL', 0.55   # Normal threshold
    elif spy_current < spy_ma50 < spy_ma200:
        return 'BEAR', 0.68   # Require higher conviction in downtrend
    else:
        return 'NEUTRAL', 0.60  # Transitioning market


def is_near_earnings(ticker):
    try:
        stock = yf.Ticker(ticker)
        earnings_dates = stock.earnings_dates
        if earnings_dates is None or len(earnings_dates) == 0:
            return False
        next_earnings = earnings_dates.index[0]
        days_to_earnings = abs((next_earnings - pd.Timestamp.now()).days)
        return days_to_earnings <= 2
    except:
        return False


def add_upgraded_features(df, spy_df, vix_df):
    # Momentum Indicators
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    RS = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + RS))

    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-9)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # Volatility Indicators
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Mid'] + 1e-9)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)

    df['TR'] = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / (df['Close'] + 1e-9)

    # Volume Indicators
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Change'] = df['OBV'].pct_change()

    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA20'] + 1e-9)

    # Trend Indicators
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    df['EMA9_vs_EMA21'] = (df['EMA9'] - df['EMA21']) / (df['Close'] + 1e-9)
    df['Price_vs_EMA9'] = (df['Close'] - df['EMA9']) / (df['Close'] + 1e-9)
    df['Price_vs_EMA21'] = (df['Close'] - df['EMA21']) / (df['Close'] + 1e-9)

    plus_DM = df['High'].diff()
    minus_DM = -df['Low'].diff()
    plus_DM[plus_DM < 0] = 0
    minus_DM[minus_DM < 0] = 0
    TR14 = df['TR'].rolling(14).sum()
    plus_DI = 100 * (plus_DM.rolling(14).sum() / (TR14 + 1e-9))
    minus_DI = 100 * (minus_DM.rolling(14).sum() / (TR14 + 1e-9))
    DX = 100 * (abs(plus_DI - minus_DI) / (plus_DI + minus_DI + 1e-9))
    df['ADX'] = DX.rolling(14).mean()

    # Price Pattern Features
    df['Candle_Body'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-9)
    df['Upper_Shadow'] = (df['High'] - df[['Close','Open']].max(axis=1)) / (df['ATR'] + 1e-9)
    df['Lower_Shadow'] = (df[['Close','Open']].min(axis=1) - df['Low']) / (df['ATR'] + 1e-9)
    df['Gap'] = (df['Open'] - df['Close'].shift()) / (df['Close'].shift() + 1e-9)

    # Keep existing features
    df['Return'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['High_Low'] = df['High'] - df['Low']
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA20_vs_MA50'] = (df['MA20'] - df['MA50']) / (df['Close'] + 1e-9)

    # Market Context Features
    df['SPY_Return'] = spy_df['Close'].pct_change().reindex(df.index)
    df['VIX_Level'] = vix_df['Close'].reindex(df.index)
    df['VIX_Change'] = vix_df['Close'].pct_change().reindex(df.index)
    df['Relative_Strength'] = df['Return'] - df['SPY_Return']
    
    df[['SPY_Return','VIX_Level','VIX_Change','Relative_Strength']] = \
        df[['SPY_Return','VIX_Level','VIX_Change','Relative_Strength']].ffill().fillna(0)

    df.dropna(inplace=True)
    return df


def fetch_live_data(ticker, config):
    print(f"Fetching live market data for {ticker}...")
    df = fetch_historical_data(
        ticker,
        start_date="2022-01-01",
        end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
    )
    
    spy_df = yf.download('SPY', period='2y', interval='1d', progress=False)
    vix_df = yf.download('^VIX', period='2y', interval='1d', progress=False)
    
    if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.droplevel(1)

    df = add_upgraded_features(df, spy_df, vix_df)

    peer_ticker = get_sector_peer(ticker)
    peer_df = fetch_historical_data(
        peer_ticker,
        start_date="2022-01-01",
        end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
    )
    peer_df = add_upgraded_features(peer_df, spy_df, vix_df)

    df_filtered = df.reindex(columns=FEATURE_COLUMNS).dropna()
    peer_filtered = peer_df.reindex(columns=FEATURE_COLUMNS).dropna()

    common_idx = df_filtered.index.intersection(peer_filtered.index)
    df_filtered = df_filtered.loc[common_idx]
    peer_filtered = peer_filtered.loc[common_idx]

    scaler = joblib.load("latest_scaler.joblib")
    time_steps = config["data"]["time_steps"]

    recent_data = df_filtered.tail(time_steps).values
    peer_recent = peer_filtered.tail(time_steps).values

    scaled_data = scaler.transform(recent_data)
    peer_scaled = scaler.transform(peer_recent)

    ts_sequence = scaled_data.reshape(1, time_steps, -1)
    peer_sequence = peer_scaled.reshape(1, time_steps, -1)
    tabular_row = scaled_data[-1].reshape(1, -1)
    
    price_series = df["Close"].squeeze()
    current_price = float(price_series.iloc[-1])

    regime, req_conf = detect_regime(spy_df)

    current_volume = df['Volume'].iloc[-1]
    avg_volume_20d = df['Volume'].rolling(20).mean().iloc[-1]
    vol_ratio = current_volume / (avg_volume_20d + 1e-9)

    tech_snapshot = {
        "RSI": round(float(df['RSI'].iloc[-1]), 2),
        "MACD": round(float(df['MACD'].iloc[-1]), 2),
        "ATR": round(float(df['ATR'].iloc[-1]), 2),
        "BB_Position": round(float(df['BB_Position'].iloc[-1]), 2),
        "ADX": round(float(df['ADX'].iloc[-1]), 2),
        "Volume_Ratio": round(float(vol_ratio), 2)
    }

    return ts_sequence, peer_sequence, tabular_row, current_price, config, regime, req_conf, vol_ratio, tech_snapshot


def fetch_live_news(ticker, tokenizer, config):
    input_ids, attention_masks, combined_text = tokenizer.tokenize_daily_news(
        "Market continues to show trend momentum.", ticker=ticker
    )
    return input_ids.reshape(1, -1), attention_masks.reshape(1, -1), combined_text


def main():
    ticker = "MSFT"
    config = load_config()

    ts_seq, peer_seq, tabular, price, config, regime, req_conf, vol_ratio, tech = fetch_live_data(ticker, config)

    config["data"]["num_features"] = len(FEATURE_COLUMNS)

    dl_model = build_fusion_model(config)
    dl_model.load_weights("latest_fusion_weights.weights.h5")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("xgb_ensemble.json")
    
    lgbm_model = joblib.load("lgbm_agent.joblib")

    dl_p = dl_model.predict([ts_seq, ts_seq, ts_seq, peer_seq], verbose=0)[2][0]
    xgb_p = xgb_model.predict_proba(tabular)[0]
    lgbm_p = lgbm_model.predict_proba(tabular)[0]

    votes = []
    def get_vote(p):
        if p[2] > req_conf: return 'BUY'
        elif p[2] < (1 - req_conf): return 'SELL'
        else: return 'HOLD'
    
    votes.append(get_vote(dl_p))
    votes.append(get_vote(xgb_p))
    votes.append(get_vote(lgbm_p))

    buy_votes = votes.count('BUY')
    sell_votes = votes.count('SELL')

    if buy_votes >= 2: final_signal = 'BUY'
    elif sell_votes >= 2: final_signal = 'SELL'
    else: final_signal = 'VETOED'

    confidence = (sum([dl_p[2], xgb_p[2], lgbm_p[2]]) / 3) * 100
    
    signal_note = None
    if is_near_earnings(ticker):
        final_signal = 'HOLD'
        signal_note = 'Suppressed: Earnings window'
    elif vol_ratio < 0.7:
        final_signal = 'HOLD'
        signal_note = 'Suppressed: Low volume (ratio: {:.2f})'.format(vol_ratio)

    print("=" * 40)
    print(f"UPGRADED HYDRA REPORT: {ticker}")
    print(f"Price: ${price:.2f} | Regime: {regime} | Vol Ratio: {vol_ratio:.2f}")
    print(f"Final Action: {final_signal} ({confidence:.1f}%)")
    if signal_note: print(f"NOTE: {signal_note}")
    print("-" * 20)
    print(f"DL Signal: {votes[0]} ({dl_p[2]:.2f})")
    print(f"XGB Signal: {votes[1]} ({xgb_p[2]:.2f})")
    print(f"LGBM Signal: {votes[2]} ({lgbm_p[2]:.2f})")
    print("=" * 40)


if __name__ == "__main__":
    main()
