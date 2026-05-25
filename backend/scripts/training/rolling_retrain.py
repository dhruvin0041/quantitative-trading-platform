import os
import json
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from src.execution.live_inference import load_config, add_upgraded_features, FEATURE_COLUMNS
from src.models.neural.fusion_network import build_fusion_model
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from src.features.sequence_builder import create_time_series_sequences
from src.data_ingestion.market_data import fetch_historical_data, apply_dynamic_triple_barrier, get_sector_peer

def run_rolling_retrain(ticker="MSFT"):
    print(f"=== HYDRA ROLLING RETRAIN: {ticker} ===")
    config = load_config()
    
    # 1. Define 24-month rolling window
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2) # 2 years
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    print(f"Window: {start_str} to {end_str}")

    # 2. Fetch and Prepare Data
    df = fetch_historical_data(ticker, start_date=start_str, end_date=end_str)
    spy_df = yf.download('SPY', start=start_str, end=end_str, progress=False)
    vix_df = yf.download('^VIX', start=start_str, end=end_str, progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.droplevel(1)

    df = add_upgraded_features(df, spy_df, vix_df)
    peer_ticker = get_sector_peer(ticker)
    peer_df = fetch_historical_data(peer_ticker, start_date=start_str, end_date=end_str)
    peer_df = add_upgraded_features(peer_df, spy_df, vix_df)
    
    # Target labeling
    df['target_direction'] = (df['Close'].shift(-5) > df['Close'] * 1.01).astype(int)
    df['target_min'] = (df['Low'].rolling(5).min().shift(-5) - df['Close']) / df['Close']
    df['target_max'] = (df['High'].rolling(5).max().shift(-5) - df['Close']) / df['Close']
    df = apply_dynamic_triple_barrier(df)
    df = df.iloc[:-10].dropna()
    peer_df = peer_df.iloc[:-10].dropna()
    
    common_idx = df.index.intersection(peer_df.index)
    df_ready = df.loc[common_idx]
    peer_ready = peer_df.loc[common_idx][FEATURE_COLUMNS]

    # 3. Process Sequences
    scaler = StandardScaler()
    scaler.fit(df_ready[FEATURE_COLUMNS])
    
    time_steps = config["data"]["time_steps"]
    
    def get_seqs(df_split, peer_split):
        ts, y_dir, y_min, y_max = create_time_series_sequences(
            df_split[FEATURE_COLUMNS + ["target_direction", "target_min", "target_max", "target_signal"]], time_steps
        )
        peer_ts, _, _, _ = create_time_series_sequences(
            pd.concat([peer_split, df_split[["target_direction"]]], axis=1), time_steps
        )
        num_s, steps, feats = ts.shape
        ts_scaled = scaler.transform(ts.reshape(-1, feats)).reshape(num_s, steps, feats)
        peer_scaled = scaler.transform(peer_ts.reshape(-1, feats)).reshape(num_s, steps, feats)
        y_sig = df_split["target_signal"].values[time_steps-1:]
        y_ran = np.column_stack((y_min, y_max))
        return ts_scaled, peer_scaled, y_sig, y_dir, y_ran

    ts_scaled, peer_scaled, y_sig, y_dir, y_ran = get_seqs(df_ready, peer_ready)
    X_tabular = ts_scaled[:, -1, :]

    # 4. Retrain Models
    print("Training models on rolling window...")
    
    # XGBoost
    xgb_params = {'objective': "multi:softprob", 'num_class': 3, 'random_state': 42}
    if os.path.exists('configs/best_xgb_params.json'):
        with open('configs/best_xgb_params.json') as f:
            best = json.load(f)
            best['learning_rate'] = best.pop('lr', 0.03)
            best['min_child_weight'] = best.pop('min_child', 1)
            best.pop('colsample', None)
            xgb_params.update(best)

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tabular, y_sig)
    xgb_model.save_model("artifacts/xgb_ensemble.json")
    
    # LightGBM
    lgbm_model = LGBMClassifier(objective='multiclass', num_class=3, random_state=42, verbose=-1)
    lgbm_model.fit(X_tabular, y_sig)
    joblib.dump(lgbm_model, "artifacts/lgbm_agent.joblib")
    
    # Fusion (Neural)
    fusion_model = build_fusion_model(config)
    X_dl = [ts_scaled, ts_scaled, ts_scaled, ts_scaled, ts_scaled, peer_scaled]
    # Correct multi-head fit: [direction, range, signal]
    fusion_model.fit(x=X_dl, y=[y_dir, y_ran, y_sig], epochs=3, verbose=0)
    fusion_model.save_weights("artifacts/latest_fusion_weights.weights.h5")
    
    # Update scaler
    joblib.dump(scaler, "artifacts/latest_scaler.joblib")
    
    print(">>> Rolling retraining complete. All artifacts updated.")

if __name__ == "__main__":
    run_rolling_retrain()
