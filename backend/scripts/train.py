import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
import yfinance as yf
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from src.execution.live_inference import load_config, add_upgraded_features, FEATURE_COLUMNS

from src.models.fusion_network import build_fusion_model
from src.models.dqn_agent import DQNAgent
from src.models.lgbm_agent import train_lgbm_agent
from src.features.sequence_builder import create_time_series_sequences
from src.data_ingestion.market_data import (
    fetch_historical_data,
    apply_dynamic_triple_barrier,
    get_sector_peer,
)

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def prepare_data(ticker, config):
    print(f"--- Preparing Data for {ticker} ---")
    end_date = datetime.now().strftime("%Y-%m-%d")
    df = fetch_historical_data(ticker, start_date="2020-01-01", end_date=end_date)
    
    spy_df = yf.download('SPY', period='5y', interval='1d', progress=False)
    vix_df = yf.download('^VIX', period='5y', interval='1d', progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.droplevel(1)

    df = add_upgraded_features(df, spy_df, vix_df)
    peer_ticker = get_sector_peer(ticker)
    peer_df = fetch_historical_data(peer_ticker, start_date="2020-01-01", end_date=end_date)
    peer_df = add_upgraded_features(peer_df, spy_df, vix_df)

    df = apply_dynamic_triple_barrier(df)
    kept_cols = FEATURE_COLUMNS
    
    with open("configs/kept_features.json", "w") as f:
        json.dump(kept_cols, f)

    # Align indices
    common_idx = df.index.intersection(peer_df.index)
    df_ready = df.loc[common_idx].copy()
    peer_ready = peer_df.loc[common_idx][kept_cols].ffill().fillna(0)

    # Mock tokenizer
    class MockTokenizer:
        def tokenize_daily_news(self, text, ticker=None):
            return np.zeros(config["data"]["max_seq_length"]), np.zeros(config["data"]["max_seq_length"]), text
            
    tokenizer = MockTokenizer()
    input_ids_list = []
    attention_masks_list = []
    for _ in range(len(df_ready)):
        ids, masks, _ = tokenizer.tokenize_daily_news("Neutral.", ticker=ticker)
        input_ids_list.append(ids)
        attention_masks_list.append(masks)

    time_steps = config["data"]["time_steps"]
    ts_sequences, y_dir, y_min, y_max = create_time_series_sequences(
        df_ready[kept_cols + ["target_direction", "target_min", "target_max", "target_signal"]], time_steps
    )
    peer_sequences, _, _, _ = create_time_series_sequences(
        pd.concat([peer_ready, df_ready[["target_direction"]]], axis=1), time_steps
    )

    aligned_input_ids = np.array(input_ids_list[time_steps - 1:])
    aligned_attention_masks = np.array(attention_masks_list[time_steps - 1:])

    scaler = StandardScaler()
    num_samples, steps, features = ts_sequences.shape
    ts_sequences_reshaped = ts_sequences.reshape(-1, features)

    split_idx = int(num_samples * 0.8)
    scaler.fit(ts_sequences_reshaped[: split_idx * steps])
    joblib.dump(scaler, "artifacts/latest_scaler.joblib")

    ts_sequences_scaled = scaler.transform(ts_sequences_reshaped).reshape(
        num_samples, steps, features
    )
    peer_sequences_scaled = scaler.transform(
        peer_sequences.reshape(-1, features)
    ).reshape(num_samples, steps, features)

    y_signal = df_ready["target_signal"].values[time_steps-1:]
    y_range = np.column_stack((y_min, y_max))

    config["data"]["num_features"] = features
    return (
        ts_sequences_scaled,
        peer_sequences_scaled,
        aligned_input_ids,
        aligned_attention_masks,
        y_signal,
        y_dir,
        y_range,
    ), config


def train_dqn(X_dl, Y_dl, dl_model, xgb_model, scaler, kept_features):
    print("\n--- Training DQN ---")
    X_tabular = X_dl[0][:, -1, :]
    dl_preds = dl_model.predict(X_dl, verbose=0)[2]
    xgb_preds = xgb_model.predict_proba(X_tabular)

    state_matrix = np.hstack((X_tabular, dl_preds, xgb_preds))
    agent = DQNAgent(state_matrix.shape[1])

    for e in range(5):
        state = state_matrix[0]
        for t in range(min(500, len(state_matrix) - 1)):
            action = agent.act(state)
            next_state = state_matrix[t + 1]
            reward = 1 if action == Y_dl[0][t] else (-1 if action != 1 else 0)
            agent.remember(state, action, reward, next_state, t == len(state_matrix)-2)
            state = next_state
            if len(agent.memory) > 32: agent.replay(32)
        print(f"DQN Episode {e+1} complete")
    agent.save("artifacts/dqn_model.pth")


def main():
    ticker = "MSFT"
    config = load_config()
    data, updated_config = prepare_data(ticker, config)

    X_ts, X_peer, X_ids, X_masks, Y_sig, Y_dir, Y_range = data
    split = int(len(X_ts) * 0.8)
    X_train = [X_ts[:split], X_ts[:split], X_ts[:split], X_peer[:split]]
    Y_train = [Y_dir[:split], Y_range[:split], Y_sig[:split]]
    X_test = [X_ts[split:], X_ts[split:], X_ts[split:], X_peer[split:]]
    Y_test = [Y_dir[split:], Y_range[split:], Y_sig[split:]]

    print("\n--- Training Deep Learning Ensemble ---")
    model = build_fusion_model(updated_config)
    model.fit(x=X_train, y=Y_train, epochs=5, validation_split=0.1, verbose=1)
    model.save_weights("artifacts/latest_fusion_weights.weights.h5")

    print("\n--- Training XGBoost Branch ---")
    X_xgb_train = X_ts[:split, -1, :]
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", num_class=3)
    xgb_model.fit(X_xgb_train, Y_sig[:split])
    xgb_model.save_model("artifacts/xgb_ensemble.json")
    
    print("\n--- Training LightGBM Branch ---")
    train_lgbm_agent(X_xgb_train, Y_sig[:split])

    train_dqn(X_test, (Y_sig[split:],), model, xgb_model, None, None)
    print("\n>>> UNIFIED 3-MODEL TRAINING COMPLETE <<<")


if __name__ == "__main__":
    main()
