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
from src.models.fusion_network import build_fusion_model
from src.models.dqn_agent import DQNAgent
from src.data_ingestion.market_data import fetch_historical_data, get_sector_peer
from src.data_ingestion.technical_indicators import add_advanced_features


def load_config():
    with open("configs/model_params.yaml", "r") as file:
        return yaml.safe_load(file)


def fetch_live_data(ticker, config):
    print(f"Fetching live market data for {ticker}...")
    df = fetch_historical_data(
        ticker,
        start_date="2023-01-01",
        end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
    )
    df = add_advanced_features(df)

    # NEW: Lead-Lag Peer Context Data
    peer_ticker = get_sector_peer(ticker)
    peer_df = fetch_historical_data(
        peer_ticker,
        start_date="2023-01-01",
        end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
    )
    peer_df = add_advanced_features(peer_df)

    with open("configs/kept_features.json", "r") as f:
        kept_features = json.load(f)

    df_filtered = df.reindex(columns=kept_features).dropna()
    peer_filtered = peer_df.reindex(columns=kept_features).dropna()

    # Align indices
    common_idx = df_filtered.index.intersection(peer_filtered.index)
    df_filtered = df_filtered.loc[common_idx]
    peer_filtered = peer_filtered.loc[common_idx]

    scaler = joblib.load("latest_scaler.joblib")

    time_steps = config["data"]["time_steps"]

    recent_data = df_filtered.tail(time_steps).values
    peer_recent = peer_filtered.tail(time_steps).values

    scaled_data = scaler.transform(recent_data)

    # NEW: Properly scale peer data independently to avoid magnitude leakage
    from sklearn.preprocessing import StandardScaler
    peer_scaler = StandardScaler()
    peer_scaled = peer_scaler.fit_transform(peer_filtered.values)[-time_steps:]

    ts_sequence = scaled_data.reshape(1, time_steps, -1)
    peer_sequence = peer_scaled.reshape(1, time_steps, -1)
    tabular_row = scaled_data[-1].reshape(1, -1)
    current_price = df["Close"].iloc[-1]

    return ts_sequence, peer_sequence, tabular_row, current_price, config


def fetch_live_news(ticker, tokenizer, config):
    # Use the SEC-aware tokenizer
    input_ids, attention_masks, combined_text = tokenizer.tokenize_daily_news(
        "Market continues to show trend momentum.", ticker=ticker
    )
    return input_ids.reshape(1, -1), attention_masks.reshape(1, -1), combined_text


def main():
    ticker = "AAPL"
    config = load_config()

    # 1. Fetch
    ts_seq, peer_seq, tabular, price, config = fetch_live_data(ticker, config)
    ids, masks, news = fetch_live_news(ticker, None, config)

    # 2. Load Models
    with open("configs/kept_features.json", "r") as f:
        kept_features = json.load(f)
    config["data"]["num_features"] = len(kept_features)

    dl_model = build_fusion_model(config)
    dl_model.load_weights("latest_fusion_weights.weights.h5")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("xgb_ensemble.json")

    dqn_agent = DQNAgent(len(kept_features) + 3 + 3)
    try:
        dqn_agent.load("dqn_model.pth")
    except Exception:
        pass

    # 3. Predict
    dl_p = dl_model.predict([ts_seq, ts_seq, ts_seq, peer_seq, ids, masks], verbose=0)[2][0]
    xgb_p = xgb_model.predict_proba(tabular)[0]

    state = np.hstack((tabular[0], dl_p, xgb_p))
    dqn_action = dqn_agent.act(state)

    # Weights
    try:
        with open("configs/model_accuracies.json", "r") as f:
            accs = json.load(f)
    except Exception:
        accs = {"dl_accuracy": 0.5, "xgb_accuracy": 0.5, "dqn_accuracy": 0.5}

    total_acc = sum(accs.values())
    w_dl = accs["dl_accuracy"] / total_acc
    w_xgb = accs["xgb_accuracy"] / total_acc
    w_dqn = accs["dqn_accuracy"] / total_acc

    ensemble_p = (dl_p * w_dl) + (xgb_p * w_xgb)
    dqn_p = np.zeros(3)
    dqn_p[dqn_action] = 1.0
    ensemble_p = (ensemble_p * (1 - w_dqn)) + (dqn_p * w_dqn)

    signal_idx = np.argmax(ensemble_p)
    confidence = ensemble_p[signal_idx]

    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    print("=" * 40)
    print(f"HYBRID ENSEMBLE REPORT: {ticker}")
    print(f"Price: ${price:.2f}")
    print(f"Final Action: {signal_map[signal_idx]} ({confidence * 100:.1f}%)")
    print("-" * 20)
    print(f"DL Suggests: {signal_map[np.argmax(dl_p)]}")
    print(f"XGB Suggests: {signal_map[np.argmax(xgb_p)]}")
    print(f"DQN Suggests: {signal_map[dqn_action]}")
    print("=" * 40)


if __name__ == "__main__":
    main()

    main()
nt(f"XGB Suggests: {signal_map[np.argmax(xgb_p)]}")
    print(f"DQN Suggests: {signal_map[dqn_action]}")
    print("=" * 40)


if __name__ == "__main__":
    main()

    main()
