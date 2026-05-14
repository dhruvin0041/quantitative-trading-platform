# train.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from live_inference import load_config

from src.models.fusion_network import build_fusion_model
from src.models.dqn_agent import DQNAgent
from src.features.sequence_builder import create_time_series_sequences
from src.data_ingestion.nlp_processor import NewsTokenizer
from src.data_ingestion.market_data import (
    fetch_historical_data,
    apply_dynamic_triple_barrier,
    get_sector_peer,
)
from src.data_ingestion.technical_indicators import (
    add_advanced_features,
    feature_deflation,
)

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def prepare_data(ticker, config):
    print(f"--- Preparing Data for {ticker} ---")
    end_date = datetime.now().strftime("%Y-%m-%d")
    df = fetch_historical_data(ticker, start_date="2019-01-01", end_date=end_date)
    df = add_advanced_features(df)

    # NEW: Fetch Peer Data
    peer_ticker = get_sector_peer(ticker)
    peer_df = fetch_historical_data(
        peer_ticker, start_date="2019-01-01", end_date=end_date
    )
    peer_df = add_advanced_features(peer_df)

    # Use default params for triple barrier
    df = apply_dynamic_triple_barrier(
        df, tp_atr_multiplier=2.0, sl_atr_multiplier=1.0, horizon=10
    )

    numerical_cols = [
        col
        for col in df.columns
        if not col.startswith("target_") and not col.startswith("future_")
    ]
    df_deflated = feature_deflation(df[numerical_cols])
    kept_cols = df_deflated.columns.tolist()

    # Reindex peer_df to match kept_cols
    peer_filtered = peer_df.reindex(columns=kept_cols).ffill().bfill()

    # Align indices
    common_idx = df_deflated.index.intersection(peer_filtered.index)
    df_ready = pd.concat(
        [
            df_deflated.loc[common_idx],
            df.loc[common_idx][
                ["target_direction", "target_min", "target_max", "target_signal"]
            ],
        ],
        axis=1,
    )
    peer_ready = peer_filtered.loc[common_idx]

    with open("configs/kept_features.json", "w") as f:
        json.dump(kept_cols, f)

    tokenizer = NewsTokenizer(max_length=config["data"]["max_seq_length"])
    input_ids_list = []
    attention_masks_list = []
    for _ in range(len(df_ready)):
        ids, masks, _ = tokenizer.tokenize_daily_news(
            "Neutral market context.", ticker=ticker
        )
        input_ids_list.append(ids)
        attention_masks_list.append(masks)

    time_steps = config["data"]["time_steps"]
    ts_sequences, y_dir, y_min, y_max = create_time_series_sequences(
        df_ready, time_steps
    )
    peer_sequences, _, _, _ = create_time_series_sequences(
        pd.concat([peer_ready, df_ready[["target_direction"]]], axis=1), time_steps
    )

    aligned_input_ids = np.array(input_ids_list[time_steps:])
    aligned_attention_masks = np.array(attention_masks_list[time_steps:])

    scaler = StandardScaler()
    num_samples, steps, features = ts_sequences.shape
    ts_sequences_reshaped = ts_sequences.reshape(-1, features)

    split_idx = int(num_samples * 0.8)
    scaler.fit(ts_sequences_reshaped[: split_idx * steps])
    joblib.dump(scaler, "latest_scaler.joblib")

    ts_sequences_scaled = scaler.transform(ts_sequences_reshaped).reshape(
        num_samples, steps, features
    )
    peer_sequences_scaled = scaler.transform(
        peer_sequences.reshape(-1, features)
    ).reshape(num_samples, steps, features)

    y_signal = df_ready["target_signal"].values[time_steps:]
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
    print("\n--- Training Reinforcement Learning Agent (DQN) ---")
    # State: Technical features + Predictions from other models
    X_tabular = X_dl[0][:, -1, :]

    # Updated model predict signature: [ts, cnn, trans, peer, ids, masks]
    dl_preds = dl_model.predict(
        [X_dl[0], X_dl[0], X_dl[0], X_dl[1], X_dl[2], X_dl[3]], verbose=0
    )[2]
    xgb_preds = xgb_model.predict_proba(X_tabular)

    state_matrix = np.hstack((X_tabular, dl_preds, xgb_preds))
    state_size = state_matrix.shape[1]
    agent = DQNAgent(state_size)

    batch_size = 32
    episodes = 10

    for e in range(episodes):
        state = state_matrix[0]
        for t in range(len(state_matrix) - 1):
            action = agent.act(state)
            next_state = state_matrix[t + 1]
            reward = 0
            # Simple reward: if BUY (2) and direction is UP (from target_signal)
            target_signal = Y_dl[0][t]
            if action == target_signal:
                reward = 1
            elif action != 1:  # Penalize wrong non-hold actions
                reward = -1

            done = t == len(state_matrix) - 2
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        print(f"DQN Episode {e + 1}/{episodes} complete.")

    agent.save("dqn_model.pth")
    return agent


def apply_optimized_params(ticker, config):
    # Load optimized params if they exist
    opt_path = f"configs/optimized_params_{ticker}.json"
    if os.path.exists(opt_path):
        with open(opt_path, "r") as f:
            opt = json.load(f)
            # Map optimization keys to model keys
            m_opt = opt.get("best_params", {})
            config["model"].update(
                {
                    "lstm_units_1": m_opt.get("lstm_u1", 64),
                    "lstm_units_2": m_opt.get("lstm_u2", 64),
                    "lstm_dropout_1": m_opt.get("lstm_d1", 0.2),
                    "lstm_dropout_2": m_opt.get("lstm_d2", 0.2),
                    "cnn_filters_1": m_opt.get("cnn_f1", 32),
                    "cnn_filters_2": m_opt.get("cnn_f2", 64),
                    "cnn_kernel": m_opt.get("cnn_k", 3),
                    "cnn_dense": m_opt.get("cnn_d", 64),
                    "trans_head_size": m_opt.get("tr_hs", 128),
                    "trans_heads": m_opt.get("tr_h", 4),
                    "trans_ff_dim": m_opt.get("tr_ff", 128),
                    "trans_dropout": m_opt.get("tr_d", 0.1),
                    "dense_units_1": m_opt.get("dense_1", 128),
                    "dense_units_2": m_opt.get("dense_2", 64),
                    "dropout_rate": m_opt.get("dropout", 0.3),
                    "learning_rate": m_opt.get("lr", 0.001),
                }
            )
    return config


def main():
    ticker = "AAPL"
    config = load_config()
    data, updated_config = prepare_data(ticker, config)

    # Apply fine-tuned hyperparameters
    updated_config = apply_optimized_params(ticker, updated_config)

    X_ts, X_peer, X_ids, X_masks, Y_sig, Y_dir, Y_range = data

    # split data
    split = int(len(X_ts) * 0.8)
    # Model inputs: [ts, cnn, trans, peer, ids, masks]
    X_train = [
        X_ts[:split],
        X_ts[:split],
        X_ts[:split],
        X_peer[:split],
        X_ids[:split],
        X_masks[:split],
    ]
    Y_train = [Y_dir[:split], Y_range[:split], Y_sig[:split]]

    X_test = [
        X_ts[split:],
        X_ts[split:],
        X_ts[split:],
        X_peer[split:],
        X_ids[split:],
        X_masks[split:],
    ]
    Y_test = [Y_dir[split:], Y_range[split:], Y_sig[split:]]

    # 1. Train Deep Learning Ensemble
    print("\n--- Training Deep Learning Ensemble ---")
    model = build_fusion_model(updated_config)

    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model.fit(
        x=X_train,
        y=Y_train,
        epochs=config["training"]["epochs"],
        validation_split=0.2,
        callbacks=[early_stopper],
        verbose=2,
    )
    model.save_weights("latest_fusion_weights.weights.h5")

    # 2. Train XGBoost
    print("\n--- Training XGBoost Branch ---")
    X_xgb_train = X_ts[:split, -1, :]
    X_xgb_test = X_ts[split:, -1, :]
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", num_class=3)
    xgb_model.fit(X_xgb_train, Y_sig[:split])
    xgb_model.save_model("xgb_ensemble.json")

    # 3. Train DQN
    train_dqn((X_ts, X_peer, X_ids, X_masks), (Y_sig,), model, xgb_model, None, None)

    # 4. Save validation accuracies for dynamic weighting
    dl_test_preds = model.predict(X_test, verbose=0)[2]
    dl_acc = accuracy_score(Y_test[2], np.argmax(dl_test_preds, axis=1))

    xgb_test_preds = xgb_model.predict(X_xgb_test)
    xgb_acc = accuracy_score(Y_test[2], xgb_test_preds)

    accuracies = {
        "dl_accuracy": float(dl_acc),
        "xgb_accuracy": float(xgb_acc),
        "dqn_accuracy": 0.5,  # Default placeholder
    }
    with open("configs/model_accuracies.json", "w") as f:
        json.dump(accuracies, f)

    print("\n>>> 5-MODEL ENSEMBLE TRAINING COMPLETE <<<")


if __name__ == "__main__":
    main()
