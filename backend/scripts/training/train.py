import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
import yfinance as yf
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import time
import argparse
from sklearn.preprocessing import StandardScaler
from src.execution.live_inference import (
    load_config,
    add_upgraded_features,
    FEATURE_COLUMNS,
)

from src.models.neural.fusion_network import build_fusion_model
from src.models.rl.dqn_agent import DQNAgent
from src.features.sequence_builder import create_time_series_sequences
from src.data_ingestion.market_data import (
    fetch_historical_data,
    apply_dynamic_triple_barrier,
    get_sector_peer,
)

from scripts.ops.clean_artifacts import main as run_cleanup
from scripts.training.optimize_models import (
    run_optimization as run_bayesian_optimization,
)
from scripts.training.optimize import run_optuna_optimization

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

mlflow.set_experiment("hydra_terminal_signals")


def prepare_data(ticker, config):
    print(f"--- Preparing Data for {ticker} ---")
    df = fetch_historical_data(ticker, start_date="2019-01-01", end_date="2024-07-01")

    spy_df = yf.download(
        "SPY", start="2019-01-01", end="2024-07-01", interval="1d", progress=False
    )
    vix_df = yf.download(
        "^VIX", start="2019-01-01", end="2024-07-01", interval="1d", progress=False
    )
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.droplevel(1)

    df = add_upgraded_features(df, spy_df, vix_df)
    peer_ticker = get_sector_peer(ticker)
    peer_df = fetch_historical_data(
        peer_ticker, start_date="2019-01-01", end_date="2024-07-01"
    )
    peer_df = add_upgraded_features(peer_df, spy_df, vix_df)

    # ROOT CAUSE 4: Fix Target Label Lookahead
    df["target_direction"] = (df["Close"].shift(-5) > df["Close"] * 1.02).astype(int)
    df["target_min"] = (df["Low"].rolling(5).min().shift(-5) - df["Close"]) / df[
        "Close"
    ]
    df["target_max"] = (df["High"].rolling(5).max().shift(-5) - df["Close"]) / df[
        "Close"
    ]
    df = apply_dynamic_triple_barrier(df)

    # Drop the last 10 rows (horizon=10) BEFORE sequence generation
    df = df.iloc[:-10]
    peer_df = peer_df.iloc[:-10]

    kept_cols = FEATURE_COLUMNS

    with open("configs/kept_features.json", "w") as f:
        json.dump(kept_cols, f)

    # Align indices
    common_idx = df.index.intersection(peer_df.index)
    df_ready = df.loc[common_idx].copy()
    peer_ready = peer_df.loc[common_idx][kept_cols].ffill().fillna(0)

    # Temporal split
    TRAIN_END = "2023-12-31"
    VAL_END = "2024-06-30"

    df_train = df_ready[df_ready.index <= TRAIN_END]
    df_val = df_ready[(df_ready.index > TRAIN_END) & (df_ready.index <= VAL_END)]

    peer_train = peer_ready[peer_ready.index <= TRAIN_END]
    peer_val = peer_ready[
        (peer_ready.index > TRAIN_END) & (peer_ready.index <= VAL_END)
    ]

    time_steps = config["data"]["time_steps"]

    # Fit scaler ONLY on train
    scaler = StandardScaler()
    scaler.fit(df_train[kept_cols])
    joblib.dump(scaler, "artifacts/latest_scaler.joblib")

    def process_split(df_split, peer_split):
        if len(df_split) <= time_steps:
            return None, None, None, None, None, None

        ts, y_dir, y_min, y_max = create_time_series_sequences(
            df_split[
                kept_cols
                + ["target_direction", "target_min", "target_max", "target_signal"]
            ],
            time_steps,
        )
        peer_ts, _, _, _ = create_time_series_sequences(
            pd.concat([peer_split, df_split[["target_direction"]]], axis=1), time_steps
        )

        # Scale
        num_s, steps, feats = ts.shape
        ts_scaled = scaler.transform(ts.reshape(-1, feats)).reshape(num_s, steps, feats)
        peer_scaled = scaler.transform(peer_ts.reshape(-1, feats)).reshape(
            num_s, steps, feats
        )

        y_sig = df_split["target_signal"].values[time_steps - 1 :]
        y_ran = np.column_stack((y_min, y_max))
        return ts_scaled, peer_scaled, y_sig, y_dir, y_ran, feats

    ts_train, peer_train, y_sig_train, y_dir_train, y_ran_train, features = (
        process_split(df_train, peer_train)
    )
    ts_val, peer_val, y_sig_val, y_dir_val, y_ran_val, _ = process_split(
        df_val, peer_val
    )

    config["data"]["num_features"] = features
    return (
        ts_train,
        peer_train,
        y_sig_train,
        y_dir_train,
        y_ran_train,
        ts_val,
        peer_val,
        y_sig_val,
        y_dir_val,
        y_ran_val,
    ), config


def train_dqn(X_dl, Y_dl, dl_model, xgb_model, scaler, kept_features):
    print("\n--- Training DQN ---")
    X_tabular = X_dl[0][:, -1, :]
    dl_preds = dl_model.predict(X_dl, verbose=0)[2]
    xgb_preds = xgb_model.predict_proba(X_tabular)

    state_matrix = np.hstack((X_tabular, dl_preds, xgb_preds))
    agent = DQNAgent(state_matrix.shape[1])

    for e in range(30):
        state = state_matrix[0]
        for t in range(min(500, len(state_matrix) - 1)):
            action = agent.act(state)
            next_state = state_matrix[t + 1]
            reward = 1 if action == Y_dl[0][t] else (-1 if action != 1 else 0)
            agent.remember(
                state, action, reward, next_state, t == len(state_matrix) - 2
            )
            state = next_state
            if len(agent.memory) > 32:
                agent.replay()
        print(f"DQN Episode {e + 1} complete")
    agent.save("artifacts/dqn_model.pth")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Unified Training Pipeline")
    parser.add_argument(
        "--ticker", type=str, default="RELIANCE.NS", help="Stock ticker symbol"
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of Optuna trials"
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    n_trials = args.trials

    pipeline_start = time.time()

    # ==========================================
    # STEP 1: CLEAN ARTIFACTS
    # ==========================================
    print(f"\n[1/4] Cleaning artifacts for {ticker}...")
    step_start = time.time()
    try:
        run_cleanup([])
        print(f"  >>> Step 1 Complete ({time.time() - step_start:.2f}s)")
    except Exception as e:
        print(f"  [FATAL ERROR] Step 1 Failed: {e}")
        return

    # ==========================================
    # STEP 2: OPTIMIZE MODELS (Bayesian)
    # ==========================================
    print("\n[2/4] Optimizing branch models (XGB, LGBM, CatBoost, RF)...")
    step_start = time.time()
    config = load_config()

    # We must prepare data FIRST because optimize_models.py depends on artifacts
    data, updated_config = prepare_data(ticker, config)
    (
        ts_train,
        peer_train,
        y_sig_train,
        y_dir_train,
        y_ran_train,
        ts_val,
        peer_val,
        y_sig_val,
        y_dir_val,
        y_ran_val,
    ) = data

    # Save training data for optimization (required by optimize_models.py)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(ts_train[:, -1, :], "artifacts/X_train_tabular.joblib")
    joblib.dump(y_sig_train, "artifacts/y_train_sig.joblib")
    joblib.dump(ts_val[:, -1, :], "artifacts/X_val_tabular.joblib")
    joblib.dump(y_sig_val, "artifacts/y_val_sig.joblib")

    if not run_bayesian_optimization():
        print("  [FATAL ERROR] Step 2 Failed.")
        return
    print(f"  >>> Step 2 Complete ({time.time() - step_start:.2f}s)")

    # ==========================================
    # STEP 3: OPTUNA OPTIMIZATION
    # ==========================================
    print(f"\n[3/4] Running Optuna optimization for {ticker}...")
    step_start = time.time()
    if not run_optuna_optimization(ticker=ticker, n_trials=n_trials):
        print("  [FATAL ERROR] Step 3 Failed.")
        return
    print(f"  >>> Step 3 Complete ({time.time() - step_start:.2f}s)")

    # ==========================================
    # STEP 4: FINAL TRAINING
    # ==========================================
    print(f"\n[4/4] Training final models for {ticker}...")
    step_start = time.time()

    print(f"Train: {ts_train.shape}, Val: {ts_val.shape}")

    # 6 Inputs: LSTM, CNN, Transformer, TCN, PatchTST, Peer
    X_train = [ts_train, ts_train, ts_train, ts_train, ts_train, peer_train]
    Y_train = [y_dir_train, y_ran_train, y_sig_train]

    X_val = [ts_val, ts_val, ts_val, ts_val, ts_val, peer_val]

    print("\n--- Training Deep Learning Ensemble ---")
    with mlflow.start_run(run_name=f"DL_FUSION_{ticker}"):
        mlflow.log_params(updated_config["model"])
        mlflow.log_param("time_steps", updated_config["data"]["time_steps"])
        model = build_fusion_model(updated_config)
        history = model.fit(
            x=X_train, y=Y_train, epochs=30, validation_split=0.1, verbose=1
        )

        # Log final metrics
        for metric, values in history.history.items():
            mlflow.log_metric(f"final_{metric}", values[-1])

        model.save_weights("artifacts/latest_fusion_weights.weights.h5")
        mlflow.tensorflow.log_model(model, "fusion_model")

    print("\n--- Training TFT Quantile Forecaster ---")
    with mlflow.start_run(run_name=f"TFT_QUANTILE_{ticker}"):
        from src.models.neural.tft_agent import build_tft_branch, total_quantile_loss

        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        tft_input, tft_output = build_tft_branch(
            time_steps=updated_config["data"]["time_steps"],
            num_features=updated_config["data"]["num_features"],
        )
        tft_model = tf.keras.Model(inputs=tft_input, outputs=tft_output)
        tft_model.compile(optimizer="adam", loss=total_quantile_loss(quantiles))

        # Train on actual price returns (Regression)
        tft_model.fit(
            X_train[0], Y_train[1][:, 1], epochs=30, validation_split=0.1, verbose=1
        )
        tft_model.save_weights("artifacts/tft_quantile_weights.weights.h5")
        mlflow.tensorflow.log_model(tft_model, "tft_model")

    print("\n--- Training XGBoost Branch ---")
    with mlflow.start_run(run_name=f"XGB_AGENT_{ticker}"):
        # Load best params if they exist
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
        }
        try:
            with open("configs/best_xgb_params.json") as f:
                best_params = json.load(f)
                best_params["learning_rate"] = best_params.pop("lr", 0.03)
                best_params["min_child_weight"] = best_params.pop("min_child", 1)
                best_params.pop("colsample", None)
                xgb_params.update(best_params)
                print(f"Using optimized XGB params: {xgb_params}")
        except Exception:
            print("Optimized XGB params not found. Using defaults.")

        X_xgb_train = ts_train[:, -1, :]
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_xgb_train, y_sig_train)
        xgb_model.save_model("artifacts/xgb_ensemble.json")
        mlflow.log_metric(
            "train_accuracy", float(xgb_model.score(X_xgb_train, y_sig_train))
        )
        mlflow.sklearn.log_model(xgb_model, "xgb_model")

    print("\n--- Training LightGBM Branch ---")
    with mlflow.start_run(run_name=f"LGBM_AGENT_{ticker}"):
        lgbm_params = {
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
            "verbose": -1,
        }
        try:
            with open("configs/best_lgbm_params.json") as f:
                best_params = json.load(f)
                best_params["learning_rate"] = best_params.pop("lr", 0.03)
                best_params["min_child_samples"] = best_params.pop("min_child", 20)
                best_params.pop("colsample", None)
                lgbm_params.update(best_params)
                print(f"Using optimized LGBM params: {lgbm_params}")
        except Exception:
            print("Optimized LGBM params not found. Using defaults.")

        from lightgbm import LGBMClassifier

        lgbm_model = LGBMClassifier(**lgbm_params)
        lgbm_model.fit(X_xgb_train, y_sig_train)
        joblib.dump(lgbm_model, "artifacts/lgbm_agent.joblib")
        mlflow.sklearn.log_model(lgbm_model, "lgbm_model")

    with mlflow.start_run(run_name=f"DQN_AGENT_{ticker}"):
        dqn_agent = train_dqn(X_val, (y_sig_val,), model, xgb_model, None, None)

    with mlflow.start_run(run_name=f"META_ENSEMBLE_{ticker}"):
        print("\n--- Training Meta-Ensemble ---")
        dl_preds = model.predict(X_val, verbose=0)[2]
        X_xgb_val = ts_val[:, -1, :]
        xgb_preds = xgb_model.predict_proba(X_xgb_val)
        lgbm_preds = lgbm_model.predict_proba(X_xgb_val)

        dqn_preds = []
        for state in np.hstack((X_xgb_val, dl_preds, xgb_preds)):
            action = dqn_agent.act(state)
            p = [0.0, 0.0, 0.0]
            p[action] = 1.0
            dqn_preds.append(p)
        dqn_preds = np.array(dqn_preds)

        from src.models.ensemble.meta_ensemble import MetaEnsemble

        meta = MetaEnsemble()

        X_meta_train = []
        for i in range(len(y_sig_val)):
            features = meta._prepare_meta_features(
                {
                    "LSTM": dl_preds[i],
                    "XGBoost": xgb_preds[i],
                    "LightGBM": lgbm_preds[i],
                    "DQN": dqn_preds[i],
                },
                1,
            )
            X_meta_train.append(features[0])

        X_meta_train = np.array(X_meta_train)
        y_meta_train = y_sig_val

        meta.fit(X_meta_train, y_meta_train)
        meta.save("artifacts/meta_ensemble.joblib")
        mlflow.sklearn.log_model(meta.meta_learner, "meta_ensemble")
        print("Meta-Ensemble saved to artifacts/meta_ensemble.joblib")

    # ==========================================
    # STEP 5: SAVE ACTIVE TICKER
    # ==========================================
    print("\n[5/5] Saving active ticker metadata for frontend...")
    try:
        from src.data_ingestion.universes import UNIVERSES_METADATA
        market = "us"
        for m_id, m_dict in UNIVERSES_METADATA.items():
            if ticker in m_dict:
                market = m_id
                break
        
        with open("configs/active_ticker.json", "w") as f:
            json.dump({"ticker": ticker, "market": market}, f)
        print(f"  >>> Active ticker saved: {ticker} ({market})")
    except Exception as e:
        print(f"  [ERROR] Could not save active ticker metadata: {e}")

    print(f"\n  >>> Steps Complete ({time.time() - pipeline_start:.2f}s)")
    print(
        f"\n>>> UNIFIED TRAINING PIPELINE COMPLETE ({time.time() - pipeline_start:.2f}s) <<<"
    )
    print("MLflow UI: run 'mlflow ui' to view experiment results")


if __name__ == "__main__":
    main()
