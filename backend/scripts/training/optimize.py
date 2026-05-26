# optimize.py
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import pandas as pd
import argparse
import optuna
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from src.data_ingestion.market_data import (
    fetch_historical_data,
    apply_dynamic_triple_barrier,
)
from src.data_ingestion.technical_indicators import (
    add_advanced_features,
    feature_deflation,
)
from src.models.neural.fusion_network import build_fusion_model
from src.features.sequence_builder import create_time_series_sequences
from datetime import datetime

# ==========================================
# UNIVERSAL CLI ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="Hybrid 5-Model Ensemble Optimizer")
parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
parser.add_argument(
    "--trials", type=int, default=50, help="Number of Optuna trials (default: 50)"
)
parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")
parser.add_argument(
    "--end", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date"
)
args = parser.parse_args()

TICKER = args.ticker.upper()
N_TRIALS = args.trials
START = args.start
END = args.end

# ==========================================
# FETCH DATA ONCE
# ==========================================
print(f"--- Preparing Optimization Data for {TICKER} ---")
df_raw = fetch_historical_data(TICKER, start_date=START, end_date=END)
df_features = add_advanced_features(df_raw.copy())


# ==========================================
# OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    # 1. Labeling Hyperparameters
    tp_mult = trial.suggest_categorical("tp_atr_multiplier", [1.0, 1.5, 2.0, 3.0])
    sl_mult = trial.suggest_categorical("sl_atr_multiplier", [0.5, 1.0, 1.5, 2.0])
    horizon = trial.suggest_categorical("horizon", [5, 8, 12, 15])

    try:
        df_labeled = apply_dynamic_triple_barrier(
            df_features.copy(),
            tp_atr_multiplier=tp_mult,
            sl_atr_multiplier=sl_mult,
            horizon=horizon,
        )
    except Exception:
        return 0.0

    # 2. Hybrid Model Hyperparameters (EXPANDED)
    dl_config = {
        "data": {"time_steps": 60, "max_seq_length": 128},
        "model": {
            # LSTM Params
            "lstm_units_1": trial.suggest_categorical("lstm_u1", [32, 64, 128, 256]),
            "lstm_units_2": trial.suggest_categorical("lstm_u2", [32, 64, 96, 128]),
            "lstm_dropout_1": trial.suggest_categorical("lstm_d1", [0.1, 0.2, 0.3, 0.4]),
            "lstm_dropout_2": trial.suggest_categorical("lstm_d2", [0.1, 0.2, 0.3, 0.4]),
            # CNN Params
            "cnn_filters_1": trial.suggest_categorical("cnn_f1", [16, 32, 48, 64]),
            "cnn_filters_2": trial.suggest_categorical("cnn_f2", [32, 64, 96, 128]),
            "cnn_kernel": trial.suggest_categorical("cnn_k", [2, 3, 5, 7]),
            "cnn_dense": trial.suggest_categorical("cnn_d", [32, 64, 96, 128]),
            # Transformer Params
            "trans_head_size": trial.suggest_categorical("tr_hs", [64, 128, 256, 512]),
            "trans_heads": trial.suggest_categorical("tr_h", [2, 4, 8, 16]),
            "trans_ff_dim": trial.suggest_categorical("tr_ff", [64, 128, 256, 512]),
            "trans_dropout": trial.suggest_categorical("tr_d", [0.05, 0.1, 0.2, 0.3]),
            # Fusion Params
            "dense_units_1": trial.suggest_categorical("dense_1", [64, 128, 256, 512]),
            "dense_units_2": trial.suggest_categorical("dense_2", [32, 64, 96, 128]),
            "dropout_rate": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4]),
            "learning_rate": trial.suggest_categorical("lr", [5e-5, 1e-4, 1e-3, 5e-3]),
        },
    }

    # 3. XGBoost Hyperparameters (EXPANDED)
    xgb_params = {
        "max_depth": trial.suggest_categorical("xgb_depth", [3, 6, 9, 12]),
        "learning_rate": trial.suggest_categorical("xgb_lr", [0.005, 0.05, 0.1, 0.3]),
        "n_estimators": trial.suggest_categorical("xgb_n", [100, 250, 500, 1000]),
        "subsample": trial.suggest_categorical("xgb_sub", [0.5, 0.7, 0.85, 1.0]),
        "colsample_bytree": trial.suggest_categorical("xgb_col", [0.5, 0.7, 0.85, 1.0]),
        "gamma": trial.suggest_categorical("xgb_gam", [0, 1, 3, 5]),
        "reg_alpha": trial.suggest_categorical("xgb_alp", [1e-8, 0.1, 1.0, 10.0]),
        "reg_lambda": trial.suggest_categorical("xgb_lam", [1e-8, 0.1, 1.0, 10.0]),
    }

    # Prepare sequences
    numerical_cols = [
        col
        for col in df_labeled.columns
        if not col.startswith("target_") and not col.startswith("future_")
    ]
    df_deflated = feature_deflation(df_labeled[numerical_cols])

    df_ready = pd.concat(
        [
            df_deflated,
            df_labeled[
                ["target_direction", "target_min", "target_max", "target_signal"]
            ],
        ],
        axis=1,
    )
    time_steps = 60
    ts_seq, y_dir, y_min, y_max = create_time_series_sequences(df_ready, time_steps)
    y_sig = df_ready["target_signal"].values[time_steps - 1 :]

    split = int(len(ts_seq) * 0.8)
    X_train_dl = [
        ts_seq[:split],
        ts_seq[:split],
        ts_seq[:split],
        ts_seq[:split],
        ts_seq[:split],
        ts_seq[:split],  # Dummy peer data
    ]
    Y_train_dl = [
        y_dir[:split],
        np.column_stack((y_min[:split], y_max[:split])),
        y_sig[:split],
    ]

    X_test_dl = [
        ts_seq[split:],
        ts_seq[split:],
        ts_seq[split:],
        ts_seq[split:],
        ts_seq[split:],
        ts_seq[split:],  # Dummy peer data
    ]
    Y_test_sig = y_sig[split:]

    # Train DL
    dl_config["data"]["num_features"] = ts_seq.shape[2]
    model = build_fusion_model(dl_config)
    model.fit(
        X_train_dl, Y_train_dl, epochs=2, verbose=0
    )  # Reduced to 2 epochs for speed

    # Train XGB
    X_xgb_train = ts_seq[:split, -1, :]
    X_xgb_test = ts_seq[split:, -1, :]
    xgb_model = xgb.XGBClassifier(**xgb_params, objective="multi:softprob", num_class=3)
    xgb_model.fit(X_xgb_train, y_sig[:split])

    # Ensemble Validation
    dl_preds = model.predict(X_test_dl, verbose=0)[2]
    xgb_preds = xgb_model.predict_proba(X_xgb_test)

    # Optimize weights
    w_dl = trial.suggest_float("weight_dl", 0.2, 0.8)
    ensemble_p = (dl_preds * w_dl) + (xgb_preds * (1 - w_dl))

    accuracy = accuracy_score(Y_test_sig, np.argmax(ensemble_p, axis=1))
    return accuracy


if __name__ == "__main__":
    from scripts.ops.clean_artifacts import clean_optimization_artifacts

    clean_optimization_artifacts(ticker=TICKER)

    print(f"\nStarting Hybrid 5-Model Optimization for {TICKER}...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBEST ACCURACY: {study.best_value * 100:.2f}%")

    os.makedirs("configs", exist_ok=True)
    with open(f"configs/optimized_params_{TICKER}.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Results saved to configs/optimized_params_{TICKER}.json")
