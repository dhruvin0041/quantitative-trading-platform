# optimize_universal.py
import json
import os
import argparse
import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from src.data_ingestion.market_data import (
    fetch_historical_data,
    apply_dynamic_triple_barrier,
)
from src.data_ingestion.technical_indicators import add_advanced_features
from src.models.neural.fusion_network import build_fusion_model
from src.features.sequence_builder import create_time_series_sequences
from datetime import datetime
import contextlib

# A diverse set of tickers for universal optimization
UNIVERSAL_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "JPM",
    "BAC",
    "XOM",
    "TSLA",
    "WMT",
    "NVDA",
    "AMD",
]

parser = argparse.ArgumentParser(description="Universal Hybrid Ensemble Optimizer")
parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
parser.add_argument("--start", type=str, default="2019-01-01", help="Start date")
parser.add_argument(
    "--end", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date"
)
args = parser.parse_args()

# ==========================================
# FETCH DATA ONCE
# ==========================================
print(f"Building Universal Dataset from {len(UNIVERSAL_TICKERS)} tickers...")
ticker_dataframes = {}
for ticker in UNIVERSAL_TICKERS:
    try:
        df = fetch_historical_data(ticker, start_date=args.start, end_date=args.end)
        if df is not None and len(df) > 500:
            ticker_dataframes[ticker] = add_advanced_features(df)
            print(f"  [SUCCESS] {ticker}")
    except Exception:
        pass


def objective(trial):
    # 1. Universal Labeling Params
    tp_mult = trial.suggest_float("tp_atr_multiplier", 1.5, 3.5)
    sl_mult = trial.suggest_float("sl_atr_multiplier", 0.5, 1.5)
    horizon = trial.suggest_int("horizon", 5, 15)

    # 2. Universal Model Architecture
    dl_config = {
        "data": {"time_steps": 60, "max_seq_length": 128},
        "model": {
            "dense_units_1": trial.suggest_int("dense_1", 128, 256),
            "dense_units_2": trial.suggest_int("dense_2", 64, 128),
            "dropout_rate": trial.suggest_float("dropout", 0.2, 0.4),
            "learning_rate": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        },
    }

    # Process all tickers
    all_ts_seq = []
    all_y_sig = []

    for ticker, df in ticker_dataframes.items():
        try:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                df_labeled = apply_dynamic_triple_barrier(
                    df.copy(),
                    tp_atr_multiplier=tp_mult,
                    sl_atr_multiplier=sl_mult,
                    horizon=horizon,
                )

                # We use a static kept_features list for universal optimization consistency
                [
                    col
                    for col in df_labeled.columns
                    if not col.startswith("target_") and not col.startswith("future_")
                ]

                ts_seq, _, _, _ = create_time_series_sequences(df_labeled, 60)
                y_sig = df_labeled["target_signal"].values[60:]

                all_ts_seq.append(ts_seq)
                all_y_sig.append(y_sig)
        except Exception:
            continue
    X_total = np.concatenate(all_ts_seq)
    Y_total = np.concatenate(all_y_sig)

    # Shuffle for universal learning
    indices = np.arange(len(X_total))
    np.random.shuffle(indices)
    X_total = X_total[indices]
    Y_total = Y_total[indices]

    split = int(len(X_total) * 0.8)
    X_train, X_test = X_total[:split], X_total[split:]
    Y_train, Y_test = Y_total[:split], Y_total[split:]

    # Dummies for news
    X_train_dl = [
        X_train,
        X_train,
        X_train,
        # np.zeros((len(X_train), 128)),
        # np.zeros((len(X_train), 128)),
        np.zeros((len(X_train), 60, X_total.shape[2])),  # Dummy peer data
    ]
    X_test_dl = [
        X_test,
        X_test,
        X_test,
        # np.zeros((len(X_test), 128)),
        # np.zeros((len(X_test), 128)),
        np.zeros((len(X_test), 60, X_total.shape[2])),  # Dummy peer data
    ]

    dl_config["data"]["num_features"] = X_total.shape[2]
    model = build_fusion_model(dl_config)

    # Fast universal training
    model.fit(
        X_train_dl, [Y_train, Y_train, Y_train], epochs=3, batch_size=64, verbose=0
    )

    preds = model.predict(X_test_dl, verbose=0)[2]
    return accuracy_score(Y_test, np.argmax(preds, axis=1))


if __name__ == "__main__":
    from clean_artifacts import clean_optimization_artifacts

    clean_optimization_artifacts(universal=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print(f"\nBEST UNIVERSAL ACCURACY: {study.best_value * 100:.2f}%")
    with open("configs/optimized_params_UNIVERSAL.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
