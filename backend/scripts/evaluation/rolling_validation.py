import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from src.execution.live_inference import add_upgraded_features, FEATURE_COLUMNS
from src.data_ingestion.market_data import (
    fetch_historical_data,
    apply_dynamic_triple_barrier,
)


def calculate_ece(y_true, y_prob, n_bins=10):
    if len(y_true) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    mask = bin_total > 0
    prob_pred = bin_sums[mask] / bin_total[mask]
    prob_true = bin_true[mask] / bin_total[mask]

    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[mask] / len(y_true)))
    return ece


def run_simulation(tickers=["AAPL", "MSFT", "NVDA"], cadence_weeks=1):
    print(f"--- STARTING SIMULATION (Cadence: {cadence_weeks} week(s)) ---")

    # 1. Fetch entire dataset first for speed
    all_data = {}
    full_start = "2021-01-01"
    full_end = "2026-05-23"

    spy_full = yf.download("SPY", start=full_start, end=full_end, progress=False)
    vix_full = yf.download("^VIX", start=full_start, end=full_end, progress=False)
    if isinstance(spy_full.columns, pd.MultiIndex):
        spy_full.columns = spy_full.columns.droplevel(1)
    if isinstance(vix_full.columns, pd.MultiIndex):
        vix_full.columns = vix_full.columns.droplevel(1)

    for t in tickers:
        df = fetch_historical_data(t, start_date=full_start, end_date=full_end)
        df = add_upgraded_features(df, spy_full, vix_full)
        df["target_signal"] = apply_dynamic_triple_barrier(df.copy())["target_signal"]
        all_data[t] = df.dropna()

    # 2. Iterative Walk-Forward
    sim_start = pd.Timestamp("2024-01-01")
    sim_end = pd.Timestamp("2026-05-01")

    current_date = sim_start
    results = []

    while current_date < sim_end:
        # Step End Date
        next_date = current_date + timedelta(weeks=cadence_weeks)
        print(f"Processing window starting {current_date.date()}...")

        # Train Window (Last 24 months)
        train_start = current_date - timedelta(days=365 * 2)

        # Build training set across tickers
        X_train_list = []
        y_train_list = []

        for t in tickers:
            df = all_data[t]
            mask = (df.index >= train_start) & (df.index < current_date)
            train_chunk = df.loc[mask]
            if len(train_chunk) > 100:
                X_train_list.append(train_chunk[FEATURE_COLUMNS].values)
                y_train_list.append(train_chunk["target_signal"].values)

        if not X_train_list:
            current_date = next_date
            continue

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Train models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
        )
        xgb_model.fit(X_train_scaled, y_train)

        # Test Window (The following week)
        for t in tickers:
            df = all_data[t]
            mask = (df.index >= current_date) & (df.index < next_date)
            test_chunk = df.loc[mask]

            if test_chunk.empty:
                continue

            X_test_scaled = scaler.transform(test_chunk[FEATURE_COLUMNS].values)
            y_test = test_chunk["target_signal"].values

            # Predict
            probs = xgb_model.predict_proba(X_test_scaled)
            preds = np.argmax(probs, axis=1)

            for i in range(len(test_chunk)):
                if preds[i] != 1:  # Only track directional signals
                    results.append(
                        {
                            "date": test_chunk.index[i],
                            "ticker": t,
                            "signal": "BUY" if preds[i] == 2 else "SELL",
                            "prob": probs[i][preds[i]],
                            "actual": y_test[i],
                            "is_correct": preds[i] == y_test[i],
                            "ret": 0.02
                            if preds[i] == y_test[i]
                            else -0.01,  # Simplified return proxy
                        }
                    )

        current_date = next_date

    # 3. Analyze Results
    res_df = pd.DataFrame(results)
    if res_df.empty:
        return None

    wr = res_df["is_correct"].mean() * 100
    # Calc ECE
    y_true_bin = (res_df["actual"] == 2).astype(int)  # Buying calibration
    y_prob_buy = res_df["prob"]  # Simplified for this check
    ece = calculate_ece(y_true_bin, y_prob_buy)

    # PnL calc
    res_df["equity"] = (1 + res_df["ret"]).cumprod()
    pf = res_df[res_df["ret"] > 0]["ret"].sum() / abs(
        res_df[res_df["ret"] < 0]["ret"].sum()
    )

    return {"wr": wr, "pf": pf, "ece": ece, "count": len(res_df)}


if __name__ == "__main__":
    # Compare cadences
    cadences = [1, 2, 4]  # Weekly, Biweekly, Monthly
    comparison = {}
    for c in cadences:
        res = run_simulation(cadence_weeks=c)
        comparison[f"{c}w"] = res

    print("\n=== EDGE RECOVERY REPORT ===")
    print(pd.DataFrame(comparison).T.to_string())
