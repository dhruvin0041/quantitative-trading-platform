import json
import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import ks_2samp


def calculate_psi(expected, actual, bins=10):
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)

    # Add epsilon to avoid division by zero or log of zero
    expected_percents = np.clip(expected_percents, 1e-6, 1.0)
    actual_percents = np.clip(actual_percents, 1e-6, 1.0)

    psi_value = np.sum(
        (actual_percents - expected_percents)
        * np.log(actual_percents / expected_percents)
    )
    return psi_value


def detect_concept_drift():
    print("=== HYDRA CONCEPT DRIFT DETECTION ===")

    # 1. Load Baseline (Train Data)
    try:
        X_train = joblib.load("artifacts/X_train_tabular.joblib")
        with open("configs/kept_features.json", "r") as f:
            features = json.load(f)
        train_df = pd.DataFrame(X_train, columns=features)
    except Exception:
        print("Training artifacts missing.")
        return

    # 2. Fetch Recent Data (Last 30 days)
    # Using SPY as a broad proxy for market drift
    recent_df_raw = yf.download("SPY", period="60d", interval="1d", progress=False)
    if isinstance(recent_df_raw.columns, pd.MultiIndex):
        recent_df_raw.columns = recent_df_raw.columns.droplevel(1)

    from src.execution.live_inference import add_upgraded_features

    # We need a dummy vix for add_upgraded_features
    vix_df = yf.download("^VIX", period="60d", progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.droplevel(1)

    recent_df = add_upgraded_features(recent_df_raw, recent_df_raw, vix_df)
    recent_df = recent_df[features].dropna()

    # 3. Compare Distributions
    drift_report = {}
    print(f"{'Feature':<20} | {'KS Stat':<10} | {'PSI':<10} | {'Status'}")
    print("-" * 60)

    for feat in features:
        train_vals = train_df[feat].values
        recent_vals = recent_df[feat].values

        ks_stat, p_value = ks_2samp(train_vals, recent_vals)
        psi = calculate_psi(train_vals, recent_vals)

        status = "OK"
        if psi > 0.2 or p_value < 0.05:
            status = "DRIFT DETECTED"

        print(f"{feat:<20} | {ks_stat:<10.4f} | {psi:<10.4f} | {status}")
        drift_report[feat] = {"ks": ks_stat, "psi": psi, "status": status}

    # Save report
    os.makedirs("backtest_results", exist_ok=True)
    with open("backtest_results/drift_report.json", "w") as f:
        json.dump(drift_report, f, indent=2)

    avg_psi = np.mean([d["psi"] for d in drift_report.values()])
    print(f"\nAverage System PSI: {avg_psi:.4f}")
    if avg_psi > 0.15:
        print(">>> WARNING: Significant concept drift detected across system features.")
    else:
        print(">>> System distributions remain relatively stable.")


if __name__ == "__main__":
    detect_concept_drift()
