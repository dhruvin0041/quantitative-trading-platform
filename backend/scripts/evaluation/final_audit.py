import argparse
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from src.execution.consensus_engine import WeightedConsensusEngine
from src.execution.live_inference import add_upgraded_features

# STRICT STATIONARY FEATURES
STATIONARY_FEATURES = [
    "MA20_vs_MA50",
    "EMA9_vs_EMA21",
    "Price_vs_EMA9",
    "Price_vs_EMA21",
    "VIX_Level",
    "BB_Width",
    "BB_Position",
    "RSI",
    "ADX",
    "MACD_Hist",
    "Relative_Strength",
    "OBV_Change",
    "Return",
    "Volume_Ratio",
]


def run_sanitized_audit(
    tickers=["AAPL", "MSFT", "NVDA"],
    w_xgb=0.5,
    w_lgbm=0.5,
    threshold=60.0,
    ablation_name="Standard Multi-Model Ensemble",
):
    print("=== QUANTITATIVE STABILITY AUDIT (STRICT SANITIZATION) ===")
    print(f"Ensemble Configuration: {ablation_name}")
    print(f"Model Weights -> XGB_AGENT: {w_xgb:.2f} | LGBM_AGENT: {w_lgbm:.2f} | Consensus Threshold: {threshold:.1f}%")

    consensus_engine = WeightedConsensusEngine()

    full_start = "2021-01-01"
    full_end = "2026-05-23"

    spy_full = yf.download("SPY", start=full_start, end=full_end, progress=False)
    vix_full = yf.download("^VIX", start=full_start, end=full_end, progress=False)
    if isinstance(spy_full.columns, pd.MultiIndex):
        spy_full.columns = spy_full.columns.droplevel(1)
    if isinstance(vix_full.columns, pd.MultiIndex):
        vix_full.columns = vix_full.columns.droplevel(1)

    all_data = {}
    for t in tickers:
        df = fetch_data_clean(t, full_start, full_end, spy_full, vix_full)
        all_data[t] = df

    # Walk-forward starting from 2024
    sim_start = pd.Timestamp("2024-01-01")
    sim_end = pd.Timestamp("2026-05-01")
    current_date = sim_start
    results = []

    model_weights_map = {"XGB_AGENT": float(w_xgb), "LGBM_AGENT": float(w_lgbm)}

    while current_date < sim_end:
        next_date = current_date + timedelta(weeks=4)  # Monthly retraining for speed
        train_start = current_date - timedelta(days=365 * 2)

        # 1. RETRAIN MODELS (Strictly on data before current_date with 10-day embargo)
        X_train_list = []
        y_train_list = []
        for t in tickers:
            df = all_data[t]
            mask = (df.index >= train_start) & (
                df.index < current_date - timedelta(days=10)
            )
            train_chunk = df.loc[mask]
            if len(train_chunk) > 100:
                X_train_list.append(train_chunk[STATIONARY_FEATURES].values)
                y_train_list.append(train_chunk["target_signal"].values)

        if not X_train_list:
            current_date = next_date
            continue

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train XGBoost Agent
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
        )
        xgb_model.fit(X_train_scaled, y_train)

        # Train LightGBM Agent
        lgbm_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            objective="multiclass",
            num_class=3,
            random_state=42,
            verbose=-1,
        )
        lgbm_model.fit(X_train_scaled, y_train)

        # 2. EVALUATE VIA WEIGHTED CONSENSUS ENGINE (On the following out-of-sample month)
        for t in tickers:
            df = all_data[t]
            test_mask = (df.index >= current_date) & (df.index < next_date)
            test_chunk = df.loc[test_mask]
            if test_chunk.empty:
                continue

            X_test_scaled = scaler.transform(test_chunk[STATIONARY_FEATURES].values)
            fwd_ret = test_chunk["forward_5d_ret"].values

            probs_xgb = xgb_model.predict_proba(X_test_scaled)
            probs_lgbm = lgbm_model.predict_proba(X_test_scaled)

            for i in range(len(test_chunk)):
                base_probs = {
                    "XGB_AGENT": probs_xgb[i],
                    "LGBM_AGENT": probs_lgbm[i],
                }
                consensus = consensus_engine.compute_agreement(
                    base_probs, model_weights_map
                )
                score = consensus["agreement_score"]
                direction = consensus["dominant_direction"]

                if score >= threshold:
                    if direction == "BUY":
                        results.append({"ret": fwd_ret[i], "correct": fwd_ret[i] > 0, "model": "CONSENSUS_BUY"})
                    elif direction == "SELL":
                        results.append({"ret": -fwd_ret[i], "correct": fwd_ret[i] < 0, "model": "CONSENSUS_SELL"})

        current_date = next_date

    # 3. FINAL STATS
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No signals fired with current threshold.")
        return

    wr = res_df["correct"].mean() * 100
    profits = res_df[res_df["ret"] > 0]["ret"].sum()
    losses = abs(res_df[res_df["ret"] < 0]["ret"].sum())
    pf = profits / losses if losses > 0 else 0

    print("\nAUDIT RESULTS:")
    print(f"Total Signals: {len(res_df)}")
    print(f"Win Rate:      {wr:.1f}%")
    print(f"Profit Factor: {pf:.2f}")

    # 4. Monte Carlo Bootstrap
    print("\n--- PHASE 6: MONTE CARLO BOOTSTRAP (1000 trials) ---")
    pfs = []
    for _ in range(1000):
        sample = res_df.sample(frac=1.0, replace=True)
        p = sample[sample["ret"] > 0]["ret"].sum()
        loss_sum = abs(sample[sample["ret"] < 0]["ret"].sum())
        pfs.append(p / loss_sum if loss_sum > 0 else 0)

    print(
        f"PF 95% Confidence Interval: [{np.percentile(pfs, 2.5):.2f}, {np.percentile(pfs, 97.5):.2f}]"
    )
    print(f"Probability PF > 1.2: {np.mean(np.array(pfs) > 1.2) * 100:.1f}%")


def fetch_data_clean(ticker, start, end, spy, vix):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = add_upgraded_features(df, spy, vix)
    df["forward_5d_ret"] = df["Close"].shift(-5) / df["Close"] - 1
    # Target signal: BUY if ret > 2%, SELL if ret < -1.5%
    df["target_signal"] = 1  # HOLD
    df.loc[df["forward_5d_ret"] > 0.02, "target_signal"] = 2
    df.loc[df["forward_5d_ret"] < -0.015, "target_signal"] = 0
    return df.dropna()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanitized Walk-Forward Stability Audit with Consensus & Ablation")
    parser.add_argument("--ablation", choices=["none", "xgb_zero", "lgbm_zero"], default="none", help="Ablation mode")
    parser.add_argument("--threshold", type=float, default=60.0, help="Consensus agreement threshold (default: 60.0)")
    args = parser.parse_args()

    if args.ablation == "xgb_zero":
        run_sanitized_audit(w_xgb=0.0, w_lgbm=1.0, threshold=args.threshold, ablation_name="Ablation: XGB_AGENT Forced to 0.0 (Pure LGBM Driven)")
    elif args.ablation == "lgbm_zero":
        run_sanitized_audit(w_xgb=1.0, w_lgbm=0.0, threshold=args.threshold, ablation_name="Ablation: LGBM_AGENT Forced to 0.0 (Pure XGB Driven)")
    else:
        run_sanitized_audit(w_xgb=0.5, w_lgbm=0.5, threshold=args.threshold, ablation_name="Active Multi-Model Soft Consensus (50% XGB, 50% LGBM)")
