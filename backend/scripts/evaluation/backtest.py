import os
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Imports from project
from src.execution.live_inference import (  # noqa: E402
    load_config,
    add_upgraded_features,
)
from src.models.neural.fusion_network import build_fusion_model  # noqa: E402
from src.models.rl.dqn_agent import DQNAgent  # noqa: E402
from src.models.ensemble.meta_ensemble import MetaEnsemble  # noqa: E402
from src.models.regime_detector import RegimeDetector  # noqa: E402
from src.optimization.objective_functions import calculate_sharpe_ratio  # noqa: E402

FEATURE_COLUMNS = [
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


def fetch_data(ticker, spy_df, vix_df, period="2y"):
    """Fetch and engineer features for backtesting."""
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Needs to match production feature engineering
    df = add_upgraded_features(df, spy_df, vix_df)

    # Return 5 days future return for evaluation
    df["future_5d_ret"] = df["Close"].shift(-5) / df["Close"] - 1

    return df.dropna()


def run_backtest():
    print("Initializing Hydra Terminal Backtest Engine...")

    # Pre-fetch SPY and VIX for the backtest period
    print("Fetching SPY and VIX data...")
    spy_df = yf.download("SPY", period="2y", progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)

    vix_df = yf.download("^VIX", period="2y", progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)

    # 1. Load Models
    config = load_config()
    with open("configs/kept_features.json", "r") as f:
        kept_features_list = json.load(f)
    actual_num_features = len(kept_features_list)
    config["data"]["num_features"] = actual_num_features

    # Models
    try:
        lstm_model = build_fusion_model(config)
        lstm_model.load_weights("artifacts/latest_fusion_weights.weights.h5")

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("artifacts/xgb_ensemble.json")

        lgbm_model = joblib.load("artifacts/lgbm_agent.joblib")

        dqn_agent = DQNAgent(state_size=actual_num_features + 6)
        dqn_agent.load("artifacts/dqn_model.pth")

        MetaEnsemble.load("artifacts/meta_ensemble.joblib")

        regime_model = RegimeDetector.load("artifacts/regime_detector.joblib")

        # Load calibrators
        xgb_calibrator = joblib.load("artifacts/xgb_calibrator.joblib")
        lgbm_calibrator = joblib.load("artifacts/lgbm_calibrator.joblib")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 2. Setup
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, help="Specific ticker to backtest")
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [
            "AAPL",
            "MSFT",
            "NVDA",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "JPM",
            "JNJ",
            "XOM",
        ]
    time_steps = config["data"]["time_steps"]

    trades = []

    print(f"Starting backtest simulation on {len(tickers)} assets...")

    from sklearn.preprocessing import StandardScaler

    def get_calibrated_probs(model, calibrators, X):
        raw_probs = model.predict_proba(X)[0]
        buy_prob = calibrators["buy"].predict([raw_probs[2]])[0]
        sell_prob = calibrators["sell"].predict([raw_probs[0]])[0]
        hold_prob = max(0, 1.0 - buy_prob - sell_prob)
        # Normalize
        total = buy_prob + sell_prob + hold_prob
        return np.array([sell_prob / total, hold_prob / total, buy_prob / total])

    def lstm_calibrated_probs(raw_probs, temperature=2.5):
        # raw_probs are softmax outputs but can still be scaled by temp to flatten overconfidence
        # a rough proxy for temperature scaling on already normalized probabilities
        logits = np.log(np.clip(raw_probs, 1e-7, 1 - 1e-7))
        scaled_logits = logits / temperature
        exps = np.exp(scaled_logits - np.max(scaled_logits))
        return exps / np.sum(exps)

    # 3. Simulation Loop
    for ticker in tickers:
        df = fetch_data(ticker, spy_df, vix_df)
        if len(df) <= time_steps + 252:
            continue

        # Get regime states for the whole dataframe for speed
        regime_states = regime_model.predict(df)

        # Load the global scaler trained during the training phase to match production
        scaler_global = joblib.load("artifacts/latest_scaler.joblib")

        for i in range(time_steps + 252, len(df)):
            window = df.iloc[i - time_steps : i]

            current_row = df.iloc[i]  # This is day t
            date = df.index[i]

            # Features
            features_df = window[FEATURE_COLUMNS].copy()
            X_flat = scaler_global.transform(features_df.iloc[-1:])

            # Scaled sequence
            scaled_window = scaler_global.transform(features_df)
            X_seq = np.expand_dims(scaled_window, axis=0)  # shape (1, 60, features)

            # Predict
            dl_preds_raw = lstm_model.predict([X_seq] * 6, verbose=0)[2][0]
            dl_preds_cal = lstm_calibrated_probs(dl_preds_raw)
            xgb_preds_cal = get_calibrated_probs(xgb_model, xgb_calibrator, X_flat)
            lgbm_preds_cal = get_calibrated_probs(lgbm_model, lgbm_calibrator, X_flat)

            # Weighted Average Consensus
            avg_probs = (
                (dl_preds_cal * 0.4) + (xgb_preds_cal * 0.4) + (lgbm_preds_cal * 0.2)
            )

            p_sell, p_hold, p_buy = avg_probs

            # Relative Directional Mass
            total_dir_mass = p_buy + p_sell
            if total_dir_mass > 0:
                rel_buy = p_buy / total_dir_mass
                rel_sell = p_sell / total_dir_mass
            else:
                rel_buy = rel_sell = 0.5

            regime_id = int(regime_states.iloc[i]["Regime_ID"])
            rsi = current_row["RSI"]
            adx = current_row["ADX"]
            vix = current_row["VIX_Level"]

            # Thresholds on relative mass
            req_rel = 0.72  # Require 72% of directional mass to be in one side (TIGHT)
            if regime_id == 2:
                req_rel = 0.68  # BULL
            elif regime_id == 0:
                req_rel = 0.75  # BEAR

            signal = "HOLD"
            confidence = max(rel_buy, rel_sell) * 100

            if rel_buy > req_rel and p_buy > p_hold * 0.9:
                signal = "BUY"
            elif rel_sell > req_rel and p_sell > p_hold * 0.9:
                signal = "SELL"

            # Final Institutional Filters
            if signal == "BUY" and (rsi > 72 or adx < 20):
                signal = "VETOED"
            if signal == "SELL" and (rsi < 28 or adx < 20):
                signal = "VETOED"
            if signal in ["BUY", "SELL"] and vix > 35:
                signal = "VETOED"

            if i % 100 == 0:  # Occasional heartbeat
                pass

            if signal in ["BUY", "SELL"]:
                actual_ret = current_row["future_5d_ret"]
                was_correct = (signal == "BUY" and actual_ret > 0) or (
                    signal == "SELL" and actual_ret < 0
                )

                trades.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "signal": signal,
                        "confidence": round(confidence, 1),
                        "regime": regime_id,
                        "actual_5day_return": round(actual_ret * 100, 2),
                        "was_correct": was_correct,
                        "atr_value": current_row.get(
                            "ATR", current_row["Close"] * 0.02
                        ),
                        "entry_price": current_row["Close"],
                    }
                )
            elif signal == "VETOED":
                trades.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "signal": "VETOED",
                        "confidence": round(confidence, 1),
                        "regime": regime_id,
                        "actual_5day_return": round(
                            current_row["future_5d_ret"] * 100, 2
                        ),
                        "was_correct": False,  # N/A
                    }
                )

    # 5. Calculate Metrics
    df_trades = pd.DataFrame(trades)

    total_signals = len(df_trades)
    active_signals = df_trades[df_trades["signal"].isin(["BUY", "SELL"])]
    vetoed_signals = df_trades[df_trades["signal"] == "VETOED"]

    num_active = len(active_signals)
    num_vetoed = len(vetoed_signals)

    if num_active > 0:
        correct_signals = active_signals[active_signals["was_correct"]]
        active_signals[~active_signals["was_correct"]]

        win_rate = len(correct_signals) / num_active * 100
        avg_conf = correct_signals["confidence"].mean()

        # Calculate strategy returns
        returns = active_signals.apply(
            lambda row: (
                row["actual_5day_return"] / 100
                if row["signal"] == "BUY"
                else -row["actual_5day_return"] / 100
            ),
            axis=1,
        )

        # Fix Root Cause 7: Position Sizing and Realistic Drawdown
        gains = returns[returns > 0]
        losses = returns[returns < 0]

        avg_win = gains.mean() if len(gains) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-9

        R = avg_win / avg_loss
        W = len(correct_signals) / num_active
        kelly = W - ((1 - W) / R) if R > 0 else 0

        portfolio_value = 1000000.0
        portfolio_history = [portfolio_value]

        # Simulate timeline
        active_signals_sorted = active_signals.sort_values(by="date")
        for _, row in active_signals_sorted.iterrows():
            ret = row["actual_5day_return"] / 100
            if row["signal"] == "SELL":
                ret = -ret

            # Position sizing
            max_position_pct = min(max(0, kelly) * 0.5, 0.10)  # Max 10%, Half-Kelly

            position_value = portfolio_value * max_position_pct
            entry_price = row["entry_price"]
            atr_value = row["atr_value"]

            # 2% risk cap
            stop_loss_price = entry_price - (2 * atr_value)
            risk_per_share = abs(entry_price - stop_loss_price)
            if risk_per_share == 0:
                risk_per_share = 0.01

            max_shares = (portfolio_value * 0.02) / risk_per_share
            shares = min(position_value / entry_price, max_shares)

            trade_pnl = shares * entry_price * ret
            portfolio_value += trade_pnl
            portfolio_history.append(portfolio_value)

        profit_factor = (
            gains.sum() / abs(losses.sum()) if len(losses) > 0 else float("inf")
        )
        sharpe = calculate_sharpe_ratio(returns.values)

        # Max Drawdown from portfolio history
        portfolio_series = pd.Series(portfolio_history)
        rolling_max = portfolio_series.cummax()
        drawdowns = (portfolio_series - rolling_max) / rolling_max
        max_dd = drawdowns.min() * 100

        # Best/Worst
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        best_signal = active_signals.loc[best_idx].to_dict()
        worst_signal = active_signals.loc[worst_idx].to_dict()
    else:
        win_rate = avg_conf = profit_factor = sharpe = max_dd = 0
        best_signal = worst_signal = None

    coverage = (num_active / total_signals * 100) if total_signals > 0 else 0
    veto_rate = (num_vetoed / total_signals * 100) if total_signals > 0 else 0

    # 6. Report
    report = f"""
    ╔══════════════════════════════════════════════╗
    ║     HYDRA TERMINAL — BACKTEST REPORT         ║
    ║     Period: 2024-01-01 to 2026-01-01         ║
    ╠══════════════════════════════════════════════╣
    ║ Total Signals:      {total_signals:<25}║
    ║ Win Rate:           {win_rate:.1f}%  ✅              ║
    ║ Avg Confidence:     {avg_conf:.1f}%                    ║
    ║ Profit Factor:      {profit_factor:.2f}                     ║
    ║ Sharpe Ratio:       {sharpe:.2f}                     ║
    ║ Max Drawdown:       {max_dd:.1f}%                    ║
    ║ Signal Coverage:    {coverage:.1f}%                    ║
    ║ VETOED Rate:        {veto_rate:.1f}%                    ║
    ╚══════════════════════════════════════════════╝
    """

    print(report)

    # 7. Save
    os.makedirs("backtest_results", exist_ok=True)

    summary = {
        "period": "2y",
        "total_signals": int(num_active),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd, 1),
        "vetoed_rate": round(veto_rate, 1),
        "coverage": round(coverage, 1),
        "best_signal": best_signal,
        "worst_signal": worst_signal,
        "monthly_win_rates": [],  # Simplified for now
    }

    with open("backtest_results/backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    df_trades.to_csv("backtest_results/backtest_trades.csv", index=False)

    with open("backtest_results/backtest_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("Results saved to backtest_results/")


if __name__ == "__main__":
    run_backtest()
