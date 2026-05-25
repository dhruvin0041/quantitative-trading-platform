# alerts.py
import os
import json
import argparse
from datetime import datetime
import pandas as pd
import yfinance as yf
import xgboost as xgb
import joblib
import numpy as np

from src.data_ingestion.technical_indicators import add_advanced_features

# ==========================================
# CONFIGURATION
# ==========================================


def setup_logger():
    import logging

    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("HydraAlerts")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"logs/alerts_{datetime.now().strftime('%Y%m')}.log")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Also log to console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


def generate_daily_signals(tickers):
    logger.info(f"Generating end-of-day signals for {len(tickers)} assets...")

    try:
        scaler = joblib.load("artifacts/latest_scaler.joblib")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("artifacts/xgb_ensemble.json")
        meta_model = joblib.load("meta_model.joblib")
        kill_switch_data = joblib.load("macro_kill_switch.joblib")
        regime_model = kill_switch_data["model"]
        panic_id = kill_switch_data["panic_cluster"]
    except Exception as e:
        logger.critical(f"Failed to load AI models: {e}")
        return []

    try:
        with open("configs/kept_features.json", "r") as f:
            kept_features = json.load(f)
    except Exception:
        logger.error("Failed to load configs/kept_features.json")
        return []

    # Check Macro Regime
    vix_df = yf.download("^VIX", period="10d", interval="1d", progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    current_vix = float(vix_df["Close"].iloc[-1])
    vix_roc = (vix_df["Close"].iloc[-1] / vix_df["Close"].iloc[-5]) - 1

    regime = regime_model.predict([[current_vix, vix_roc]])[0]
    is_panic = regime == panic_id

    if is_panic:
        logger.warning(
            f"MACRO KILL-SWITCH ACTIVE. Market is in a Panic Regime. (VIX: {current_vix:.2f})"
        )

    alerts = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = add_advanced_features(df)
            df = df.reindex(columns=kept_features).dropna()

            current_price = float(
                df["Close"].iloc[-1]
                if "Close" in df.columns
                else yf.Ticker(ticker).fast_info["lastPrice"]
            )

            recent_data = df.tail(1).values
            scaled_data = scaler.transform(recent_data)

            xgb_probs = xgb_model.predict_proba(scaled_data)
            prob_sell = xgb_probs[0][0]
            prob_buy = xgb_probs[0][2]

            X_meta = np.hstack((scaled_data, xgb_probs))
            meta_approval = meta_model.predict(X_meta)

            action = "HOLD"
            confidence = 0

            if is_panic:
                if prob_sell > 0.30:
                    action = "SELL (PANIC LIQUIDATION)"
                    confidence = 100
                else:
                    action = "BLOCK (PANIC)"
            elif meta_approval[0] == 1 or prob_buy > 0.40 or prob_sell > 0.55:
                if prob_buy > 0.32:
                    action = "BUY"
                    confidence = int(prob_buy * 100)
                elif prob_sell > 0.50:
                    action = "SELL"
                    confidence = int(prob_sell * 100)

            if action not in ["HOLD", "BLOCK (PANIC)"]:
                alerts.append(
                    {
                        "ticker": ticker,
                        "action": action,
                        "price": current_price,
                        "confidence": confidence,
                        "vix": current_vix,
                    }
                )
                logger.info(
                    f"Signal Generated: {ticker} -> {action} ({confidence}%) @ ${current_price:.2f}"
                )

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

    return alerts, is_panic, current_vix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydra Alert System")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"],
        help="Tickers to scan",
    )
    args = parser.parse_args()

    logger.info("--- Starting Daily Alert Scan ---")
    alerts, is_panic, current_vix = generate_daily_signals(args.tickers)

    if is_panic:
        logger.warning("🚨 HYDRA PANIC ALERT: Market Veto Active")

    if alerts:
        logger.info(f"Generated {len(alerts)} actionable signals.")
    else:
        logger.info("No actionable signals detected today.")

    logger.info("--- Scan Complete ---")
