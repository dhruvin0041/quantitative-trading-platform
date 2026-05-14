# alerts.py
import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
# You MUST set these environment variables or fill them in securely
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.environ.get("HYDRA_EMAIL", "your_email@gmail.com")
SENDER_PASSWORD = os.environ.get("HYDRA_APP_PASSWORD", "your_app_password")
RECEIVER_EMAIL = os.environ.get("HYDRA_ALERT_EMAIL", "your_email@gmail.com")


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


def send_email_alert(subject, body):
    if SENDER_EMAIL == "your_email@gmail.com":
        logger.warning(
            "Email credentials not configured. Skipping email send. (Set HYDRA_EMAIL and HYDRA_APP_PASSWORD)"
        )
        print("\n--- EMAIL ALERT SIMULATION ---")
        print(f"Subject: {subject}")
        print(f"Body:\n{body}")
        print("------------------------------\n")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "html"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info(f"Successfully sent alert email to {RECEIVER_EMAIL}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def generate_daily_signals(tickers):
    logger.info(f"Generating end-of-day signals for {len(tickers)} assets...")

    try:
        scaler = joblib.load("latest_scaler.joblib")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("xgb_ensemble.json")
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


def format_html_email(alerts, is_panic, current_vix):
    date_str = datetime.now().strftime("%B %d, %Y")

    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
        <h2 style="color: #1e293b; border-bottom: 2px solid #3b82f6; padding-bottom: 10px;">Hydra Terminal - Daily AI Signals</h2>
        <p><strong>Date:</strong> {date_str}</p>
    """

    if is_panic:
        html += f"""
        <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 10px; margin-bottom: 20px;">
            <h3 style="color: #b91c1c; margin-top: 0;">🚨 MACRO KILL-SWITCH ACTIVE 🚨</h3>
            <p>The market is currently in a <strong>High Volatility / Panic Regime</strong> (VIX: {current_vix:.2f}). 
            All standard BUY signals have been blocked by the Risk Manager. Only protective SELL signals will be issued.</p>
        </div>
        """
    else:
        html += f"""
        <div style="background-color: #f0fdf4; border-left: 4px solid #22c55e; padding: 10px; margin-bottom: 20px;">
            <h3 style="color: #15803d; margin-top: 0;">✅ Market Regime: Normal</h3>
            <p>Volatility (VIX) is at {current_vix:.2f}. Standard algorithmic trading rules apply.</p>
        </div>
        """

    if not alerts:
        html += (
            "<p><em>No actionable trade signals generated today. Standing by.</em></p>"
        )
    else:
        html += """
        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
            <tr style="background-color: #f8fafc; text-align: left;">
                <th style="padding: 10px; border-bottom: 2px solid #cbd5e1;">Ticker</th>
                <th style="padding: 10px; border-bottom: 2px solid #cbd5e1;">Action</th>
                <th style="padding: 10px; border-bottom: 2px solid #cbd5e1;">Confidence</th>
                <th style="padding: 10px; border-bottom: 2px solid #cbd5e1;">Price</th>
            </tr>
        """

        for a in alerts:
            action_color = "#22c55e" if "BUY" in a["action"] else "#ef4444"
            html += f"""
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;"><strong>{a["ticker"]}</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; color: {action_color}; font-weight: bold;">{a["action"]}</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{a["confidence"]}%</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">${a["price"]:.2f}</td>
            </tr>
            """

        html += "</table>"

    html += """
        <p style="font-size: 0.8em; color: #64748b; margin-top: 30px;">
            Generated autonomously by Project Hydra. Not financial advice.
        </p>
      </body>
    </html>
    """
    return html


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

    if alerts or is_panic:
        html_body = format_html_email(alerts, is_panic, current_vix)
        subject = f"Hydra Signals [{datetime.now().strftime('%m/%d')}]: {len(alerts)} Actions Required"
        if is_panic:
            subject = "🚨 HYDRA PANIC ALERT: Market Veto Active"

        send_email_alert(subject, html_body)
    else:
        logger.info(
            "No actionable signals or panic regimes detected today. No email sent."
        )

    logger.info("--- Scan Complete ---")
