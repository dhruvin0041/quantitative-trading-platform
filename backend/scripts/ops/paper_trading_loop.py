import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Institutional Constants
try:
    API_URL = os.environ.get("API_URL", "http://localhost:8000")
    API_KEY = os.environ["API_KEY"]
except KeyError:
    print("CRITICAL: API_KEY must be set in environment.")
    exit(1)
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "XOM"]
INTERVAL = 3600  # Check every hour (institutional scan)


def run_paper_trading_loop():
    print(f"[{datetime.now()}] Starting Hydra Paper Trading Loop...")
    headers = {"X-API-Key": API_KEY}

    while True:
        print(f"\n--- Scanning Universe at {datetime.now()} ---")

        for ticker in TICKERS:
            try:
                # Trigger prediction (which now also executes paper trades)
                response = requests.get(
                    f"{API_URL}/predict?ticker={ticker}", headers=headers, timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    signal = data.get("signal")
                    price = data.get("current_price")
                    conf = data.get("confidence_score")

                    if signal in ["BUY", "SELL"]:
                        print(f"[SIGNAL] {ticker}: {signal} at ${price} ({conf:.1f}%)")
                        if "paper_trade" in data:
                            trade = data["paper_trade"]
                            print(
                                f" >>> EXECUTED: {trade['action']} {trade['shares']} shares"
                            )
                    elif signal == "VETOED":
                        print(f"[VETOED] {ticker}: {data.get('signal_note')}")
                else:
                    print(
                        f"[ERROR] Failed to fetch prediction for {ticker}: {response.status_code}"
                    )
            except Exception as e:
                print(f"[ERROR] Loop error for {ticker}: {str(e)}")

        # Periodic Summary
        try:
            perf_resp = requests.get(f"{API_URL}/performance", headers=headers)
            if perf_resp.status_code == 200:
                perf = perf_resp.json()
                summary = perf.get("summary", {})
                equity = perf.get("current_equity", 0)
                win_rate = summary.get("Win Rate", 0)

                print("\n" + "=" * 45)
                print(" HYDRA PORTFOLIO STATUS ")
                print(f" Total Equity: ${equity:,.2f}")
                print(f" Live Win Rate: {win_rate:.1f}%")
                print(" Backtest WR:   54.6%")
                print(f" Alpha Gap:     {win_rate - 54.6:+.1f}%")
                print("=" * 45)
        except Exception:
            pass

        print(f"\nSleeping for {INTERVAL / 60:.0f} minutes...")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    run_paper_trading_loop()
