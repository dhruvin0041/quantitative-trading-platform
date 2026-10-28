import json
import urllib.request

TICKERS = [
    # US Tech
    "AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    # US Defensives & Financials
    "COST", "WMT", "JPM", "LLY",
    # Indian NSE Giants
    "RELIANCE.NS", "TCS.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "LT.NS", "ITC.NS"
]

results = {}

for ticker in TICKERS:
    url = f"http://127.0.0.1:8000/predict?ticker={ticker}"
    req = urllib.request.Request(url, headers={"X-API-Key": "hydra-secure-key-2026"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            markers = data.get("historical_markers") or data.get("markers") or []
            candles = data.get("candles") or []
            current_price = data.get("current_price") or (candles[-1]["close"] if candles else 0.0)

            # Sort markers chronologically
            markers = sorted(markers, key=lambda m: m.get("time") or m.get("timestamp") or "")

            # Filter to actionable signals (BUY / SELL)
            actionable = []
            for m in markers:
                act = str(m.get("action") or m.get("signal") or "").upper()
                if "BUY" in act:
                    actionable.append({"time": m.get("time"), "action": "BUY", "price": float(m.get("price") or m.get("execution_price") or 0.0)})
                elif "SELL" in act:
                    actionable.append({"time": m.get("time"), "action": "SELL", "price": float(m.get("price") or m.get("execution_price") or 0.0)})

            # If price is missing or 0 in marker, look up close price from candles
            candle_map = {c["time"]: c["close"] for c in candles}
            for a in actionable:
                if a["price"] <= 0 and a["time"] in candle_map:
                    a["price"] = candle_map[a["time"]]

            # Simulate $1000 trading
            initial_capital = 1000.0
            cash = initial_capital
            shares = 0.0
            trades = []
            last_buy = None

            for s in actionable:
                p = s["price"]
                if p <= 0:
                    continue
                if s["action"] == "BUY" and shares == 0:
                    shares = cash / p
                    cash = 0.0
                    last_buy = {"buy_time": s["time"], "buy_price": p}
                elif s["action"] == "SELL" and shares > 0:
                    sell_proceeds = shares * p
                    pnl = sell_proceeds - (shares * last_buy["buy_price"])
                    ret_pct = ((p / last_buy["buy_price"]) - 1.0) * 100
                    trades.append({
                        "buy_time": last_buy["buy_time"],
                        "buy_price": last_buy["buy_price"],
                        "sell_time": s["time"],
                        "sell_price": p,
                        "pnl": pnl,
                        "return_pct": ret_pct
                    })
                    cash = sell_proceeds
                    shares = 0.0
                    last_buy = None

            # Mark to market if still holding
            open_trade = None
            if shares > 0 and current_price > 0:
                final_val = shares * current_price
                open_pnl = final_val - (shares * last_buy["buy_price"])
                open_ret = ((current_price / last_buy["buy_price"]) - 1.0) * 100
                open_trade = {
                    "buy_time": last_buy["buy_time"],
                    "buy_price": last_buy["buy_price"],
                    "current_price": current_price,
                    "unrealized_pnl": open_pnl,
                    "unrealized_return_pct": open_ret
                }
            else:
                final_val = cash

            profit = final_val - initial_capital
            total_return_pct = (profit / initial_capital) * 100

            results[ticker] = {
                "initial_capital": initial_capital,
                "final_value": round(final_val, 2),
                "profit": round(profit, 2),
                "total_return_pct": round(total_return_pct, 2),
                "total_markers": len(markers),
                "actionable_signals": len(actionable),
                "completed_trades": len(trades),
                "trades": trades,
                "open_trade": open_trade,
                "current_price": current_price,
                "first_candle": candles[0]["time"] if candles else "N/A",
                "last_candle": candles[-1]["time"] if candles else "N/A",
            }
            print(f"[{ticker}] Signals: {len(actionable)} | Final Value: ${final_val:.2f} ({total_return_pct:+.2f}%)")
    except Exception as e:
        print(f"[{ticker}] Error: {e}")

with open("backend/scratch/chart_returns_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSummary saved to backend/scratch/chart_returns_summary.json")
