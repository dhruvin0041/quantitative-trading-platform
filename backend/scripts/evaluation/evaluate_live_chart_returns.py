import json
import urllib.request

TICKERS = [
    # US Stable Champions (Top) & Universe
    "AAPL", "JNJ", "V", "BRK-B", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    "COST", "WMT", "JPM", "LLY",
    # Indian Stable Champions (Top) & Universe
    "LT.NS", "TITAN.NS", "SUNPHARMA.NS", "RELIANCE.NS", "TCS.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS"
]

results = []

print(f"{'Ticker':<14} | {'Strategy $':<12} | {'Strategy %':<12} | {'B&H $':<12} | {'B&H %':<12} | {'Alpha %':<10} | {'Trades':<8} | {'Win Rate':<10}")
print("-" * 105)

total_strat_val = 0.0
total_bh_val = 0.0
total_trades = 0
total_wins = 0

for ticker in TICKERS:
    url = f"http://127.0.0.1:8000/predict?ticker={ticker}"
    req = urllib.request.Request(url, headers={"X-API-Key": "hydra-secure-key-2026"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            markers = data.get("historical_markers") or data.get("markers") or []
            candles = data.get("candles") or []
            current_price = float(data.get("current_price") or (candles[-1]["close"] if candles else 0.0))

            if not candles:
                print(f"{ticker:<14} | No candles available")
                continue

            first_close = float(candles[0]["close"])
            last_close = float(candles[-1]["close"])
            bh_return_pct = ((last_close / first_close) - 1.0) * 100.0
            bh_final_val = 1000.0 * (last_close / first_close)

            # Sort markers chronologically
            markers = sorted(markers, key=lambda m: m.get("time") or m.get("timestamp") or "")

            actionable = []
            for m in markers:
                act = str(m.get("action") or m.get("signal") or "").upper()
                p = float(m.get("price") or m.get("execution_price") or 0.0)
                t = m.get("time") or m.get("timestamp") or ""
                if "BUY" in act:
                    actionable.append({"time": t, "action": "BUY", "price": p})
                elif "SELL" in act:
                    actionable.append({"time": t, "action": "SELL", "price": p})

            candle_map = {c["time"]: float(c["close"]) for c in candles}
            for a in actionable:
                if a["price"] <= 0 and a["time"] in candle_map:
                    a["price"] = candle_map[a["time"]]

            # Simulate $1,000 trading
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
                    ret_pct = ((p / last_buy["buy_price"]) - 1.0) * 100.0
                    trades.append({
                        "buy_time": last_buy["buy_time"],
                        "buy_price": last_buy["buy_price"],
                        "sell_time": s["time"],
                        "sell_price": p,
                        "pnl": pnl,
                        "return_pct": ret_pct,
                    })
                    cash = sell_proceeds
                    shares = 0.0
                    last_buy = None

            if shares > 0 and current_price > 0:
                final_val = shares * current_price
                open_ret = ((current_price / last_buy["buy_price"]) - 1.0) * 100.0
                trades.append({
                    "buy_time": last_buy["buy_time"],
                    "buy_price": last_buy["buy_price"],
                    "sell_time": "OPEN",
                    "sell_price": current_price,
                    "pnl": final_val - (shares * last_buy["buy_price"]),
                    "return_pct": open_ret,
                })
            else:
                final_val = cash

            strat_return_pct = ((final_val / initial_capital) - 1.0) * 100.0
            alpha_pct = strat_return_pct - bh_return_pct

            num_trades = len(trades)
            wins = sum(1 for t in trades if t["return_pct"] > 0)
            win_rate = (wins / num_trades * 100.0) if num_trades > 0 else 0.0

            total_strat_val += final_val
            total_bh_val += bh_final_val
            total_trades += num_trades
            total_wins += wins

            res_obj = {
                "ticker": ticker,
                "strategy_final": round(final_val, 2),
                "strategy_return_pct": round(strat_return_pct, 2),
                "bh_final": round(bh_final_val, 2),
                "bh_return_pct": round(bh_return_pct, 2),
                "alpha_pct": round(alpha_pct, 2),
                "trades": num_trades,
                "win_rate_pct": round(win_rate, 1),
            }
            results.append(res_obj)

            print(f"{ticker:<14} | ${final_val:>10.2f} | {strat_return_pct:>+10.2f}% | ${bh_final_val:>10.2f} | {bh_return_pct:>+10.2f}% | {alpha_pct:>+8.2f}% | {num_trades:>6} | {win_rate:>8.1f}%")

    except Exception as e:
        print(f"{ticker:<14} | Error: {e}")

print("=" * 105)
overall_strat_ret = ((total_strat_val / (len(results) * 1000.0)) - 1.0) * 100.0
overall_bh_ret = ((total_bh_val / (len(results) * 1000.0)) - 1.0) * 100.0
overall_alpha = overall_strat_ret - overall_bh_ret
overall_win_rate = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0

print(f"{'PORTFOLIO TOTAL':<14} | ${total_strat_val:>10.2f} | {overall_strat_ret:>+10.2f}% | ${total_bh_val:>10.2f} | {overall_bh_ret:>+10.2f}% | {overall_alpha:>+8.2f}% | {total_trades:>6} | {overall_win_rate:>8.1f}%")

from pathlib import Path

out_path = Path(__file__).resolve().parent.parent.parent / "reports" / "live_chart_returns_leaderboard.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "tickers": results,
        "portfolio": {
            "strategy_final": round(total_strat_val, 2),
            "strategy_return_pct": round(overall_strat_ret, 2),
            "bh_final": round(total_bh_val, 2),
            "bh_return_pct": round(overall_bh_ret, 2),
            "alpha_pct": round(overall_alpha, 2),
            "trades": total_trades,
            "win_rate_pct": round(overall_win_rate, 1),
        }
    }, f, indent=2)
