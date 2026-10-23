"""
Institutional Performance & Signal Accuracy Audit for All 18 Assets.
Measures Win Rate, Profit Factor, Strategy Return, Benchmark B&H Return, and Max Drawdown
across the entire institutional universe over trailing 2 years.
"""
import json
import logging
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from unittest.mock import MagicMock

import pandas as pd
import yfinance as yf

from src.execution.asset_intelligence import get_champion_strategy_for_ticker
from src.execution.inference_service import InferenceService
from src.execution.live_inference import load_config
from src.execution.signal_ledger import SignalLedger
from src.models.model_loader import ModelManager

logging.basicConfig(level=logging.WARNING)

ALL_TICKERS = [
    # Original 8 US Mega-Caps
    "AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    # 4 New US Defensives / Leaders
    "COST", "WMT", "JPM", "LLY",
    # 6 New Indian Nifty-50 Anchors
    "RELIANCE.NS", "TCS.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "LT.NS", "ITC.NS"
]

def run_audit():
    config = load_config()
    kf_path = backend_dir / "configs" / "kept_features.json"
    with open(kf_path) as f:
        kept_features = json.load(f)

    config["data"]["num_features"] = len(kept_features)
    mm = ModelManager(config, kept_features)
    mm.load_all_models()

    ledger = SignalLedger()
    service = InferenceService(
        mm,
        MagicMock(), MagicMock(), MagicMock(),
        MagicMock(), MagicMock(), MagicMock(),
        MagicMock(), MagicMock(),
        signal_ledger=ledger,
    )

    spy_df = yf.download("SPY", period="2y", progress=False)
    if hasattr(spy_df.columns, "droplevel"):
        try:
            spy_df.columns = spy_df.columns.droplevel(1)
        except Exception:
            pass

    records = []
    print("=" * 115)
    print(f"{'TICKER':<14} | {'STRATEGY MODE':<26} | {'TRADES':<6} | {'WIN RATE':<9} | {'PROFIT FACTOR':<13} | {'STRAT RET%':<10} | {'B&H RET%':<9} | {'MAX DD%':<8}")
    print("=" * 115)

    for sym in ALL_TICKERS:
        pq_path = backend_dir / "data" / "cache" / f"{sym}_2y_features.parquet"
        if not pq_path.exists():
            continue
        df = pd.read_parquet(pq_path)
        markers = service.get_causal_chart_markers(sym, df, spy_df=spy_df, force_recompute=True)

        strat_cfg = get_champion_strategy_for_ticker(sym)
        strat_mode = strat_cfg["strategy_type"]

        # Calculate closed trade statistics
        trades = []
        entry = None
        for m in markers:
            act = m.get("action")
            price = m.get("price")
            date = m.get("time") or m.get("bar_timestamp")
            if act == "BUY" and entry is None:
                entry = {"price": price, "date": date}
            elif act == "SELL" and entry is not None:
                pnl = (price - entry["price"]) / entry["price"]
                trades.append({
                    "entry_date": entry["date"],
                    "exit_date": date,
                    "entry_price": entry["price"],
                    "exit_price": price,
                    "ret": pnl,
                })
                entry = None

        # Cumulative performance
        if trades:
            wins = [t for t in trades if t["ret"] > 0]
            losses = [t for t in trades if t["ret"] <= 0]
            win_rate = len(wins) / len(trades) * 100.0
            gross_win = sum(t["ret"] for t in wins)
            gross_loss = abs(sum(t["ret"] for t in losses))
            pf = (gross_win / gross_loss) if gross_loss > 0 else (9.99 if gross_win > 0 else 1.0)

            # Cumulative equity curve
            eq = 1.0
            peak = 1.0
            max_dd = 0.0
            for t in trades:
                eq *= (1.0 + t["ret"])
                peak = max(peak, eq)
                dd = (peak - eq) / peak * 100.0
                max_dd = max(max_dd, dd)
            strat_ret = (eq - 1.0) * 100.0
        else:
            win_rate = 0.0
            pf = 0.0
            strat_ret = 0.0
            max_dd = 0.0

        # Buy & Hold return
        close_first = float(df["Close"].iloc[0])
        close_last = float(df["Close"].iloc[-1])
        bh_ret = ((close_last - close_first) / close_first) * 100.0

        records.append({
            "ticker": sym,
            "mode": strat_mode,
            "trades": len(trades),
            "win_rate": round(win_rate, 1),
            "pf": round(pf, 2),
            "strat_ret": round(strat_ret, 1),
            "bh_ret": round(bh_ret, 1),
            "max_dd": round(max_dd, 1),
        })

        print(f"{sym:<14} | {strat_mode:<26} | {len(trades):<6} | {win_rate:>7.1f}% | {pf:>11.2f}x | {strat_ret:>9.1f}% | {bh_ret:>8.1f}% | {max_dd:>7.1f}%")

    print("=" * 115)

    # Save results to evaluation folder
    out_path = backend_dir / "artifacts" / "universe_accuracy_audit.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=4)
    print(f"\n[+] Full 18-Asset Audit exported to {out_path}")
    return records

if __name__ == "__main__":
    run_audit()
