"""
Test Alpha Performance on Top Stable Candidates.
Simulates Hydra V2 Adaptive Mesh on BRK-B, V, JNJ, SUNPHARMA.NS, TITAN.NS.
"""
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import yfinance as yf

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from scripts.evaluation.backtest import fetch_and_prepare_data
from src.execution.inference_service import InferenceService
from src.execution.live_inference import load_config
from src.execution.signal_ledger import SignalLedger
from src.models.model_loader import ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANDIDATES = [
    # Top Outside US
    "BRK-B", "V", "JNJ",
    # Top Outside India
    "SUNPHARMA.NS", "TITAN.NS"
]


def test_candidates():
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
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        signal_ledger=ledger,
    )

    spy_df = yf.download("SPY", period="2y", progress=False)
    if hasattr(spy_df.columns, "droplevel"):
        try:
            spy_df.columns = spy_df.columns.droplevel(1)
        except Exception:
            pass

    vix_df = yf.download("^VIX", period="2y", progress=False)
    if hasattr(vix_df.columns, "droplevel"):
        try:
            vix_df.columns = vix_df.columns.droplevel(1)
        except Exception:
            pass

    print(f"\n{'Ticker':<14} | {'Strategy $':<12} | {'Strategy %':<12} | {'B&H $':<12} | {'B&H %':<12} | {'Alpha %':<10} | {'Trades':<8} | {'Win Rate':<10}")
    print("-" * 105)

    results = []
    for sym in CANDIDATES:
        try:
            df = fetch_and_prepare_data(
                ticker=sym,
                spy_df=spy_df,
                vix_df=vix_df,
                period="2y",
                cache_dir=backend_dir / "data" / "cache",
            )
            markers = service.get_causal_chart_markers(sym, df, spy_df=spy_df, force_recompute=True)

            # Replay $1000 portfolio on these markers
            first_close = float(df["Close"].iloc[0])
            last_close = float(df["Close"].iloc[-1])
            bh_ret = ((last_close / first_close) - 1.0) * 100.0
            bh_val = 1000.0 * (last_close / first_close)

            # Sort markers chronologically
            sorted_m = sorted(markers, key=lambda m: m.get("time") or m.get("timestamp") or "")
            actionable = []
            for m in sorted_m:
                act = str(m.get("action") or m.get("signal") or "").upper()
                p = float(m.get("price") or m.get("execution_price") or 0.0)
                t = m.get("time") or m.get("timestamp") or ""
                if "BUY" in act:
                    actionable.append({"time": t, "action": "BUY", "price": p})
                elif "SELL" in act:
                    actionable.append({"time": t, "action": "SELL", "price": p})

            candle_map = {str(idx)[:10]: float(row["Close"]) for idx, row in df.iterrows()}
            for a in actionable:
                if a["price"] <= 0 and a["time"] in candle_map:
                    a["price"] = candle_map[a["time"]]

            cash = 1000.0
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

            if shares > 0:
                final_val = shares * last_close
                open_ret = ((last_close / last_buy["buy_price"]) - 1.0) * 100.0
                trades.append({
                    "buy_time": last_buy["buy_time"],
                    "buy_price": last_buy["buy_price"],
                    "sell_time": "OPEN",
                    "sell_price": last_close,
                    "pnl": final_val - (shares * last_buy["buy_price"]),
                    "return_pct": open_ret,
                })
            else:
                final_val = cash

            strat_ret = ((final_val / 1000.0) - 1.0) * 100.0
            alpha = strat_ret - bh_ret
            num_trades = len(trades)
            wins = sum(1 for tr in trades if tr["return_pct"] > 0)
            win_rate = (wins / num_trades * 100.0) if num_trades > 0 else 0.0

            print(f"{sym:<14} | $ {final_val:>9.2f} | {strat_ret:>+10.2f}% | $ {bh_val:>9.2f} | {bh_ret:>+10.2f}% | {alpha:>+9.2f}% | {num_trades:>6} | {win_rate:>8.1f}%")

            results.append({
                "ticker": sym,
                "strategy_val": final_val,
                "strategy_pct": strat_ret,
                "bh_val": bh_val,
                "bh_pct": bh_ret,
                "alpha_pct": alpha,
                "trades": num_trades,
                "win_rate": win_rate,
                "markers_count": len(markers)
            })

        except Exception as e:
            logger.error(f"Failed to test {sym}: {e}", exc_info=True)

    return results


if __name__ == "__main__":
    test_candidates()

