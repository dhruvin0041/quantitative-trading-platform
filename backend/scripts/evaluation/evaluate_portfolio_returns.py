"""
Evaluate Portfolio 1-Year Net Returns ($1,000 Investment)
==========================================================
Executes chronological historical backtests across all 8 institutional tickers
(AAPL, NVDA, MSFT, AMZN, META, GOOGL, TSLA, AMD) and the combined portfolio
from 2025-10-22 to 2026-10-22 (exactly 1 year / 251 trading days).
Compares HYDRA Core + Tactical Alpha, Dynamic Triple Barrier Rotation, and Buy & Hold.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.evaluation.backtest import run_chronological_backtest

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Portfolio1YBacktest")

TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD"]
START_DATE = "2025-10-22"
END_DATE = "2026-10-22"
INITIAL_CAPITAL = 1000.0


def compute_buy_and_hold(ticker: str, start_date: str, end_date: str, capital: float) -> dict:
    """Computes buy-and-hold performance from start_date Open to end_date Close."""
    try:
        df = yf.download(ticker, start=start_date, end=pd.Timestamp(end_date) + pd.Timedelta(days=1), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 5:
            return {"final_equity": capital, "net_return_pct": 0.0, "net_profit": 0.0}

        entry_price = float(df.iloc[0]["Open"])
        exit_price = float(df.iloc[-1]["Close"])
        shares = capital / entry_price
        final_equity = shares * exit_price
        net_return_pct = ((final_equity / capital) - 1.0) * 100.0
        return {
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "final_equity": round(final_equity, 2),
            "net_profit": round(final_equity - capital, 2),
            "net_return_pct": round(net_return_pct, 2),
        }
    except Exception as e:
        logger.error(f"Error computing buy and hold for {ticker}: {e}")
        return {"final_equity": capital, "net_return_pct": 0.0, "net_profit": 0.0}


def main():
    print("=" * 80)
    print("  HYDRA 1-YEAR $1,000 INSTITUTIONAL BACKTEST EVALUATION")
    print(f"  Window: {START_DATE} to {END_DATE} (Exactly 1 Year)")
    print(f"  Capital per asset / portfolio: ${INITIAL_CAPITAL:,.2f}")
    print("=" * 80)

    results = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "initial_capital_per_asset": INITIAL_CAPITAL,
        "tickers": {},
        "portfolio": {},
        "benchmarks": {},
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

    # 1. SPY Benchmark
    spy_bnh = compute_buy_and_hold("SPY", START_DATE, END_DATE, INITIAL_CAPITAL)
    results["benchmarks"]["SPY"] = spy_bnh
    print(f"\n[BENCHMARK] SPY 1Y Buy & Hold: ${INITIAL_CAPITAL:,.2f} -> ${spy_bnh['final_equity']:,.2f} ({spy_bnh['net_return_pct']:+.2f}%)")

    # 2. Individual Tickers Evaluation
    for ticker in TICKERS:
        print(f"\n--> Running Backtest for {ticker} ($1,000 Capital)...")
        bnh = compute_buy_and_hold(ticker, START_DATE, END_DATE, INITIAL_CAPITAL)

        # Core + Tactical Alpha
        out_core = BACKEND_DIR / "artifacts" / "backtest" / f"{ticker}_core_tactical"
        summary_core = run_chronological_backtest(
            tickers=[ticker],
            period="2y",
            start_date=START_DATE,
            end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL,
            strategy_mode="core_tactical",
            min_commission=0.0,
            commission_per_share=0.0,
            output_dir=str(out_core),
        )

        # Dynamic Triple Barrier Rotation
        out_rot = BACKEND_DIR / "artifacts" / "backtest" / f"{ticker}_rotation"
        summary_rot = run_chronological_backtest(
            tickers=[ticker],
            period="2y",
            start_date=START_DATE,
            end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL,
            allocation_per_trade=1.0,
            strategy_mode="rotation",
            min_commission=0.0,
            commission_per_share=0.0,
            output_dir=str(out_rot),
        )

        # Capitulation Oversold Rebound (Bear / Chop Regime Specialist)
        out_rebound = BACKEND_DIR / "artifacts" / "backtest" / f"{ticker}_oversold_rebound"
        summary_rebound = run_chronological_backtest(
            tickers=[ticker],
            period="2y",
            start_date=START_DATE,
            end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL,
            strategy_mode="oversold_rebound",
            min_commission=0.0,
            commission_per_share=0.0,
            output_dir=str(out_rebound),
        )

        # Institutional Adaptive Mesh (Regime-Matched Engine)
        out_adapt = BACKEND_DIR / "artifacts" / "backtest" / f"{ticker}_adaptive"
        summary_adapt = run_chronological_backtest(
            tickers=[ticker],
            period="2y",
            start_date=START_DATE,
            end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL,
            allocation_per_trade=1.0,
            strategy_mode="adaptive",
            min_commission=0.0,
            commission_per_share=0.0,
            output_dir=str(out_adapt),
        )

        results["tickers"][ticker] = {
            "buy_and_hold": bnh,
            "core_tactical": summary_core,
            "rotation": summary_rot,
            "oversold_rebound": summary_rebound,
            "adaptive": summary_adapt,
        }

    # 3. Combined Multi-Asset 8-Ticker Portfolio ($1,000 Total Capital)
    print("\n--> Running Combined 8-Ticker Portfolio Backtest ($1,000 Total Capital)...")
    out_port_core = BACKEND_DIR / "artifacts" / "backtest" / "portfolio_core_tactical"
    port_core = run_chronological_backtest(
        tickers=TICKERS,
        period="2y",
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        strategy_mode="core_tactical",
        min_commission=0.0,
        commission_per_share=0.0,
        output_dir=str(out_port_core),
    )

    out_port_rot = BACKEND_DIR / "artifacts" / "backtest" / "portfolio_rotation"
    port_rot = run_chronological_backtest(
        tickers=TICKERS,
        period="2y",
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        allocation_per_trade=0.25,
        strategy_mode="rotation",
        min_commission=0.0,
        commission_per_share=0.0,
        output_dir=str(out_port_rot),
    )

    out_port_reb = BACKEND_DIR / "artifacts" / "backtest" / "portfolio_oversold_rebound"
    port_reb = run_chronological_backtest(
        tickers=TICKERS,
        period="2y",
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        strategy_mode="oversold_rebound",
        min_commission=0.0,
        commission_per_share=0.0,
        output_dir=str(out_port_reb),
    )

    out_port_adapt = BACKEND_DIR / "artifacts" / "backtest" / "portfolio_adaptive"
    port_adapt = run_chronological_backtest(
        tickers=TICKERS,
        period="2y",
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        strategy_mode="adaptive",
        min_commission=0.0,
        commission_per_share=0.0,
        output_dir=str(out_port_adapt),
    )

    results["portfolio"]["core_tactical"] = port_core
    results["portfolio"]["rotation"] = port_rot
    results["portfolio"]["oversold_rebound"] = port_reb
    results["portfolio"]["adaptive"] = port_adapt

    # Save summary
    save_path = BACKEND_DIR / "artifacts" / "1year_backtest_results_1000usd.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print Institutional Summary Table
    print("\n" + "=" * 162)
    print("                    HYDRA 1-YEAR $1,000 INSTITUTIONAL BACKTEST RESULTS (2025-10-22 to 2026-10-22)")
    print("=" * 162)
    header = (
        f"{'Ticker':<7} | {'Buy & Hold (1Y)':<18} | {'Core + Tactical Alpha':<26} | "
        f"{'Dynamic Rotation':<26} | {'Oversold Rebound':<26} | {'Adaptive Mesh (Optimal)':<26}"
    )
    print(header)
    print("-" * 162)
    sub = (
        f"{'':<7} | {'Equity':>9} {'Return':>7} | {'Equity':>9} {'Return':>7} {'MaxDD':>7} | "
        f"{'Equity':>9} {'Return':>7} {'MaxDD':>7} | {'Equity':>9} {'Return':>7} {'MaxDD':>7} | "
        f"{'Equity':>9} {'Return':>7} {'MaxDD':>7}"
    )
    print(sub)
    print("-" * 162)

    for ticker in TICKERS:
        t_data = results["tickers"][ticker]
        bnh = t_data["buy_and_hold"]
        core = t_data["core_tactical"] or {}
        rot = t_data["rotation"] or {}
        reb = t_data["oversold_rebound"] or {}
        adapt = t_data["adaptive"] or {}

        bnh_eq = f"${bnh.get('final_equity', 1000):,.2f}"
        bnh_ret = f"{bnh.get('net_return_pct', 0):+.2f}%"

        core_eq = f"${core.get('final_equity', 1000):,.2f}"
        core_ret = f"{core.get('total_net_return_pct', core.get('total_return_pct', 0)):+.2f}%"
        core_dd = f"{core.get('max_drawdown_pct', 0):.2f}%"

        rot_eq = f"${rot.get('final_equity', 1000):,.2f}"
        rot_ret = f"{rot.get('total_net_return_pct', rot.get('total_return_pct', 0)):+.2f}%"
        rot_dd = f"{rot.get('max_drawdown_pct', 0):.2f}%"

        reb_eq = f"${reb.get('final_equity', 1000):,.2f}"
        reb_ret = f"{reb.get('total_net_return_pct', reb.get('total_return_pct', 0)):+.2f}%"
        reb_dd = f"{reb.get('max_drawdown_pct', 0):.2f}%"

        adapt_eq = f"${adapt.get('final_equity', 1000):,.2f}"
        adapt_ret = f"{adapt.get('total_net_return_pct', adapt.get('total_return_pct', 0)):+.2f}%"
        adapt_dd = f"{adapt.get('max_drawdown_pct', 0):.2f}%"

        print(
            f"{ticker:<7} | {bnh_eq:>9} {bnh_ret:>7} | {core_eq:>9} {core_ret:>7} {core_dd:>7} | "
            f"{rot_eq:>9} {rot_ret:>7} {rot_dd:>7} | {reb_eq:>9} {reb_ret:>7} {reb_dd:>7} | "
            f"{adapt_eq:>9} {adapt_ret:>7} {adapt_dd:>7}"
        )

    print("-" * 162)
    port_c = results["portfolio"]["core_tactical"] or {}
    port_r = results["portfolio"]["rotation"] or {}
    port_b = results["portfolio"]["oversold_rebound"] or {}
    port_a = results["portfolio"]["adaptive"] or {}
    p_core_eq = f"${port_c.get('final_equity', 1000):,.2f}"
    p_core_ret = f"{port_c.get('total_net_return_pct', port_c.get('total_return_pct', 0)):+.2f}%"
    p_core_dd = f"{port_c.get('max_drawdown_pct', 0):.2f}%"
    p_rot_eq = f"${port_r.get('final_equity', 1000):,.2f}"
    p_rot_ret = f"{port_r.get('total_net_return_pct', port_r.get('total_return_pct', 0)):+.2f}%"
    p_rot_dd = f"{port_r.get('max_drawdown_pct', 0):.2f}%"
    p_reb_eq = f"${port_b.get('final_equity', 1000):,.2f}"
    p_reb_ret = f"{port_b.get('total_net_return_pct', port_b.get('total_return_pct', 0)):+.2f}%"
    p_reb_dd = f"{port_b.get('max_drawdown_pct', 0):.2f}%"
    p_adapt_eq = f"${port_a.get('final_equity', 1000):,.2f}"
    p_adapt_ret = f"{port_a.get('total_net_return_pct', port_a.get('total_return_pct', 0)):+.2f}%"
    p_adapt_dd = f"{port_a.get('max_drawdown_pct', 0):.2f}%"
    spy_ret = f"{spy_bnh.get('net_return_pct', 0):+.2f}%"
    spy_eq = f"${spy_bnh.get('final_equity', 1000):,.2f}"

    print(
        f"{'PORTFOLIO':<7} | {spy_eq:>9} {spy_ret:>7} | {p_core_eq:>9} {p_core_ret:>7} {p_core_dd:>7} | "
        f"{p_rot_eq:>9} {p_rot_ret:>7} {p_rot_dd:>7} | {p_reb_eq:>9} {p_reb_ret:>7} {p_reb_dd:>7} | "
        f"{p_adapt_eq:>9} {p_adapt_ret:>7} {p_adapt_dd:>7}"
    )
    print("=" * 162)
    print(f"  Artifact successfully saved to {save_path}")
    print("=" * 162)


if __name__ == "__main__":
    main()
