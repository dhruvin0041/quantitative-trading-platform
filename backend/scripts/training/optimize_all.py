"""
Master Institutional Multi-Ticker Bayesian Optimization Runner.
Sequentially runs Optuna Multivariate TPESampler with Purged Walk-Forward Cross-Validation
for the entire institutional universe (AAPL, NVDA, MSFT, AMZN, META, GOOGL, TSLA, AMD)
and outputs an institutional optimization summary matrix.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from scripts.training.optimize import run_optuna_optimization
from src.utils.date_window import resolve_training_dates

INSTITUTIONAL_UNIVERSE = [
    # US Stable Champions & Mega-Cap Tech
    "AAPL",
    "JNJ",
    "V",
    "BRK-B",
    "COST",
    "JPM",
    "WMT",
    "MSFT",
    "GOOGL",
    "META",
    "NVDA",
    "AMZN",
    "LLY",
    "TSLA",
    "AMD",
    # India (NSE) Institutional Giants & Stable Champions
    "LT.NS",
    "TITAN.NS",
    "SUNPHARMA.NS",
    "RELIANCE.NS",
    "TCS.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
]


def run_optimize_all(
    tickers=None,
    trials: int = 50,
    start_date: str = "auto",
    end_date: str = "auto",
):
    if tickers is None:
        tickers = INSTITUTIONAL_UNIVERSE

    start_date, end_date, is_auto = resolve_training_dates(start_date, end_date)
    total_start = time.time()
    print("================================================================================")
    print("      HYDRA INSTITUTIONAL MULTI-TICKER BAYESIAN OPTIMIZATION RUNNER")
    print(f" Universe ({len(tickers)} assets): {', '.join(tickers)}")
    print(f" Historical Range: {start_date} to {end_date} (Auto-Rolling: {is_auto})")
    print(f" Optuna Trials per Asset: {trials}")
    print("================================================================================")

    results = []
    for idx, ticker in enumerate(tickers, start=1):
        print(f"\n[{idx}/{len(tickers)}] >>> Launching Bayesian Optimization for {ticker} ({trials} trials) <<<")
        t0 = time.time()
        try:
            best_config = run_optuna_optimization(
                ticker=ticker,
                n_trials=trials,
                start=start_date,
                end=end_date,
            )
            elapsed = time.time() - t0
            best_acc = best_config.get("_provenance", {}).get("best_accuracy", 0.0) if isinstance(best_config, dict) else 0.0
            results.append({
                "ticker": ticker,
                "best_trading_score": best_acc,
                "duration_sec": round(elapsed, 2),
                "config_file": f"configs/optimized_params_{ticker}.json",
                "status": "SUCCESS",
            })
            print(f"  [+] {ticker} OPTIMIZATION COMPLETE -> Best Trading Score: {best_acc*100:.2f}% ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [-] {ticker} OPTIMIZATION FAILED: {e}")
            results.append({
                "ticker": ticker,
                "best_trading_score": 0.0,
                "duration_sec": round(elapsed, 2),
                "config_file": f"configs/optimized_params_{ticker}.json",
                "error": str(e),
                "status": "FAILED",
            })

    total_duration = time.time() - total_start

    print("\n================================================================================")
    print("              MULTI-TICKER OPTIMIZATION SUMMARY MATRIX")
    print("================================================================================")
    header = f"{'TICKER':<8} | {'BEST SCORE':<12} | {'TIME':<10} | {'CONFIG FILE':<35} | {'STATUS'}"
    print(header)
    print("-" * len(header))
    for r in results:
        line = (
            f"{r['ticker']:<8} | "
            f"{r.get('best_trading_score', 0.0)*100:>10.2f}% | "
            f"{r.get('duration_sec', 0.0):>8.1f}s | "
            f"{r.get('config_file', ''):<35} | "
            f"{r.get('status', 'UNKNOWN')}"
        )
        print(line)
    print("================================================================================")
    print(f"Total Portfolio Optimization Time: {total_duration:.2f}s ({total_duration/60:.2f} mins)")
    print("================================================================================")

    # Save summary report
    summary_path = BACKEND_DIR / "artifacts" / "multi_ticker_optimization_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_duration_sec": total_duration,
                "universe": tickers,
                "trials_per_ticker": trials,
                "results": results,
            },
            f,
            indent=4,
        )
    print(f"Summary JSON saved to {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Master Multi-Ticker Bayesian Optimization Runner"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=INSTITUTIONAL_UNIVERSE,
        help="Space-separated list of tickers (default: AAPL NVDA MSFT AMZN META GOOGL TSLA AMD)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials per ticker (default: 50)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="auto",
        help="Start date for optimization data (default: auto dynamic rolling window)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="auto",
        help="End date for optimization data (default: auto dynamic rolling window)",
    )
    args = parser.parse_args()

    run_optimize_all(
        tickers=[t.upper() for t in args.tickers],
        trials=args.trials,
        start_date=args.start,
        end_date=args.end,
    )


if __name__ == "__main__":
    main()
