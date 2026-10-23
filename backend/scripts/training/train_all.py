"""
Master Institutional Multi-Ticker Training Runner.
Sequentially trains isolated XGBoost, LightGBM, and probability calibrator models
for the complete institutional universe (AAPL, NVDA, MSFT, AMZN, META, GOOGL, TSLA, AMD)
and outputs a comparative multi-asset performance matrix.
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

from scripts.training.train import train_ticker
from src.utils.date_window import resolve_training_dates

INSTITUTIONAL_UNIVERSE = [
    # Core US Mega-Cap Tech
    "AAPL",
    "NVDA",
    "MSFT",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AMD",
    # US Low-Beta / Defensives & Financials
    "COST",
    "WMT",
    "JPM",
    "LLY",
    # India (NSE) Institutional Giants
    "RELIANCE.NS",
    "TCS.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "LT.NS",
    "ITC.NS",
]


def run_train_all(
    tickers=None,
    trials: int = 50,
    optimize: bool = False,
    start_date: str = "auto",
    end_date: str = "auto",
):
    if tickers is None:
        tickers = INSTITUTIONAL_UNIVERSE

    start_date, end_date, is_auto = resolve_training_dates(start_date, end_date)
    total_start = time.time()
    print("================================================================================")
    print("      HYDRA INSTITUTIONAL MULTI-TICKER TRAINING RUNNER (SOTA 2026)")
    print(f" Universe ({len(tickers)} assets): {', '.join(tickers)}")
    print(f" Historical Range: {start_date} to {end_date} (Auto-Rolling: {is_auto})")
    print(f" Run Bayesian Optimization: {optimize} (Trials per asset: {trials})")
    print("================================================================================")

    results = []
    for idx, ticker in enumerate(tickers, start=1):
        print(f"\n[{idx}/{len(tickers)}] >>> Launching Dedicated Training Pipeline for {ticker} <<<")
        try:
            summary = train_ticker(
                ticker=ticker,
                trials=trials,
                skip_optimization=not optimize,
                start_date=start_date,
                end_date=end_date,
            )
            results.append(summary)
            print(f"  [+] {ticker} SUCCESS -> XGB Test: {summary['xgb_test_acc']*100:.2f}%, LGBM Test: {summary['lgbm_test_acc']*100:.2f}% ({summary['duration_sec']}s)")
        except Exception as e:
            print(f"  [-] {ticker} FAILED: {e}")
            results.append({
                "ticker": ticker,
                "xgb_train_acc": 0.0,
                "xgb_test_acc": 0.0,
                "lgbm_train_acc": 0.0,
                "lgbm_test_acc": 0.0,
                "duration_sec": 0.0,
                "error": str(e),
                "model_dir": f"artifacts/models/{ticker}",
            })

    total_duration = time.time() - total_start

    # Print Summary Table
    print("\n================================================================================")
    print("               MULTI-TICKER TRAINING PORTFOLIO SUMMARY MATRIX")
    print("================================================================================")
    header = f"{'TICKER':<8} | {'XGB TRAIN':<10} | {'XGB TEST':<10} | {'LGBM TRAIN':<10} | {'LGBM TEST':<10} | {'TIME':<8} | {'STATUS'}"
    print(header)
    print("-" * len(header))
    for r in results:
        status = "OK" if "error" not in r else "ERROR"
        line = (
            f"{r['ticker']:<8} | "
            f"{r.get('xgb_train_acc', 0.0)*100:>8.2f}% | "
            f"{r.get('xgb_test_acc', 0.0)*100:>8.2f}% | "
            f"{r.get('lgbm_train_acc', 0.0)*100:>9.2f}% | "
            f"{r.get('lgbm_test_acc', 0.0)*100:>8.2f}% | "
            f"{r.get('duration_sec', 0.0):>6.1f}s | "
            f"{status}"
        )
        print(line)
    print("================================================================================")
    print(f"All {len(tickers)} models isolated in: backend/artifacts/models/<TICKER>/")
    print(f"Total Execution Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print("================================================================================")

    # Save summary report
    summary_path = BACKEND_DIR / "artifacts" / "multi_ticker_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_duration_sec": total_duration,
                "universe": tickers,
                "results": results,
            },
            f,
            indent=4,
        )
    print(f"Summary JSON saved to {summary_path}")

    # Reset active ticker to primary benchmark asset (AAPL)
    active_path = BACKEND_DIR / "configs" / "active_ticker.json"
    with open(active_path, "w") as f:
        json.dump({"ticker": "AAPL", "market": "us"}, f, indent=4)
    print("Default active ticker restored to AAPL.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Master Multi-Ticker Training Runner for Hydra Institutional Platform"
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
        help="Number of Optuna optimization trials per ticker (if --optimize is active)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=False,
        help="Run Optuna Bayesian hyperparameter optimization prior to training (default: False)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="auto",
        help="Start date for training data (default: auto dynamic rolling window)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="auto",
        help="End date for training data (default: auto dynamic rolling window)",
    )
    args = parser.parse_args()

    run_train_all(
        tickers=[t.upper() for t in args.tickers],
        trials=args.trials,
        optimize=args.optimize,
        start_date=args.start,
        end_date=args.end,
    )


if __name__ == "__main__":
    main()
