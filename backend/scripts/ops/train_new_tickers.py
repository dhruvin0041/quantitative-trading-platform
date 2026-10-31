"""
Institutional Training & Optimization Pipeline for Untrained / New Universe Assets.
Runs Purged Walk-Forward Bayesian Optimization (XGBoost + LightGBM) with user-defined trials,
trains isolated model ensembles, fits probability calibrators, and stores dedicated weights.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from scripts.training.train import train_ticker_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainAndOptimize")

# The newly introduced outside assets requiring Bayesian Hyperparameter Optimization
UNTRAINED_TICKERS = [
    # US Stable Outside Additions
    "JNJ",
    "V",
    "BRK-B",
    # Indian NSE Stable Outside Additions
    "TITAN.NS",
    "SUNPHARMA.NS",
]


def train_and_optimize_untrained(tickers=None, trials: int = 250, skip_optimization: bool = False):
    if tickers is None or len(tickers) == 0:
        tickers = UNTRAINED_TICKERS

    logger.info("=" * 90)
    logger.info("STARTING BAYESIAN HYPERPARAMETER OPTIMIZATION & TRAINING FOR UNTRAINED ASSETS")
    logger.info(f"Target Assets ({len(tickers)}): {', '.join(tickers)}")
    logger.info(f"Optuna Trials per Model: {trials} (Total {trials * 2} trials per asset)")
    logger.info(f"Optimization Mode: {'SKIPPED' if skip_optimization else 'ACTIVE (Purged Walk-Forward CV)'}")
    logger.info("=" * 90)

    total_start = time.time()
    results = []

    for i, sym in enumerate(tickers, start=1):
        logger.info(f"\n[{i:02d}/{len(tickers):02d}] >>> Processing {sym} (Optimization: {not skip_optimization}, Trials: {trials}) <<<")
        t0 = time.time()
        try:
            res = train_ticker_pipeline(
                ticker=sym,
                trials=trials,
                skip_optimization=skip_optimization,
            )
            elapsed = round(time.time() - t0, 2)
            results.append({
                "ticker": sym,
                "status": "SUCCESS",
                "xgb_test_acc": res.get("xgb_test_acc", 0.0),
                "lgbm_test_acc": res.get("lgbm_test_acc", 0.0),
                "duration_sec": elapsed,
            })
            logger.info(
                f"[SUCCESS] {sym} completed in {elapsed}s | "
                f"XGB Test Acc: {res.get('xgb_test_acc', 0.0) * 100:.2f}% | "
                f"LGBM Test Acc: {res.get('lgbm_test_acc', 0.0) * 100:.2f}%"
            )
        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            logger.error(f"[FAILED] {sym} encountered error after {elapsed}s: {exc}", exc_info=True)
            results.append({
                "ticker": sym,
                "status": "FAILED",
                "error": str(exc),
                "duration_sec": elapsed,
            })

    # Reset active ticker to AAPL
    active_path = backend_dir / "configs" / "active_ticker.json"
    if active_path.exists():
        with open(active_path, "w") as f:
            json.dump({"ticker": "AAPL", "market": "us"}, f, indent=2)

    total_elapsed = round(time.time() - total_start, 2)
    logger.info("\n" + "=" * 90)
    logger.info(f"BATCH EXECUTION COMPLETE IN {total_elapsed}s")
    logger.info("=" * 90)
    for r in results:
        status_str = f"Test Acc: XGB={r.get('xgb_test_acc', 0.0)*100:.2f}%, LGBM={r.get('lgbm_test_acc', 0.0)*100:.2f}%" if r["status"] == "SUCCESS" else f"Error: {r.get('error')}"
        logger.info(f"  {r['ticker']:<14} | {r['status']:<8} | {r['duration_sec']:>6.1f}s | {status_str}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Institutional Training & Optimization for Untrained Assets")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=UNTRAINED_TICKERS,
        help="List of tickers to train and optimize (default: untrained stable assets JNJ, V, BRK-B, TITAN.NS, SUNPHARMA.NS)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=250,
        help="Number of Optuna trials per model (default: 250)",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip optimization and use fallback parameters",
    )
    args = parser.parse_args()

    train_and_optimize_untrained(
        tickers=args.tickers,
        trials=args.trials,
        skip_optimization=args.skip_optimization,
    )


if __name__ == "__main__":
    main()
