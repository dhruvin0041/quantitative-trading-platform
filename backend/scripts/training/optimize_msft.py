"""
Dedicated Bayesian Hyperparameter Optimizer for MSFT.
Runs Optuna Multivariate TPESampler with Purged Walk-Forward Cross-Validation
and saves optimal parameters into backend/configs/optimized_params_MSFT.json.
"""
import argparse
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from scripts.training.optimize import run_optuna_optimization

TICKER = "MSFT"


def main():
    parser = argparse.ArgumentParser(
        description=f"Dedicated Bayesian Hyperparameter Optimizer for {TICKER}"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna optimization trials (default: 50)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2016-01-01",
        help="Start date for training data (default: 2016-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date for training data (default: 2024-12-31)",
    )
    args = parser.parse_args()

    print("=================================================================")
    print(f" STARTING BAYESIAN HYPERPARAMETER OPTIMIZATION FOR {TICKER}")
    print(f" Trials: {args.trials} | Date Range: {args.start} to {args.end}")
    print(f" Destination: backend/configs/optimized_params_{TICKER}.json")
    print("=================================================================")

    run_optuna_optimization(
        ticker=TICKER,
        n_trials=args.trials,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
