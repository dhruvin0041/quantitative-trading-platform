"""
Dedicated Institutional Training Pipeline for AMD.
Trains XGBoost and LightGBM models, fits scalers, calibrates probabilities,
and saves isolated weights into backend/artifacts/models/AMD/.
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

from scripts.training.train import train_ticker

TICKER = "AMD"


def main():
    parser = argparse.ArgumentParser(
        description=f"Dedicated {TICKER} Institutional Training Pipeline"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna optimization trials if optimization is run",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=False,
        help="Run Optuna Bayesian optimization prior to training (default: False)",
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
    print(f" STARTING DEDICATED TRAINING PIPELINE FOR {TICKER}")
    print(f" Target Directory: backend/artifacts/models/{TICKER}/")
    print(f" Optimize First: {args.optimize} | Start: {args.start} | End: {args.end}")
    print("=================================================================")

    metrics = train_ticker(
        ticker=TICKER,
        trials=args.trials,
        skip_optimization=not args.optimize,
        start_date=args.start,
        end_date=args.end,
    )
    print(f"\n[SUCCESS] Completed {TICKER} Training:")
    print(f"  - XGBoost Test Acc:  {metrics.get('xgb_test_acc', 0.0) * 100:.2f}%")
    print(f"  - LightGBM Test Acc: {metrics.get('lgbm_test_acc', 0.0) * 100:.2f}%")
    print(f"  - Saved to:          {metrics.get('model_dir')}")


if __name__ == "__main__":
    main()
