# clean_artifacts.py
import os
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


def _resolve_path(f: str) -> Path:
    p = Path(f)
    if p.is_absolute():
        return p
    return BACKEND_DIR / f


def clean_cache():
    """Clears python cache folders recursively."""
    print("--- Clearing Caches ---")
    for root, dirs, files in os.walk(BACKEND_DIR):
        dirs[:] = [
            d for d in dirs if d not in ("venv", ".venv", ".git", "node_modules")
        ]
        for d in dirs:
            if d == "__pycache__":
                path = os.path.join(root, d)
                try:
                    shutil.rmtree(path)
                    print(f"  [CLEANED] {path}")
                except Exception:
                    pass


def clean_training_artifacts(ticker=None, all_tickers=False):
    """
    Deletes models and scalers.
    - If ticker is specified: purges that ticker's isolated directory backend/artifacts/models/{ticker}/
      as well as ticker-specific scalers.
    - If all_tickers is True or (ticker is None and global clean): purges all ticker models in
      artifacts/models/ and multi-asset training summaries.
    """
    files = [
        "artifacts/xgb_ensemble.json",
        "artifacts/meta_model.joblib",
        "artifacts/meta_ensemble.joblib",
        "artifacts/model_calibrator.joblib",
        "artifacts/latest_scaler.joblib",
        "artifacts/latest_fusion_weights.weights.h5",
        "artifacts/best_fusion_weights.weights.h5",
        "artifacts/tft_quantile_weights.weights.h5",
        "artifacts/dqn_model.pth",
        "artifacts/lgbm_agent.joblib",
        "artifacts/macro_kill_switch.joblib",
        "artifacts/calibrator.joblib",
        "artifacts/xgb_calibrator.joblib",
        "artifacts/lgbm_calibrator.joblib",
        "artifacts/X_train_tabular.joblib",
        "artifacts/y_train_sig.joblib",
        "artifacts/X_val_tabular.joblib",
        "artifacts/y_val_sig.joblib",
        "artifacts/best_lgbm_params.json",
        "artifacts/best_xgb_params.json",
        "configs/active_ticker.json",
    ]

    if ticker:
        t = ticker.upper()
        files.append(f"artifacts/scaler_{t}.joblib")
        ticker_dir = BACKEND_DIR / "artifacts" / "models" / t
        if ticker_dir.exists():
            try:
                shutil.rmtree(ticker_dir)
                print(f"  [CLEANED DIRECTORY] artifacts/models/{t}/")
            except Exception as e:
                print(f"  [ERROR] Could not remove artifacts/models/{t}/: {e}")
    elif all_tickers:
        files.append("artifacts/multi_ticker_training_summary.json")
        models_dir = BACKEND_DIR / "artifacts" / "models"
        if models_dir.exists():
            for item in models_dir.iterdir():
                if item.is_dir():
                    try:
                        shutil.rmtree(item)
                        print(f"  [CLEANED DIRECTORY] {item.relative_to(BACKEND_DIR)}")
                    except Exception as e:
                        print(f"  [ERROR] Could not remove {item}: {e}")

    print(f"--- Cleaning Training Artifacts (Ticker: {ticker}, All Tickers: {all_tickers}) ---")
    for f in files:
        target = _resolve_path(f)
        if target.exists():
            try:
                target.unlink()
                print(f"  [DELETED] {f}")
            except Exception as e:
                print(f"  [ERROR] Could not delete {f}: {e}")
    clean_cache()


def clean_optimization_artifacts(ticker=None, universal=False, all_tickers=False):
    """
    Deletes optimization artifacts, studies, and parameter files.
    - If ticker is specified: removes configs/optimized_params_{ticker}.json,
      optuna_studies/{ticker}.db, and artifacts/models/{ticker}/optimized_params.json.
    - If universal or all_tickers: removes all ticker optimization parameter files,
      multi-ticker optimization summary, and optuna databases.
    """
    print(
        f"--- Cleaning Optimization Artifacts (Ticker: {ticker}, Universal: {universal}, All Tickers: {all_tickers}) ---"
    )

    # Generic artifacts
    files = [
        "artifacts/xgb_ensemble.json",
        "artifacts/meta_ensemble.joblib",
        "artifacts/model_calibrator.joblib",
        "artifacts/latest_scaler.joblib",
        "artifacts/latest_fusion_weights.weights.h5",
        "artifacts/best_fusion_weights.weights.h5",
        "artifacts/tft_quantile_weights.weights.h5",
        "artifacts/dqn_model.pth",
        "artifacts/lgbm_agent.joblib",
        "artifacts/best_xgb_params.json",
        "artifacts/best_lgbm_params.json",
        "artifacts/v3/xgb_ensemble.json",
        "artifacts/v3/lgbm_agent.joblib",
        "artifacts/v3/latest_scaler.joblib",
        "artifacts/v3/model_calibrator.joblib",
        "artifacts/v3/kept_features.json",
        "configs/best_xgb_params.json",
        "configs/best_lgbm_params.json",
        "configs/best_catboost_params.json",
        "configs/best_rf_params.json",
    ]

    # Target specific ticker or universal files
    if ticker:
        t = ticker.upper()
        files.append(f"configs/optimized_params_{t}.json")
        files.append(f"optuna_studies/{t}.db")
        files.append(f"artifacts/models/{t}/optimized_params.json")

    if universal or all_tickers:
        files.append("configs/optimized_params_UNIVERSAL.json")
        files.append("optuna_studies/UNIVERSAL.db")
        files.append("artifacts/multi_ticker_optimization_summary.json")

        # Clean all per-ticker optimized_params in configs/ and artifacts/models/
        configs_dir = BACKEND_DIR / "configs"
        if configs_dir.exists():
            for p in configs_dir.glob("optimized_params_*.json"):
                files.append(str(p.relative_to(BACKEND_DIR)))

        optuna_dir = BACKEND_DIR / "optuna_studies"
        if optuna_dir.exists():
            for p in optuna_dir.glob("*.db"):
                files.append(str(p.relative_to(BACKEND_DIR)))

        models_dir = BACKEND_DIR / "artifacts" / "models"
        if models_dir.exists():
            for p in models_dir.glob("*/optimized_params.json"):
                files.append(str(p.relative_to(BACKEND_DIR)))

    for f in files:
        target = _resolve_path(f)
        if target.exists():
            try:
                target.unlink()
                print(f"  [DELETED] {f}")
            except Exception as e:
                print(f"  [ERROR] Could not delete {f}: {e}")

    clean_cache()


def clean_data_state():
    """Deletes institutional data state: Signal Journal and Paper Trading history."""
    files = [
        "data/empirical_validation.db",
        "data/paper_trading.json",
        "data/paper_trading.db",
        "data/backtest_log.db",
        "artifacts/paper_execution_state.db",
    ]
    print("--- Cleaning Data State (Zero-State Protocol) ---")
    for f in files:
        target = _resolve_path(f)
        if target.exists():
            try:
                target.unlink()
                print(f"  [DELETED] {f}")
            except Exception as e:
                print(f"  [ERROR] Could not delete {f}: {e}")


def main(argv=None):
    import argparse
    import sys

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Clean system artifacts")
    parser.add_argument(
        "--full", action="store_true", help="Enforce full institutional zero-state"
    )
    parser.add_argument(
        "--ticker", type=str, default=None, help="Stock ticker symbol (purges only this ticker)"
    )
    parser.add_argument(
        "--all-tickers", action="store_true", help="Purge artifacts across all tickers in universe"
    )
    parser.add_argument(
        "--universal", action="store_true", help="Clean universal artifacts and all ticker models"
    )
    args = parser.parse_args(argv)

    if args.full:
        clean_data_state()

    is_all = args.universal or args.all_tickers or args.full

    if args.ticker:
        clean_optimization_artifacts(ticker=args.ticker)
        clean_training_artifacts(ticker=args.ticker)
    elif is_all:
        clean_optimization_artifacts(all_tickers=True, universal=True)
        clean_training_artifacts(all_tickers=True)
    else:
        # Default manual run without flags cleans root training artifacts and all ticker models
        clean_training_artifacts(all_tickers=True)

    print("\n>>> System artifacts cleaned. <<<")
    if args.full:
        print(">>> Institutional Zero-State Protocol Enforced. <<<")


if __name__ == "__main__":
    main()

