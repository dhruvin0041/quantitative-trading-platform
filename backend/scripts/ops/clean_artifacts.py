# clean_artifacts.py
import os
import shutil


def clean_cache():
    """Clears python cache folders recursively."""
    print("--- Clearing Caches ---")
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                path = os.path.join(root, d)
                try:
                    shutil.rmtree(path)
                    print(f"  [CLEANED] {path}")
                except Exception:
                    pass


def clean_training_artifacts():
    """Deletes models and scalers but KEEPS optimized parameters."""
    files = [
        "artifacts/xgb_ensemble.json",
        "artifacts/meta_model.joblib",
        "artifacts/latest_scaler.joblib",
        "artifacts/latest_fusion_weights.weights.h5",
        "artifacts/best_fusion_weights.weights.h5",
        "artifacts/dqn_model.pth",
        "artifacts/lgbm_agent.joblib",
        "artifacts/macro_kill_switch.joblib",
        "configs/active_ticker.json",
    ]
    print("--- Cleaning Training Artifacts ---")
    for f in files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  [DELETED] {f}")
    clean_cache()


def clean_optimization_artifacts(ticker=None, universal=False):
    """Deletes models and specific Optuna databases/configs to force a fresh search."""
    print(
        f"--- Cleaning Optimization Artifacts (Ticker: {ticker}, Universal: {universal}) ---"
    )

    # Generic artifacts
    files = [
        "artifacts/xgb_ensemble.json",
        "artifacts/latest_scaler.joblib",
        "artifacts/latest_fusion_weights.weights.h5",
        "artifacts/best_fusion_weights.weights.h5",
        "artifacts/dqn_model.pth",
        "artifacts/lgbm_agent.joblib",
    ]

    # Target specific ticker or universal files
    if universal:
        files.append("configs/optimized_params_UNIVERSAL.json")
        files.append("optuna_studies/UNIVERSAL.db")
    elif ticker:
        files.append(f"configs/optimized_params_{ticker}.json")
        files.append(f"optuna_studies/{ticker}.db")

    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"  [DELETED] {f}")
            except Exception as e:
                print(f"  [ERROR] Could not delete {f}: {e}")

    clean_cache()


def clean_data_state():
    """Deletes institutional data state: Signal Journal and Paper Trading history."""
    files = [
        "data/empirical_validation.db",
        "data/paper_trading.json",
    ]
    print("--- Cleaning Data State (Zero-State Protocol) ---")
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"  [DELETED] {f}")
            except Exception as e:
                print(f"  [ERROR] Could not delete {f}: {e}")


def main(argv=None):
    import sys

    if argv is None:
        argv = sys.argv[1:]

    # Check for --full flag
    if "--full" in argv:
        clean_data_state()

    # Default behavior for manual execution
    clean_training_artifacts()
    print("\n>>> System artifacts cleaned. <<<")
    if "--full" in argv:
        print(">>> Institutional Zero-State Protocol Enforced. <<<")


if __name__ == "__main__":
    main()
