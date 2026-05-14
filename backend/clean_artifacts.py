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
        "xgb_ensemble.json",
        "meta_model.joblib",
        "latest_scaler.joblib",
        "latest_fusion_weights.weights.h5",
        "best_fusion_weights.weights.h5",
        "dqn_model.pth",
        "macro_kill_switch.joblib",
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
        "xgb_ensemble.json",
        "latest_scaler.joblib",
        "latest_fusion_weights.weights.h5",
        "best_fusion_weights.weights.h5",
        "dqn_model.pth",
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


if __name__ == "__main__":
    # Default behavior for manual execution
    clean_training_artifacts()
    print("\n>>> System artifacts cleaned. <<<")
