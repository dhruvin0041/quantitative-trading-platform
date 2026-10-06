import argparse
import json
import os
import sys
from pathlib import Path

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import joblib
import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.gpu_utils import (
    get_catboost_gpu_params,
    get_lightgbm_gpu_params,
    get_xgboost_gpu_params,
)

# Ensure models dir exists for saving params
os.makedirs("models", exist_ok=True)
os.makedirs("configs", exist_ok=True)


def purged_walk_forward_cv(X_raw, y_raw, fit_and_predict_fn, n_splits=5, embargo=15):
    """
    Purged chronological expanding-window splits with post-train embargo.
    1. Expanding training windows enforce temporal causality.
    2. 15-bar post-train embargo purges Triple Barrier forward label lookahead.
    3. Scaler fit strictly inside the training fold (zero scaling leakage).
    """
    n_samples = len(X_raw)
    fold_size = n_samples // (n_splits + 1)
    scores = []
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        effective_train_end = max(1, train_end - embargo)
        val_start = train_end
        val_end = min(n_samples, (i + 1) * fold_size)
        if val_end <= val_start or effective_train_end <= 0:
            continue
        train_idx = np.arange(0, effective_train_end)
        val_idx = np.arange(val_start, val_end)

        scaler = StandardScaler()
        X_t = scaler.fit_transform(X_raw[train_idx])
        X_v = scaler.transform(X_raw[val_idx])
        y_t = y_raw[train_idx]
        y_v = y_raw[val_idx]

        preds = fit_and_predict_fn(X_t, y_t, X_v)
        scores.append(f1_score(y_v, preds, average="macro", zero_division=0))
    return float(np.mean(scores)) if scores else 0.33


def objective_xgb(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [100, 300, 600, 1000]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 9]),
        "learning_rate": trial.suggest_categorical("lr", [0.001, 0.01, 0.05, 0.1]),
        "subsample": trial.suggest_categorical("subsample", [0.5, 0.7, 0.85, 1.0]),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.5, 0.7, 0.85, 1.0]
        ),
        "min_child_weight": trial.suggest_categorical(
            "min_child_weight", [1, 3, 5, 10]
        ),
        "gamma": trial.suggest_categorical("gamma", [0, 0.1, 0.5, 1.0]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [1e-8, 0.1, 1.0, 10.0]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [1e-8, 0.1, 1.0, 10.0]),
        "max_bin": trial.suggest_categorical("max_bin", [64, 128, 192, 255]),
        "colsample_bylevel": trial.suggest_categorical(
            "colsample_bylevel", [0.5, 0.7, 0.85, 1.0]
        ),
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        **get_xgboost_gpu_params()
    }

    def fit_and_predict(X_t, y_t, X_v):
        model = XGBClassifier(**params)
        model.fit(X_t, y_t)
        return model.predict(X_v)

    return purged_walk_forward_cv(X_train, y_train, fit_and_predict)


def objective_lgbm(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [100, 300, 600, 1000]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 10, 15]),
        "learning_rate": trial.suggest_categorical("lr", [0.001, 0.01, 0.1, 0.2]),
        "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 127, 255]),
        "subsample": trial.suggest_categorical("subsample", [0.5, 0.7, 0.85, 1.0]),
        "subsample_freq": trial.suggest_categorical("subsample_freq", [1, 3, 5, 10]),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.5, 0.7, 0.85, 1.0]
        ),
        "min_child_samples": trial.suggest_categorical(
            "min_child_samples", [5, 20, 50, 100]
        ),
        "min_split_gain": trial.suggest_categorical(
            "min_split_gain", [1e-8, 0.01, 0.1, 1.0]
        ),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [1e-8, 0.1, 1.0, 10.0]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [1e-8, 0.1, 1.0, 10.0]),
        "max_bin": trial.suggest_categorical("max_bin", [64, 128, 192, 255]),
        "feature_fraction": trial.suggest_categorical(
            "feature_fraction", [0.5, 0.7, 0.85, 1.0]
        ),
        "bagging_fraction": trial.suggest_categorical(
            "bagging_fraction", [0.5, 0.7, 0.85, 1.0]
        ),
        "min_data_in_leaf": trial.suggest_categorical(
            "min_data_in_leaf", [10, 20, 50, 100]
        ),
        "objective": "multiclass",
        "num_class": 3,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
        **get_lightgbm_gpu_params()
    }

    def fit_and_predict(X_t, y_t, X_v):
        model = LGBMClassifier(**params)
        model.fit(X_t, y_t)
        return model.predict(X_v)

    return purged_walk_forward_cv(X_train, y_train, fit_and_predict)


def objective_catboost(trial, X_train, y_train):
    params = {
        "iterations": trial.suggest_categorical("iterations", [100, 300, 600, 1000]),
        "depth": trial.suggest_categorical("depth", [4, 6, 8, 10]),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.001, 0.01, 0.1, 0.2]
        ),
        "l2_leaf_reg": trial.suggest_categorical("l2_leaf_reg", [1, 3, 5, 10]),
        "random_strength": trial.suggest_categorical(
            "random_strength", [0.1, 1.0, 2.0, 5.0]
        ),
        "bagging_temperature": trial.suggest_categorical(
            "bagging_temperature", [0.0, 0.5, 1.0, 2.0]
        ),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide", "Depthwise"]
        ),
        "loss_function": "MultiClass",
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
        **get_catboost_gpu_params()
    }

    def fit_and_predict(X_t, y_t, X_v):
        model = CatBoostClassifier(**params)
        model.fit(X_t, y_t)
        return model.predict(X_v)

    return purged_walk_forward_cv(X_train, y_train, fit_and_predict)


def objective_rf(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [100, 200, 500, 1000]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [5, 10, 20, 50]),
        "min_samples_split": trial.suggest_categorical(
            "min_samples_split", [2, 5, 10, 20]
        ),
        "min_samples_leaf": trial.suggest_categorical(
            "min_samples_leaf", [1, 2, 4, 10]
        ),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None, "sqrt"]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False, True, False]),
        "random_state": 42,
        "n_jobs": -1,
    }

    def fit_and_predict(X_t, y_t, X_v):
        model = RandomForestClassifier(**params)
        model.fit(X_t, y_t)
        return model.predict(X_v)

    return purged_walk_forward_cv(X_train, y_train, fit_and_predict)


def run_optimization(n_trials: int = 50) -> bool:
    from src.utils.gpu_utils import get_compute_backend

    get_compute_backend()
    artifacts_dir = BACKEND_DIR / "artifacts"
    configs_dir = BACKEND_DIR / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    x_train_path = artifacts_dir / "X_train_tabular.joblib"
    y_train_path = artifacts_dir / "y_train_sig.joblib"

    if not x_train_path.exists() or not y_train_path.exists():
        print(" [ERROR] Required data artifacts not found. Run data preparation first.")
        return False

    print("--- Loading Data for Bayesian Optimization ---")
    X_train = joblib.load(x_train_path)
    y_train = joblib.load(y_train_path)

    print(
        f"Running Bayesian Optimization for 4 models with {n_trials} trials each..."
    )
    for model_name, obj_func in [
        ("xgb", objective_xgb),
        ("lgbm", objective_lgbm),
        ("catboost", objective_catboost),
        ("rf", objective_rf),
    ]:
        print(f"\nOptimizing {model_name} ({n_trials} trials)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: obj_func(t, X_train, y_train), n_trials=n_trials, n_jobs=1
        )
        print(f"Best {model_name} Macro-F1: {study.best_value:.4f}")
        with open(configs_dir / f"best_{model_name}_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian Hyperparameter Optimization for Branch Models"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model (default: 50)",
    )
    args = parser.parse_args()
    run_optimization(n_trials=args.trials)
