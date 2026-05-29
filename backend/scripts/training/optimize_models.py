import optuna
import joblib
import numpy as np
import json
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# Ensure models dir exists for saving params
os.makedirs("models", exist_ok=True)
os.makedirs("configs", exist_ok=True)


def objective_xgb(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [100, 300, 600, 1000]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [3, 6, 9, 12]),
        "learning_rate": trial.suggest_categorical("lr", [0.001, 0.01, 0.1, 0.2]),
        "subsample": trial.suggest_categorical("subsample", [0.5, 0.7, 0.85, 1.0]),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.5, 0.7, 0.85, 1.0]
        ),
        "min_child_weight": trial.suggest_categorical(
            "min_child_weight", [1, 5, 10, 20]
        ),
        "gamma": trial.suggest_categorical("gamma", [1e-8, 0.1, 1.0, 5.0]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [1e-8, 0.1, 1.0, 10.0]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [1e-8, 0.1, 1.0, 10.0]),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 384, 512]),
        "colsample_bylevel": trial.suggest_categorical(
            "colsample_bylevel", [0.5, 0.7, 0.85, 1.0]
        ),
        "max_delta_step": trial.suggest_categorical("max_delta_step", [0, 1, 5, 10]),
        "scale_pos_weight": trial.suggest_categorical(
            "scale_pos_weight", [1, 2, 5, 10]
        ),
        "objective": "multi:softprob",
        "num_class": 3,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_t, y_t)
        prob = model.predict_proba(X_v)[:, 2]
        y_v_bin = (y_v == 2).astype(int)
        if len(np.unique(y_v_bin)) > 1:
            scores.append(roc_auc_score(y_v_bin, prob))

    return np.mean(scores) if scores else 0.5


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
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 384, 512]),
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
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]
        model = LGBMClassifier(**params)
        model.fit(X_t, y_t)
        prob = model.predict_proba(X_v)[:, 2]
        y_v_bin = (y_v == 2).astype(int)
        if len(np.unique(y_v_bin)) > 1:
            scores.append(roc_auc_score(y_v_bin, prob))

    return np.mean(scores) if scores else 0.5


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
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]
        model = CatBoostClassifier(**params)
        model.fit(X_t, y_t)
        prob = model.predict_proba(X_v)[:, 2]
        y_v_bin = (y_v == 2).astype(int)
        if len(np.unique(y_v_bin)) > 1:
            scores.append(roc_auc_score(y_v_bin, prob))
    return np.mean(scores) if scores else 0.5


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

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]
        model = RandomForestClassifier(**params)
        model.fit(X_t, y_t)
        prob = model.predict_proba(X_v)[:, 2]
        y_v_bin = (y_v == 2).astype(int)
        if len(np.unique(y_v_bin)) > 1:
            scores.append(roc_auc_score(y_v_bin, prob))
    return np.mean(scores) if scores else 0.5


def run_optimization():
    # Load training data saved by train.py
    if not os.path.exists("artifacts/X_train_tabular.joblib") or not os.path.exists(
        "artifacts/y_train_sig.joblib"
    ):
        print(" [ERROR] Required data artifacts not found. Run data preparation first.")
        return False

    print("--- Loading Data for Bayesian Optimization ---")
    X_train = joblib.load("artifacts/X_train_tabular.joblib")
    y_train = joblib.load("artifacts/y_train_sig.joblib")

    print("Running Bayesian Optimization for 4 models with 4 choices per parameter...")
    for model_name, obj_func in [
        ("xgb", objective_xgb),
        ("lgbm", objective_lgbm),
        ("catboost", objective_catboost),
        ("rf", objective_rf),
    ]:
        print(f"\nOptimizing {model_name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: obj_func(t, X_train, y_train), n_trials=50, n_jobs=-1)
        print(f"Best {model_name} AUC: {study.best_value}")
        with open(f"configs/best_{model_name}_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2)
    return True


if __name__ == "__main__":
    run_optimization()
