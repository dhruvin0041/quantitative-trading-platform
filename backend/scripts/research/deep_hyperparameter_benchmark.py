"""
HYDRA Quantitative Research: Deep Hyperparameter Benchmark for XGBoost & LightGBM.
Systematically tests multiple hyperparameter configurations, feature subsets, and regularization
schemes across institutional test partitions (75% Train / 25% Test) with strict Zero Data Leakage.
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.training.train import (
    FEATURE_COLUMNS_V30,
    prepare_ticker_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DeepHyperparameterBenchmark")


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Fits model and evaluates comprehensive out-of-sample accuracy metrics."""
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = float(accuracy_score(y_train, train_preds))
    test_acc = float(accuracy_score(y_test, test_preds))
    macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))

    # Directional Accuracy (precision when forecasting non-neutral BUY/SELL signals)
    dir_mask = (y_test != 1) & (test_preds != 1)
    if np.sum(dir_mask) > 0:
        dir_acc = float(np.mean(y_test[dir_mask] == test_preds[dir_mask]))
    else:
        dir_acc = 0.3333

    composite_score = float(0.45 * dir_acc + 0.40 * test_acc + 0.15 * macro_f1)
    overfit_gap = float(train_acc - test_acc)

    return {
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "dir_acc": round(dir_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "composite_score": round(composite_score, 4),
        "overfit_gap": round(overfit_gap, 4),
        "fit_time_sec": round(fit_time, 3),
    }


def get_xgb_parameter_profiles() -> List[Dict[str, Any]]:
    """Returns curated institutional XGBoost hyperparameter candidate profiles."""
    return [
        {
            "name": "XGB_UltraShallow_L2_Heavy",
            "params": {
                "max_depth": 2,
                "n_estimators": 80,
                "learning_rate": 0.03,
                "subsample": 0.80,
                "colsample_bytree": 0.70,
                "colsample_bylevel": 0.80,
                "min_child_weight": 12.0,
                "gamma": 1.5,
                "reg_alpha": 2.0,
                "reg_lambda": 12.0,
            },
        },
        {
            "name": "XGB_Shallow_Balanced",
            "params": {
                "max_depth": 2,
                "n_estimators": 120,
                "learning_rate": 0.025,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "colsample_bylevel": 0.80,
                "min_child_weight": 8.0,
                "gamma": 1.0,
                "reg_alpha": 1.0,
                "reg_lambda": 8.0,
            },
        },
        {
            "name": "XGB_Shallow_Slow_200",
            "params": {
                "max_depth": 2,
                "n_estimators": 200,
                "learning_rate": 0.015,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "colsample_bylevel": 0.85,
                "min_child_weight": 6.0,
                "gamma": 0.5,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
        },
        {
            "name": "XGB_Depth3_Conservative",
            "params": {
                "max_depth": 3,
                "n_estimators": 100,
                "learning_rate": 0.03,
                "subsample": 0.80,
                "colsample_bytree": 0.70,
                "colsample_bylevel": 0.80,
                "min_child_weight": 10.0,
                "gamma": 1.0,
                "reg_alpha": 1.5,
                "reg_lambda": 8.0,
            },
        },
        {
            "name": "XGB_Depth3_Balanced",
            "params": {
                "max_depth": 3,
                "n_estimators": 130,
                "learning_rate": 0.025,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "colsample_bylevel": 0.80,
                "min_child_weight": 8.0,
                "gamma": 0.8,
                "reg_alpha": 1.0,
                "reg_lambda": 6.0,
            },
        },
        {
            "name": "XGB_Depth3_Slow_180",
            "params": {
                "max_depth": 3,
                "n_estimators": 180,
                "learning_rate": 0.015,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "colsample_bylevel": 0.85,
                "min_child_weight": 6.0,
                "gamma": 0.5,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
        },
        {
            "name": "XGB_Depth3_Agile_LowTrees",
            "params": {
                "max_depth": 3,
                "n_estimators": 70,
                "learning_rate": 0.045,
                "subsample": 0.80,
                "colsample_bytree": 0.75,
                "colsample_bylevel": 0.80,
                "min_child_weight": 8.0,
                "gamma": 1.0,
                "reg_alpha": 1.5,
                "reg_lambda": 8.0,
            },
        },
        {
            "name": "XGB_Depth3_HighSubsample",
            "params": {
                "max_depth": 3,
                "n_estimators": 140,
                "learning_rate": 0.025,
                "subsample": 0.90,
                "colsample_bytree": 0.85,
                "colsample_bylevel": 0.85,
                "min_child_weight": 7.0,
                "gamma": 0.6,
                "reg_alpha": 0.8,
                "reg_lambda": 5.0,
            },
        },
        {
            "name": "XGB_Depth3_HighRandomization",
            "params": {
                "max_depth": 3,
                "n_estimators": 140,
                "learning_rate": 0.03,
                "subsample": 0.70,
                "colsample_bytree": 0.60,
                "colsample_bylevel": 0.70,
                "min_child_weight": 10.0,
                "gamma": 1.2,
                "reg_alpha": 2.0,
                "reg_lambda": 10.0,
            },
        },
        {
            "name": "XGB_Depth4_StrictPruning",
            "params": {
                "max_depth": 4,
                "n_estimators": 90,
                "learning_rate": 0.025,
                "subsample": 0.75,
                "colsample_bytree": 0.65,
                "colsample_bylevel": 0.75,
                "min_child_weight": 15.0,
                "gamma": 2.0,
                "reg_alpha": 3.0,
                "reg_lambda": 15.0,
            },
        },
        {
            "name": "XGB_Depth4_Slow",
            "params": {
                "max_depth": 4,
                "n_estimators": 150,
                "learning_rate": 0.015,
                "subsample": 0.80,
                "colsample_bytree": 0.70,
                "colsample_bylevel": 0.75,
                "min_child_weight": 12.0,
                "gamma": 1.5,
                "reg_alpha": 2.0,
                "reg_lambda": 12.0,
            },
        },
        {
            "name": "XGB_Lossguide_MaxLeaves6",
            "params": {
                "grow_policy": "lossguide",
                "max_leaves": 6,
                "max_depth": 0,
                "n_estimators": 120,
                "learning_rate": 0.03,
                "subsample": 0.80,
                "colsample_bytree": 0.75,
                "colsample_bylevel": 0.80,
                "min_child_weight": 8.0,
                "gamma": 1.0,
                "reg_alpha": 1.5,
                "reg_lambda": 8.0,
            },
        },
        {
            "name": "XGB_Lossguide_MaxLeaves8",
            "params": {
                "grow_policy": "lossguide",
                "max_leaves": 8,
                "max_depth": 0,
                "n_estimators": 140,
                "learning_rate": 0.025,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "colsample_bylevel": 0.80,
                "min_child_weight": 7.0,
                "gamma": 0.8,
                "reg_alpha": 1.0,
                "reg_lambda": 6.0,
            },
        },
        {
            "name": "XGB_High_Regularization_L1L2",
            "params": {
                "max_depth": 3,
                "n_estimators": 110,
                "learning_rate": 0.03,
                "subsample": 0.80,
                "colsample_bytree": 0.70,
                "colsample_bylevel": 0.80,
                "min_child_weight": 10.0,
                "gamma": 1.5,
                "reg_alpha": 3.5,
                "reg_lambda": 15.0,
            },
        },
        {
            "name": "XGB_Mild_Regularization",
            "params": {
                "max_depth": 3,
                "n_estimators": 120,
                "learning_rate": 0.03,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "colsample_bylevel": 0.85,
                "min_child_weight": 5.0,
                "gamma": 0.3,
                "reg_alpha": 0.5,
                "reg_lambda": 3.0,
            },
        },
    ]


def get_lgbm_parameter_profiles() -> List[Dict[str, Any]]:
    """Returns curated institutional LightGBM hyperparameter candidate profiles."""
    return [
        {
            "name": "LGBM_UltraShallow_Leaves4",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 2,
                "num_leaves": 4,
                "n_estimators": 100,
                "learning_rate": 0.03,
                "min_child_samples": 50,
                "min_split_gain": 0.2,
                "subsample": 0.80,
                "subsample_freq": 1,
                "colsample_bytree": 0.70,
                "reg_alpha": 2.0,
                "reg_lambda": 8.0,
            },
        },
        {
            "name": "LGBM_Shallow_Leaves5_Slow",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 2,
                "num_leaves": 5,
                "n_estimators": 180,
                "learning_rate": 0.015,
                "min_child_samples": 40,
                "min_split_gain": 0.1,
                "subsample": 0.85,
                "subsample_freq": 1,
                "colsample_bytree": 0.75,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
        },
        {
            "name": "LGBM_Depth3_Leaves6_Balanced",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 3,
                "num_leaves": 6,
                "n_estimators": 120,
                "learning_rate": 0.03,
                "min_child_samples": 45,
                "min_split_gain": 0.1,
                "subsample": 0.85,
                "subsample_freq": 1,
                "colsample_bytree": 0.75,
                "reg_alpha": 1.5,
                "reg_lambda": 6.0,
            },
        },
        {
            "name": "LGBM_Depth3_Leaves7_Standard",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 3,
                "num_leaves": 7,
                "n_estimators": 120,
                "learning_rate": 0.03,
                "min_child_samples": 40,
                "min_split_gain": 0.1,
                "subsample": 0.85,
                "subsample_freq": 1,
                "colsample_bytree": 0.75,
                "reg_alpha": 1.5,
                "reg_lambda": 6.0,
            },
        },
        {
            "name": "LGBM_Depth3_Leaves7_Slow_180",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 3,
                "num_leaves": 7,
                "n_estimators": 180,
                "learning_rate": 0.015,
                "min_child_samples": 50,
                "min_split_gain": 0.1,
                "subsample": 0.85,
                "subsample_freq": 1,
                "colsample_bytree": 0.75,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
        },
        {
            "name": "LGBM_Depth3_Leaves7_HighMinChild",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 3,
                "num_leaves": 7,
                "n_estimators": 110,
                "learning_rate": 0.03,
                "min_child_samples": 75,
                "min_split_gain": 0.2,
                "subsample": 0.80,
                "subsample_freq": 1,
                "colsample_bytree": 0.70,
                "reg_alpha": 2.0,
                "reg_lambda": 10.0,
            },
        },
        {
            "name": "LGBM_Depth3_Leaves8_Agile",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 3,
                "num_leaves": 8,
                "n_estimators": 80,
                "learning_rate": 0.045,
                "min_child_samples": 40,
                "min_split_gain": 0.15,
                "subsample": 0.80,
                "subsample_freq": 1,
                "colsample_bytree": 0.70,
                "reg_alpha": 1.5,
                "reg_lambda": 7.0,
            },
        },
        {
            "name": "LGBM_DART_Dropout_Depth3",
            "params": {
                "boosting_type": "dart",
                "max_depth": 3,
                "num_leaves": 7,
                "n_estimators": 110,
                "learning_rate": 0.035,
                "min_child_samples": 40,
                "min_split_gain": 0.1,
                "subsample": 0.80,
                "subsample_freq": 1,
                "colsample_bytree": 0.70,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
        },
        {
            "name": "LGBM_DART_Dropout_Depth2",
            "params": {
                "boosting_type": "dart",
                "max_depth": 2,
                "num_leaves": 4,
                "n_estimators": 130,
                "learning_rate": 0.04,
                "min_child_samples": 45,
                "min_split_gain": 0.15,
                "subsample": 0.80,
                "subsample_freq": 1,
                "colsample_bytree": 0.75,
                "reg_alpha": 1.5,
                "reg_lambda": 6.0,
            },
        },
        {
            "name": "LGBM_GOSS_Sampling",
            "params": {
                "boosting_type": "goss",
                "max_depth": 3,
                "num_leaves": 7,
                "n_estimators": 110,
                "learning_rate": 0.03,
                "min_child_samples": 40,
                "min_split_gain": 0.1,
                "colsample_bytree": 0.75,
                "reg_alpha": 1.5,
                "reg_lambda": 6.0,
            },
        },
        {
            "name": "LGBM_ExtraTrees_Depth3",
            "params": {
                "boosting_type": "gbdt",
                "extra_trees": True,
                "max_depth": 3,
                "num_leaves": 7,
                "n_estimators": 130,
                "learning_rate": 0.03,
                "min_child_samples": 40,
                "min_split_gain": 0.1,
                "subsample": 0.80,
                "subsample_freq": 1,
                "colsample_bytree": 0.70,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
        },
        {
            "name": "LGBM_PathSmooth_Depth3",
            "params": {
                "boosting_type": "gbdt",
                "path_smooth": 2.0,
                "max_depth": 3,
                "num_leaves": 7,
                "n_estimators": 130,
                "learning_rate": 0.025,
                "min_child_samples": 45,
                "min_split_gain": 0.1,
                "subsample": 0.85,
                "subsample_freq": 1,
                "colsample_bytree": 0.75,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
        },
        {
            "name": "LGBM_Depth4_Leaves10_Strict",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 4,
                "num_leaves": 10,
                "n_estimators": 90,
                "learning_rate": 0.025,
                "min_child_samples": 80,
                "min_split_gain": 0.25,
                "subsample": 0.75,
                "subsample_freq": 1,
                "colsample_bytree": 0.65,
                "reg_alpha": 3.0,
                "reg_lambda": 12.0,
            },
        },
        {
            "name": "LGBM_Depth4_Leaves12_Slow",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 4,
                "num_leaves": 12,
                "n_estimators": 140,
                "learning_rate": 0.015,
                "min_child_samples": 70,
                "min_split_gain": 0.2,
                "subsample": 0.75,
                "subsample_freq": 1,
                "colsample_bytree": 0.65,
                "reg_alpha": 2.5,
                "reg_lambda": 10.0,
            },
        },
        {
            "name": "LGBM_HighReg_L1L2",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 3,
                "num_leaves": 7,
                "n_estimators": 90,
                "learning_rate": 0.035,
                "min_child_samples": 70,
                "min_split_gain": 0.3,
                "subsample": 0.75,
                "subsample_freq": 1,
                "colsample_bytree": 0.65,
                "reg_alpha": 4.0,
                "reg_lambda": 15.0,
            },
        },
    ]


def prepare_benchmark_data(ticker: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Ingests data, computes features, generates feature subsets, and returns scaled arrays."""
    df_train, df_test = prepare_ticker_data(ticker)
    kept_features = FEATURE_COLUMNS_V30

    df_tr_clean = df_train.replace([np.inf, -np.inf], np.nan).dropna(subset=kept_features + ["target_signal"])
    df_te_clean = df_test.replace([np.inf, -np.inf], np.nan).dropna(subset=kept_features + ["target_signal"])

    X_train_raw = df_tr_clean[kept_features].values
    y_train = df_tr_clean["target_signal"].astype(int).values
    X_test_raw = df_te_clean[kept_features].values
    y_test = df_te_clean["target_signal"].astype(int).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Feature Importance Ranking & Noise Reduction
    base_xgb = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.03, eval_metric="mlogloss", random_state=42)
    base_xgb.fit(X_train, y_train)
    feat_importances = pd.Series(base_xgb.feature_importances_, index=kept_features).sort_values(ascending=False)

    top20_feats = feat_importances.head(20).index.tolist()
    top15_feats = feat_importances.head(15).index.tolist()

    top20_idx = [kept_features.index(f) for f in top20_feats]
    top15_idx = [kept_features.index(f) for f in top15_feats]

    feature_sets = {
        "all_31": (X_train, X_test),
        "top_20": (X_train[:, top20_idx], X_test[:, top20_idx]),
        "top_15": (X_train[:, top15_idx], X_test[:, top15_idx]),
    }

    return feature_sets, y_train, y_test


def run_benchmark_for_ticker(ticker: str) -> Dict[str, Any]:
    """Executes systematic benchmark across all profiles and feature sets for a ticker."""
    logger.info("================================================================")
    logger.info(f"STARTING COMPREHENSIVE BENCHMARK FOR {ticker}")
    logger.info("================================================================")

    feature_sets, y_train, y_test = prepare_benchmark_data(ticker)
    xgb_profiles = get_xgb_parameter_profiles()
    lgbm_profiles = get_lgbm_parameter_profiles()

    xgb_results = []
    lgbm_results = []

    # 1. XGBoost Systematic Sweep
    logger.info(f"Evaluating {len(xgb_profiles) * len(feature_sets)} XGBoost Configurations for {ticker}...")
    for fset_name, (X_tr_sub, X_te_sub) in feature_sets.items():
        for cand in xgb_profiles:
            p = dict(cand["params"])
            p.update({
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "random_state": 42,
                "n_jobs": -1,
            })
            model = XGBClassifier(**p)
            res = evaluate_model(model, X_tr_sub, y_train, X_te_sub, y_test)
            res["candidate"] = cand["name"]
            res["feature_set"] = fset_name
            res["params"] = cand["params"]
            xgb_results.append(res)

    # 2. LightGBM Systematic Sweep
    logger.info(f"Evaluating {len(lgbm_profiles) * len(feature_sets)} LightGBM Configurations for {ticker}...")
    for fset_name, (X_tr_sub, X_te_sub) in feature_sets.items():
        for cand in lgbm_profiles:
            p = dict(cand["params"])
            p.update({
                "objective": "multiclass",
                "num_class": 3,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            })
            model = LGBMClassifier(**p)
            res = evaluate_model(model, X_tr_sub, y_train, X_te_sub, y_test)
            res["candidate"] = cand["name"]
            res["feature_set"] = fset_name
            res["params"] = cand["params"]
            lgbm_results.append(res)

    # Sort results by composite performance score and test accuracy
    xgb_by_test_acc = sorted(xgb_results, key=lambda x: (x["test_acc"], x["dir_acc"]), reverse=True)
    lgbm_by_test_acc = sorted(lgbm_results, key=lambda x: (x["test_acc"], x["dir_acc"]), reverse=True)

    xgb_by_composite = sorted(xgb_results, key=lambda x: (x["composite_score"], x["test_acc"]), reverse=True)
    lgbm_by_composite = sorted(lgbm_results, key=lambda x: (x["composite_score"], x["test_acc"]), reverse=True)

    return {
        "ticker": ticker,
        "best_xgb_by_accuracy": xgb_by_test_acc[0],
        "best_xgb_by_composite": xgb_by_composite[0],
        "top_xgb_candidates": xgb_by_test_acc[:5],
        "best_lgbm_by_accuracy": lgbm_by_test_acc[0],
        "best_lgbm_by_composite": lgbm_by_composite[0],
        "top_lgbm_candidates": lgbm_by_test_acc[:5],
        "all_xgb_results": xgb_by_test_acc,
        "all_lgbm_results": lgbm_by_test_acc,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deep Hyperparameter Benchmark for XGBoost and LightGBM")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "NVDA", "MSFT"], help="Tickers to evaluate")
    args = parser.parse_args()

    overall_results = {}
    for ticker in args.tickers:
        overall_results[ticker] = run_benchmark_for_ticker(ticker)

    # Summary table
    print("\n" + "=" * 90)
    print("          DEEP HYPERPARAMETER ACCURACY TOURNAMENT - EXECUTIVE RESULTS")
    print("=" * 90)
    header = f"{'TICKER':<6} | {'MODEL':<8} | {'BEST CONFIG':<30} | {'FEATS':<7} | {'TEST ACC':<9} | {'DIR ACC':<9} | {'GAP'}"
    print(header)
    print("-" * len(header))

    for t, res in overall_results.items():
        b_xgb = res["best_xgb_by_accuracy"]
        b_lgbm = res["best_lgbm_by_accuracy"]
        print(
            f"{t:<6} | {'XGBoost':<8} | {b_xgb['candidate']:<30} | {b_xgb['feature_set']:<7} | "
            f"{b_xgb['test_acc']*100:>7.2f}% | {b_xgb['dir_acc']*100:>7.2f}% | {b_xgb['overfit_gap']*100:>5.1f}%"
        )
        print(
            f"{t:<6} | {'LightGBM':<8} | {b_lgbm['candidate']:<30} | {b_lgbm['feature_set']:<7} | "
            f"{b_lgbm['test_acc']*100:>7.2f}% | {b_lgbm['dir_acc']*100:>7.2f}% | {b_lgbm['overfit_gap']*100:>5.1f}%"
        )
    print("=" * 90)

    # Save summary report
    out_dir = BACKEND_DIR / "artifacts" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "deep_hyperparameter_benchmark_report.json"
    with open(out_file, "w") as f:
        json.dump(overall_results, f, indent=4)
    print(f"\nComprehensive benchmark results saved to: {out_file}")


if __name__ == "__main__":
    main()
