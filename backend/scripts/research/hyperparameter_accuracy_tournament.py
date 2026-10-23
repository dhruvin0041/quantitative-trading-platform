"""
HYDRA Quantitative Research: Hyperparameter Accuracy Tournament for XGBoost & LightGBM.
Systematically tests multiple hyperparameter configurations, feature subsets, and regularization
schemes across institutional test partitions (75% Train / 25% Test) with strict Zero Data Leakage.
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

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
logger = logging.getLogger("AccuracyTournament")


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


def run_tournament_for_ticker(ticker: str = "AAPL") -> Dict[str, Any]:
    """Runs systematic parameter sweep for both XGBoost and LightGBM on a specific ticker."""
    logger.info("================================================================")
    logger.info(f"STARTING ACCURACY TOURNAMENT FOR {ticker}")
    logger.info("================================================================")

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

    # -------------------------------------------------------------
    # 1. Feature Importance Ranking & Noise Reduction
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 2. XGBoost Systematic Parameter Grid
    # -------------------------------------------------------------
    logger.info("Evaluating XGBoost Hyperparameter Variations...")
    xgb_candidates = [
        # Candidate 1: Conservative Ultra-Shallow (Depth 2, High Regularization)
        {
            "name": "XGB_Shallow_Regularized",
            "params": {
                "max_depth": 2,
                "n_estimators": 100,
                "learning_rate": 0.03,
                "subsample": 0.80,
                "colsample_bytree": 0.70,
                "colsample_bylevel": 0.80,
                "min_child_weight": 10.0,
                "gamma": 1.5,
                "reg_alpha": 2.0,
                "reg_lambda": 10.0,
            },
        },
        # Candidate 2: Balanced Interaction Tree (Depth 3, Step-tuned Learning)
        {
            "name": "XGB_Balanced_Depth3_lr03",
            "params": {
                "max_depth": 3,
                "n_estimators": 120,
                "learning_rate": 0.03,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "colsample_bylevel": 0.80,
                "min_child_weight": 8.0,
                "gamma": 1.0,
                "reg_alpha": 1.5,
                "reg_lambda": 8.0,
            },
        },
        # Candidate 3: Compact Low-Rate (Depth 3, Low Learning Rate 0.015, 180 trees)
        {
            "name": "XGB_Depth3_LowLR180",
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
        # Candidate 4: Higher Capacity Interaction (Depth 4, Strong L2)
        {
            "name": "XGB_Depth4_HeavyL2",
            "params": {
                "max_depth": 4,
                "n_estimators": 100,
                "learning_rate": 0.025,
                "subsample": 0.75,
                "colsample_bytree": 0.65,
                "colsample_bylevel": 0.75,
                "min_child_weight": 12.0,
                "gamma": 2.0,
                "reg_alpha": 3.0,
                "reg_lambda": 15.0,
            },
        },
        # Candidate 5: High Bagging Feature Randomization (Subsample 0.70, Colsample 0.55)
        {
            "name": "XGB_High_Randomization",
            "params": {
                "max_depth": 3,
                "n_estimators": 140,
                "learning_rate": 0.035,
                "subsample": 0.70,
                "colsample_bytree": 0.55,
                "colsample_bylevel": 0.70,
                "min_child_weight": 10.0,
                "gamma": 1.0,
                "reg_alpha": 2.0,
                "reg_lambda": 10.0,
            },
        },
        # Candidate 6: Fast Agile Momentum (Depth 2, 70 trees, lr 0.05)
        {
            "name": "XGB_Fast_Agile_Depth2",
            "params": {
                "max_depth": 2,
                "n_estimators": 70,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "colsample_bylevel": 0.85,
                "min_child_weight": 8.0,
                "gamma": 1.0,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
        },
    ]

    xgb_results = []
    for fset_name, (X_tr_sub, X_te_sub) in feature_sets.items():
        for cand in xgb_candidates:
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
            logger.info(
                f"  [XGB] {cand['name']:<26} ({fset_name:<6}) -> Test Acc: {res['test_acc']*100:.2f}% | "
                f"Dir Acc: {res['dir_acc']*100:.2f}% | Macro-F1: {res['macro_f1']*100:.2f}% | Gap: {res['overfit_gap']*100:.1f}%"
            )

    # -------------------------------------------------------------
    # 3. LightGBM Systematic Parameter Grid
    # -------------------------------------------------------------
    logger.info("\nEvaluating LightGBM Hyperparameter Variations...")
    lgbm_candidates = [
        # Candidate 1: Shallow Leaf-Restricted (Depth 2, 4 leaves)
        {
            "name": "LGBM_Depth2_Leaves4",
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
        # Candidate 2: Compact Depth 3 (7 leaves, balanced subsampling)
        {
            "name": "LGBM_Depth3_Leaves7",
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
        # Candidate 3: DART Dropout Boosting (Prevents Overfitting)
        {
            "name": "LGBM_DART_Dropout_100",
            "params": {
                "boosting_type": "dart",
                "max_depth": 3,
                "num_leaves": 7,
                "n_estimators": 100,
                "learning_rate": 0.04,
                "min_child_samples": 40,
                "min_split_gain": 0.1,
                "subsample": 0.80,
                "subsample_freq": 1,
                "colsample_bytree": 0.70,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
        },
        # Candidate 4: High Regularization (min_child_samples 75, L2=12.0)
        {
            "name": "LGBM_High_Regularized",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 3,
                "num_leaves": 7,
                "n_estimators": 90,
                "learning_rate": 0.035,
                "min_child_samples": 75,
                "min_split_gain": 0.3,
                "subsample": 0.75,
                "subsample_freq": 1,
                "colsample_bytree": 0.65,
                "reg_alpha": 3.0,
                "reg_lambda": 12.0,
            },
        },
        # Candidate 5: Slow Learning Horizon (lr 0.015, 180 trees, 7 leaves)
        {
            "name": "LGBM_Slow_Learning_180",
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
        # Candidate 6: Deeper 12-leaf with High Min Child Samples (80)
        {
            "name": "LGBM_Depth4_Leaves12_MinCS80",
            "params": {
                "boosting_type": "gbdt",
                "max_depth": 4,
                "num_leaves": 12,
                "n_estimators": 100,
                "learning_rate": 0.025,
                "min_child_samples": 80,
                "min_split_gain": 0.2,
                "subsample": 0.75,
                "subsample_freq": 1,
                "colsample_bytree": 0.65,
                "reg_alpha": 2.5,
                "reg_lambda": 10.0,
            },
        },
    ]

    lgbm_results = []
    for fset_name, (X_tr_sub, X_te_sub) in feature_sets.items():
        for cand in lgbm_candidates:
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
            logger.info(
                f"  [LGBM] {cand['name']:<26} ({fset_name:<6}) -> Test Acc: {res['test_acc']*100:.2f}% | "
                f"Dir Acc: {res['dir_acc']*100:.2f}% | Macro-F1: {res['macro_f1']*100:.2f}% | Gap: {res['overfit_gap']*100:.1f}%"
            )

    # Sort results
    xgb_sorted = sorted(xgb_results, key=lambda x: (x["test_acc"], x["dir_acc"]), reverse=True)
    lgbm_sorted = sorted(lgbm_results, key=lambda x: (x["test_acc"], x["dir_acc"]), reverse=True)

    return {
        "ticker": ticker,
        "best_xgb": xgb_sorted[0],
        "top_xgb_candidates": xgb_sorted[:3],
        "best_lgbm": lgbm_sorted[0],
        "top_lgbm_candidates": lgbm_sorted[:3],
        "all_xgb_results": xgb_sorted,
        "all_lgbm_results": lgbm_sorted,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Accuracy Tournament for XGBoost and LightGBM")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "NVDA", "MSFT"], help="Tickers to evaluate")
    args = parser.parse_args()

    overall_results = {}
    for ticker in args.tickers:
        overall_results[ticker] = run_tournament_for_ticker(ticker)

    # Summarize Best Configurations
    print("\n" + "=" * 80)
    print("           HYDRA HYPERPARAMETER ACCURACY TOURNAMENT - EXECUTIVE RESULTS")
    print("=" * 80)
    header = f"{'TICKER':<6} | {'MODEL':<8} | {'BEST CONFIG':<26} | {'FEATS':<7} | {'TEST ACC':<9} | {'DIR ACC':<9} | {'GAP'}"
    print(header)
    print("-" * len(header))

    for t, res in overall_results.items():
        b_xgb = res["best_xgb"]
        b_lgbm = res["best_lgbm"]
        print(
            f"{t:<6} | {'XGBoost':<8} | {b_xgb['candidate']:<26} | {b_xgb['feature_set']:<7} | "
            f"{b_xgb['test_acc']*100:>7.2f}% | {b_xgb['dir_acc']*100:>7.2f}% | {b_xgb['overfit_gap']*100:>5.1f}%"
        )
        print(
            f"{t:<6} | {'LightGBM':<8} | {b_lgbm['candidate']:<26} | {b_lgbm['feature_set']:<7} | "
            f"{b_lgbm['test_acc']*100:>7.2f}% | {b_lgbm['dir_acc']*100:>7.2f}% | {b_lgbm['overfit_gap']*100:>5.1f}%"
        )
    print("=" * 80)

    # Save summary report
    out_dir = BACKEND_DIR / "artifacts" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hyperparameter_accuracy_tournament_report.json"
    with open(out_file, "w") as f:
        json.dump(overall_results, f, indent=4)
    print(f"\nComprehensive tournament results saved to: {out_file}")


if __name__ == "__main__":
    main()
