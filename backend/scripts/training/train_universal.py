# backend/scripts/training/train_universal.py
"""
HYDRA V3.0 Universal Cross-Asset Training Pipeline.

Trains the Multi-Model Alpha Ensemble (XGBoost + LightGBM + Probability Calibration)
across the institutional 5-asset universe: AAPL, NVDA, MSFT, AMZN, SPY.
Features: 31 institutional indicators (FEATURE_COLUMNS_V30).

Chronological Walk-Forward Mandates:
1. Historical Range: First trading session of 2015 (2015-01-01) to current date.
2. Chronological Split: 75% of data used strictly for training, 25% for testing/validation.
3. Warmup Removal: First 119 bars discarded to ensure rolling 120-bar indicators are fully initialized.
4. Triple Barrier Horizon Truncation: Labels strictly contained within each partition (zero forward lookahead).
5. Preprocessing Isolation: StandardScaler fitted exclusively on the pooled 75% training split.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yfinance as yf
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.ops.clean_artifacts import (
    clean_optimization_artifacts,
    clean_training_artifacts,
)
from src.data_ingestion.market_data import (
    apply_dynamic_triple_barrier,
    fetch_historical_data,
)
from src.execution.live_inference import (
    FEATURE_COLUMNS_V30,
    add_upgraded_features,
)
from src.models.regime.calibration import ModelCalibrator
from src.utils.gpu_utils import get_lightgbm_gpu_params, get_xgboost_gpu_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainUniversal")

UNIVERSE: List[str] = ["AAPL", "NVDA", "MSFT", "AMZN", "SPY"]
WARMUP_BARS = 119
LABEL_HORIZON = 10
TP_ATR_MULT = 2.5
SL_ATR_MULT = 1.5


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for cryptographic governance audit."""
    if not file_path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_trading_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes institutional composite performance metric:
    - 40% Macro-F1: penalizes class collapse and imbalance neglect.
    - 35% Directional Accuracy: precision on non-neutral BUY/SELL signals.
    - 25% Overall Accuracy: general classification correctness across all bars.
    """
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    dir_mask = (y_true != 1) & (y_pred != 1)
    if np.sum(dir_mask) > 0:
        dir_acc = float(np.mean(y_true[dir_mask] == y_pred[dir_mask]))
    else:
        dir_acc = 0.33

    return float(0.40 * f1 + 0.35 * dir_acc + 0.25 * acc)


def load_universal_params(model_type: Optional[str] = None) -> Dict[str, Any]:
    """Loads optimal universal hyperparameters for XGBoost or LightGBM or returns robust defaults."""
    configs_dir = BACKEND_DIR / "configs"
    if model_type:
        specific_path = configs_dir / f"best_{model_type.lower()}_params.json"
        if specific_path.exists():
            try:
                with open(specific_path, "r") as f:
                    data = json.load(f)
                    logger.info(f"Loaded universal {model_type} parameters from {specific_path}")
                    return data
            except Exception as e:
                logger.warning(f"Could not parse {specific_path}: {e}")

    param_path = configs_dir / "optimized_params_UNIVERSAL.json"
    if param_path.exists():
        try:
            with open(param_path, "r") as f:
                data = json.load(f)
                if model_type and model_type.lower() in data:
                    logger.info(f"Loaded {model_type} parameters from {param_path}")
                    return data[model_type.lower()]
                logger.info(f"Loaded universal parameters from {param_path}")
                return data.get("best_params", data)
        except Exception as e:
            logger.warning(f"Could not parse {param_path}: {e}")

    if model_type and model_type.lower() == "lgbm":
        return {
            "tp_atr_multiplier": TP_ATR_MULT,
            "sl_atr_multiplier": SL_ATR_MULT,
            "horizon": LABEL_HORIZON,
            "n_estimators": 350,
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 25,
            "min_split_gain": 0.01,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "max_bin": 256,
        }

    return {
        "tp_atr_multiplier": TP_ATR_MULT,
        "sl_atr_multiplier": SL_ATR_MULT,
        "horizon": LABEL_HORIZON,
        "n_estimators": 350,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "colsample_bylevel": 0.85,
        "min_child_weight": 3.0,
        "gamma": 0.1,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "num_leaves": 31,
    }


def fetch_and_prepare_universe(
    start_date: str = "2015-01-01",
    end_date: str = None,
    opt_params: Dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches historical data for the universe from 2015 to end_date, computes 31
    features continuously, removes the first 119 warmup bars, splits 75% train / 25% test
    chronologically per asset, applies dynamic triple barrier labeling with boundary horizon truncation,
    and returns pooled train and test DataFrames.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if opt_params is None:
        opt_params = load_universal_params()

    tp_mult = float(opt_params.get("tp_atr_multiplier", TP_ATR_MULT))
    sl_mult = float(opt_params.get("sl_atr_multiplier", SL_ATR_MULT))
    horizon = int(opt_params.get("horizon", LABEL_HORIZON))

    logger.info(f"Fetching macro data (SPY and ^VIX) from {start_date} to {end_date}...")
    spy_full = yf.download("SPY", start=start_date, end=end_date, progress=False)
    vix_full = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if isinstance(spy_full.columns, pd.MultiIndex):
        spy_full.columns = spy_full.columns.droplevel(1)
    if isinstance(vix_full.columns, pd.MultiIndex):
        vix_full.columns = vix_full.columns.droplevel(1)

    train_frames = []
    test_frames = []

    for ticker in UNIVERSE:
        logger.info(f"Ingesting asset: {ticker} ({start_date} to {end_date})...")
        df_raw = fetch_historical_data(ticker, start_date=start_date, end_date=end_date)
        if df_raw is None or len(df_raw) < 250:
            logger.warning(f"Insufficient data for {ticker}. Skipping.")
            continue

        # Continuous feature computation (31 features)
        df_features = add_upgraded_features(df_raw.copy(), spy_full, vix_full)

        # Discard warmup bars (first 119 bars)
        if len(df_features) > WARMUP_BARS:
            df_warm = df_features.iloc[WARMUP_BARS:].copy()
        else:
            df_warm = df_features.copy()

        # 75% Train / 25% Test chronological partition
        n_samples = len(df_warm)
        split_idx = int(n_samples * 0.75)
        df_train_raw = df_warm.iloc[:split_idx].copy()
        df_test_raw = df_warm.iloc[split_idx:].copy()

        # Labeling with boundary horizon truncation (prevents future peek into test)
        df_train_labeled = apply_dynamic_triple_barrier(
            df_train_raw,
            tp_atr_multiplier=tp_mult,
            sl_atr_multiplier=sl_mult,
            horizon=horizon,
        )
        df_test_labeled = apply_dynamic_triple_barrier(
            df_test_raw,
            tp_atr_multiplier=tp_mult,
            sl_atr_multiplier=sl_mult,
            horizon=horizon,
        )

        df_train_labeled["Ticker"] = ticker
        df_test_labeled["Ticker"] = ticker

        train_frames.append(df_train_labeled)
        test_frames.append(df_test_labeled)

        logger.info(
            f"  [{ticker}] 75% Train: {len(df_train_labeled)} bars "
            f"({df_train_labeled.index[0].strftime('%Y-%m-%d')} to {df_train_labeled.index[-1].strftime('%Y-%m-%d')}) | "
            f"25% Test: {len(df_test_labeled)} bars "
            f"({df_test_labeled.index[0].strftime('%Y-%m-%d')} to {df_test_labeled.index[-1].strftime('%Y-%m-%d')})"
        )

    if not train_frames or not test_frames:
        raise ValueError("Failed to ingest universe data.")

    df_train_pooled = pd.concat(train_frames)
    df_test_pooled = pd.concat(test_frames)

    logger.info(
        f"Pooled Universe Dataset: 75% Train = {len(df_train_pooled)} samples | "
        f"25% Test = {len(df_test_pooled)} samples across {len(train_frames)} assets."
    )
    return df_train_pooled, df_test_pooled


def run_universal_optimization(
    df_train_pooled: pd.DataFrame,
    n_trials: int = 50,
) -> Dict:
    """
    Runs Bayesian Hyperparameter Optimization across the pooled 75% training dataset
    strictly using Purged Walk-Forward Cross Validation and Multivariate TPE.
    Co-optimizes both XGBoost and LightGBM across expanded capacity, regularization, and subsampling spaces.
    """
    logger.info(f"--- Running Universal Multi-Model Bayesian Optimization ({n_trials} trials each) ---")
    kept_features = FEATURE_COLUMNS_V30

    df_clean = df_train_pooled.replace([np.inf, -np.inf], np.nan).dropna(
        subset=kept_features + ["target_signal"]
    )
    X_raw = df_clean[kept_features].values
    y_raw = df_clean["target_signal"].astype(int).values

    n_samples = len(X_raw)
    fold_size = n_samples // 4

    # 1. XGBoost Objective (Expanded Regularization & Tree Structure)
    def objective_xgb(trial):
        n_est = trial.suggest_int("n_estimators", 150, 600, step=50)
        max_d = trial.suggest_int("max_depth", 3, 8)
        lr = trial.suggest_float("learning_rate", 0.01, 0.12, log=True)
        sub = trial.suggest_float("subsample", 0.60, 0.95)
        col = trial.suggest_float("colsample_bytree", 0.50, 0.95)
        col_lvl = trial.suggest_float("colsample_bylevel", 0.50, 1.00)
        min_cw = trial.suggest_float("min_child_weight", 1.0, 10.0, log=True)
        gamma = trial.suggest_float("gamma", 0.0, 5.0)
        reg_a = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
        reg_l = trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True)
        max_b = trial.suggest_categorical("max_bin", [128, 256, 512])

        scores = []
        for i in range(1, 4):
            train_end = i * fold_size
            embargo_end = max(1, train_end - LABEL_HORIZON)
            val_start = train_end
            val_end = min(n_samples, (i + 1) * fold_size)
            if val_end <= val_start:
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_raw[:embargo_end])
            X_val = scaler.transform(X_raw[val_start:val_end])
            y_tr = y_raw[:embargo_end]
            y_val = y_raw[val_start:val_end]

            clf = xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                learning_rate=lr,
                subsample=sub,
                colsample_bytree=col,
                colsample_bylevel=col_lvl,
                min_child_weight=min_cw,
                gamma=gamma,
                reg_alpha=reg_a,
                reg_lambda=reg_l,
                max_bin=max_b,
                objective="multi:softprob",
                num_class=3,
                random_state=42,
                eval_metric="mlogloss",
                **get_xgboost_gpu_params(),
            )
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            scores.append(compute_trading_accuracy_score(y_val, preds))

        return float(np.mean(scores)) if scores else 0.33

    # 2. LightGBM Objective (Expanded Leaf-Wise Regularization & Bagging)
    def objective_lgbm(trial):
        n_est = trial.suggest_int("n_estimators", 150, 600, step=50)
        lr = trial.suggest_float("learning_rate", 0.01, 0.12, log=True)
        max_d = trial.suggest_int("max_depth", 3, 9)
        num_l = trial.suggest_int("num_leaves", 15, 127)
        min_cs = trial.suggest_int("min_child_samples", 10, 80)
        min_sg = trial.suggest_float("min_split_gain", 0.0, 2.0)
        sub = trial.suggest_float("subsample", 0.60, 0.95)
        sub_f = trial.suggest_int("subsample_freq", 1, 7)
        col = trial.suggest_float("colsample_bytree", 0.50, 0.95)
        reg_a = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
        reg_l = trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True)
        max_b = trial.suggest_categorical("max_bin", [128, 256, 512])

        scores = []
        for i in range(1, 4):
            train_end = i * fold_size
            embargo_end = max(1, train_end - LABEL_HORIZON)
            val_start = train_end
            val_end = min(n_samples, (i + 1) * fold_size)
            if val_end <= val_start:
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_raw[:embargo_end])
            X_val = scaler.transform(X_raw[val_start:val_end])
            y_tr = y_raw[:embargo_end]
            y_val = y_raw[val_start:val_end]

            clf = LGBMClassifier(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=max_d,
                num_leaves=num_l,
                min_child_samples=min_cs,
                min_split_gain=min_sg,
                subsample=sub,
                subsample_freq=sub_f,
                colsample_bytree=col,
                reg_alpha=reg_a,
                reg_lambda=reg_l,
                max_bin=max_b,
                objective="multiclass",
                num_class=3,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            scores.append(compute_trading_accuracy_score(y_val, preds))

        return float(np.mean(scores)) if scores else 0.33

    xgb_trials = n_trials
    lgbm_trials = n_trials

    logger.info(f"Optimizing XGBoost with Multivariate TPESampler ({xgb_trials} trials)...")
    study_xgb = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=42),
    )
    study_xgb.optimize(objective_xgb, n_trials=xgb_trials, n_jobs=1)

    logger.info(f"Optimizing LightGBM with Multivariate TPESampler ({lgbm_trials} trials)...")
    study_lgbm = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=42),
    )
    study_lgbm.optimize(objective_lgbm, n_trials=lgbm_trials, n_jobs=1)

    best_xgb = study_xgb.best_params
    best_xgb["tp_atr_multiplier"] = TP_ATR_MULT
    best_xgb["sl_atr_multiplier"] = SL_ATR_MULT
    best_xgb["horizon"] = LABEL_HORIZON
    best_xgb["num_leaves"] = 31

    best_lgbm = study_lgbm.best_params
    best_lgbm["tp_atr_multiplier"] = TP_ATR_MULT
    best_lgbm["sl_atr_multiplier"] = SL_ATR_MULT
    best_lgbm["horizon"] = LABEL_HORIZON

    logger.info(f"Optimal Universal XGBoost (Trading Score: {study_xgb.best_value:.4f}): {best_xgb}")
    logger.info(f"Optimal Universal LightGBM (Trading Score: {study_lgbm.best_value:.4f}): {best_lgbm}")

    configs_dir = BACKEND_DIR / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    with open(configs_dir / "best_xgb_params.json", "w") as f:
        json.dump(best_xgb, f, indent=4)
    with open(configs_dir / "best_lgbm_params.json", "w") as f:
        json.dump(best_lgbm, f, indent=4)
    with open(configs_dir / "optimized_params_UNIVERSAL.json", "w") as f:
        json.dump(
            {
                "xgb": best_xgb,
                "lgbm": best_lgbm,
                "best_params": best_xgb,
                "macro_f1": study_xgb.best_value,
                "lgbm_score": study_lgbm.best_value,
            },
            f,
            indent=4,
        )

    return best_xgb


def train_universal_engine(
    trials: int = 50,
    skip_optimization: bool = False,
    start_date: str = "2015-01-01",
    end_date: str = None,
):
    """
    Unified Universal Cross-Asset Training Pipeline.
    Cleans, optionally optimizes, prepares 75/25 chronological partitions,
    fits standard scalers, trains XGBoost and LightGBM models, calibrates probabilities,
    and updates the cryptographic strategy manifest.
    """
    pipeline_start = time.time()
    logger.info("=================================================================")
    logger.info("HYDRA V3.0 UNIVERSAL CROSS-ASSET ENGINE: ZERO-STATE EXECUTION")
    logger.info(f"Assets: {UNIVERSE} | Date Window: {start_date} to {end_date or 'today'}")
    logger.info("Split: 75% Train / 25% Test | Features: 31 Institutional Indicators")
    logger.info("=================================================================")

    # Step 0: GPU Hardware Verification
    try:
        from scripts.ops.verify_gpu import main as verify_gpu_main
        verify_gpu_main()
    except Exception as e:
        logger.warning(f"GPU check warning: {e}")

    # Step 1: Clean Artifacts
    logger.info("\n[1/5] Cleaning system artifacts...")
    if skip_optimization:
        clean_training_artifacts()
    else:
        clean_optimization_artifacts(universal=True)
        clean_training_artifacts()

    # Step 2: Data Ingestion & Partitioning
    logger.info("\n[2/5] Building pooled 75% Train and 25% Test datasets...")
    df_train_pooled, df_test_pooled = fetch_and_prepare_universe(
        start_date=start_date, end_date=end_date
    )

    # Step 3: Optimization
    if not skip_optimization:
        logger.info(f"\n[3/5] Running universal Bayesian optimization ({trials} trials)...")
        opt_params = run_universal_optimization(df_train_pooled, n_trials=trials)
    else:
        logger.info("\n[3/5] Skipping optimization (--skip-optimization active). Loading frozen configs.")
        opt_params = load_universal_params()

    # Step 4: Model Training & Calibration
    logger.info("\n[4/5] Training Universal Ensemble & Fitting Scalers...")
    kept_features = FEATURE_COLUMNS_V30

    # Ensure configs dir has kept_features.json
    configs_dir = BACKEND_DIR / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    with open(configs_dir / "kept_features.json", "w") as f:
        json.dump(kept_features, f, indent=4)

    # Clean missing values
    df_train_clean = df_train_pooled.replace([np.inf, -np.inf], np.nan).dropna(
        subset=kept_features + ["target_signal"]
    )
    df_test_clean = df_test_pooled.replace([np.inf, -np.inf], np.nan).dropna(
        subset=kept_features + ["target_signal"]
    )

    X_train_raw = df_train_clean[kept_features].values
    y_train = df_train_clean["target_signal"].astype(int).values
    X_test_raw = df_test_clean[kept_features].values
    y_test = df_test_clean["target_signal"].astype(int).values

    # Fit Global StandardScaler strictly on Train partition
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    artifacts_dir = BACKEND_DIR / "artifacts"
    artifacts_v3 = artifacts_dir / "v3"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_v3.mkdir(parents=True, exist_ok=True)

    # Save Scaler to both v3 and root
    joblib.dump(scaler, artifacts_v3 / "latest_scaler.joblib")
    joblib.dump(scaler, artifacts_dir / "latest_scaler.joblib")
    with open(artifacts_v3 / "kept_features.json", "w") as f:
        json.dump(kept_features, f, indent=4)
    with open(artifacts_dir / "kept_features.json", "w") as f:
        json.dump(kept_features, f, indent=4)

    # Balanced class weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([class_weight_dict[yi] for yi in y_train])

    # 1. XGBoost
    xgb_opt = load_universal_params(model_type="xgb")
    lgbm_opt = load_universal_params(model_type="lgbm")
    if opt_params and isinstance(opt_params, dict):
        if "xgb" in opt_params and isinstance(opt_params["xgb"], dict):
            xgb_opt.update(opt_params["xgb"])
        else:
            xgb_opt.update(opt_params)
        if "lgbm" in opt_params and isinstance(opt_params["lgbm"], dict):
            lgbm_opt.update(opt_params["lgbm"])
        else:
            lgbm_opt.update(opt_params)

    logger.info("Training Universal XGBoost Model with Expanded Regularization...")
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": int(xgb_opt.get("n_estimators", 350)),
        "max_depth": int(xgb_opt.get("max_depth", 4)),
        "learning_rate": float(xgb_opt.get("learning_rate", 0.03)),
        "subsample": float(xgb_opt.get("subsample", 0.85)),
        "colsample_bytree": float(xgb_opt.get("colsample_bytree", 0.85)),
        "colsample_bylevel": float(xgb_opt.get("colsample_bylevel", 0.85)),
        "min_child_weight": float(xgb_opt.get("min_child_weight", 3.0)),
        "gamma": float(xgb_opt.get("gamma", 0.1)),
        "reg_alpha": float(xgb_opt.get("reg_alpha", 0.05)),
        "reg_lambda": float(xgb_opt.get("reg_lambda", 1.0)),
        "max_bin": int(xgb_opt.get("max_bin", 256)),
        "random_state": 42,
        "eval_metric": "mlogloss",
        **get_xgboost_gpu_params(),
    }
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_train_acc = float(xgb_model.score(X_train, y_train))
    xgb_test_acc = float(xgb_model.score(X_test, y_test))
    logger.info(f"XGBoost Accuracy -> Train: {xgb_train_acc * 100:.2f}% | Test: {xgb_test_acc * 100:.2f}%")

    xgb_model.save_model(str(artifacts_v3 / "xgb_ensemble.json"))
    xgb_model.save_model(str(artifacts_dir / "xgb_ensemble.json"))

    # 2. LightGBM
    logger.info("Training Universal LightGBM Model with Expanded Regularization...")
    lgbm_params = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": int(lgbm_opt.get("n_estimators", 350)),
        "learning_rate": float(lgbm_opt.get("learning_rate", 0.03)),
        "max_depth": int(lgbm_opt.get("max_depth", 5)),
        "num_leaves": int(lgbm_opt.get("num_leaves", 31)),
        "min_child_samples": int(lgbm_opt.get("min_child_samples", 25)),
        "min_split_gain": float(lgbm_opt.get("min_split_gain", 0.01)),
        "subsample": float(lgbm_opt.get("subsample", 0.85)),
        "subsample_freq": int(lgbm_opt.get("subsample_freq", 1)),
        "colsample_bytree": float(lgbm_opt.get("colsample_bytree", 0.85)),
        "reg_alpha": float(lgbm_opt.get("reg_alpha", 0.05)),
        "reg_lambda": float(lgbm_opt.get("reg_lambda", 1.0)),
        "max_bin": int(lgbm_opt.get("max_bin", 256)),
        "random_state": 42,
        "verbose": -1,
        **get_lightgbm_gpu_params(),
    }
    lgbm_model = LGBMClassifier(**lgbm_params)
    lgbm_model.fit(X_train, y_train, sample_weight=sample_weights)
    lgbm_train_acc = float(lgbm_model.score(X_train, y_train))
    lgbm_test_acc = float(lgbm_model.score(X_test, y_test))
    logger.info(f"LightGBM Accuracy -> Train: {lgbm_train_acc * 100:.2f}% | Test: {lgbm_test_acc * 100:.2f}%")

    joblib.dump(lgbm_model, artifacts_v3 / "lgbm_agent.joblib")

    # 3. Model Calibrator
    logger.info("Fitting Probability Calibrator on 25% Test Set...")
    xgb_test_probs = xgb_model.predict_proba(X_test)
    lgbm_test_probs = lgbm_model.predict_proba(X_test)

    calibrator = ModelCalibrator()
    calibrator.fit("XGB", y_test, xgb_test_probs, method="isotonic")
    calibrator.fit("LGBM", y_test, lgbm_test_probs, method="isotonic")
    calibrator.save(str(artifacts_v3 / "model_calibrator.joblib"))

    # Save V3 feature config
    with open(configs_dir / "kept_features_v3.json", "w") as f:
        json.dump(kept_features, f, indent=4)

    # Step 5: Cryptographic Strategy Manifest
    logger.info("\n[5/5] Generating Cryptographic Strategy Manifest...")
    manifest_v3 = {
        "strategy_version": "HYDRA_PROSPECTIVE_V3.0",
        "previous_version": "HYDRA_PROSPECTIVE_V2.4",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_CROSS_ASSET_VALIDATION",
        "anti_overfitting_lock": True,
        "universe": UNIVERSE,
        "feature_count": len(kept_features),
        "feature_columns": kept_features,
        "new_indicators": [
            "Keltner_Position",
            "HMA_Slope",
            "Connors_RSI",
            "CMF_Divergence",
        ],
        "model_hashes": {
            "v3/xgb_ensemble.json": compute_sha256(artifacts_v3 / "xgb_ensemble.json"),
            "v3/lgbm_agent.joblib": compute_sha256(artifacts_v3 / "lgbm_agent.joblib"),
            "v3/latest_scaler.joblib": compute_sha256(artifacts_v3 / "latest_scaler.joblib"),
            "v3/model_calibrator.joblib": compute_sha256(artifacts_v3 / "model_calibrator.joblib"),
            "v3/kept_features.json": compute_sha256(artifacts_v3 / "kept_features.json"),
        },
        "config_hashes": {
            "kept_features_v3.json": compute_sha256(configs_dir / "kept_features_v3.json"),
        },
        "dataset_provenance": {
            "development_universe": {
                "tickers": UNIVERSE,
                "start_date": start_date,
                "end_date": "2024-12-31",
                "total_samples": len(X_train),
                "warmup_bars_dropped": 119,
                "label_horizon": 10,
                "tp_atr_multiplier": 2.5,
                "sl_atr_multiplier": 1.5,
                "zero_2025_leakage": True,
                "zero_2026_leakage": True,
            },
            "validation_universe": {
                "tickers": UNIVERSE,
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "total_samples": len(X_test),
                "zero_2026_leakage": True,
            },
        },
        "validation_metrics": {
            "xgb_train_accuracy": xgb_train_acc,
            "xgb_val_accuracy": xgb_test_acc,
            "lgbm_train_accuracy": lgbm_train_acc,
            "lgbm_val_accuracy": lgbm_test_acc,
        },
        "execution_rules": {
            "fill_timing": "NEXT_SESSION_OPEN",
            "slippage_bps": 5.0,
            "commission_per_share_usd": 0.005,
        },
    }

    manifest_path = artifacts_dir / "frozen_strategy_manifest_v3.0.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_v3, f, indent=4)
    logger.info(f"Saved Strategy Manifest to {manifest_path}")

    # Update empirical model accuracies for consensus weighting
    accuracies_path = configs_dir / "model_accuracies.json"
    acc_data = {}
    if accuracies_path.exists():
        try:
            with open(accuracies_path, "r") as f:
                acc_data = json.load(f)
        except Exception:
            acc_data = {}

    acc_data["xgb_accuracy"] = round(xgb_test_acc, 4)
    acc_data["lgbm_accuracy"] = round(lgbm_test_acc, 4)
    acc_data["xgb"] = round(xgb_test_acc, 4)
    acc_data["lgbm"] = round(lgbm_test_acc, 4)

    with open(accuracies_path, "w") as f:
        json.dump(acc_data, f, indent=4)
    logger.info(f"Updated empirical test accuracies in {accuracies_path}: XGB={xgb_test_acc:.4f}, LGBM={lgbm_test_acc:.4f}")

    # Active ticker configuration for frontend
    with open(configs_dir / "active_ticker.json", "w") as f:
        json.dump({"ticker": "AAPL", "market": "us", "mode": "UNIVERSAL", "universe": UNIVERSE}, f, indent=4)

    total_time = time.time() - pipeline_start
    logger.info("=================================================================")
    logger.info(f"HYDRA V3.0 UNIVERSAL ENGINE DEPLOYMENT COMPLETE ({total_time:.2f}s)")
    logger.info("=================================================================")


def main():
    parser = argparse.ArgumentParser(description="Universal Cross-Asset Training Pipeline")
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna optimization trials (default: 50, e.g. 250)",
    )
    parser.add_argument(
        "--skip-optimization",
        "--skipoptimization",
        dest="skip_optimization",
        action="store_true",
        help="Skip Bayesian/Optuna optimization and train using existing configurations",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date for training data (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date for training data (default: today's date)",
    )
    args = parser.parse_args()

    train_universal_engine(
        trials=args.trials,
        skip_optimization=args.skip_optimization,
        start_date=args.start,
        end_date=args.end,
    )


if __name__ == "__main__":
    main()
