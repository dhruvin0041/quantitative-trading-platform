# backend/scripts/training/train.py
"""
HYDRA V3.0 Asset-Specific Training Pipeline.

Trains the Multi-Model Alpha Ensemble (XGBoost + LightGBM + Probability Calibration)
for a user-specified stock ticker.
Features: 31 institutional indicators (FEATURE_COLUMNS_V30).

Chronological Walk-Forward Mandates:
1. Historical Range: First trading session of 2015 (2015-01-01) to current date.
2. Chronological Split: 75% of data used strictly for training, 25% for testing/validation.
3. Warmup Removal: First 119 bars discarded to ensure rolling 120-bar indicators are fully initialized.
4. Triple Barrier Horizon Truncation: Labels strictly contained within each partition (zero forward lookahead).
5. Preprocessing Isolation: StandardScaler fitted exclusively on the 75% training split.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yfinance as yf
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
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
    get_sector_peer,
)
from src.execution.live_inference import (
    FEATURE_COLUMNS_V30,
    add_upgraded_features,
)
from src.models.regime.calibration import ModelCalibrator
from src.utils.gpu_utils import get_xgboost_gpu_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainSpecificTicker")

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


def load_ticker_params(ticker: str) -> Dict[str, float]:
    """Loads optimal hyperparameters for ticker or returns robust defaults."""
    param_path = BACKEND_DIR / "configs" / f"optimized_params_{ticker}.json"
    if param_path.exists():
        try:
            with open(param_path, "r") as f:
                data = json.load(f)
                logger.info(f"Loaded optimized parameters from {param_path}")
                return data.get("best_params", data)
        except Exception as e:
            logger.warning(f"Could not parse {param_path}: {e}")
    return {
        "tp_atr_multiplier": TP_ATR_MULT,
        "sl_atr_multiplier": SL_ATR_MULT,
        "horizon": LABEL_HORIZON,
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "num_leaves": 31,
    }


def prepare_ticker_data(
    ticker: str,
    start_date: str = "2015-01-01",
    end_date: str = None,
    opt_params: Dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches historical data for ticker, SPY, ^VIX, and sector peer.
    Computes 31 features continuously, removes first 119 warmup bars,
    partitions into 75% Train and 25% Test chronologically,
    and applies dynamic triple barrier labeling with boundary horizon truncation.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if opt_params is None:
        opt_params = load_ticker_params(ticker)

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

    logger.info(f"Ingesting asset: {ticker} ({start_date} to {end_date})...")
    df_raw = fetch_historical_data(ticker, start_date=start_date, end_date=end_date)
    if df_raw is None or len(df_raw) < 250:
        raise ValueError(f"Insufficient data for {ticker}. At least 250 bars required.")

    peer_ticker = get_sector_peer(ticker)
    peer_raw = fetch_historical_data(peer_ticker, start_date=start_date, end_date=end_date)

    # Compute 31 features continuously
    df_features = add_upgraded_features(df_raw.copy(), spy_full, vix_full)
    if peer_raw is not None and len(peer_raw) > 250:
        peer_features = add_upgraded_features(peer_raw.copy(), spy_full, vix_full)
        common_idx = df_features.index.intersection(peer_features.index)
        df_features = df_features.loc[common_idx].copy()

    # Discard warmup bars (first 119 bars for 120-bar rolling z-score stability)
    if len(df_features) > WARMUP_BARS:
        df_warm = df_features.iloc[WARMUP_BARS:].copy()
    else:
        df_warm = df_features.copy()

    # 75% Train / 25% Test chronological partition
    n_samples = len(df_warm)
    split_idx = int(n_samples * 0.75)
    df_train_raw = df_warm.iloc[:split_idx].copy()
    df_test_raw = df_warm.iloc[split_idx:].copy()

    # Labeling with boundary horizon truncation (zero forward lookahead into test)
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

    logger.info(
        f"  [{ticker}] 75% Train: {len(df_train_labeled)} bars "
        f"({df_train_labeled.index[0].strftime('%Y-%m-%d')} to {df_train_labeled.index[-1].strftime('%Y-%m-%d')}) | "
        f"25% Test: {len(df_test_labeled)} bars "
        f"({df_test_labeled.index[0].strftime('%Y-%m-%d')} to {df_test_labeled.index[-1].strftime('%Y-%m-%d')})"
    )

    return df_train_labeled, df_test_labeled


def run_ticker_optimization(
    ticker: str,
    df_train: pd.DataFrame,
    n_trials: int = 50,
) -> Dict:
    """
    Runs Bayesian Hyperparameter Optimization for ticker strictly within the 75% train partition
    using Purged Walk-Forward Cross Validation.
    """
    logger.info(f"--- Running Bayesian Optimization for {ticker} ({n_trials} trials) ---")
    kept_features = FEATURE_COLUMNS_V30

    df_clean = df_train.replace([np.inf, -np.inf], np.nan).dropna(
        subset=kept_features + ["target_signal"]
    )
    X_raw = df_clean[kept_features].values
    y_raw = df_clean["target_signal"].astype(int).values

    def objective(trial):
        n_est = trial.suggest_categorical("n_estimators", [150, 300, 500])
        max_d = trial.suggest_int("max_depth", 3, 7)
        lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        sub = trial.suggest_float("subsample", 0.6, 0.9)
        col = trial.suggest_float("colsample_bytree", 0.6, 0.9)

        # 3-Fold Chronological Purged Expanding Window CV on Train split
        n_samples = len(X_raw)
        fold_size = n_samples // 4
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
                objective="multi:softprob",
                num_class=3,
                random_state=42,
                eval_metric="mlogloss",
                **get_xgboost_gpu_params(),
            )
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            scores.append(f1_score(y_val, preds, average="macro", zero_division=0))

        return float(np.mean(scores)) if scores else 0.33

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = study.best_params
    best_params["tp_atr_multiplier"] = TP_ATR_MULT
    best_params["sl_atr_multiplier"] = SL_ATR_MULT
    best_params["horizon"] = LABEL_HORIZON
    best_params["num_leaves"] = 31

    logger.info(f"Optimal Parameters for {ticker} (Macro-F1: {study.best_value:.4f}): {best_params}")

    configs_dir = BACKEND_DIR / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    with open(configs_dir / f"optimized_params_{ticker}.json", "w") as f:
        json.dump({"best_params": best_params, "macro_f1": study.best_value}, f, indent=4)
    with open(configs_dir / "best_xgb_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    with open(configs_dir / "best_lgbm_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    return best_params


def train_ticker_pipeline(
    ticker: str = "AAPL",
    trials: int = 50,
    skip_optimization: bool = False,
    start_date: str = "2015-01-01",
    end_date: str = None,
):
    """
    Unified Single-Ticker Training Pipeline.
    Cleans artifacts, optionally optimizes, prepares 75/25 chronological partitions,
    fits standard scalers, trains XGBoost and LightGBM models, calibrates probabilities,
    and updates active ticker and strategy manifest.
    """
    ticker = ticker.upper()
    pipeline_start = time.time()
    logger.info("=================================================================")
    logger.info(f"HYDRA V3.0 SINGLE-TICKER ENGINE: ZERO-STATE EXECUTION [{ticker}]")
    logger.info(f"Date Window: {start_date} to {end_date or 'today'}")
    logger.info("Split: 75% Train / 25% Test | Features: 31 Institutional Indicators")
    logger.info("=================================================================")

    # Step 0: GPU Hardware Verification
    try:
        from scripts.ops.verify_gpu import main as verify_gpu_main
        verify_gpu_main()
    except Exception as e:
        logger.warning(f"GPU check warning: {e}")

    # Step 1: Clean Artifacts
    logger.info(f"\n[1/5] Cleaning system artifacts for {ticker}...")
    if skip_optimization:
        clean_training_artifacts()
    else:
        clean_optimization_artifacts(ticker=ticker)
        clean_training_artifacts()

    # Step 2: Data Ingestion & Partitioning
    logger.info(f"\n[2/5] Building 75% Train and 25% Test datasets for {ticker}...")
    df_train, df_test = prepare_ticker_data(
        ticker=ticker, start_date=start_date, end_date=end_date
    )

    # Step 3: Optimization
    if not skip_optimization:
        logger.info(f"\n[3/5] Running Bayesian optimization for {ticker} ({trials} trials)...")
        opt_params = run_ticker_optimization(ticker, df_train, n_trials=trials)
    else:
        logger.info(f"\n[3/5] Skipping optimization (--skip-optimization active). Loading configs for {ticker}.")
        opt_params = load_ticker_params(ticker)

    # Step 4: Model Training & Calibration
    logger.info(f"\n[4/5] Training Technical Brain Ensemble & Fitting Scalers for {ticker}...")
    kept_features = FEATURE_COLUMNS_V30

    configs_dir = BACKEND_DIR / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    with open(configs_dir / "kept_features.json", "w") as f:
        json.dump(kept_features, f, indent=4)

    # Clean missing values
    df_train_clean = df_train.replace([np.inf, -np.inf], np.nan).dropna(
        subset=kept_features + ["target_signal"]
    )
    df_test_clean = df_test.replace([np.inf, -np.inf], np.nan).dropna(
        subset=kept_features + ["target_signal"]
    )

    X_train_raw = df_train_clean[kept_features].values
    y_train = df_train_clean["target_signal"].astype(int).values
    X_test_raw = df_test_clean[kept_features].values
    y_test = df_test_clean["target_signal"].astype(int).values

    # Fit StandardScaler strictly on Train partition
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    artifacts_dir = BACKEND_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, artifacts_dir / "latest_scaler.joblib")
    joblib.dump(scaler, artifacts_dir / f"scaler_{ticker}.joblib")
    with open(artifacts_dir / "kept_features.json", "w") as f:
        json.dump(kept_features, f, indent=4)

    scaler_metadata = {
        "architecture": "asset_specific_standard_scaler",
        "asset": ticker,
        "feature_count": len(kept_features),
        "feature_names": kept_features,
        "sample_count": len(X_train),
        "training_dates": {
            "start": df_train.index[0].strftime("%Y-%m-%d"),
            "end": df_train.index[-1].strftime("%Y-%m-%d"),
        },
        "sha256": compute_sha256(artifacts_dir / "latest_scaler.joblib"),
    }
    with open(artifacts_dir / "scaler_metadata.json", "w") as f:
        json.dump(scaler_metadata, f, indent=4)

    # Balanced class weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([class_weight_dict[yi] for yi in y_train])

    # 1. XGBoost
    logger.info(f"Training XGBoost Model for {ticker}...")
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": int(opt_params.get("n_estimators", 300)),
        "max_depth": int(opt_params.get("max_depth", 5)),
        "learning_rate": float(opt_params.get("learning_rate", 0.03)),
        "subsample": float(opt_params.get("subsample", 0.85)),
        "colsample_bytree": float(opt_params.get("colsample_bytree", 0.85)),
        "random_state": 42,
        "eval_metric": "mlogloss",
        **get_xgboost_gpu_params(),
    }
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_train_acc = float(xgb_model.score(X_train, y_train))
    xgb_test_acc = float(xgb_model.score(X_test, y_test))
    logger.info(f"XGBoost Accuracy -> Train: {xgb_train_acc * 100:.2f}% | Test: {xgb_test_acc * 100:.2f}%")

    xgb_model.save_model(str(artifacts_dir / "xgb_ensemble.json"))

    # 2. LightGBM
    logger.info(f"Training LightGBM Model for {ticker}...")
    lgbm_params = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": int(opt_params.get("n_estimators", 300)),
        "learning_rate": float(opt_params.get("learning_rate", 0.03)),
        "max_depth": int(opt_params.get("max_depth", 5)),
        "num_leaves": int(opt_params.get("num_leaves", 31)),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    lgbm_model = LGBMClassifier(**lgbm_params)
    lgbm_model.fit(X_train, y_train, sample_weight=sample_weights)
    lgbm_train_acc = float(lgbm_model.score(X_train, y_train))
    lgbm_test_acc = float(lgbm_model.score(X_test, y_test))
    logger.info(f"LightGBM Accuracy -> Train: {lgbm_train_acc * 100:.2f}% | Test: {lgbm_test_acc * 100:.2f}%")

    joblib.dump(lgbm_model, artifacts_dir / "lgbm_agent.joblib")

    # 3. Model Calibrator
    logger.info(f"Fitting Probability Calibrator on 25% Test Set for {ticker}...")
    xgb_test_probs = xgb_model.predict_proba(X_test)
    lgbm_test_probs = lgbm_model.predict_proba(X_test)

    calibrator = ModelCalibrator()
    calibrator.fit("XGB", y_test, xgb_test_probs, method="isotonic")
    calibrator.fit("LGBM", y_test, lgbm_test_probs, method="isotonic")
    calibrator.save(str(artifacts_dir / "model_calibrator.joblib"))

    # Step 5: Save Active Ticker & Update Governance Manifest
    logger.info("\n[5/5] Saving Active Ticker and Strategy Metadata...")
    market = "us"
    if ".NS" in ticker or ".BO" in ticker:
        market = "in"

    with open(configs_dir / "active_ticker.json", "w") as f:
        json.dump({"ticker": ticker, "market": market}, f, indent=4)
    logger.info(f"Active ticker set: {ticker} ({market})")

    # Update manifest model hashes if manifest exists
    manifest_v24_path = artifacts_dir / "frozen_strategy_manifest_v2.4.json"
    if manifest_v24_path.exists():
        try:
            with open(manifest_v24_path, "r") as f:
                manifest_v24 = json.load(f)
            manifest_v24["model_hashes"]["xgb_ensemble.json"] = compute_sha256(artifacts_dir / "xgb_ensemble.json")
            manifest_v24["model_hashes"]["lgbm_agent.joblib"] = compute_sha256(artifacts_dir / "lgbm_agent.joblib")
            manifest_v24["model_hashes"]["latest_scaler.joblib"] = compute_sha256(artifacts_dir / "latest_scaler.joblib")
            manifest_v24["model_hashes"]["model_calibrator.joblib"] = compute_sha256(artifacts_dir / "model_calibrator.joblib")
            with open(manifest_v24_path, "w") as f:
                json.dump(manifest_v24, f, indent=4)
        except Exception as e:
            logger.warning(f"Could not refresh manifest hashes: {e}")

    total_time = time.time() - pipeline_start
    logger.info("=================================================================")
    logger.info(f"HYDRA V3.0 TRAINING PIPELINE COMPLETE FOR {ticker} ({total_time:.2f}s)")
    logger.info("=================================================================")


def main():
    parser = argparse.ArgumentParser(description="Unified Asset-Specific Training Pipeline")
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)",
    )
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

    train_ticker_pipeline(
        ticker=args.ticker,
        trials=args.trials,
        skip_optimization=args.skip_optimization,
        start_date=args.start,
        end_date=args.end,
    )


if __name__ == "__main__":
    main()
