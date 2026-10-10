# backend/scripts/training/train_v3_cross_asset.py
"""
HYDRA V3.0 Universal Cross-Asset Training Pipeline.

Trains the Multi-Model Alpha Ensemble (XGBoost + LightGBM + Isotonic Calibration)
across the 5-asset universe: AAPL, NVDA, MSFT, AMZN, SPY.
Expands the feature matrix to 31 institutional indicators:
- 27 baseline features (momentum, trend, volatility, rolling z-scores, ATR regime)
- 4 enhanced indicators: Keltner_Position, HMA_Slope, Connors_RSI, CMF_Divergence

Temporal Firewall & Causality Mandates:
1. Development Dataset: 2016-01-01 to 2024-12-31 (warmup 119 bars discarded).
2. Validation Dataset: 2025-01-01 to 2025-12-31.
3. Hard 2026 Firewall: Zero 2026 price or feature contamination.
4. Point-in-time dynamic triple barrier labeling with zero future lookahead.
5. Standard Scaler fitted strictly on pooled development features.
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.data_ingestion.market_data import (
    apply_dynamic_triple_barrier,
    fetch_historical_data,
)
from src.execution.data_firewall import TemporalFirewall
from src.execution.live_inference import (
    FEATURE_COLUMNS,
    add_upgraded_features,
)
from src.models.regime.calibration import ModelCalibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainV3CrossAsset")

# 31-Feature Matrix
FEATURE_COLUMNS_V3: List[str] = FEATURE_COLUMNS + [
    "Keltner_Position",
    "HMA_Slope",
    "Connors_RSI",
    "CMF_Divergence",
]

UNIVERSE: List[str] = ["AAPL", "NVDA", "MSFT", "AMZN", "SPY"]
DEV_START = "2016-01-01"
DEV_END = "2024-12-31"
VAL_START = "2025-01-01"
VAL_END = "2025-12-31"
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


def fetch_and_prepare_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches historical data for the universe, computes continuous features,
    applies temporal firewall separation, removes warmup bars, and applies
    dynamic triple barrier labeling.
    """
    logger.info("Fetching macro reference data: SPY and ^VIX...")
    spy_full = yf.download("SPY", start="2015-06-01", end=VAL_END, progress=False)
    vix_full = yf.download("^VIX", start="2015-06-01", end=VAL_END, progress=False)
    if isinstance(spy_full.columns, pd.MultiIndex):
        spy_full.columns = spy_full.columns.droplevel(1)
    if isinstance(vix_full.columns, pd.MultiIndex):
        vix_full.columns = vix_full.columns.droplevel(1)

    dev_frames = []
    val_frames = []

    for ticker in UNIVERSE:
        logger.info(f"Processing universe asset: {ticker}...")
        df_raw = fetch_historical_data(ticker, start_date="2015-06-01", end_date=VAL_END)
        TemporalFirewall.validate_no_2026_leakage(df_raw, f"{ticker}_raw")

        # Feature computation across continuous history for robust indicator warmup
        df_feat = add_upgraded_features(df_raw, spy_full, vix_full, lag_vix=True)
        df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()].copy()

        # Isolate Dev slice (<= 2024-12-31)
        df_dev_raw = df_feat[(df_feat.index >= DEV_START) & (df_feat.index <= DEV_END)].copy()
        TemporalFirewall.validate_development_data(df_dev_raw, f"{ticker}_dev")

        # Discard warmup bars (first 119 bars)
        if len(df_dev_raw) > WARMUP_BARS:
            df_dev_warm = df_dev_raw.iloc[WARMUP_BARS:].copy()
        else:
            df_dev_warm = df_dev_raw.copy()

        # Dynamic Triple Barrier Labeling on Dev
        # Last horizon bars naturally drop because future prices are absent
        df_dev_labeled = apply_dynamic_triple_barrier(
            df_dev_warm,
            tp_atr_multiplier=TP_ATR_MULT,
            sl_atr_multiplier=SL_ATR_MULT,
            horizon=LABEL_HORIZON,
        )
        df_dev_labeled["Ticker"] = ticker
        dev_frames.append(df_dev_labeled)
        logger.info(
            f"  -> {ticker} Dev labeled bars: {len(df_dev_labeled)} "
            f"({df_dev_labeled.index[0].strftime('%Y-%m-%d')} to {df_dev_labeled.index[-1].strftime('%Y-%m-%d')})"
        )

        # Isolate Val slice (2025-01-01 to 2025-12-31)
        df_val_raw = df_feat[(df_feat.index >= VAL_START) & (df_feat.index <= VAL_END)].copy()
        TemporalFirewall.validate_validation_data(df_val_raw, f"{ticker}_val")

        df_val_labeled = apply_dynamic_triple_barrier(
            df_val_raw,
            tp_atr_multiplier=TP_ATR_MULT,
            sl_atr_multiplier=SL_ATR_MULT,
            horizon=LABEL_HORIZON,
        )
        df_val_labeled["Ticker"] = ticker
        val_frames.append(df_val_labeled)
        logger.info(
            f"  -> {ticker} Val labeled bars: {len(df_val_labeled)} "
            f"({df_val_labeled.index[0].strftime('%Y-%m-%d')} to {df_val_labeled.index[-1].strftime('%Y-%m-%d')})"
        )

    master_dev = pd.concat(dev_frames)
    master_val = pd.concat(val_frames)

    logger.info(
        f"Cross-Asset Panel Assembled! Dev rows: {len(master_dev)}, Val rows: {len(master_val)}"
    )
    return master_dev, master_val


def train_universal_v3() -> Dict[str, any]:
    """Executes end-to-end training, scaling, calibration, and governance manifest creation."""
    artifacts_v3_dir = BACKEND_DIR / "artifacts" / "v3"
    artifacts_v3_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data
    master_dev, master_val = fetch_and_prepare_universe()

    # Clean and check nulls
    master_dev = master_dev.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLUMNS_V3 + ["target_signal"]
    )
    master_val = master_val.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLUMNS_V3 + ["target_signal"]
    )

    X_train_raw = master_dev[FEATURE_COLUMNS_V3].values
    y_train = master_dev["target_signal"].astype(int).values

    X_val_raw = master_val[FEATURE_COLUMNS_V3].values
    y_val = master_val["target_signal"].astype(int).values

    # 2. Fit Scaler Strictly on Dev Set
    logger.info(f"Fitting StandardScaler on {len(X_train_raw)} development bars across 31 features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    scaler_path = artifacts_v3_dir / "latest_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # Save kept features
    kept_features_path = artifacts_v3_dir / "kept_features.json"
    with open(kept_features_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS_V3, f, indent=2)

    # Also save to configs/kept_features_v3.json
    configs_v3_path = BACKEND_DIR / "configs" / "kept_features_v3.json"
    with open(configs_v3_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS_V3, f, indent=2)

    # 3. Class Balance Weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([class_weight_dict[yi] for yi in y_train])
    logger.info(f"Class distribution in train: {np.bincount(y_train)}, weights: {class_weight_dict}")

    # 4. Train Universal XGBoost
    logger.info("Training Universal XGBoost Multi-Classifier...")
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 350,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "gamma": 1.2,
        "min_child_weight": 6,
        "random_state": 42,
        "n_jobs": -1,
    }
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    train_acc_xgb = float(xgb_model.score(X_train, y_train))
    val_acc_xgb = float(xgb_model.score(X_val, y_val))
    logger.info(f"XGBoost Accuracy -> Train: {train_acc_xgb * 100:.2f}% | Val: {val_acc_xgb * 100:.2f}%")

    xgb_path = artifacts_v3_dir / "xgb_ensemble.json"
    xgb_model.save_model(str(xgb_path))
    logger.info(f"XGBoost model saved to {xgb_path}")

    # 5. Train Universal LightGBM
    logger.info("Training Universal LightGBM Agent...")
    lgbm_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        verbose=-1,
    )
    lgbm_model.fit(X_train, y_train)

    train_acc_lgbm = float(lgbm_model.score(X_train, y_train))
    val_acc_lgbm = float(lgbm_model.score(X_val, y_val))
    logger.info(f"LightGBM Accuracy -> Train: {train_acc_lgbm * 100:.2f}% | Val: {val_acc_lgbm * 100:.2f}%")

    lgbm_path = artifacts_v3_dir / "lgbm_agent.joblib"
    joblib.dump(lgbm_model, lgbm_path)
    logger.info(f"LightGBM model saved to {lgbm_path}")

    # 6. Fit Model Calibrator on Held-Out 2025 Validation Data
    logger.info("Calibrating probabilities on 2025 validation set...")
    val_xgb_probs = xgb_model.predict_proba(X_val)
    val_lgbm_probs = lgbm_model.predict_proba(X_val)

    calibrator = ModelCalibrator()
    calibrator.fit("XGB", y_val, val_xgb_probs, method="isotonic")
    calibrator.fit("LGBM", y_val, val_lgbm_probs, method="isotonic")

    calibrator_path = artifacts_v3_dir / "model_calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)
    logger.info(f"ModelCalibrator saved to {calibrator_path}")

    # 7. Generate HYDRA V3.0 Strategy Manifest
    logger.info("Generating Cryptographic Strategy Manifest for HYDRA V3.0...")
    manifest_v3 = {
        "strategy_version": "HYDRA_PROSPECTIVE_V3.0",
        "previous_version": "HYDRA_PROSPECTIVE_V2.4",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_CROSS_ASSET_VALIDATION",
        "anti_overfitting_lock": True,
        "universe": UNIVERSE,
        "feature_count": len(FEATURE_COLUMNS_V3),
        "feature_columns": FEATURE_COLUMNS_V3,
        "new_indicators": [
            "Keltner_Position",
            "HMA_Slope",
            "Connors_RSI",
            "CMF_Divergence",
        ],
        "model_hashes": {
            "v3/xgb_ensemble.json": compute_sha256(xgb_path),
            "v3/lgbm_agent.joblib": compute_sha256(lgbm_path),
            "v3/latest_scaler.joblib": compute_sha256(scaler_path),
            "v3/model_calibrator.joblib": compute_sha256(calibrator_path),
            "v3/kept_features.json": compute_sha256(kept_features_path),
        },
        "config_hashes": {
            "kept_features_v3.json": compute_sha256(configs_v3_path),
        },
        "dataset_provenance": {
            "development_universe": {
                "tickers": UNIVERSE,
                "start_date": DEV_START,
                "end_date": DEV_END,
                "total_samples": len(X_train),
                "warmup_bars_dropped": WARMUP_BARS,
                "label_horizon": LABEL_HORIZON,
                "tp_atr_multiplier": TP_ATR_MULT,
                "sl_atr_multiplier": SL_ATR_MULT,
                "zero_2025_leakage": True,
                "zero_2026_leakage": True,
            },
            "validation_universe": {
                "tickers": UNIVERSE,
                "start_date": VAL_START,
                "end_date": VAL_END,
                "total_samples": len(X_val),
                "zero_2026_leakage": True,
            },
        },
        "validation_metrics": {
            "xgb_train_accuracy": train_acc_xgb,
            "xgb_val_accuracy": val_acc_xgb,
            "lgbm_train_accuracy": train_acc_lgbm,
            "lgbm_val_accuracy": val_acc_lgbm,
        },
        "execution_rules": {
            "fill_timing": "NEXT_SESSION_OPEN",
            "slippage_bps": 5.0,
            "commission_per_share_usd": 0.005,
        },
    }

    manifest_path = BACKEND_DIR / "artifacts" / "frozen_strategy_manifest_v3.0.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_v3, f, indent=2)
    logger.info(f"HYDRA V3.0 Manifest generated at {manifest_path}")

    return {
        "status": "SUCCESS",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "xgb_train_acc": train_acc_xgb,
        "xgb_val_acc": val_acc_xgb,
        "lgbm_train_acc": train_acc_lgbm,
        "lgbm_val_acc": val_acc_lgbm,
        "manifest_path": str(manifest_path),
    }


if __name__ == "__main__":
    results = train_universal_v3()
    print("\n--- Training Complete ---")
    print(json.dumps(results, indent=2))
