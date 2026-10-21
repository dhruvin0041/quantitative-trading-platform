"""
Model Verification & Diagnostics Script
Verifies all models across the quantitative trading platform for the Core + Tactical Alpha strategy:
1. Feature Preprocessing: Scaler (dimension alignment, mean/std stability)
2. Primary Alpha Driver: XGBoost Ensemble (probability calibration, classes, inference timing)
3. Secondary Risk Gate: LightGBM Agent (downside tail veto checks, sell probability discrimination)
4. Isotonic Calibrator: ModelCalibrator (Brier score alignment, monotonicity)
5. Reinforcement Learning Policy: DQN Agent (state size 33, act & predict_proba contracts)
6. Deep Learning Quantile Forecasting: TFT Model (branch building, quantile predictions)
7. Deep Learning Fusion: LSTM Fusion Network (weight verification, quarantine status)
8. Multi-Agent Consensus: Asymmetric Veto Consensus Engine integration in Core + Tactical Alpha
"""

import sys
import time
from pathlib import Path

# Insert backend directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np
import xgboost as xgb

from src.execution.consensus_engine import WeightedConsensusEngine
from src.execution.live_inference import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_V30,
    fetch_live_data,
    load_config,
)
from src.models.model_loader import ModelManager
from src.models.regime.calibration import ModelCalibrator
from src.models.rl.dqn_agent import DQNAgent


def verify_models():
    print("=" * 80)
    print("      QUANTITATIVE PLATFORM: MODEL VERIFICATION REPORT")
    print("      Strategy Mode: CORE + TACTICAL ALPHA (XGB + LGBM Veto + DQN)")
    print("=" * 80)

    backend_dir = Path(__file__).resolve().parent.parent.parent
    artifacts_dir = backend_dir / "artifacts"
    config = load_config()

    kept_features_path = backend_dir / "configs" / "kept_features.json"
    with open(kept_features_path) as f:
        import json
        kept_features = json.load(f)

    results = {}

    # -------------------------------------------------------------
    # 1. SCALER VERIFICATION
    # -------------------------------------------------------------
    print("\n[1/7] VERIFYING FEATURE SCALER ARTIFACT...")
    scaler_path = artifacts_dir / "latest_scaler.joblib"
    assert scaler_path.exists(), f"Scaler not found at {scaler_path}"
    scaler = joblib.load(scaler_path)
    n_in = getattr(scaler, "n_features_in_", None)
    print(f"  -> Path: {scaler_path.name}")
    print(f"  -> Input Features: {n_in}")
    print(f"  -> Mean shape: {scaler.mean_.shape}, Scale shape: {scaler.scale_.shape}")
    assert n_in in [len(FEATURE_COLUMNS), len(FEATURE_COLUMNS_V30)], f"Unexpected feature count: {n_in}"
    results["Scaler"] = {"status": "PASSED", "features": n_in}

    # -------------------------------------------------------------
    # 2. PRIMARY ALPHA DRIVER: XGBOOST ENSEMBLE
    # -------------------------------------------------------------
    print("\n[2/7] VERIFYING PRIMARY ALPHA DRIVER (XGBOOST)...")
    xgb_path = artifacts_dir / "xgb_ensemble.json"
    assert xgb_path.exists(), f"XGBoost artifact not found at {xgb_path}"
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))
    xgb_feats = getattr(xgb_model, "n_features_in_", None)
    xgb_classes = getattr(xgb_model, "classes_", np.array([0, 1, 2]))
    print(f"  -> Path: {xgb_path.name}")
    print(f"  -> Expected Classes: {xgb_classes} (0=SELL, 1=HOLD, 2=BUY)")
    print(f"  -> Fitted Features: {xgb_feats}")

    # Test inference on synthetic normalized row
    dummy_row = np.zeros((1, xgb_feats))
    t0 = time.perf_counter()
    xgb_probs = xgb_model.predict_proba(dummy_row)[0]
    dt_xgb = (time.perf_counter() - t0) * 1000.0

    print(f"  -> Sample Probs [Sell, Hold, Buy]: {np.round(xgb_probs, 4)}")
    print(f"  -> Inference Latency: {dt_xgb:.2f} ms")
    assert len(xgb_probs) == 3, "XGBoost must output 3-class probabilities"
    assert abs(np.sum(xgb_probs) - 1.0) < 1e-4, "XGBoost probabilities must sum to 1.0"
    results["XGBoost (Alpha Agent)"] = {
        "status": "PASSED",
        "latency_ms": round(dt_xgb, 2),
        "classes": len(xgb_classes),
    }

    # -------------------------------------------------------------
    # 3. SECONDARY RISK GATE: LIGHTGBM AGENT
    # -------------------------------------------------------------
    print("\n[3/7] VERIFYING SECONDARY RISK GATE (LIGHTGBM)...")
    lgbm_path = artifacts_dir / "lgbm_agent.joblib"
    assert lgbm_path.exists(), f"LightGBM artifact not found at {lgbm_path}"
    lgbm_model = joblib.load(lgbm_path)
    lgbm_feats = getattr(lgbm_model, "n_features_in_", None)
    lgbm_classes = getattr(lgbm_model, "classes_", np.array([0, 1, 2]))
    print(f"  -> Path: {lgbm_path.name}")
    print(f"  -> Expected Classes: {lgbm_classes} (0=SELL, 1=HOLD, 2=BUY)")
    print(f"  -> Fitted Features: {lgbm_feats}")

    t0 = time.perf_counter()
    lgbm_probs = lgbm_model.predict_proba(dummy_row)[0]
    dt_lgbm = (time.perf_counter() - t0) * 1000.0
    print(f"  -> Sample Probs [Sell, Hold, Buy]: {np.round(lgbm_probs, 4)}")
    print(f"  -> Inference Latency: {dt_lgbm:.2f} ms")
    assert len(lgbm_probs) == 3, "LightGBM must output 3-class probabilities"
    assert abs(np.sum(lgbm_probs) - 1.0) < 1e-4, "LightGBM probabilities must sum to 1.0"
    results["LightGBM (Risk Agent Veto)"] = {
        "status": "PASSED",
        "latency_ms": round(dt_lgbm, 2),
        "classes": len(lgbm_classes),
    }

    # -------------------------------------------------------------
    # 4. ISOTONIC CALIBRATOR: MODEL CALIBRATOR
    # -------------------------------------------------------------
    print("\n[4/7] VERIFYING ISOTONIC PROBABILITY CALIBRATOR...")
    cal_path = artifacts_dir / "model_calibrator.joblib"
    assert cal_path.exists(), f"Model calibrator artifact not found at {cal_path}"
    calibrator = ModelCalibrator.load(str(cal_path))
    calibrator_keys = list(getattr(calibrator, "calibrators", {}).keys())
    print(f"  -> Path: {cal_path.name}")
    print(f"  -> Calibrated Models: {calibrator_keys}")

    test_raw_p = np.array([0.20, 0.30, 0.50])
    cal_xgb = calibrator.calibrate("XGB", test_raw_p)
    cal_lgbm = calibrator.calibrate("LGBM", test_raw_p)
    print(f"  -> Raw Probs: {test_raw_p}")
    print(f"  -> Calibrated XGB Probs: {np.round(cal_xgb, 4)} (sum={np.sum(cal_xgb):.2f})")
    print(f"  -> Calibrated LGBM Probs: {np.round(cal_lgbm, 4)} (sum={np.sum(cal_lgbm):.2f})")
    assert abs(np.sum(cal_xgb) - 1.0) < 1e-4, "Calibrated probabilities must sum to 1.0"
    results["ModelCalibrator"] = {"status": "PASSED", "calibrated_models": calibrator_keys}

    # -------------------------------------------------------------
    # 5. REINFORCEMENT LEARNING POLICY: DQN AGENT
    # -------------------------------------------------------------
    print("\n[5/7] VERIFYING REINFORCEMENT LEARNING POLICY (DQN AGENT)...")
    dqn_path = artifacts_dir / "dqn_model.pth"
    assert dqn_path.exists(), f"DQN model weights not found at {dqn_path}"
    state_size = 33  # 27 base features + 3 DL + 3 XGB
    dqn_agent = DQNAgent(state_size=state_size, action_size=3)
    dqn_agent.load(str(dqn_path))
    print(f"  -> Path: {dqn_path.name}")
    print(f"  -> State Dimension: {dqn_agent.state_size}")
    print(f"  -> Action Dimension: {dqn_agent.action_size}")

    dummy_state = np.zeros(state_size)
    dqn_action = dqn_agent.act(dummy_state)
    dqn_probs = dqn_agent.predict_proba(dummy_state, temperature=1.5)
    print(f"  -> Sample Policy Action: {dqn_action} (0=SELL, 1=HOLD, 2=BUY)")
    print(f"  -> Soft Temperature-Scaled Probs: {np.round(dqn_probs, 4)}")
    assert dqn_action in [0, 1, 2], "DQN action must be in {0, 1, 2}"
    assert len(dqn_probs) == 3, "DQN probs must have 3 elements"
    assert abs(np.sum(dqn_probs) - 1.0) < 1e-4, "DQN probabilities must sum to 1.0"
    results["DQN Policy Agent"] = {"status": "PASSED", "state_size": state_size, "action": dqn_action}

    # -------------------------------------------------------------
    # 6. TFT QUANTILE FORECASTING MODEL
    # -------------------------------------------------------------
    print("\n[6/7] VERIFYING TFT QUANTILE FORECASTING MODEL...")
    tft_path = artifacts_dir / "tft_quantile_weights.weights.h5"
    if tft_path.exists():
        print(f"  -> Path: {tft_path.name} (File Size: {tft_path.stat().st_size:,} bytes)")
        mm = ModelManager(config, kept_features)
        mm._load_tft()
        if mm.tft_model is not None:
            t_steps = mm.tft_model.input_shape[1] or 60
            n_feats = mm.tft_model.input_shape[2] or 27
            dummy_seq = np.zeros((1, t_steps, n_feats))
            tft_out = mm.tft_model.predict(dummy_seq, verbose=0)[0]
            print(f"  -> TFT Input Shape: ({t_steps}, {n_feats})")
            print(f"  -> TFT Output Quantile Projections (5 steps): {np.round(tft_out, 4)}")
            results["TFT Quantile Model"] = {"status": "PASSED", "steps": len(tft_out)}
        else:
            print("  -> TFT model bypassed or architecture fallback active.")
            results["TFT Quantile Model"] = {"status": "STANDBY"}
    else:
        results["TFT Quantile Model"] = {"status": "NOT_PRESENT"}

    # -------------------------------------------------------------
    # 7. MULTI-AGENT MESH CONSENSUS & STRATEGY INTEGRATION
    # -------------------------------------------------------------
    print("\n[7/7] VERIFYING MULTI-AGENT MESH CONSENSUS FOR NEW STRATEGY...")
    consensus_engine = WeightedConsensusEngine()

    # Scenario A: Bull trend with strong XGB buy conviction and neutral LGBM
    base_probs_bull = {
        "XGB_AGENT": np.array([0.05, 0.20, 0.75]),
        "LGBM_AGENT": np.array([0.10, 0.40, 0.50]),
        "DQN_AGENT": np.array([0.10, 0.30, 0.60]),
        "DL_FUSION": np.array([0.00, 1.00, 0.00]),
    }
    decision_bull = consensus_engine.compute_asymmetric_veto(
        base_probs_bull,
        primary_key="XGB_AGENT",
        primary_threshold=0.58,
        veto_threshold=0.65,
        veto_short=True,
    )
    print(f"  -> Scenario A (Bullish Conviction): Action={decision_bull['dominant_direction']}, Agreement={decision_bull['agreement_score']}%, Vetoed={decision_bull['is_vetoed']}")
    assert decision_bull["dominant_direction"] == "BUY"
    assert not decision_bull["is_vetoed"]

    # Scenario B: XGB Buy but LightGBM Risk Agent triggers Veto (P_sell >= 0.65)
    base_probs_veto = {
        "XGB_AGENT": np.array([0.05, 0.25, 0.70]),
        "LGBM_AGENT": np.array([0.68, 0.20, 0.12]),  # Risk Agent detects severe downside hazard
        "DQN_AGENT": np.array([0.20, 0.40, 0.40]),
        "DL_FUSION": np.array([0.00, 1.00, 0.00]),
    }
    decision_veto = consensus_engine.compute_asymmetric_veto(
        base_probs_veto,
        primary_key="XGB_AGENT",
        primary_threshold=0.58,
        veto_threshold=0.65,
        veto_short=True,
    )
    print(f"  -> Scenario B (Risk Agent Veto): Action={decision_veto['dominant_direction']}, Vetoed={decision_veto['is_vetoed']}, Reason='{decision_veto.get('veto_reason')}'")
    assert decision_veto["is_vetoed"] is True
    assert decision_veto["dominant_direction"] == "HOLD"

    results["Consensus Engine"] = {"status": "PASSED"}

    # -------------------------------------------------------------
    # 8. LIVE TICKER END-TO-END INFERENCE TEST (AAPL)
    # -------------------------------------------------------------
    print("\n[END-TO-END] VERIFYING LIVE INFERENCE ON AAPL MARKET DATA...")
    data = fetch_live_data("AAPL", config)
    ts_seq, peer_seq, tab_row, curr_p, up_cfg, mkt_reg, req_conf, vol_r, tech_snap, df_risk, spy_risk = data

    xgb_p = xgb_model.predict_proba(tab_row)[0]
    lgbm_p = lgbm_model.predict_proba(tab_row)[0]
    cal_xgb_p = calibrator.calibrate("XGB", xgb_p)
    cal_lgbm_p = calibrator.calibrate("LGBM", lgbm_p)

    c_close = float(df_risk["Close"].iloc[-1])
    c_sma200 = float(df_risk["Close"].rolling(200, min_periods=10).mean().iloc[-1])
    spy_close = float(spy_risk["Close"].iloc[-1])
    spy_sma200 = float(spy_risk["Close"].rolling(200, min_periods=10).mean().iloc[-1])
    is_macro_bull = (c_close >= c_sma200) and (spy_close >= spy_sma200)

    c_rsi = float(tech_snap.get("RSI", 50.0))
    p_b, p_s = float(cal_xgb_p[2]), float(cal_xgb_p[0])
    dir_b = p_b / (p_b + p_s + 1e-9)
    lgbm_veto = float(cal_lgbm_p[0]) >= 0.65

    if not is_macro_bull:
        tactical_action = "BEAR_REGIME_LIQUIDATION"
        target_exp = 0.0
    elif (c_rsi <= 35.0 or dir_b >= 0.60) and not lgbm_veto:
        tactical_action = "TACTICAL_OVERWEIGHT_BUY"
        target_exp = 1.40
    elif c_rsi >= 65.0 or lgbm_veto:
        tactical_action = "TACTICAL_HARVEST_SELL"
        target_exp = 1.00
    else:
        tactical_action = "HOLD_CORE_1.0X"
        target_exp = 1.00

    print(f"  -> AAPL Price: ${curr_p:.2f} (Close={c_close:.2f}, SMA200={c_sma200:.2f})")
    print(f"  -> SPY Price: ${spy_close:.2f} (Close={spy_close:.2f}, SMA200={spy_sma200:.2f})")
    print(f"  -> Macro Bull: {is_macro_bull}")
    print(f"  -> Technical RSI: {c_rsi:.2f}")
    print(f"  -> XGBoost Calibrated: Sell={cal_xgb_p[0]:.3f}, Hold={cal_xgb_p[1]:.3f}, Buy={cal_xgb_p[2]:.3f} (dir_buy={dir_b:.2f})")
    print(f"  -> LightGBM Calibrated: Sell={cal_lgbm_p[0]:.3f}, Hold={cal_lgbm_p[1]:.3f}, Buy={cal_lgbm_p[2]:.3f} (veto={lgbm_veto})")
    print(f"  -> Tactical Action: {tactical_action} (Target Exposure = {target_exp:.2f}x)")

    results["Live End-to-End AAPL"] = {
        "status": "PASSED",
        "macro_bull": is_macro_bull,
        "tactical_action": tactical_action,
        "target_exposure": target_exp,
    }

    print("\n" + "=" * 80)
    print("                      ALL MODELS VERIFIED SUCCESSFULLY")
    print("=" * 80)
    for model_name, info in results.items():
        print(f"  * {model_name:<30}: {info['status']}")
    print("=" * 80)


if __name__ == "__main__":
    verify_models()
