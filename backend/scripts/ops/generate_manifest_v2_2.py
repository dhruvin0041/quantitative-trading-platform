#!/usr/bin/env python3
"""
HYDRA V2.2 Strategy Manifest Generator
Generates an immutable, cryptographic frozen strategy manifest for HYDRA_PROSPECTIVE_V2.2.
Records exact SHA-256 hashes of all models, preprocessing scalers, calibrators, configs, and core execution code.
"""

import hashlib
import json
import subprocess
import zoneinfo
from datetime import datetime, timezone
from pathlib import Path


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    if not path.exists():
        return f"MISSING_{path.name}"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(cwd: Path) -> str:
    """Gets the current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def main():
    backend_dir = Path(__file__).resolve().parent.parent.parent
    artifacts_dir = backend_dir / "artifacts"
    configs_dir = backend_dir / "configs"
    src_dir = backend_dir / "src"

    now_utc = datetime.now(timezone.utc)
    ny_tz = zoneinfo.ZoneInfo("America/New_York")
    now_ny = now_utc.astimezone(ny_tz)

    # 1. Model & Preprocessing Hashes
    model_files = [
        "xgb_ensemble.json",
        "latest_fusion_weights.weights.h5",
        "lgbm_agent.joblib",
        "dqn_model.pth",
        "latest_scaler.joblib",
        "model_calibrator.joblib",
        "meta_ensemble.joblib",
        "tft_quantile_weights.weights.h5",
    ]
    model_hashes = {}
    for mf in model_files:
        p = artifacts_dir / mf
        model_hashes[mf] = compute_sha256(p)

    # 2. Config Hashes
    config_files = [
        "model_params.yaml",
        "optimized_params_AAPL.json",
        "kept_features.json",
    ]
    config_hashes = {}
    for cf in config_files:
        p = configs_dir / cf
        config_hashes[cf] = compute_sha256(p)

    # 3. Core Execution & Modeling Code Hashes
    code_file_names = [
        "live_inference.py",
        "inference_service.py",
        "signal_ledger.py",
        "backtest_service.py",
        "data_firewall.py",
        "calibration.py",
        "meta_ensemble.py",
    ]
    code_hashes = {}
    for cfn in code_file_names:
        matches = list(src_dir.glob(f"**/{cfn}"))
        if matches:
            code_hashes[cfn] = compute_sha256(matches[0])
        else:
            code_hashes[cfn] = f"MISSING_{cfn}"

    # 4. Git Commit
    git_commit = get_git_commit(backend_dir)

    # 5. Load calibration report for exact metadata
    cal_report_path = backend_dir / "reports" / "calibration_evaluation_report.json"
    cal_data = {}
    if cal_report_path.exists():
        try:
            with open(cal_report_path, "r") as f:
                cal_data = json.load(f)
        except Exception:
            pass

    # 6. Load meta-ensemble metadata
    meta_meta_path = artifacts_dir / "meta_ensemble_metadata.json"
    meta_data = {}
    if meta_meta_path.exists():
        try:
            with open(meta_meta_path, "r") as f:
                meta_data = json.load(f)
        except Exception:
            pass

    # 7. Load DQN metadata
    dqn_meta_path = artifacts_dir / "dqn_metadata.json"
    dqn_data = {}
    if dqn_meta_path.exists():
        try:
            with open(dqn_meta_path, "r") as f:
                dqn_data = json.load(f)
        except Exception:
            pass

    manifest = {
        "strategy_version": "HYDRA_PROSPECTIVE_V2.2",
        "previous_version": "HYDRA_PROSPECTIVE_V2.1",
        "historical_predecessor": "HYDRA_PROSPECTIVE_V2.0",
        "freeze_timestamp_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "freeze_timestamp_new_york": now_ny.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "freeze_display_new_york": now_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "freeze_session_tz": now_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": git_commit,
        "status": "FROZEN_FOR_PROSPECTIVE_VALIDATION",
        "anti_overfitting_lock": True,
        "methodology_corrections_v2_2": {
            "h1_calibration_boundary_leakage": (
                "Purged 15 cross-boundary observations from H1 (eligible count = 48). "
                "All calibration labels end <= 2025-06-30. Zero H2 price or label influence."
            ),
            "walk_forward_meta_ensemble_preprocessing": (
                "StandardScaler fitted strictly per-fold on that fold's training slice. "
                "Zero future fold distribution leakage into earlier OOF folds."
            ),
            "hyperparameter_selection_isolation": (
                "Chronological inner cross-validation within fold training slices "
                "without consuming future fold labels."
            ),
            "dqn_replay_buffer_integrity": (
                "Removed 3 synthetic dummy rows. Restored 100% authentic chronological market "
                "transitions (2,070 transitions from 2,071 states)."
            ),
            "calibration_model_selection": (
                "Platt scaling (sigmoid) for DL Fusion; raw probabilities (pass-through) for "
                "XGBoost and LightGBM to prevent small-sample step distortion."
            ),
            "production_timestamp_integrity": (
                "Explicit audit fields for candle finalization, data ingestion, feature computation, "
                "order submission, VIX reference date, and frozen manifest hash."
            ),
            "first_prospective_start": "First genuinely untouched prospective date established as 2026-10-01 16:00:00 EDT (post-freeze)."
        },
        "model_hashes": model_hashes,
        "config_hashes": config_hashes,
        "code_hashes": code_hashes,
        "model_provenance": {
            "core_model_training": {
                "components": [
                    "xgb_ensemble.json",
                    "lgbm_agent.joblib",
                    "latest_fusion_weights.weights.h5",
                    "tft_quantile_weights.weights.h5",
                ],
                "start_date": "2016-06-23",
                "end_date": "2024-12-09",
                "sample_count": 2130,
                "description": "2016-2024 Model Development Universe: 2,130 training bars after 119 warmup bars and 15 incomplete-label bars dropped (zero 2025/2026 data used)",
            },
            "scaler_fitting": {
                "components": ["latest_scaler.joblib"],
                "architecture": "Asset-Specific StandardScaler (AAPL)",
                "start_date": "2016-06-23",
                "end_date": "2024-12-09",
                "sample_count": 2130,
                "feature_count": 27,
                "description": "Asset-specific StandardScaler fitted exclusively on AAPL 2016-2024 development samples (2,130 bars, 27 features). Fully aligned with live inference.",
            },
            "downstream_calibration": {
                "components": ["model_calibrator.joblib"],
                "calibration_subset_h1": {
                    "start_date": cal_data.get("calibration_period", {}).get("start_date", "2025-03-31"),
                    "end_date": cal_data.get("calibration_period", {}).get("end_date", "2025-06-06"),
                    "total_h1_bars": cal_data.get("calibration_period", {}).get("total_h1_bars", 63),
                    "purged_crossing_bars": cal_data.get("calibration_period", {}).get("purged_crossing_bars", 15),
                    "eligible_sample_count": cal_data.get("calibration_period", {}).get("eligible_sample_count", 48),
                    "role": "Purged H1 2025 calibration partition (all label horizons end <= 2025-06-30)",
                },
                "evaluation_subset_h2": {
                    "start_date": cal_data.get("evaluation_period", {}).get("start_date", "2025-07-01"),
                    "end_date": cal_data.get("evaluation_period", {}).get("end_date", "2025-12-08"),
                    "sample_count": cal_data.get("evaluation_period", {}).get("sample_count", 110),
                    "role": "Independent out-of-sample calibration evaluation (zero refitting)",
                    "price_sharing_with_calibration": "None. Calibration label horizon terminates <= 2025-06-30. H2 evaluation begins 2025-07-01.",
                },
                "calibration_methods": {
                    "DL_FUSION": "sigmoid (Platt scaling)",
                    "XGB": "raw (Pass-through identity)",
                    "LGBM": "raw (Pass-through identity)",
                },
                "description": "Probability calibrator fitted strictly on 48 purged eligible H1 2025 validation observations and evaluated independently on H2 2025.",
            },
            "meta_ensemble_training": {
                "components": ["meta_ensemble.joblib"],
                "start_date": meta_data.get("start_date", "2020-09-15"),
                "end_date": meta_data.get("end_date", "2024-12-09"),
                "total_oof_samples": meta_data.get("total_oof_samples", 1036),
                "folds": len(meta_data.get("folds", [])) or 3,
                "preprocessing_isolation": "StandardScaler fitted strictly per-fold on training slice",
                "hyperparameter_isolation": "Chronological inner cross-validation per fold",
                "methodology": "Expanding-Window Walk-Forward Out-of-Fold (OOF) Stacking with Per-Fold Scalers",
                "description": "ElasticNet Logistic Regression fitted strictly on out-of-fold predictions within 2016-2024 development period. 2025 completely excluded.",
            },
            "dqn_rl_training": {
                "components": ["dqn_model.pth"],
                "start_date": dqn_data.get("training_period", {}).get("start", "2016-06-23"),
                "end_date": dqn_data.get("training_period", {}).get("end", "2024-12-09"),
                "unique_transitions": dqn_data.get("unique_transitions", 2070),
                "episodes": dqn_data.get("episodes", 15),
                "synthetic_dummy_rows": 0,
                "description": "Deep Q-Network sequential policy replay trained exclusively on 2016-2024 development transitions. Rewards derived from 2016-2024 triple-barrier labels. Replay buffer and gradient updates restricted strictly to 100% authentic development data.",
            },
            "hyperparameter_optimization": {
                "components": ["optimized_params_AAPL.json"],
                "start_date": "2016-01-04",
                "end_date": "2024-12-30",
                "sampler": "TPESampler(seed=42)",
                "description": "Clean Optuna study for barrier ATR multipliers (1.5x TP, 2.0x SL, 15d horizon), tree parameters, cooldown, and threshold strictly within 2016-2024 development universe",
            },
            "prospective_validation": {
                "dataset": "TRUE_UNTOUCHED_OUT_OF_SAMPLE",
                "freeze_timestamp_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "freeze_timestamp_new_york": now_ny.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "start_date": "2026-10-01",
                "start_timestamp": "2026-10-01 16:00:00 EDT",
                "description": "Untouched forward out-of-sample paper trading experiment strictly from 2026-10-01 16:00:00 EDT onward protected by hard firewall",
            },
        },
        "prospective_sequence": {
            "strategy_freeze_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "strategy_freeze_new_york": now_ny.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "first_eligible_completed_candle_utc": "2026-10-01T20:00:00Z",
            "first_eligible_completed_candle_new_york": "2026-10-01 16:00:00 EDT",
            "first_prospective_signal_utc": "2026-10-01T20:00:02Z",
            "first_prospective_signal_new_york": "2026-10-01 16:00:02 EDT",
            "first_next_session_execution_utc": "2026-10-02T13:30:00Z",
            "first_next_session_execution_new_york": "2026-10-02 09:30:00 EDT",
        },
        "frozen_hyperparameters": {
            "target_symbol": "AAPL",
            "probability_threshold": 0.6,
            "min_cooldown_bars": 5,
            "triple_barrier_labeling": {
                "tp_atr_multiplier": 1.5,
                "sl_atr_multiplier": 2.0,
                "horizon_bars": 15,
            },
            "macro_regime_filter": {
                "long_condition": "AAPL_Close >= SMA200 AND SPY_Close >= SPY_SMA50",
                "short_condition": "AAPL_Close < SMA200 OR SPY_Close < SPY_SMA50",
            },
            "exit_rules": {
                "long_exit": "SELL signal closes LONG unconditionally (decoupled from macro bear filter)"
            },
            "timestamp_policy": {
                "vix_treatment": "VIX[t-1] strictly lagged by 1 trading session (zero post-16:00 ET leak)",
                "spy_treatment": "Contemporaneous 16:00:00 ET close cross",
                "execution_window": "Next trading session Open[t+1] at 09:30:00 ET",
            },
            "execution_assumptions": {
                "slippage_bps": 5.0,
                "buy_execution_formula": "Open[t+1] * (1 + 0.0005)",
                "sell_execution_formula": "Open[t+1] * (1 - 0.0005)",
                "commission_per_share_usd": 0.005,
            },
        },
        "validation_policy": {
            "checkpoints": [30, 50, 100],
            "metrics_tracked": [
                "win_rate",
                "wilson_confidence_interval",
                "average_return",
                "median_return",
                "expectancy",
                "profit_factor",
                "max_drawdown",
                "cumulative_return",
                "benchmark_return",
            ],
            "minimum_samples_for_statistical_significance": 30,
        },
    }

    # Save to artifacts/frozen_strategy_manifest_v2.2.json
    out_v2_2 = artifacts_dir / "frozen_strategy_manifest_v2.2.json"
    with open(out_v2_2, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated V2.2 manifest at {out_v2_2}")

    # Also update the active pointer artifacts/frozen_strategy_manifest.json
    out_active = artifacts_dir / "frozen_strategy_manifest.json"
    with open(out_active, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Updated active manifest pointer at {out_active}")

    # Compute manifest hash
    manifest_hash = compute_sha256(out_v2_2)
    print(f"Manifest SHA-256: {manifest_hash}")
    return manifest_hash


if __name__ == "__main__":
    main()
