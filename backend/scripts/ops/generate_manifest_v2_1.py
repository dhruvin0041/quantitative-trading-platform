import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def compute_file_hash(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_current_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(BACKEND_DIR.parent),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def generate_v2_1_manifest():
    print("=" * 80)
    print("GENERATING IMMUTABLE STRATEGY MANIFEST: HYDRA_PROSPECTIVE_V2.1")
    print("=" * 80)

    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    now_ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    is_dst = bool(now_ny.dst())
    tz_abbr = "EDT" if is_dst else "EST"

    # Exact timestamp representations
    utc_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    ny_iso = now_ny.isoformat()
    ny_display = f"{now_ny.strftime('%Y-%m-%d %H:%M:%S')} {tz_abbr}"

    models = {
        "xgb_ensemble.json": BACKEND_DIR / "artifacts" / "xgb_ensemble.json",
        "latest_fusion_weights.weights.h5": BACKEND_DIR / "artifacts" / "latest_fusion_weights.weights.h5",
        "lgbm_agent.joblib": BACKEND_DIR / "artifacts" / "lgbm_agent.joblib",
        "dqn_model.pth": BACKEND_DIR / "artifacts" / "dqn_model.pth",
        "latest_scaler.joblib": BACKEND_DIR / "artifacts" / "latest_scaler.joblib",
        "model_calibrator.joblib": BACKEND_DIR / "artifacts" / "model_calibrator.joblib",
        "meta_ensemble.joblib": BACKEND_DIR / "artifacts" / "meta_ensemble.joblib",
        "tft_quantile_weights.weights.h5": BACKEND_DIR / "artifacts" / "tft_quantile_weights.weights.h5",
    }

    configs = {
        "model_params.yaml": BACKEND_DIR / "configs" / "model_params.yaml",
        "optimized_params_AAPL.json": BACKEND_DIR / "configs" / "optimized_params_AAPL.json",
        "kept_features.json": BACKEND_DIR / "configs" / "kept_features.json",
    }

    code = {
        "live_inference.py": BACKEND_DIR / "src" / "execution" / "live_inference.py",
        "inference_service.py": BACKEND_DIR / "src" / "execution" / "inference_service.py",
        "signal_ledger.py": BACKEND_DIR / "src" / "execution" / "signal_ledger.py",
        "backtest_service.py": BACKEND_DIR / "src" / "execution" / "backtest_service.py",
        "data_firewall.py": BACKEND_DIR / "src" / "execution" / "data_firewall.py",
    }

    model_hashes = {k: compute_file_hash(v) for k, v in models.items()}
    config_hashes = {k: compute_file_hash(v) for k, v in configs.items()}
    code_hashes = {k: compute_file_hash(v) for k, v in code.items()}

    manifest = {
        "strategy_version": "HYDRA_PROSPECTIVE_V2.1",
        "previous_version": "HYDRA_PROSPECTIVE_V2.0",
        "historical_predecessor": "HYDRA_PROSPECTIVE_V1.0",
        "freeze_timestamp_utc": utc_str,
        "freeze_timestamp_new_york": ny_iso,
        "freeze_display_new_york": ny_display,
        "freeze_session_tz": ny_display,
        "git_commit": get_current_git_commit(),
        "status": "FROZEN_FOR_PROSPECTIVE_VALIDATION",
        "anti_overfitting_lock": True,
        "methodology_corrections_v2_1": {
            "dqn_retraining": "Trained exclusively on 2016-2024 development transitions. Zero 2025 observations or rewards.",
            "meta_ensemble_stacking": "Fitted exclusively on chronological walk-forward out-of-fold (OOF) predictions within 2016-2024. 2025 completely excluded.",
            "probability_calibration_split": "2025 partitioned chronologically: H1 2025 (fit isotonic calibrator) and H2 2025 (independent evaluation with zero refitting).",
            "scaler_architecture_alignment": "Formally established Asset-Specific StandardScaler (Option B, AAPL). Declared sample count matches artifact exactly (2,130 bars).",
            "hyperparameter_provenance_audit": "Disclosed that historical studies before commit 4b3013b accessed 2025-2026 data via datetime.now(). Parameters re-optimized cleanly strictly on 2016-2024.",
            "first_prospective_start": "First genuinely untouched prospective date established as 2026-10-01 16:00:00 EDT (post-freeze).",
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
                "components": ["latest_scaler.joblib", "scaler_AAPL.joblib"],
                "architecture": "Asset-Specific StandardScaler (Option B, AAPL)",
                "start_date": "2016-06-23",
                "end_date": "2024-12-09",
                "sample_count": 2130,
                "feature_count": 27,
                "description": "Asset-specific StandardScaler fitted exclusively on AAPL 2016-2024 development samples (2,130 bars, 27 features). Fully aligned with live inference.",
            },
            "downstream_calibration": {
                "components": ["model_calibrator.joblib"],
                "calibration_subset_h1": {
                    "start_date": "2025-01-02",
                    "end_date": "2025-06-30",
                    "sample_count": 65,
                    "role": "Isotonic regression fit partition",
                },
                "evaluation_subset_h2": {
                    "start_date": "2025-07-01",
                    "end_date": "2025-12-08",
                    "sample_count": 110,
                    "role": "Independent out-of-sample calibration evaluation (zero refitting)",
                },
                "description": "Probability calibrator fitted strictly on H1 2025 validation subset and evaluated independently on H2 2025.",
            },
            "meta_ensemble_training": {
                "components": ["meta_ensemble.joblib"],
                "start_date": "2020-09-15",
                "end_date": "2024-12-09",
                "total_oof_samples": 1065,
                "folds": 3,
                "methodology": "Expanding-Window Walk-Forward Out-of-Fold (OOF) Stacking",
                "description": "ElasticNet Logistic Regression fitted strictly on out-of-fold predictions within 2016-2024 development period. 2025 completely excluded.",
            },
            "dqn_rl_training": {
                "components": ["dqn_model.pth"],
                "start_date": "2016-06-23",
                "end_date": "2024-12-09",
                "unique_transitions": 2070,
                "episodes": 15,
                "description": "Deep Q-Network sequential policy replay trained exclusively on 2016-2024 development transitions. Rewards derived from 2016-2024 triple-barrier labels. Replay buffer and gradient updates restricted strictly to development data.",
            },
            "hyperparameter_optimization": {
                "components": [
                    "optimized_params_AAPL.json",
                ],
                "start_date": "2016-01-04",
                "end_date": "2024-12-30",
                "sampler": "TPESampler(seed=42)",
                "description": "Clean Optuna study for barrier ATR multipliers (1.5x TP, 2.0x SL, 15d horizon), tree parameters, cooldown, and threshold strictly within 2016-2024 development universe",
            },
            "historical_optimization_provenance_audit": {
                "finding": "Historical runs prior to commit 4b3013b used datetime.now(), consuming 2025-2026 data. Retrained cleanly on 2016-2024.",
                "status": "FIXED_IN_V2.1",
            },
            "prospective_validation": {
                "dataset": "TRUE_UNTOUCHED_OUT_OF_SAMPLE",
                "freeze_timestamp_utc": utc_str,
                "freeze_timestamp_new_york": ny_iso,
                "start_date": "2026-10-01",
                "start_timestamp": "2026-10-01 16:00:00 EDT",
                "description": "Untouched forward out-of-sample paper trading experiment strictly from 2026-10-01 16:00:00 EDT onward protected by hard firewall",
            },
        },
        "prospective_sequence": {
            "strategy_freeze_utc": utc_str,
            "strategy_freeze_new_york": ny_iso,
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
                "long_exit": "SELL signal closes LONG unconditionally (decoupled from macro bear filter)",
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

    # Write both V2.1 explicit manifest and active manifest
    v2_1_path = BACKEND_DIR / "artifacts" / "frozen_strategy_manifest_v2.1.json"
    active_path = BACKEND_DIR / "artifacts" / "frozen_strategy_manifest.json"

    with open(v2_1_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {v2_1_path}")

    with open(active_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {active_path}")

    return manifest


if __name__ == "__main__":
    generate_v2_1_manifest()
