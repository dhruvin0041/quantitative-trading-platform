# HYDRA COMPLETE ARCHITECTURE AUDIT
**Document Version:** 1.1.0 (Forensic Baseline & V2.3 Evolution)  
**Classification:** Institutional Quantitative Trading Platform Architecture (Unvalidated Research Baseline)  
**Author:** Quantitative Research & Architecture Team  
**Date:** October 2026  
**Repository Baseline Commit:** `60e0705a56c01a9eb1569dbc618dac55fe3289eb` (V2.2 Frozen Release)  
**Repository Branch:** `main` (Tracking `origin/main`)

---

## 1. Executive Architecture Summary

HYDRA is an institutional quantitative trading and signal intelligence system designed to ingest multi-modal physical and financial time series, construct stationarized features, infer predictive probability distributions across three market actions (`Class 0: SELL`, `Class 1: HOLD`, `Class 2: BUY`), execute asymmetric multi-agent risk consensus, simulate broker execution with rigorous transaction friction accounting, and track prospective out-of-sample forward observations within an immutable cryptographic ledger.

> [!WARNING]
> **RECONCILIATION NOTICE:** Following forensic audit reconciliation, HYDRA V2.3 is classified as **Unvalidated Research-Only**. Previous claims of "institutional certification" and specific backtest performance metrics have been formally withdrawn. The system architecture described herein represents the reconciled engineering foundation.

The platform is partitioned into two primary functional domains:
1. **The Backend Engine (`backend/`):** A high-performance Python 3.11 asynchronous micro-mesh exposing a FastAPI interface at `http://localhost:8000`, containing automated feature extraction, model inference, probability calibration, multi-agent consensus, execution paper runners, and SQLite transactional accounting.
2. **The Command Center (`frontend/`):** A Next.js 16.2 / React 19 institutional web dashboard at `http://localhost:3000` providing real-time signal monitoring, multi-model probability transparency, reliability calibration diagrams, prospective ledger audit inspection, portfolio equity tracking, and agent debate streams.

---

## 2. Directory & Component Structure

```
Stock_Indicator/
├── backend/
│   ├── artifacts/                 <- Production model weights, scalers, calibrators, SQLite ledgers
│   │   ├── frozen_strategy_manifest_v2.2.json  <- Immutable V2.2 strategy fingerprint
│   │   ├── signal_ledger.db       <- Authoritative SQLite execution & prospective ledger
│   │   ├── lgbm_agent.joblib      <- Trained LightGBM booster
│   │   ├── xgb_ensemble.json      <- Trained XGBoost booster
│   │   ├── latest_scaler.joblib   <- Fitted StandardScaler on 2016-2024 train partition
│   │   ├── model_calibrator.joblib<- Calibration artifact (raw tree pass-through, DL Platt sigmoid)
│   │   ├── meta_ensemble.joblib   <- Stacking meta-classifier
│   │   ├── dqn_model.pth          <- PyTorch Deep Q-Network weights (Quarantined)
│   │   └── latest_fusion_weights.weights.h5 <- Keras DL Fusion weights (Quarantined)
│   ├── configs/                   <- Configuration files, parameters, and feature definitions
│   │   ├── kept_features.json     <- 27 stationarized feature columns (canonical order)
│   │   ├── model_accuracies.json  <- Historical validation accuracies and model weights
│   │   ├── model_params.yaml      <- Model architectures and inference hyperparameters
│   │   └── trading_config.yaml    <- Risk thresholds, veto logic, Kelly sizing
│   ├── reports/                   <- Static evaluation reports, diagnostic plots, prospective status
│   │   ├── v2_2_dashboard/        <- Pre-generated forensic charts and JSON data
│   │   └── v2_2_prospective/      <- Immutable prospective ledger verification reports
│   ├── scripts/                   <- Operational, research, training, and evaluation scripts
│   │   ├── ops/                   <- Ledger audit, paper runner, clean artifacts
│   │   ├── research/              <- Physical proxy research, timegan stress test
│   │   ├── training/              <- train.py, optimize.py, calibrate_models.py
│   │   └── evaluation/            <- backtest.py (chronological event-driven simulator), reconcile_prospective_ledger.py
│   ├── src/                       <- Core backend source code
│   │   ├── agents/                <- Multi-agent mesh: Alpha, Risk, Execution, Orchestrator
│   │   ├── api/                   <- FastAPI routes, endpoints, SSE streams, middleware
│   │   ├── data/                  <- Ingestion: OHLCV, SEC EDGAR, Google Trends, Weather, Port proxies
│   │   ├── execution/             <- Broker interface, paper trading, risk manager, signal ledger
│   │   ├── features/              <- Sequence builders, technical indicators, stationarization
│   │   ├── models/                <- Model wrappers: boosting, neural, RL, regime, calibration
│   │   ├── optimization/          <- Optuna objectives, search spaces, pruning rules
│   │   └── utils/                 <- GPU device utilities, caching, timezone normalization
│   └── tests/                     <- Comprehensive unit, integration, and integrity test suite (17 suites)
├── frontend/
│   ├── app/                       <- Next.js App Router (pages: /, /agents, /performance, /validation)
│   ├── components/                <- React UI components (AnalystGrid, TradeCard, PriceChart, etc.)
│   │   ├── agents/                <- Agent consensus visualization, debate feed, risk panel
│   │   ├── dashboard/             <- Signal accuracy, prospective audit, backtest explorer
│   │   └── ui/                    <- Institutional UI primitives (badges, cards, dialogs, tables)
│   ├── hooks/                     <- Custom React hooks (useAgentStream)
│   ├── lib/                       <- Client config, export utilities, formatters
│   └── types/                     <- TypeScript interfaces and API schemas
├── graphify-out/                  <- Static dependency graph analysis (local-only asset)
└── GEMINI.md                      <- Institutional coding mandates and system protocol (local-only)
```

---

## 3. End-to-End System Lifecycle & Execution Flow

```
                     +---------------------------------------+
                     |         EXTERNAL DATA SOURCES         |
                     |  - Yahoo Finance (OHLCV)              |
                     |  - SEC EDGAR (8-K / 10-Q Filings)     |
                     |  - Open-Meteo (Supply Chain Weather)  |
                     |  - Google Trends (Retail Interest)    |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |        DATA INGESTION LAYER           |
                     |  (src/data/data_loader.py)            |
                     |  - Strict chronological sorting       |
                     |  - Forward-fill & calendar align      |
                     |  - Stationarity transforms            |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |      FEATURE ENGINEERING LAYER        |
                     |  (src/api/live_inference.py)          |
                     |  - 27 Stationarized Features          |
                     |  - Rolling Z-scores, Returns, Vol     |
                     |  - Enforced kept_features.json schema |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |      FEATURE SCALING & PARTITION      |
                     |  - StandardScaler fitted on Train     |
                     |  - No lookahead from Val / OOS        |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |        ACTIVE MODEL INFERENCE         |
                     |  (src/api/asset_intelligence.py)      |
                     |  - XGBoost Ensemble (Primary Engine)  |
                     |  - LightGBM Booster (Veto Candidate)  |
                     |  - (DL Fusion & DQN Quarantined)      |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |        PROBABILITY CALIBRATION        |
                     |  (src/models/regime/calibration.py)   |
                     |  - ModelCalibrator dictionary:        |
                     |    * XGB & LGBM: Raw pass-through     |
                     |    * DL Fusion: Sigmoid Platt scaling |
                     |  - Output: P(SELL), P(HOLD), P(BUY)   |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |       MULTI-AGENT RISK CONSENSUS      |
                     |  (src/agents/orchestrator.py)         |
                     |  - Primary Engine: XGBoost (>= 0.60)  |
                     |  - Secondary Veto: LGBM (>= 0.65)     |
                     |  - Macro Regime Gate (SPY 200 SMA)    |
                     |  - Execution Cooldown (5 bars)        |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |      DECISION & SIGNAL GENERATION     |
                     |  - Action: BUY / SELL / HOLD          |
                     |  - Pending execution recorded         |
                     |  - No same-close price lookahead      |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |      EXECUTION SIMULATION & LEDGER    |
                     |  (src/execution/paper_trading.py)     |
                     |  - Causal Execution at T+1 Open       |
                     |  - Friction: 5 bps Slippage + Comm    |
                     |  - Authoritative: prospective_obs     |
                     |  - Cryptographic SHA-256 Hash Chaining|
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |        INSTITUTIONAL DASHBOARD        |
                     |  (Next.js Frontend on port 3000)      |
                     |  - Real-time SSE Agent Stream         |
                     |  - Model Governance Registry          |
                     |  - Honest unvalidated metrics         |
                     |  - Prospective Ledger Verification    |
                     +---------------------------------------+
```

---

## 4. Subsystem Deep-Dive

### 4.1 Data & Ingestion Subsystem
- **Modules:** [data_loader.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/data/data_loader.py), [edgar_client.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/data/edgar_client.py), [trends_client.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/data/trends_client.py), [weather_client.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/data/weather_client.py).
- **Temporal Firewall:** [data_firewall.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/data/data_firewall.py) strictly enforces development (<= 2024-12-31), validation (2025-01-01 to 2025-12-31), and out-of-sample prospective (>= 2026-01-01).
- **Stationarity:** Strict prohibition against raw non-stationary price series (Open, High, Low, Close, Volume). All series are converted to log-returns, percentage ATR, normalized volume ratios, and rolling volatility.

### 4.2 Feature Engineering Subsystem
- **Modules:** [live_inference.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/api/live_inference.py), [sequence_builder.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/features/sequence_builder.py).
- **Feature Set:** Canonical 27 features stored in [kept_features.json](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/configs/kept_features.json).
- **Ordering Rigidity:** Inference vectors must match the exact 27-column list in `kept_features.json`. Any discrepancy immediately triggers a feature shape error.

### 4.3 Model Architecture & Ensemble Subsystem
- **Modules:** [xgb_agent.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/boosting/xgb_agent.py), [lgbm_agent.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/boosting/lgbm_agent.py), [meta_ensemble.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/ensemble/meta_ensemble.py), [calibration.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/regime/calibration.py), [model_loader.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/model_loader.py).
- **Active Boosters:**
  - **XGBoost:** Gradient boosted tree ensemble with multi-softprob objective across 3 classes.
  - **LightGBM:** Fast histogram-based gradient booster with multiclass objective.
- **Calibrator:** Multinomial logistic calibrator ([model_calibrator.joblib](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/artifacts/model_calibrator.joblib)) trained on validation out-of-fold outputs to transform raw confidence scores into calibrated probabilities summing to 1.0.
- **Quarantined Components:**
  - **DL Fusion (`latest_fusion_weights.weights.h5`):** Quarantined due to probability collapse (>0.99 constant BUY).
  - **DQN (`dqn_model.pth`):** Quarantined from active consensus due to pseudo-RL training rewards and input distribution mismatch.

### 4.4 Multi-Agent Consensus & Governance
- **Modules:** [orchestrator.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/agents/orchestrator.py), [alpha_agent.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/agents/alpha_agent.py), [risk_agent.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/agents/risk_agent.py), [execution_agent.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/agents/execution_agent.py), [strategy_governance.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/strategy_governance.py).
- **Asymmetric Veto Protocol:**
  - If `P(BUY) > threshold` but `P(BUY) - P(SELL) < 0.15`, the trade is vetoed to prevent low-conviction entries during heightened directional conflict.
  - Risk agent evaluates 20-day historical volatility, maximum drawdown tolerance, and macro trend regime (SPY 200-day SMA).
  - 5-bar post-trade cooldown prevents churn and fee bleeding.

### 4.5 Execution, Paper Trading & Ledger Subsystem
- **Modules:** [paper_trading.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/paper_trading.py), [broker_interface.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/broker_interface.py), [signal_ledger.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/signal_ledger.py), [paper_runner.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/paper_runner.py).
- **Database Architecture:** SQLite file [signal_ledger.db](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/artifacts/signal_ledger.db).
  - Table `prospective_observations`: Immutable forward predictions recorded with SHA-256 hash chaining prior to market open.
  - Table `execution_events`: Simulated order executions with slippage (5 bps) and commission accounting.
  - Table `portfolio_states`: Mark-to-market NAV, cash balance, and exposure records.
  - Table `prospective_signals` (DEPRECATED): Retained for backward audit compatibility, clearly isolated from authoritative ledger.

### 4.6 Frontend Institutional Command Center
- **Framework:** Next.js 16.2.0 (App Router), React 19, TypeScript 5.8, Tailwind CSS 3.4.
- **Pages:**
  - `/`: Main command center with interactive price chart, signal intelligence, technical snapshot, trade card, and model reliability breakdown.
  - `/agents`: Multi-agent mesh monitoring, analyst consensus cards, risk execution metrics, and real-time SSE debate feed.
  - `/performance`: Historical backtesting metrics, cumulative return curves, drawdown profiles, and benchmark comparisons.
  - `/validation`: Prospective paper trading ledger audit, SHA-256 manifest verification, and forward prediction verification.

---

## 5. External & Operational Dependencies

1. **Market Data Provider:** Yahoo Finance via `yfinance` library. Requires internet egress. Cached locally in memory / disk cache to prevent rate-limiting.
2. **PyTorch & TensorFlow Dual Runtime:** The backend operates both PyTorch (for DQN/TCN/PatchTST) and TensorFlow/Keras (for DL Fusion & TFT). GPU detection via CUDA is supported with CPU fallbacks.
3. **Optuna:** Storage-backed Bayesian optimization framework using SQLite or in-memory studies for hyperparameter tuning.
4. **FastAPI & Uvicorn:** ASGI web server supporting REST endpoints and asynchronous SSE streaming.

---

## 6. Identified Architectural Problems & System Weaknesses

1. **Path Fragility in Auxiliary Scripts:** Multiple scripts in `backend/scripts/` rely on relative paths (e.g. `open('configs/model_params.yaml')`), failing when invoked from repository root.
2. **Broken Evaluation Script (`backend/scripts/evaluation/backtest.py`):** Attempts to load non-existent `xgb_calibrator.joblib` and `lgbm_calibrator.joblib` instead of the canonical `model_calibrator.joblib`.
3. **Data Leakage in Optimization Pipeline (`backend/scripts/training/optimize.py`):** Fits `StandardScaler` on the entire feature dataset prior to splitting into train/validation folds.
4. **Flawed Multi-Class Optimization Objective (`backend/scripts/training/optimize_models.py`):** Calculates binary ROC-AUC on Class 2 only for 3-class models, completely ignoring Class 0 and Class 1, and passes invalid `scale_pos_weight` to multiclass XGBoost.
5. **Unit Inconsistency in Model Accuracies:** `model_accuracies.json` stores decimal values (`0.4968`), but fallbacks in `model_loader.py` and `api.py` store percentage floats (`52.1`).
6. **DL Fusion Architecture Over-Parameterization:** 4-branch multi-modal neural network exhibits gradient saturation and complete collapse to Class 2 BUY predictions.
7. **DQN Training As Supervised Approximation:** The reinforcement learning agent was trained using a static reward proxy matching the triple-barrier label rather than dynamic sequential portfolio rewards, introducing distributional shift.

---

## 7. Architectural Target for HYDRA V2.3

The V2.3 architecture must enforce:
- Robust absolute path resolution across all submodules and scripts.
- Unification of the backtesting engine to use the single authoritative `model_calibrator.joblib` and execution accounting rules.
- Per-fold feature scaling inside Optuna optimization loops to eliminate any information leakage.
- Multi-class Macro-F1 / Multi-class Brier score optimization objectives.
- Explicit pruning or architectural isolation of collapsed DL/RL models, leaving a clean, highly reliable, well-calibrated boosting ensemble with multi-agent consensus governance.
- Complete data integrity and parity between backend execution math and frontend dashboard presentation.
