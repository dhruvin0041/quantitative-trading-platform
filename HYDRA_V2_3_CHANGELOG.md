# HYDRA V2.3 ENGINEERING CHANGELOG & MIGRATION SPECIFICATION
**Version:** HYDRA V2.3 (Reconciled Tooling & Research Baseline)  
**Code-Freeze Baseline Commit:** `e687e2321da9159dca2b10c6744f2166fb44a506` (V2.2 Frozen Strategy Manifest)  
**Evaluation Reporting Commit:** `60e0705a56c01a9eb1569dbc618dac55fe3289eb` (V2.2 Evaluation Documentation Update)  
**Repository Branch:** `main` (Tracking `origin/main`)  
**Classification:** Institutional Platform Engineering Overhaul & Quantitative Tooling Reclassification  
**Date:** October 2026

---

## 1. Overview & Baseline Preservation

HYDRA V2.3 delivers critical engineering fixes across model evaluation, cross-validation, and risk consensus. Following a comprehensive reconciliation audit, the release has been formally reclassified as **Unvalidated Research-Only**. All previously reported positive backtest figures have been formally withdrawn.

> [!WARNING]
> **RECONCILIATION VERDICT & RECLASSIFICATION:**
> - **Release Status:** Release Candidate Rejected / Unvalidated Research Baseline.
> - **Performance Status:** All reported V2.3 backtest metrics (+6.03% return, 1.53 Sharpe, 59.03% win rate) are **formally withdrawn** due to historical non-chronological accumulation, single-sided friction, and uncalibrated tree scores.
> - **Deployment Status:** Live capital deployment remains strictly prohibited until prospective out-of-sample evidence is accumulated.

### Absolute Preservation Verification
- **V2.2 Frozen Manifest:** `backend/artifacts/frozen_strategy_manifest_v2.2.json` (SHA-256: `e34d9d472505d6b2d7a79924cbdf1fc6c0aa2346f67c72f9c0afd94b5352b195`) remains 100% untouched and byte-identical across all 8 model/scaler artifacts, 3 configuration files, and 7 frozen source files.
- **Authoritative Prospective Ledger:** `backend/artifacts/signal_ledger.db` preserves the immutable forward observation `PROP-AAPL-20261001-DEE62FA6` without modification.
- **Repository Branch:** All work committed and pushed directly to `main`.

---

## 2. Inventory of Issues Resolved in V2.3

### 2.1 Critical Engineering Remediations

#### [CRIT-01] Overhauled Evaluation & Backtesting Engine (`backend/scripts/evaluation/backtest.py`)
- **Original Behavior:** Script crashed on startup due to missing calibrator files (`xgb_calibrator.joblib`, `lgbm_calibrator.joblib`), accumulated trades ticker-by-ticker non-chronologically, and applied one-sided slippage without broker commissions.
- **V2.3 Behavior:** Fully rewritten into a true **chronological multi-asset event-driven portfolio simulator**:
  - Dynamically loads authoritative `model_calibrator.joblib`.
  - Simulates portfolio calendar sequentially across all assets simultaneously.
  - Enforces explicit cash and position accounting with equal-weight cash allocation.
  - Implements dynamic ATR triple barriers ($1.5 \times \text{ATR}$ TP, $2.0 \times \text{ATR}$ SL, 15-day horizon).
  - Models realistic two-sided friction: 5 bps adverse slippage on both entry and exit, plus $0.005/share brokerage commission ($1.00 min).
  - Computes Calmar ratio correctly as annualized return divided by maximum drawdown.
- **Status:** Evaluated and running cleanly; previous performance claims withdrawn.

#### [CRIT-02] Eliminated Data Leakage in Hyperparameter Optimization (`backend/scripts/training/optimize.py`)
- **Original Behavior:** Scaled features globally using `StandardScaler.fit_transform()` on the entire dataset prior to splitting into train/test sets, leaking test set distribution moments into Optuna trials.
- **V2.3 Behavior:** Refactored scaler fitting to occur strictly on training observations `df_ready.iloc[:split_idx]` before transforming the sequence partitions. Switched trial objective to Multiclass Macro-F1.

#### [CRIT-03] Fixed Flawed Multi-Class Optimization Objective & Cross-Validation (`backend/scripts/training/optimize_models.py`)
- **Original Behavior:** Evaluated 3-class models using binary ROC-AUC on Class 2 (BUY) only, passed invalid `scale_pos_weight` to multiclass XGBoost, and used standard unpurged `TimeSeriesSplit`.
- **V2.3 Behavior:**
  - Replaced binary AUC with Multiclass Macro-F1 across all 3 classes (`f1_score(average="macro")`).
  - Removed invalid `scale_pos_weight` parameter.
  - Implemented `purged_walk_forward_cv` with 15-bar post-training embargo and fold-level `StandardScaler` fitting (zero scaling leakage across folds).

#### [CRIT-04] Standardized Model Accuracy Configurations & Aliases (`backend/configs/model_accuracies.json`)
- **Original Behavior:** Stored decimal fractions (`0.4968`) while fallbacks in `model_loader.py` and `api.py` stored percentage values (`52.1`), and key names differed (`xgb` vs `xgb_accuracy`).
- **V2.3 Behavior:** Standardized `model_accuracies.json` with both canonical short keys and `_accuracy` aliases in uniform decimal format (`0.0 to 1.0`), and updated `model_loader.py` and `consensus_engine.py` to support all key variants.

#### [CRIT-05] Resolved Path Fragility in Auxiliary Scripts
- **Original Behavior:** Scripts in `backend/scripts/` assumed `cwd` was `backend/` and failed when executed from project root or external runners due to relative `configs/` or `artifacts/` paths.
- **V2.3 Behavior:** Anchored paths using `Path(__file__).resolve()` across operational scripts and added path resolution fallbacks in `backtest.py`, `optimize.py`, `optimize_models.py`, `model_loader.py`, and `consensus_engine.py`.

#### [CRIT-06] Formally Quarantined Degenerate Models (`DL_FUSION` & `DQN_AGENT`)
- **Original Behavior:** DL Fusion network collapsed into constant >0.99 BUY predictions; DQN agent was trained with mismatched inputs and static reward approximations. Despite changelog claims, DQN remained active with a secondary role in `asset_intelligence.py`.
- **V2.3 Behavior:**
  - `asset_intelligence.py`: Updated `DQN_AGENT` to `ModelRole.QUARANTINED` and `ModelStatus.QUARANTINED`.
  - `model_loader.py`: Updated `_load_dqn` and `_load_lstm` to bypass weight loading when marked as quarantined.
  - `consensus_engine.py`: Excluded quarantined models from veto authority (`veto_candidates = ["LGBM_AGENT"]`).

#### [CRIT-07] Harmonized Authoritative Production Inference Policy (`backend/src/execution/inference_service.py`)
- **Original Behavior:** Production inference defaulted to `veto_threshold=1.01`, disabling secondary veto logic.
- **V2.3 Behavior:** Unified production consensus configuration to Primary `XGB_AGENT` (threshold 0.60) and Secondary `LGBM_AGENT` veto authority (`veto_threshold=0.65`).

#### [CRIT-08] Fixed Causal Execution Price Timing (`backend/src/execution/inference_service.py`)
- **Original Behavior:** Pending execution records synthesized fill prices using current bar's Close * 1.0005.
- **V2.3 Behavior:** Pending next-session orders are marked `signal_state="PENDING_EXECUTION"` without synthesizing fill prices; actual fill prices are recorded upon next-session Open bar arrival.

#### [CRIT-09] Hard-Disabled Mutable Legacy Table Cross-Logging (`backend/scripts/ops/run_prospective_validation.py`)
- **Original Behavior:** Hardened prospective validation cross-logged observations into the mutable `prospective_signals` table.
- **V2.3 Behavior:** Cross-logging to `prospective_signals` is hard-disabled. The cryptographic `prospective_observations` table is established as the sole prospective authority.

#### [CRIT-10] Removed Mock and Placeholder Metrics from Dashboard and Backend
- **Original Behavior:** `ModelReliabilityDashboard.tsx` contained hardcoded `mockModels` array (69.5% win rate, 92 reliability score). `signal_intelligence.py` returned hardcoded mock Brier (0.18) and ECE (0.05) metrics.
- **V2.3 Behavior:**
  - `ModelReliabilityDashboard.tsx`: Removed mock models; implemented **Model Governance Registry** displaying authentic active/quarantined statuses, roles, and research disclaimers.
  - `signal_intelligence.py`: Removed fake Brier/ECE numbers; returns `None` and honest `UNVALIDATED_PROVISIONAL` indicators.

---

## 3. Inventory of Modified and Created Files

### 3.1 Created Audit Deliverables (17 Core Documents - Reconciled)
1. `HYDRA_COMPLETE_ARCHITECTURE_AUDIT.md`: Complete subsystem and dataflow mapping.
2. `HYDRA_FULL_CODEBASE_AUDIT.md`: Static and dynamic codebase review, defect register.
3. `HYDRA_DATA_INTEGRITY_AUDIT.md`: Data ingestion, stationarity, 27 canonical features.
4. `HYDRA_DATA_LEAKAGE_AUDIT.md`: Temporal boundary verification, leakage vector audit.
5. `HYDRA_MODEL_BY_MODEL_AUDIT.md`: Individual architecture, training, and empirical utility review.
6. `HYDRA_FEATURE_ENGINEERING_AUDIT.md`: Canonical 27 features, formulas, importance analysis.
7. `HYDRA_CALIBRATION_AND_ENSEMBLE_AUDIT.md`: Calibrator artifact dictionary structure, raw pass-through.
8. `HYDRA_HYPERPARAMETER_OPTIMIZATION_AUDIT.md`: Optuna search spaces, walk-forward CV design.
9. `HYDRA_STRATEGY_AND_RISK_AUDIT.md`: Strategy rules, asymmetric veto, triple barriers, Kelly sizing.
10. `HYDRA_BACKTEST_AND_VALIDATION_REPORT.md`: Reconciled event-driven portfolio engine; performance withdrawn.
11. `HYDRA_PRODUCTION_INFERENCE_AUDIT.md`: Live inference lifecycle, preprocessing parity, ledger.
12. `HYDRA_FRONTEND_AND_DASHBOARD_AUDIT.md`: Next.js 16 command center, removal of mock metrics.
13. `HYDRA_SECURITY_AND_DEPENDENCY_AUDIT.md`: Cybersecurity, deserialization safety, dependencies.
14. `HYDRA_PERFORMANCE_BENCHMARK_REPORT.md`: Benchmark comparisons, Jensen's Alpha, cost sensitivity.
15. `HYDRA_TEST_AND_REGRESSION_REPORT.md`: Automated test execution report (139 runtime test cases).
16. `HYDRA_V2_3_CHANGELOG.md`: Detailed changelog and migration specification.
17. `HYDRA_FINAL_RELEASE_READINESS_REPORT.md`: Release candidate rejected; reconciliation review.

### 3.2 Modified Source & Script Files
- `backend/scripts/evaluation/backtest.py`: Full rewrite to chronological event-driven portfolio simulator.
- `backend/scripts/training/optimize.py`: Fixed global scaler leakage; macro-F1 scoring; anchored configs.
- `backend/scripts/training/optimize_models.py`: Upgraded to purged walk-forward CV, 15-bar embargo, macro-F1.
- `backend/configs/model_accuracies.json`: Standardized decimal representations and key aliases.
- `backend/src/models/model_loader.py`: Updated fallback accuracies, quarantined model bypass, path resolution.
- `backend/src/models/regime/calibration.py`: Added LogisticRegression import, documented Platt scaling mechanics.
- `backend/src/execution/asset_intelligence.py`: Quarantined DQN and DL Fusion.
- `backend/src/execution/consensus_engine.py`: Excluded DQN/DL from veto candidates.
- `backend/src/execution/inference_service.py`: Harmonized consensus thresholds (0.60/0.65); fixed causal fill prices.
- `backend/src/execution/signal_intelligence.py`: Cleaned placeholder Brier/ECE metrics.
- `backend/scripts/ops/run_prospective_validation.py`: Hard-disabled cross-logging to mutable table.
- `backend/tests/test_api_portfolio_status.py`: Added dynamic sys.path insertion for test discovery.
- `frontend/components/dashboard/ModelReliabilityDashboard.tsx`: Removed mock metrics; Model Governance Registry.

---

## 4. Operational Migration Guide

1. **Verify Main Branch:**
   ```bash
   git branch
   # Confirms branch: main
   ```
2. **Execute Full Test Suite Verification:**
   ```powershell
   cd backend
   $env:PYTHONPATH=".;src"
   .\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
   ```
   *Expected outcome: 139 passed tests in ~19.8 seconds.*
3. **Execute Historical Chronological Backtest (Research Tooling):**
   ```powershell
   .\venv\Scripts\python.exe scripts/evaluation/backtest.py --period 2y
   ```
4. **Launch Frontend Command Center:**
   ```bash
   cd frontend
   npm run dev
   ```
   *Verify institutional terminal at http://localhost:3000.*
