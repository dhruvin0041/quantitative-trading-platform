# HYDRA MODEL-BY-MODEL FORENSIC AUDIT
**Document Version:** 1.0.0  
**Classification:** Individual Model Architecture, Performance & Empirical Value Audit  
**Repository Branch:** `hydra-v2.3`  
**Date:** October 2026

---

## 1. Executive Summary

This audit evaluates every machine learning and quantitative predictive model implemented in the HYDRA repository. Each model is assessed for theoretical soundness, training stability, calibration, empirical value, and operational status in production.

### Model Status Overview
| Model | Type | Architecture / Implementation | Status in V2.2 | Recommended Status in V2.3 |
|---|---|---|---|---|
| **XGBoost** | Gradient Boosted Trees | Multi-softprob 3-class classifier | **ACTIVE** | **ACTIVE (Primary Engine)** |
| **LightGBM** | Histogram Boosted Trees | Multiclass objective with early stopping | **ACTIVE** | **ACTIVE (Primary Engine)** |
| **DL Fusion** | Multi-Branch Neural Net | Keras CNN + LSTM + Transformer + Tabular | **QUARANTINED** | **DEPRECATED / RETIRED** |
| **DQN** | Deep Reinforcement Learning | PyTorch Deep Q-Network with replay buffer | **ACTIVE (Mesh Veto)** | **SUPPRESSED IN FROZEN INFERENCE** |
| **Meta-Ensemble** | Stacking Classifier | Ridge / Logistic Regression on base models | **OPTIONAL / FALLBACK** | **STREAMLINED / CONSERVATIVE** |
| **TFT** | Temporal Fusion Transformer | Quantile regression price boundary forecaster | **EXPERIMENTAL** | **EXPERIMENTAL (Research Only)** |

---

## 2. Detailed Audit by Model

### 2.1 XGBoost Ensemble
- **Implementation:** [backend/src/models/boosting/xgb_agent.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/boosting/xgb_agent.py)
- **Artifact:** `backend/artifacts/xgb_ensemble.json`
- **Objective:** `multi:softprob`, `num_class: 3`
- **Features:** Canonical 27 stationarized features from `kept_features.json`.
- **Key Hyperparameters:**
  - `max_depth`: 4–6
  - `learning_rate`: 0.03–0.05
  - `n_estimators`: 250 (with early stopping at 30 rounds)
  - `subsample`: 0.8, `colsample_bytree`: 0.8
  - `reg_alpha` (L1): 0.1, `reg_lambda` (L2): 1.0
- **Validation Metrics (2025 Historical Out-of-Sample):**
  - Balanced Accuracy: 49.68%
  - Multiclass Brier Score: 0.5841
  - Log Loss: 0.9823
- **Strengths:**
  - Highly robust against over-fitting on noisy financial features.
  - Generates monotonic feature importances (e.g. `volatility_20d`, `returns_5d`, `bb_position`).
  - Serializes natively to JSON, ensuring cross-platform reproducibility without pickle vulnerabilities.
- **Weaknesses:**
  - Tendency to produce uncalibrated probabilities in tail distributions (addressed via downstream calibration).

---

### 2.2 LightGBM Booster
- **Implementation:** [backend/src/models/boosting/lgbm_agent.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/boosting/lgbm_agent.py)
- **Artifact:** `backend/artifacts/lgbm_agent.joblib`
- **Objective:** `multiclass`, `num_class: 3`
- **Features:** Identical 27 stationarized features.
- **Key Hyperparameters:**
  - `num_leaves`: 31
  - `learning_rate`: 0.03
  - `n_estimators`: 200
  - `min_child_samples`: 20
  - `subsample`: 0.85, `colsample_bytree`: 0.80
- **Validation Metrics (2025 Historical Out-of-Sample):**
  - Balanced Accuracy: 45.59%
  - Multiclass Brier Score: 0.6120
  - Log Loss: 1.0214
- **Strengths:**
  - Extremely fast inference latency (< 5ms).
  - Excellent complement to XGBoost; decision paths exhibit useful orthogonal diversity.
- **Weaknesses:**
  - Sensitive to small sample regimes; requires conservative `min_child_samples` to avoid leaf-level overfitting.

---

### 2.3 Deep Learning Multi-Branch Fusion Network
- **Implementation:** [backend/src/models/neural/fusion_network.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/neural/fusion_network.py)
- **Artifact:** `backend/artifacts/latest_fusion_weights.weights.h5`
- **Architecture:**
  - Branch 1: 1D Temporal CNN (Kernel sizes 3, 5 with batch norm).
  - Branch 2: Bidirectional LSTM (64 units) with dropout (0.3).
  - Branch 3: Multi-Head Self-Attention (4 heads, key dimension 16).
  - Branch 4: Dense Tabular projection.
  - Fusion Layer: Concatenation -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(3, Softmax).
- **Audit Findings & Model Collapse:**
  - The model collapsed into predicting Class 2 (BUY) with >0.99 confidence on virtually all inputs.
  - In `prospective_operations_status.json`, raw inference output for AAPL on 2026-10-01 was:
    `P(SELL) = 0.0012, P(HOLD) = 0.0027, P(BUY) = 0.9961`.
  - Cause: Gradient saturation during training due to over-parameterization relative to sample size, combined with severe triple-barrier class imbalance.
- **Verdict:** **DEPRECATED.** Deep learning adds zero positive out-of-sample alpha in this tabular time series configuration and introduces severe fragility. Correctly quarantined in V2.2 and recommended for formal retirement in V2.3.

---

### 2.4 Deep Q-Network (DQN) Reinforcement Learning
- **Implementation:** [backend/src/models/rl/dqn_agent.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/rl/dqn_agent.py)
- **Artifact:** `backend/artifacts/dqn_model.pth`
- **Architecture:** PyTorch MLP: Input(30) -> Linear(128) -> LayerNorm -> ReLU -> Linear(64) -> ReLU -> Linear(3).
- **Audit Findings:**
  1. **Reward Formulation Flaw:** In `train.py` line 288:
     `reward = 1.0 if action == target_sig else (-1.0 if action != 1 else 0.0)`.
     This is not true reinforcement learning based on sequential portfolio returns or Sharpe optimization; it is an inefficient supervised classification approximation using Q-learning.
  2. **State Vector Train-Inference Mismatch:** The training state vector included predictions from the collapsed DL model. In production, DL is quarantined (replaced with dummy `[0, 1, 0]`), meaning the DQN policy network receives out-of-distribution inputs.
  3. **Role & Baseline Freeze Reality:** In multi-agent mesh intelligence (`asset_intelligence.py:104`, `consensus_engine.py:236`), DQN remains marked as `ACTIVE` with `SECONDARY_VETO` authority (threshold 0.65). In frozen baseline `inference_service.py:537`, secondary vetoes are effectively bypassed by default (`veto_threshold=1.01`). DQN was preserved rather than modified under the anti-overfitting baseline freeze.
- **Verdict:** **EMPIRICALLY DEFECTIVE / OPERATIONALLY SUPPRESSED.** While retained in baseline mesh code, DQN action outputs suffer from static-label approximation and out-of-distribution inputs; live deployment remains strictly prohibited.

---

### 2.5 Probability Calibrator & Meta-Ensemble
- **Implementation:** [backend/src/models/regime/calibration.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/regime/calibration.py) & [backend/src/models/ensemble/meta_ensemble.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/ensemble/meta_ensemble.py)
- **Artifacts:** `backend/artifacts/model_calibrator.joblib` and `backend/artifacts/meta_ensemble.joblib`
- **Actual Calibrator Artifact Structure:**
  Inspection of `model_calibrator.joblib` confirms it is a dictionary defining:
  ```python
  {
      'DL_FUSION': {'method': 'sigmoid', 'models': {0: LogReg, 1: LogReg, 2: LogReg}},
      'XGB': {'method': 'raw', 'models': {}},
      'LGBM': {'method': 'raw', 'models': {}}
  }
  ```
  - `XGB` and `LGBM` are configured for **raw pass-through**, while `DL_FUSION` uses per-class Sigmoid Platt scaling.
  - Previous documentation claiming a unified "3-class multinomial logistic regression calibrator" was inaccurate.
- **Verdict:** Accurately documented in V2.3. Tree models operate with raw calibrated probabilities from their boosting objectives; DL Fusion is quarantined.

---

## 3. Comparative Model Performance Status

> [!WARNING]
> **RECONCILIATION WITHDRAWAL:** The previously published performance metrics (51.84% accuracy, Brier 0.5512, Log Loss 0.9340) derived from flawed single-ticker accumulation and single 80/20 train/test splits are **formally withdrawn**. The confusion matrix previously reported was mathematically incompatible with the published metrics.

| Model / Configuration | Historical Estimate (Withdrawn) | Reconciliation Status | Operational Action in V2.3 |
|---|---|---|---|
| **XGBoost (Standalone)** | 49.68% | Unvalidated Baseline | **ACTIVE (Primary Engine, 0.60 threshold)** |
| **LightGBM (Standalone)** | 45.59% | Unvalidated Baseline | **ACTIVE (Veto Candidate, 0.65 threshold)** |
| **Active Consensus** | *51.84% (Withdrawn)* | Unvalidated Research-Only | Production Policy Defined |
| **Deep Learning Fusion** | *Collapsed (>0.99 BUY)* | Degenerate Local Optimum | **QUARANTINED** |
| **Deep Q-Network (RL)** | *Degraded* | Mismatched Reward / State | **QUARANTINED** |

**Key Research Finding:** Forcing over-parameterized neural networks and pseudo-RL approximations into the tabular pipeline degraded generalization. In V2.3, the system isolates to the robust tree models (XGBoost and LightGBM) under strict multi-agent governance, pending rigorous walk-forward empirical validation.
