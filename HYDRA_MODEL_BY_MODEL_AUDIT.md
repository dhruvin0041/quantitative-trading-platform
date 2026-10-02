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
| **DQN** | Deep Reinforcement Learning | PyTorch Deep Q-Network with replay buffer | **QUARANTINED** | **DEPRECATED / ISOLATED** |
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
- **Verdict:** **DEPRECATED / ISOLATED.** Reinforcement learning under this configuration degrades consensus performance. It was properly excluded from active voting in V2.2 and should remain quarantined.

---

### 2.5 Probability Calibrator & Meta-Ensemble
- **Implementation:** [backend/src/models/regime/calibration.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/regime/calibration.py) & [backend/src/models/ensemble/meta_ensemble.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/ensemble/meta_ensemble.py)
- **Artifacts:** `backend/artifacts/model_calibrator.joblib` and `backend/artifacts/meta_ensemble.joblib`
- **Methodology:** Multinomial Logistic Regression calibrated on out-of-fold validation predictions.
- **Output:** Calibrated 3-class posterior distribution:
  $$\sum_{c=0}^2 P(c) = 1.0, \quad P(c) \in [0, 1]$$
- **Audit Verification:** Output probabilities are rigorously checked for numerical bounds and unit-sum property.
- **Verdict:** **HIGH VALUE.** Calibration transforms raw, overconfident tree scores into reliable risk probabilities, enabling institutional conviction filtering.

---

## 3. Comparative Model Performance Summary

| Model / Configuration | 2025 Out-of-Sample Balanced Accuracy | Multiclass Brier Score | Log Loss | Status / Action |
|---|---|---|---|---|
| **XGBoost (Standalone)** | 49.68% | 0.5841 | 0.9823 | Active |
| **LightGBM (Standalone)** | 45.59% | 0.6120 | 1.0214 | Active |
| **Active Consensus (XGB + LGBM + Calibrator)** | **51.84%** | **0.5512** | **0.9340** | **Production Baseline** |
| **Full Ensemble (inc. DL & DQN)** | 37.10% | 0.7420 | 1.4890 | Degraded (Collapsed) |
| **Buy & Hold Baseline** | 33.33% | N/A | N/A | Benchmark |

**Key Research Finding:** The pruned, well-calibrated ensemble combining XGBoost and LightGBM with asymmetric risk veto delivers the highest out-of-sample predictive accuracy and lowest calibration error. Forcing over-parameterized neural networks and flawed RL approximations into the ensemble degrades predictive alpha by over 14 percentage points.
