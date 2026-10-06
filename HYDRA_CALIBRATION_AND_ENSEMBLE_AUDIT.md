# HYDRA MODEL CALIBRATION & ENSEMBLE OPTIMIZATION AUDIT
**Document Version:** 2.0.0 (Post-Reconciliation Audit)  
**Classification:** Multiclass Probability Calibration & Stacking Ensemble Forensic Audit  
**Repository Branch:** `main` (Preserving V2.2 Frozen Release)  
**Date:** October 2026

---

## 1. Executive Summary & Reconciliation Notice

> [!IMPORTANT]
> **RECONCILIATION CORRECTION:**  
> Previous versions of this document incorrectly claimed that production models used a "multinomial logistic calibration matrix." Independent forensic inspection of the frozen artifact [backend/artifacts/model_calibrator.joblib](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/artifacts/model_calibrator.joblib) and source code in [backend/src/models/regime/calibration.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/regime/calibration.py) demonstrates the actual architecture:
> 1. **Tree Models (XGBoost & LightGBM):** Configured as **raw pass-through** (`method: "raw"`, empty models dictionary). Softmax probabilities from boosting trees are uncalibrated in the baseline release.
> 2. **Deep Learning Fusion Network:** Utilizes **per-class sigmoid (Platt) scaling** (`method: "sigmoid"`), using three independent binary `LogisticRegression` models, followed by row re-normalization.
> 3. **ModelCalibrator Implementation:** Implements per-class Isotonic Regression and per-class Sigmoid Platt scaling. It does NOT implement multinomial logistic regression.

---

## 2. Actual Calibration Architecture & Artifact Contents

Programmatic inspection of the frozen production artifact `model_calibrator.joblib` reveals:

```python
{
    "DL_FUSION": {
        "method": "sigmoid",
        "models": {
            0: LogisticRegression(C=1.0),  # Binary P(SELL) calibrator
            1: LogisticRegression(C=1.0),  # Binary P(HOLD) calibrator
            2: LogisticRegression(C=1.0)   # Binary P(BUY) calibrator
        }
    },
    "XGB": {
        "method": "raw",
        "models": {}  # Pass-through uncalibrated
    },
    "LGBM": {
        "method": "raw",
        "models": {}  # Pass-through uncalibrated
    }
}
```

### 2.1 Mathematical Calibration Mechanics in `ModelCalibrator`

For a 3-class prediction vector $y_{\text{prob}} = [P_0, P_1, P_2]$:
1. **Raw Pass-Through (`method: raw`):**
   $$P_{\text{cal}} = y_{\text{prob}}$$
2. **Per-Class Sigmoid Platt Scaling (`method: sigmoid`):**
   $$P_c^* = \sigma(w_c P_c + b_c) = \frac{1}{1 + \exp(-(w_c P_c + b_c))}, \quad c \in \{0, 1, 2\}$$
   Followed by row normalization to ensure the vector sums to 1:
   $$P_c^{\text{cal}} = \frac{P_c^*}{\sum_{j=0}^2 P_j^*}$$
3. **Per-Class Isotonic Regression (`method: isotonic`):**
   $$P_c^* = f_c^{\text{iso}}(P_c), \quad c \in \{0, 1, 2\}$$
   where $f_c^{\text{iso}}$ is a piecewise constant non-decreasing step function, followed by row normalization.

### 2.2 Engineering Defect Resolved
Prior to this reconciliation, `backend/src/models/regime/calibration.py` called `LogisticRegression` inside a nested conditional without an explicit module-level import. This has been resolved by importing `LogisticRegression` from `sklearn.linear_model` at the top of the file.

---

## 3. Ensemble Architecture & Model Roles

The ensemble is structured under the single authoritative production hierarchy:
* **Primary Alpha Driver:** `XGB_AGENT` (Active, conviction hurdle $\ge 0.60$).
* **Secondary Asymmetric Veto:** `LGBM_AGENT` (Active, counter-trend veto hurdle $\ge 0.65$).
* **Quarantined Architectures:**
  * `DL_FUSION`: Permanently quarantined due to class collapse (>0.99 BUY concentration).
  * `DQN_AGENT`: Permanently quarantined due to environment disconnect and uncalibrated action-preferences.
* **Forecast Oracle:** `TFT_AGENT` (Quantile trajectory & volatility oracle).

### 4. Calibration Status & Conclusion
Because XGBoost and LightGBM currently operate in raw pass-through mode in the baseline artifact, calibration metrics reported in prior research documents represent experimental evaluations rather than active production calibration. The release is reclassified as **Unvalidated Research-Only**.
