# HYDRA HYPERPARAMETER OPTIMIZATION AUDIT
**Document Version:** 1.1.0  
**Classification:** Bayesian Optimization, Search Space & Objective Function Audit (Reconciled Research Baseline)  
**Repository Branch:** `main`  
**Date:** October 2026

---

## 1. Executive Summary

Hyperparameter optimization in quantitative trading is a double-edged sword: properly executed, it finds hyperparameter regimes that maximize out-of-sample generalization; poorly executed, it acts as an aggressive over-fitting engine that memorizes noise and inflates historical backtest metrics.

This audit evaluates the Bayesian optimization infrastructure implemented in [backend/scripts/training/optimize.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/scripts/training/optimize.py) and [backend/scripts/training/optimize_models.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/scripts/training/optimize_models.py), powered by Optuna.

> [!WARNING]
> **RECONCILIATION NOTICE:** In accordance with the Reconciliation Verdict, all claims of "certified" optimization performance are withdrawn. Optimization tools are classified as **Research-Only Utilities**.

### Key Audit Findings & Remediations
1. **Critical Flaw 1 (Leakage):** `optimize.py` previously fit `StandardScaler` globally on the entire dataset prior to splitting into train and test sets. **Remediated:** Scaler fitting moved strictly to the pre-split training slice.
2. **Critical Flaw 2 (Flawed Objective):** `optimize_models.py` previously used binary ROC-AUC on Class 2 only and passed `scale_pos_weight` to multiclass XGBoost. **Remediated:** Replaced with Multiclass Macro-F1 across all 3 classes, and removed `scale_pos_weight`.
3. **Critical Flaw 3 (Unpurged CV):** Standard `TimeSeriesSplit` allowed forward label overlap near fold boundaries. **Remediated:** Implemented `purged_walk_forward_cv` with 15-bar post-training embargo and fold-level scaling.
4. **Study Scoring Metric:** Updated `optimize.py` trial objective from raw accuracy to Multiclass Macro-F1 (`f1_score(average="macro")`).

---

## 2. In-Depth Analysis of Identified Flaws

### 2.1 Global Scaler Fitting Leakage in `optimize.py`
In [backend/scripts/training/optimize.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/scripts/training/optimize.py#L99-L108):
```python
# Flawed implementation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_ready[FEATURE_COLUMNS])
...
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)
```
- **Why this is defective:** The validation set's mean $\mu_{\text{val}}$ and variance $\sigma^2_{\text{val}}$ are embedded within the scaled features $X_{\text{train}}$. In time-series regimes undergoing volatility expansion, this artificially dampens extreme validation values, presenting the optimizer with an artificially predictable test set.
- **Remediation in V2.3:**
```python
# Correct implementation
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df_ready[FEATURE_COLUMNS], y, test_size=0.2, shuffle=False
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
```

---

### 2.2 Flawed Multiclass Optimization Objective in `optimize_models.py`
In [backend/scripts/training/optimize_models.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/scripts/training/optimize_models.py#L73-L76):
```python
# Flawed objective: Binary ROC-AUC on Class 2 only
prob = model.predict_proba(X_v)[:, 2]
val_auc = roc_auc_score((y_v == 2).astype(int), prob)
return val_auc
```
And lines 62-63:
```python
# Invalid parameter for multiclass XGBoost
scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 3.0)
params["scale_pos_weight"] = scale_pos_weight  # Ignored or errors in multiclass!
```
- **Why this is defective:**
  1. HYDRA is a 3-class trading system (`0: SELL`, `1: HOLD`, `2: BUY`). Evaluating solely on Class 2 binary AUC rewards models that classify every SELL as a HOLD, as long as BUY rank ordering is partially preserved.
  2. In XGBoost, `scale_pos_weight` is only supported for `binary:logistic`. Passing it to `multi:softprob` is invalid and creates warnings or undefined weighting.
- **Remediation in V2.3:**
  1. Optimize for **Multiclass Macro F1** or **Multiclass Log Loss / Brier Score**:
     $$\text{Macro F1} = \frac{1}{3} \sum_{c=0}^2 F1(c)$$
     $$\text{Multiclass Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{c=0}^2 (P_{ic} - y_{ic})^2$$
  2. Remove `scale_pos_weight` and utilize `class_weight='balanced'` in LightGBM or custom sample weights derived from class frequencies in XGBoost.

---

## 3. Search Space Specification & Regularization

The recommended search spaces for tree boosting architectures:

### 3.1 XGBoost Multi-Class Search Space
```python
params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": trial.suggest_int("max_depth", 3, 6),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
    "n_estimators": trial.suggest_int("n_estimators", 100, 300),
    "subsample": trial.suggest_float("subsample", 0.65, 0.90),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.90),
    "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    "random_state": 42,
    "n_jobs": -1
}
```

### 3.2 LightGBM Multi-Class Search Space
```python
params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
    "max_depth": trial.suggest_int("max_depth", 3, 7),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
    "n_estimators": trial.suggest_int("n_estimators", 100, 300),
    "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
    "subsample": trial.suggest_float("subsample", 0.65, 0.90),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.90),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}
```

---

## 4. Validation Framework & Cross-Validation Design

To prevent overfitting to a single validation period, the optimization engine in V2.3 must use **Purged Walk-Forward Time-Series Split**:
- Split development data (2016–2024) into 4 expanding walk-forward folds:
  - Fold 1: Train 2016–2018 | Val 2019 (Purge 15 bars)
  - Fold 2: Train 2016–2020 | Val 2021 (Purge 15 bars)
  - Fold 3: Train 2016–2022 | Val 2023 (Purge 15 bars)
  - Fold 4: Train 2016–2023 | Val 2024 (Purge 15 bars)
- Score each trial by the mean validation Macro-F1 across all 4 walk-forward folds.
- The 2025 validation set remains completely quarantined during hyperparameter search.

---

## 5. Summary of Remediations Implemented in V2.3

1. **`backend/scripts/training/optimize.py`:**
   - Refactored `StandardScaler` to fit strictly on `X_train_raw` pre-split.
   - Replaced raw validation accuracy with Multiclass Macro-F1 (`f1_score(y_test, y_pred, average="macro")`).
   - Anchored configuration paths with `Path(__file__).resolve()`.
2. **`backend/scripts/training/optimize_models.py`:**
   - Replaced binary Class 2 AUC with Multiclass Macro-F1 across all 3 classes (`f1_score(y_val, val_preds, average="macro")`).
   - Removed invalid `scale_pos_weight` parameter for multiclass XGBoost.
   - Implemented `purged_walk_forward_cv` enforcing a 15-bar post-training embargo and fold-level `StandardScaler` fitting (zero scaling leakage across folds).
   - Applied across all tree model objectives: XGBoost, LightGBM, CatBoost, and RandomForest.
