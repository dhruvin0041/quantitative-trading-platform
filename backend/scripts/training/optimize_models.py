import optuna
import joblib
import numpy as np
import json
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# Ensure models dir exists for saving params
os.makedirs('models', exist_ok=True)

# Load training data saved by train.py
X_train = joblib.load('artifacts/X_train_tabular.joblib')
y_train = joblib.load('artifacts/y_train_sig.joblib')

# We only care about BUY vs REST or SELL vs REST? 
# The request asks for general accuracy. Multi-class AUC can be tricky.
# We'll optimize for overall multi-class performance or just class 2 (BUY) AUC as a proxy for alpha.

def objective_xgb(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 100, 600),
        'max_depth':       trial.suggest_int('max_depth', 3, 8),
        'learning_rate':   trial.suggest_float('lr', 0.01, 0.15, log=True),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':trial.suggest_float('colsample', 0.6, 1.0),
        'min_child_weight':trial.suggest_int('min_child', 1, 10),
        'reg_alpha':       trial.suggest_float('alpha', 0.0, 1.0),
        'reg_lambda':      trial.suggest_float('lambda', 0.5, 3.0),
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'n_jobs': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]
        
        model = XGBClassifier(**params)
        model.fit(X_t, y_t)
        
        # Calculate AUC for BUY signals (class 2)
        prob = model.predict_proba(X_v)[:, 2]
        # Target for binary AUC (1 if BUY, else 0)
        y_v_bin = (y_v == 2).astype(int)
        
        if len(np.unique(y_v_bin)) > 1:
            scores.append(roc_auc_score(y_v_bin, prob))
            
    return np.mean(scores) if scores else 0.5

def objective_lgbm(trial):
    params = {
        'n_estimators':   trial.suggest_int('n_estimators', 100, 600),
        'max_depth':      trial.suggest_int('max_depth', 3, 8),
        'learning_rate':  trial.suggest_float('lr', 0.01, 0.15, log=True),
        'num_leaves':     trial.suggest_int('num_leaves', 20, 100),
        'subsample':      trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':trial.suggest_float('colsample', 0.6, 1.0),
        'min_child_samples':trial.suggest_int('min_child', 5, 50),
        'reg_alpha':      trial.suggest_float('alpha', 0.0, 1.0),
        'reg_lambda':     trial.suggest_float('lambda', 0.5, 3.0),
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_t, y_t)
        
        # Calculate AUC for BUY signals (class 2)
        prob = model.predict_proba(X_v)[:, 2]
        y_v_bin = (y_v == 2).astype(int)
        
        if len(np.unique(y_v_bin)) > 1:
            scores.append(roc_auc_score(y_v_bin, prob))
            
    return np.mean(scores) if scores else 0.5

def run_optimization():
    print("Running Bayesian Optimization for XGBoost...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=50) # Reduced to 50 for time
    
    print(f"Best XGB AUC: {study_xgb.best_value}")
    with open('configs/best_xgb_params.json', 'w') as f:
        json.dump(study_xgb.best_params, f, indent=2)
        
    print("\nRunning Bayesian Optimization for LightGBM...")
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(objective_lgbm, n_trials=50)
    
    print(f"Best LGBM AUC: {study_lgbm.best_value}")
    with open('configs/best_lgbm_params.json', 'w') as f:
        json.dump(study_lgbm.best_params, f, indent=2)

if __name__ == "__main__":
    run_optimization()
