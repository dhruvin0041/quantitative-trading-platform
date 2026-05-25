import optuna
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
from src.optimization.objective_functions import sharpe_objective
from src.optimization.search_spaces import get_lgbm_search_space, get_xgb_search_space

class PurgedTimeSeriesSplit:
    """
    Walk-forward validation that purges embargo periods to prevent lookahead bias
    commonly found in financial time series data.
    """
    def __init__(self, n_splits=5, purge_gap=5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        for train_idx, test_idx in self.tscv.split(X):
            # Apply purge gap
            if len(train_idx) > self.purge_gap:
                train_idx = train_idx[:-self.purge_gap]
                yield train_idx, test_idx

def optimize_lgbm(X, y, forward_returns, n_trials=50):
    """
    Optimizes LightGBM using Optuna TPE.
    Objective: Maximize Sharpe Ratio.
    """
    def objective(trial):
        params = get_lgbm_search_space(trial)
        params['objective'] = 'multiclass'
        params['num_class'] = 3
        params['verbose'] = -1
        params['random_state'] = 42

        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap=10)
        sharpe_scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            val_returns = forward_returns[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predict probabilities
            y_pred_probs = model.predict_proba(X_val)
            
            # Calculate Sharpe Ratio on validation set
            sharpe = sharpe_objective(y_val, y_pred_probs, val_returns, threshold=0.55)
            sharpe_scores.append(sharpe)
            
        return np.mean(sharpe_scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
    return study.best_trial.params

def optimize_xgb(X, y, forward_returns, n_trials=50):
    """
    Optimizes XGBoost using Optuna TPE.
    Objective: Maximize Sharpe Ratio.
    """
    def objective(trial):
        params = get_xgb_search_space(trial)
        params['objective'] = 'multi:softprob'
        params['num_class'] = 3
        params['n_jobs'] = -1
        params['random_state'] = 42

        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap=10)
        sharpe_scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            val_returns = forward_returns[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predict probabilities
            y_pred_probs = model.predict_proba(X_val)
            
            # Calculate Sharpe Ratio on validation set
            sharpe = sharpe_objective(y_val, y_pred_probs, val_returns, threshold=0.55)
            sharpe_scores.append(sharpe)
            
        return np.mean(sharpe_scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
    return study.best_trial.params
