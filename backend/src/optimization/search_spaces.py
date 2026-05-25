def get_lgbm_search_space(trial):
    """
    Defines the hyperparameter search space for LightGBM.
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

def get_xgb_search_space(trial):
    """
    Defines the hyperparameter search space for XGBoost.
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
    }

def get_fusion_nn_search_space(trial):
    """
    Defines the hyperparameter search space for the Keras Fusion Network.
    """
    return {
        'lstm_units_1': trial.suggest_categorical('lstm_units_1', [32, 64, 128]),
        'lstm_units_2': trial.suggest_categorical('lstm_units_2', [32, 64, 128]),
        'lstm_dropout_1': trial.suggest_float('lstm_dropout_1', 0.1, 0.5),
        
        'cnn_filters_1': trial.suggest_categorical('cnn_filters_1', [16, 32, 64]),
        'cnn_kernel': trial.suggest_categorical('cnn_kernel', [3, 5, 7]),
        
        'trans_head_size': trial.suggest_categorical('trans_head_size', [32, 64, 128]),
        'trans_heads': trial.suggest_categorical('trans_heads', [2, 4, 8]),
        
        'dense_units_1': trial.suggest_categorical('dense_units_1', [64, 128, 256]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
    }

def get_ensemble_search_space(trial):
    """
    Defines the hyperparameter search space for the ElasticNet Meta-Learner.
    """
    return {
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        'C': trial.suggest_float('C', 0.01, 10.0, log=True),
    }
