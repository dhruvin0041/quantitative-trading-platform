def get_lgbm_search_space(trial):
    """
    Defines the hyperparameter search space for LightGBM.
    """
    return {
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [100, 300, 600, 1000]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.01, 0.05, 0.1, 0.2]
        ),
        "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127, 255]),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 10, 15]),
        "min_child_samples": trial.suggest_categorical(
            "min_child_samples", [5, 20, 50, 100]
        ),
        "subsample": trial.suggest_categorical("subsample", [0.5, 0.7, 0.85, 1.0]),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.5, 0.7, 0.85, 1.0]
        ),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [1e-8, 0.1, 1.0, 10.0]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [1e-8, 0.1, 1.0, 10.0]),
    }


def get_xgb_search_space(trial):
    """
    Defines the hyperparameter search space for XGBoost.
    """
    return {
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [100, 300, 600, 1000]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.01, 0.05, 0.1, 0.2]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [3, 6, 9, 12]),
        "min_child_weight": trial.suggest_categorical(
            "min_child_weight", [1, 5, 10, 20]
        ),
        "subsample": trial.suggest_categorical("subsample", [0.5, 0.7, 0.85, 1.0]),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.5, 0.7, 0.85, 1.0]
        ),
        "gamma": trial.suggest_categorical("gamma", [1e-8, 0.1, 1.0, 5.0]),
        "lambda": trial.suggest_categorical("lambda", [1e-8, 0.1, 1.0, 10.0]),
        "alpha": trial.suggest_categorical("alpha", [1e-8, 0.1, 1.0, 10.0]),
    }


def get_fusion_nn_search_space(trial):
    """
    Defines the hyperparameter search space for the Keras Fusion Network.
    """
    return {
        "lstm_units_1": trial.suggest_categorical("lstm_units_1", [32, 64, 128, 256]),
        "lstm_units_2": trial.suggest_categorical("lstm_units_2", [32, 64, 96, 128]),
        "lstm_dropout_1": trial.suggest_categorical(
            "lstm_dropout_1", [0.1, 0.2, 0.3, 0.4]
        ),
        "cnn_filters_1": trial.suggest_categorical("cnn_filters_1", [16, 32, 48, 64]),
        "cnn_kernel": trial.suggest_categorical("cnn_kernel", [2, 3, 5, 7]),
        "trans_head_size": trial.suggest_categorical(
            "trans_head_size", [32, 64, 128, 256]
        ),
        "trans_heads": trial.suggest_categorical("trans_heads", [2, 4, 8, 16]),
        "dense_units_1": trial.suggest_categorical(
            "dense_units_1", [64, 128, 256, 512]
        ),
        "dropout_rate": trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3, 0.4]),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [1e-4, 5e-4, 1e-3, 5e-3]
        ),
    }


def get_ensemble_search_space(trial):
    """
    Defines the hyperparameter search space for the ElasticNet Meta-Learner.
    """
    return {
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "C": trial.suggest_float("C", 0.01, 10.0, log=True),
    }
