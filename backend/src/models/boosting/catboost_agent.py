import joblib
from catboost import CatBoostClassifier


def train_catboost_agent(X_train, y_train, save_path="artifacts/catboost_agent.joblib"):
    """
    Trains a CatBoost Classifier on provided scaled data and saves the artifact.
    Highly resistant to overfitting.
    """
    print(f"--- Training CatBoost Agent ({len(X_train)} samples) ---")
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=6,
        loss_function="MultiClass",
        random_seed=42,
        verbose=0,
    )
    # CatBoost can natively handle categorical features,
    # but we assume X_train is fully numerical and preprocessed for now.
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"CatBoost model saved to {save_path}")
    return model
