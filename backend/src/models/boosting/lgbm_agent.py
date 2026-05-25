import joblib
from lightgbm import LGBMClassifier


def train_lgbm_agent(X_train, y_train, save_path="artifacts/lgbm_agent.joblib"):
    """
    Trains a LightGBM Classifier on provided scaled data and saves the artifact.
    """
    print(f"--- Training LightGBM Agent ({len(X_train)} samples) ---")
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        objective="multiclass",
        num_class=3,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"LightGBM model saved to {save_path}")
    return model
