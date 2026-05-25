import joblib
from xgboost import XGBClassifier


def train_xgb_agent(X_train, y_train, save_path="artifacts/xgb_agent.joblib"):
    """
    Trains an XGBoost Classifier on provided scaled data and saves the artifact.
    """
    print(f"--- Training XGBoost Agent ({len(X_train)} samples) ---")
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"XGBoost model saved to {save_path}")
    return model
