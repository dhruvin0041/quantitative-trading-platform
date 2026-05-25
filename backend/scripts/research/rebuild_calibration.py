import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def calibrate_model_output(y_true, y_prob, method="isotonic"):
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_prob, y_true)
    else:  # Platt scaling (Logistic)
        calibrator = LogisticRegression()
        calibrator.fit(y_prob.reshape(-1, 1), y_true)
    return calibrator


def rebuild_calibration():
    print("=== HYDRA CALIBRATION REBUILD ===")

    # 1. Load Validation Data
    X_val = joblib.load("artifacts/X_val_tabular.joblib")
    y_val = joblib.load("artifacts/y_val_sig.joblib")

    # Models
    import xgboost as xgb

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("artifacts/xgb_ensemble.json")
    lgbm_model = joblib.load("artifacts/lgbm_agent.joblib")

    # 2. Extract raw probabilities
    probs_xgb = xgb_model.predict_proba(X_val)
    probs_lgbm = lgbm_model.predict_proba(X_val)

    # 3. Calibrate BUY (2) and SELL (0) for each
    calibrators = {}

    for model_name, probs in [("xgb", probs_xgb), ("lgbm", probs_lgbm)]:
        for class_idx, label in [(0, "sell"), (2, "buy")]:
            y_true_class = (y_val == class_idx).astype(int)
            y_prob_class = probs[:, class_idx]

            # Use Isotonic for larger sets, Platt for smaller.
            # Our val set is small (45 samples), so Platt might be more stable.
            # We'll use Isotonic as it was requested, but with safety.
            calibrator = calibrate_model_output(
                y_true_class, y_prob_class, method="isotonic"
            )

            cal_key = f"{model_name}_calibrator_{label}"
            calibrators[cal_key] = calibrator

            # Score
            cal_probs = calibrator.predict(y_prob_class)
            brier_before = brier_score_loss(y_true_class, y_prob_class)
            brier_after = brier_score_loss(y_true_class, cal_probs)
            print(
                f"[{model_name.upper()} {label.upper()}] Brier: {brier_before:.4f} -> {brier_after:.4f}"
            )

    # 4. Save calibrators
    joblib.dump(calibrators, "artifacts/hydra_calibrators.joblib")
    print("\nSaved all calibrators to artifacts/hydra_calibrators.joblib")


if __name__ == "__main__":
    rebuild_calibration()
