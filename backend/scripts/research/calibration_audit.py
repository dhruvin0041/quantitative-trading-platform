import joblib
import numpy as np
import json
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


def calculate_ece(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1))[0]
    non_empty_bins = bin_totals > 0

    bin_weights = bin_totals[non_empty_bins] / len(y_prob)
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece


def run_calibration_audit():
    print("=== HYDRA MODEL CALIBRATION AUDIT ===")

    # 1. Load Validation Data
    # For a real audit, we need the validation features and true labels.
    # I'll try to find artifacts or reconstruct from training script logic.
    try:
        X_val_tabular = joblib.load("artifacts/X_val_tabular.joblib")
        y_val_sig = joblib.load("artifacts/y_val_sig.joblib")
        print(f"Loaded {len(y_val_sig)} validation samples.")
    except Exception:
        print(
            "Validation data artifacts not found. Please run scripts/train.py to generate them."
        )
        return

    # 2. Load Models
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("artifacts/xgb_ensemble.json")
        lgbm_model = joblib.load("artifacts/lgbm_agent.joblib")
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    # 3. Audit XGBoost (Class 2 = BUY)
    print("\n--- Model: XGBoost (BUY) ---")
    y_true_buy = (y_val_sig == 2).astype(int)
    probs_xgb = xgb_model.predict_proba(X_val_tabular)[:, 2]

    brier_xgb = brier_score_loss(y_true_buy, probs_xgb)
    ece_xgb = calculate_ece(y_true_buy, probs_xgb)

    print(f"Brier Score: {brier_xgb:.4f}")
    print(f"Expected Calibration Error (ECE): {ece_xgb:.4f}")

    # 4. Audit LightGBM (BUY)
    print("\n--- Model: LightGBM (BUY) ---")
    probs_lgbm = lgbm_model.predict_proba(X_val_tabular)[:, 2]

    brier_lgbm = brier_score_loss(y_true_buy, probs_lgbm)
    ece_lgbm = calculate_ece(y_true_buy, probs_lgbm)

    print(f"Brier Score: {brier_lgbm:.4f}")
    print(f"Expected Calibration Error (ECE): {ece_lgbm:.4f}")

    # 5. Combined Meta-Stats (Average Probability)
    print("\n--- Model: Weighted Average (Consensus) ---")
    # Using previous consensus weights
    probs_combined = (probs_xgb * 0.5) + (probs_lgbm * 0.5)
    brier_comb = brier_score_loss(y_true_buy, probs_combined)
    ece_comb = calculate_ece(y_true_buy, probs_combined)

    print(f"Brier Score: {brier_comb:.4f}")
    print(f"Expected Calibration Error (ECE): {ece_comb:.4f}")

    # Summary JSON for Sprint
    results = {
        "xgboost": {"brier": brier_xgb, "ece": ece_xgb},
        "lightgbm": {"brier": brier_lgbm, "ece": ece_lgbm},
        "consensus": {"brier": brier_comb, "ece": ece_comb},
    }
    with open("backtest_results/calibration_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAudit saved to backtest_results/calibration_audit.json")


if __name__ == "__main__":
    run_calibration_audit()
