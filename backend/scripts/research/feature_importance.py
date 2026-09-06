import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt

# Ensure artifacts and backtest_results exist
os.makedirs("artifacts", exist_ok=True)
os.makedirs("backtest_results", exist_ok=True)

try:
    import xgboost as xgb

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("artifacts/xgb_ensemble.json")
except Exception:
    xgb_model = joblib.load("artifacts/xgb_ensemble.json")

lgbm_model = joblib.load("artifacts/lgbm_agent.joblib")

from src.execution.live_inference import FEATURE_COLUMNS

# XGBoost importances (normalized 0-1)
xgb_imp_raw = xgb_model.feature_importances_
xgb_imp = pd.Series(xgb_imp_raw / np.sum(xgb_imp_raw), index=FEATURE_COLUMNS)

# LightGBM importances (normalized 0-1)
lgbm_imp_raw = lgbm_model.feature_importances_
lgbm_imp = pd.Series(lgbm_imp_raw / np.sum(lgbm_imp_raw), index=FEATURE_COLUMNS)

# Combined score
combined = (xgb_imp + lgbm_imp) / 2
combined = combined.sort_values(ascending=False)

print("\n=== NORMALIZED FEATURE IMPORTANCE REPORT ===")
print("\nTop 10 Most Important Features:")
print(combined.head(10).to_string())
print("\nBottom 10 Least Important Features (candidates for removal):")
print(combined.tail(10).to_string())
print("\nFeatures with importance < 0.015 (noise candidates):")
# Thresholding at 0.015 to be slightly more aggressive if needed
noise = combined[combined < 0.015]
if not noise.empty:
    print(noise.to_string())
else:
    print("None found with current threshold.")
print(f"\nTotal noise features: {len(noise)}")

# Save importances
combined.to_csv("backtest_results/feature_importances.csv")

# Plot
plt.figure(figsize=(12, 10))
combined.plot(kind="barh")
plt.title("Normalized Feature Importance — XGB + LGBM Combined")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("backtest_results/feature_importances.png")
print("\nSaved: backtest_results/feature_importances.png")
