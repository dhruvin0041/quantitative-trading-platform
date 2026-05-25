import xgboost as xgb
import shap
import pandas as pd
import numpy as np

# Create dummy model
X = np.random.rand(10, 14)
y = np.random.randint(0, 3, 10)
model = xgb.XGBClassifier()
model.fit(X, y)

X_flat = np.random.rand(1, 14)
FEATURE_COLUMNS = [
    'MA20_vs_MA50', 'EMA9_vs_EMA21', 'Price_vs_EMA9', 'Price_vs_EMA21',
    'VIX_Level', 'BB_Width', 'BB_Position', 'RSI', 'ADX', 'MACD_Hist', 
    'Relative_Strength', 'OBV_Change', 'Return', 'Volume_Ratio'
]
X_df = pd.DataFrame(X_flat, columns=FEATURE_COLUMNS)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_df)
print(type(shap_values))
if isinstance(shap_values, list):
    print(len(shap_values))
    print(shap_values[0].shape)
else:
    print(shap_values.shape)
    print(shap_values[0])
