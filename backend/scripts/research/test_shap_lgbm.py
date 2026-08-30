import lightgbm as lgb
import shap
import pandas as pd
import numpy as np

from src.execution.live_inference import FEATURE_COLUMNS

# Create dummy model
num_feats = len(FEATURE_COLUMNS)
X = np.random.rand(10, num_feats)
y = np.random.randint(0, 3, 10)
model = lgb.LGBMClassifier()
model.fit(X, y)

X_flat = np.random.rand(1, num_feats)

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
