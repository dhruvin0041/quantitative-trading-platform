import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

class MetaEnsemble:
    """
    Institutional-grade Stacked Generalization Ensemble.
    Uses an ElasticNet Logistic Regression as a meta-learner, integrating 
    predictions from base models and the current market regime to generate 
    a robust consensus signal.
    """
    def __init__(self, l1_ratio=0.5, C=1.0):
        # ElasticNet Logistic Regression:
        # l1_ratio=1 is Lasso, l1_ratio=0 is Ridge. 0.5 is balanced.
        self.meta_learner = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=l1_ratio,
            C=C,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        self.is_fitted = False
        self.model_names = [
            "LSTM", "CNN", "Transformer", "TFT", "PatchTST", "TCN", 
            "XGBoost", "CatBoost", "LightGBM", "DQN"
        ]
        
    def _prepare_meta_features(self, base_predictions: dict, regime_id: int):
        """
        Flattens base model predictions into a feature vector.
        base_predictions: {model_name: np.array([prob_sell, prob_hold, prob_buy])}
        """
        features = []
        # Ensure consistent ordering
        for name in self.model_names:
            if name in base_predictions:
                # Append the 3 probabilities
                features.extend(base_predictions[name])
            else:
                # Impute neutral if a model is missing
                features.extend([0.0, 1.0, 0.0])
                
        # Inject regime as a categorical one-hot encoded feature
        regime_one_hot = [0.0, 0.0, 0.0]
        if 0 <= regime_id < 3:
            regime_one_hot[regime_id] = 1.0
        features.extend(regime_one_hot)
        
        return np.array(features).reshape(1, -1)
        
    def fit(self, X_meta, y_meta):
        """
        X_meta: shape (n_samples, n_models * 3 + 3)
        y_meta: shape (n_samples,)
        """
        print(f"Training Meta-Ensemble (ElasticNet) on {len(X_meta)} samples...")
        self.meta_learner.fit(X_meta, y_meta)
        self.is_fitted = True
        
    def predict_proba(self, base_predictions: dict, regime_id: int):
        """
        Returns the consensus probability distribution.
        """
        if not self.is_fitted:
            # Fallback to simple average if not fitted
            return self._fallback_predict(base_predictions)
            
        X = self._prepare_meta_features(base_predictions, regime_id)
        return self.meta_learner.predict_proba(X)[0]
        
    def _fallback_predict(self, base_predictions: dict):
        probs = np.zeros(3)
        count = 0
        for name, p in base_predictions.items():
            probs += p
            count += 1
        if count > 0:
            probs /= count
        else:
            probs = np.array([0.0, 1.0, 0.0])
        return probs

    def get_model_contributions(self):
        """
        Analyzes the meta-learner coefficients to determine which base models 
        are contributing the most predictive power.
        """
        if not self.is_fitted:
            return {}
            
        # coef_ shape for multiclass is (n_classes, n_features)
        # We average the absolute coefficients across classes
        avg_coefs = np.mean(np.abs(self.meta_learner.coef_), axis=0)
        
        contributions = {}
        idx = 0
        for name in self.model_names:
            # Sum the importance of the 3 probability outputs for each model
            importance = np.sum(avg_coefs[idx:idx+3])
            contributions[name] = float(importance)
            idx += 3
            
        # Add regime importance
        regime_importance = np.sum(avg_coefs[idx:idx+3])
        contributions["Market_Regime"] = float(regime_importance)
        
        # Normalize
        total = sum(contributions.values())
        if total > 0:
            for k in contributions:
                contributions[k] = round((contributions[k] / total) * 100, 2)
                
        return contributions

    def calculate_uncertainty(self, base_predictions: dict, ensemble_prob: np.ndarray):
        """
        Calculates prediction dispersion (disagreement between base models).
        High dispersion = High uncertainty.
        """
        predictions = []
        for name in self.model_names:
            if name in base_predictions:
                predictions.append(base_predictions[name])
                
        if not predictions:
            return 1.0 # Max uncertainty
            
        # Calculate standard deviation of probabilities across models
        pred_matrix = np.array(predictions) # shape (n_models, 3)
        dispersion = np.mean(np.std(pred_matrix, axis=0))
        
        # Scale to 0-1 range (approximate)
        uncertainty_score = min(1.0, dispersion * 2)
        return float(uncertainty_score)
        
    def save(self, filepath="artifacts/meta_ensemble.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath="artifacts/meta_ensemble.joblib"):
        return joblib.load(filepath)
