import numpy as np

class MetaEnsemble:
    """
    Dynamically weights individual ML models based on recent out-of-sample performance.
    """
    def __init__(self):
        # Initial equal weighting
        self.weights = {
            "LSTM": 1.0,
            "XGBoost": 1.0,
            "LightGBM": 1.0,
            "DQN": 1.0,
            "TFT": 1.0,
            "Informer": 1.0,
            "PatchTST": 1.0
        }
        self.normalize_weights()

    def normalize_weights(self):
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total

    def update_weights(self, recent_accuracies: dict):
        """
        Updates weights using an exponential scaling factor to reward high performers.
        """
        for model_name, acc in recent_accuracies.items():
            if model_name in self.weights:
                # Exponential reward for accuracy > 50%
                self.weights[model_name] = np.exp(max(0, acc - 0.5) * 5)
        
        self.normalize_weights()

    def aggregate_predictions(self, model_probs: dict) -> np.ndarray:
        """
        model_probs: dict of {model_name: np.array([prob_sell, prob_hold, prob_buy])}
        """
        ensemble_prob = np.zeros(3)
        for model_name, probs in model_probs.items():
            if model_name in self.weights:
                ensemble_prob += probs * self.weights[model_name]
        
        return ensemble_prob
