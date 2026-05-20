import numpy as np
from scipy.stats import ks_2samp

class DriftMonitor:
    """
    Monitors data and model drift in production.
    Alerts if live inference distributions diverge from training distributions.
    """
    
    def __init__(self, p_value_threshold=0.05):
        self.p_value_threshold = p_value_threshold
        
    def check_covariate_drift(self, train_features: np.ndarray, live_features: np.ndarray):
        """
        Uses Kolmogorov-Smirnov test to detect data drift on a per-feature basis.
        Returns a list of indices for features that have drifted.
        """
        drifted_features = []
        for i in range(train_features.shape[1]):
            stat, p_value = ks_2samp(train_features[:, i], live_features[:, i])
            if p_value < self.p_value_threshold:
                drifted_features.append(i)
                
        is_drifting = len(drifted_features) > (train_features.shape[1] * 0.2) # Alert if >20% features drift
        return {
            "is_drifting": is_drifting,
            "drifted_feature_indices": drifted_features,
            "drift_ratio": len(drifted_features) / train_features.shape[1]
        }
