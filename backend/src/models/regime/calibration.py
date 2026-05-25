import numpy as np
from sklearn.calibration import IsotonicRegression


class ModelCalibrator:
    """
    Calibrates model output probabilities to reflect true confidence using Isotonic Regression.
    """

    def __init__(self):
        self.calibrators = {}

    def fit(self, model_name, y_true, y_prob):
        """
        Fits an isotonic regression model to calibrate probabilities.
        y_prob should be the predicted probability of the positive class.
        """
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(y_prob, y_true)
        self.calibrators[model_name] = ir

    def calibrate(self, model_name, y_prob):
        if model_name not in self.calibrators:
            return y_prob
        return self.calibrators[model_name].predict(y_prob)

    def calculate_confidence_interval(self, ensemble_preds, percentile=95):
        """
        Calculates confidence intervals using ensemble variance.
        """
        mean_pred = np.mean(ensemble_preds, axis=0)
        std_pred = np.std(ensemble_preds, axis=0)
        z_score = 1.96 if percentile == 95 else 2.58
        lower_bound = np.clip(mean_pred - z_score * std_pred, 0, 1)
        upper_bound = np.clip(mean_pred + z_score * std_pred, 0, 1)
        return lower_bound, upper_bound
