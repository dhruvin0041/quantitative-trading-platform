import numpy as np
import joblib
import logging
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class ModelCalibrator:
    """
    Calibrates model output probabilities using per-class Isotonic Regression.

    For a 3-class problem (SELL=0, HOLD=1, BUY=2), we fit one isotonic
    regressor per class per model.  This converts raw softmax outputs into
    probabilities that reflect the *true* empirical frequency of each class
    at that predicted-probability level.

    Fitting protocol:
        * Called ONLY on held-out validation predictions (never on training data)
          to prevent double-dipping / over-calibration.
        * Saved alongside the model checkpoint so inference can load & apply.
    """

    NUM_CLASSES = 3
    CLASS_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}

    def __init__(self):
        # calibrators[model_name][class_idx] = IsotonicRegression
        self.calibrators: dict[str, dict[int, IsotonicRegression]] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, model_name: str, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Fit per-class isotonic regressors.

        Parameters
        ----------
        model_name : str
            Identifier (e.g. "DL_FUSION", "XGB", "LGBM").
        y_true : np.ndarray, shape (n_samples,)
            Integer class labels {0, 1, 2}.
        y_prob : np.ndarray, shape (n_samples, 3)
            Predicted probability matrix [P(SELL), P(HOLD), P(BUY)].
        """
        if y_prob.ndim == 1:
            raise ValueError(
                f"y_prob must be 2-D (n_samples, {self.NUM_CLASSES}), got 1-D."
            )

        self.calibrators[model_name] = {}
        for cls_idx in range(self.NUM_CLASSES):
            ir = IsotonicRegression(out_of_bounds="clip")
            binary_target = (y_true == cls_idx).astype(float)
            ir.fit(y_prob[:, cls_idx], binary_target)
            self.calibrators[model_name][cls_idx] = ir
            logger.info(
                "Fitted calibrator for %s / class %s (%s) on %d samples",
                model_name,
                cls_idx,
                self.CLASS_NAMES[cls_idx],
                len(y_true),
            )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, model_name: str, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibrate a probability vector (or matrix) through isotonic regressors.

        Parameters
        ----------
        model_name : str
        y_prob : np.ndarray, shape (3,) or (n_samples, 3)

        Returns
        -------
        np.ndarray of the same shape, re-normalised so rows sum to 1.
        """
        if model_name not in self.calibrators:
            logger.warning("No calibrator for model '%s'. Returning raw probs.", model_name)
            return y_prob

        single = y_prob.ndim == 1
        if single:
            y_prob = y_prob.reshape(1, -1)

        calibrated = np.zeros_like(y_prob)
        for cls_idx in range(self.NUM_CLASSES):
            calibrated[:, cls_idx] = self.calibrators[model_name][cls_idx].predict(
                y_prob[:, cls_idx]
            )

        # Re-normalise so each row sums to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        calibrated = calibrated / row_sums

        if single:
            calibrated = calibrated[0]

        return calibrated

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str = "artifacts/model_calibrator.joblib"):
        """Persist the fitted calibrators to disk."""
        joblib.dump(self.calibrators, path)
        logger.info("Saved calibrator to %s", path)

    @classmethod
    def load(cls, path: str = "artifacts/model_calibrator.joblib") -> "ModelCalibrator":
        """Load a persisted calibrator."""
        instance = cls()
        instance.calibrators = joblib.load(path)
        logger.info("Loaded calibrator from %s with models: %s", path, list(instance.calibrators.keys()))
        return instance

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
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
