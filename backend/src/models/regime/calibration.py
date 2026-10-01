import logging

import joblib
import numpy as np
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
        # calibrators[model_name] = {"method": str, "models": dict[int, Any]}
        self.calibrators: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, model_name: str, y_true: np.ndarray, y_prob: np.ndarray, method: str = "isotonic"):
        """
        Fit per-class calibrators using the specified method: 'isotonic', 'sigmoid', or 'raw'.

        Parameters
        ----------
        model_name : str
            Identifier (e.g. "DL_FUSION", "XGB", "LGBM").
        y_true : np.ndarray, shape (n_samples,)
            Integer class labels {0, 1, 2}.
        y_prob : np.ndarray, shape (n_samples, 3)
            Predicted probability matrix [P(SELL), P(HOLD), P(BUY)].
        method : str, default="isotonic"
            One of 'isotonic', 'sigmoid' (Platt scaling), or 'raw'.
        """
        if method == "raw":
            self.calibrators[model_name] = {"method": "raw", "models": {}}
            logger.info("Configured raw pass-through calibration for %s (N=%d)", model_name, len(y_true))
            return

        if y_prob.ndim == 1:
            raise ValueError(
                f"y_prob must be 2-D (n_samples, {self.NUM_CLASSES}), got 1-D."
            )

        from sklearn.linear_model import LogisticRegression

        class_models = {}
        for cls_idx in range(self.NUM_CLASSES):
            binary_target = (y_true == cls_idx).astype(float)
            if method == "sigmoid":
                clf = LogisticRegression(C=1.0)
                clf.fit(y_prob[:, cls_idx : cls_idx + 1], binary_target.astype(int))
                class_models[cls_idx] = clf
            elif method == "isotonic":
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(y_prob[:, cls_idx], binary_target)
                class_models[cls_idx] = ir
            else:
                raise ValueError(f"Unknown calibration method: {method}. Must be 'isotonic', 'sigmoid', or 'raw'.")

        self.calibrators[model_name] = {"method": method, "models": class_models}
        logger.info(
            "Fitted %s calibrator for %s on %d samples",
            method,
            model_name,
            len(y_true),
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, model_name: str, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibrate a probability vector (or matrix) through fitted calibrators.

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

        entry = self.calibrators[model_name]
        # Backward compatibility for legacy dictionary structure
        if isinstance(entry, dict) and "method" not in entry:
            entry = {"method": "isotonic", "models": entry}

        method = entry.get("method", "isotonic")
        if method == "raw":
            return y_prob

        single = y_prob.ndim == 1
        if single:
            y_prob = y_prob.reshape(1, -1)

        calibrated = np.zeros_like(y_prob)
        models_dict = entry.get("models", {})
        for cls_idx in range(self.NUM_CLASSES):
            est = models_dict.get(cls_idx)
            if est is None:
                calibrated[:, cls_idx] = y_prob[:, cls_idx]
            elif method == "sigmoid":
                calibrated[:, cls_idx] = est.predict_proba(y_prob[:, cls_idx : cls_idx + 1])[:, 1]
            elif method == "isotonic":
                calibrated[:, cls_idx] = est.predict(y_prob[:, cls_idx])

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
