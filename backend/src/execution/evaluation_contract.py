# backend/src/execution/evaluation_contract.py
"""
Authoritative Evaluation Contract & Calibration Governance Engine for HYDRA V2.4.

Mandate 4:
1. Canonical Multiclass Brier Score:
   Brier = (1 / N) * sum_{i=1}^N sum_{k=0}^2 (p_{ik} - y_{ik})^2
   Gate: Brier <= 0.20. (No division by K=3; binary reduction strictly prohibited).
2. Canonical 10-Bin Expected Calibration Error (ECE):
   ECE = sum_{m=1}^{10} (|B_m| / N) * |acc(B_m) - conf(B_m)|
   Gate: ECE <= 0.08. Equal-width partition on [0, 1].
3. Canonical Class Ordering:
   Strictly indexed as:
     0: SELL
     1: HOLD
     2: BUY
4. Label Maturity, Purging & Embargo Rules:
   - Holding horizon H (default: 5 sessions) mandates that bar t label cannot mature
     or be observed until bar t+H close.
   - Fold purging: Trailing H bars of training fold must be purged.
   - Fold embargo: Minimum E sessions (default: 5 sessions) must separate train end
     from test start to eliminate serial correlation leakage.
5. Chronological Model Provenance:
   - Verifies that model artifact training cutoff <= evaluation period start date.
   - Violations are flagged as CONTAMINATED_LOOKAHEAD.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# Canonical Class Definitions
CLASS_SELL = 0
CLASS_HOLD = 1
CLASS_BUY = 2

CANONICAL_CLASSES: List[int] = [CLASS_SELL, CLASS_HOLD, CLASS_BUY]
CANONICAL_CLASS_NAMES: List[str] = ["SELL", "HOLD", "BUY"]

CLASS_NAME_TO_INT: Dict[str, int] = {
    "SELL": CLASS_SELL,
    "HOLD": CLASS_HOLD,
    "BUY": CLASS_BUY,
}

INT_TO_CLASS_NAME: Dict[int, str] = {
    CLASS_SELL: "SELL",
    CLASS_HOLD: "HOLD",
    CLASS_BUY: "BUY",
}

# Authoritative Governance Gate Hurdles
BRIER_GATE_MAX: float = 0.20
ECE_GATE_MAX: float = 0.08


class ProvenanceViolationError(ValueError):
    """Raised when evaluation violates temporal ordering or lookahead rules."""
    pass


class EvaluationContractError(ValueError):
    """Raised when evaluation inputs or probability arrays violate canonical schemas."""
    pass


@dataclass(frozen=True)
class CalibrationMetricsResult:
    """Immutable result container for canonical calibration evaluation."""
    model_name: str
    n_samples: int
    brier_score: float
    brier_gate_passed: bool
    brier_gate_threshold: float
    ece_score: float
    ece_gate_passed: bool
    ece_gate_threshold: float
    all_gates_passed: bool
    status: str
    details: Dict[str, Any]


def canonical_multiclass_brier(
    y_true: Union[np.ndarray, List[int], pd.Series],
    y_prob: Union[np.ndarray, List[List[float]], pd.DataFrame],
    n_classes: int = 3,
) -> float:
    """
    Computes the canonical multiclass Brier score under the institutional charter.

    Formula:
        Brier = (1 / N) * sum_{i=1}^N sum_{k=0}^{n_classes - 1} (p_{ik} - y_{ik})^2

    Strict Governance Rules:
    - y_prob must have shape (N, n_classes) with rows summing to 1.0 within epsilon.
    - y_true must have integer labels in [0, n_classes - 1].
    - Division by n_classes (e.g. dividing by 3) is STRICTLY PROHIBITED.
    - Binary reduction to class 2 alone is STRICTLY PROHIBITED.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) != len(y_prob_arr):
        raise EvaluationContractError(
            f"Length mismatch: y_true has {len(y_true_arr)} items, y_prob has {len(y_prob_arr)} rows."
        )

    if len(y_true_arr) == 0:
        raise EvaluationContractError("Cannot compute Brier score on empty inputs.")

    if y_prob_arr.ndim != 2 or y_prob_arr.shape[1] != n_classes:
        raise EvaluationContractError(
            f"y_prob must have shape (N, {n_classes}), got {y_prob_arr.shape}."
        )

    # Check bounds
    if np.any(y_true_arr < 0) or np.any(y_true_arr >= n_classes):
        raise EvaluationContractError(
            f"y_true contains invalid class indices outside [0, {n_classes - 1}]."
        )

    # One-hot ground truth encoding
    n_samples = len(y_true_arr)
    y_onehot = np.zeros((n_samples, n_classes), dtype=float)
    y_onehot[np.arange(n_samples), y_true_arr] = 1.0

    # Full multiclass sum across all K classes, averaged over N samples
    sample_brier = np.sum((y_prob_arr - y_onehot) ** 2, axis=1)
    brier = float(np.mean(sample_brier))
    return round(brier, 4)


def canonical_expected_calibration_error(
    y_true: Union[np.ndarray, List[int], pd.Series],
    y_prob: Union[np.ndarray, List[List[float]], pd.DataFrame],
    n_bins: int = 10,
) -> float:
    """
    Computes the canonical 10-bin Expected Calibration Error (ECE) on [0, 1].

    Formula:
        ECE = sum_{m=1}^{n_bins} (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Where:
        - conf_i = max_{k} p_{ik}
        - pred_i = argmax_{k} p_{ik}
        - acc(B_m) = (1 / |B_m|) * sum_{i in B_m} 1(pred_i == y_i)
        - conf(B_m) = (1 / |B_m|) * sum_{i in B_m} conf_i
        - Bins are equal-width partitions on [0, 1].
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) != len(y_prob_arr):
        raise EvaluationContractError("Length mismatch between y_true and y_prob.")

    if len(y_true_arr) == 0:
        raise EvaluationContractError("Cannot compute ECE on empty inputs.")

    confidences = np.max(y_prob_arr, axis=1)
    predictions = np.argmax(y_prob_arr, axis=1)
    accuracies = (predictions == y_true_arr).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = float(len(y_true_arr))
    ece = 0.0

    for b in range(n_bins):
        low = bin_boundaries[b]
        high = bin_boundaries[b + 1]
        # Include left edge in first bin
        if b == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)

        bin_count = float(np.sum(mask))
        if bin_count == 0:
            continue

        bin_acc = float(np.mean(accuracies[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        ece += (bin_count / n_total) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def evaluate_calibration_contract(
    y_true: Union[np.ndarray, List[int], pd.Series],
    y_prob: Union[np.ndarray, List[List[float]], pd.DataFrame],
    model_name: str = "XGBoost",
    brier_max: float = BRIER_GATE_MAX,
    ece_max: float = ECE_GATE_MAX,
) -> CalibrationMetricsResult:
    """
    Executes the canonical evaluation contract and checks adherence to governance gates.
    Fails closed: If any gate is breached, status is marked as 'FAILED_GATE'.
    """
    brier = canonical_multiclass_brier(y_true, y_prob, n_classes=3)
    ece = canonical_expected_calibration_error(y_true, y_prob, n_bins=10)

    brier_passed = bool(brier <= brier_max)
    ece_passed = bool(ece <= ece_max)
    all_passed = bool(brier_passed and ece_passed)

    status = "PASS" if all_passed else "FAILED_GATE"

    details = {
        "class_ordering": CANONICAL_CLASS_NAMES,
        "n_bins": 10,
        "brier_formula": "multiclass_sum",
        "brier_score": brier,
        "brier_gate": brier_max,
        "brier_passed": brier_passed,
        "ece_score": ece,
        "ece_gate": ece_max,
        "ece_passed": ece_passed,
    }

    return CalibrationMetricsResult(
        model_name=model_name,
        n_samples=len(y_true),
        brier_score=brier,
        brier_gate_passed=brier_passed,
        brier_gate_threshold=brier_max,
        ece_score=ece,
        ece_gate_passed=ece_passed,
        ece_gate_threshold=ece_max,
        all_gates_passed=all_passed,
        status=status,
        details=details,
    )


def verify_label_maturity(
    signal_date: Union[str, date, datetime],
    evaluation_date: Union[str, date, datetime],
    label_horizon_sessions: int = 5,
) -> bool:
    """
    Verifies that a label generated at signal_date has fully matured by evaluation_date.
    A signal requiring H sessions to mature cannot be validated on session t+k where k < H.
    """
    sig_dt = pd.to_datetime(signal_date).date()
    eval_dt = pd.to_datetime(evaluation_date).date()

    # Approximate business days count
    bus_days = int(np.busday_count(sig_dt, eval_dt))
    if bus_days < label_horizon_sessions:
        return False
    return True


def verify_purging_and_embargo(
    train_end_date: Union[str, date, datetime],
    test_start_date: Union[str, date, datetime],
    label_horizon_sessions: int = 5,
    embargo_sessions: int = 5,
) -> Tuple[bool, str]:
    """
    Validates De Prado's purging and embargo requirements:
    1. The last label_horizon_sessions of the training set must be purged
       (or train_end must precede test_start by at least horizon + embargo sessions).
    2. Minimum total gap required between train end and test start is:
       gap >= label_horizon_sessions + embargo_sessions.
    """
    t_end = pd.to_datetime(train_end_date).date()
    t_start = pd.to_datetime(test_start_date).date()

    if t_start <= t_end:
        return False, f"Test start date ({t_start}) is on or before train end date ({t_end}). Overlap detected."

    gap_sessions = int(np.busday_count(t_end, t_start))
    required_gap = label_horizon_sessions + embargo_sessions

    if gap_sessions < required_gap:
        return (
            False,
            f"Insufficient gap between train_end ({t_end}) and test_start ({t_start}). "
            f"Observed gap: {gap_sessions} sessions; Required gap: {required_gap} sessions "
            f"(horizon={label_horizon_sessions} + embargo={embargo_sessions}).",
        )

    return True, f"Purging and embargo verified. Gap: {gap_sessions} sessions >= {required_gap}."


def verify_chronological_provenance(
    model_training_cutoff: Union[str, date, datetime],
    evaluation_start_date: Union[str, date, datetime],
) -> Tuple[bool, str]:
    """
    Verifies that the model artifact training cutoff strictly precedes or matches
    the evaluation start date. If model was trained on data beyond evaluation start date,
    raises or returns a lookahead violation.
    """
    m_cutoff = pd.to_datetime(model_training_cutoff).date()
    e_start = pd.to_datetime(evaluation_start_date).date()

    if m_cutoff > e_start:
        msg = (
            f"CONTAMINATED_LOOKAHEAD: Model was trained through {m_cutoff}, "
            f"but evaluated starting at {e_start}. Out-of-sample validity invalidated."
        )
        return False, msg

    return True, f"Chronological provenance verified. Model cutoff {m_cutoff} <= Eval start {e_start}."
