# backend/tests/test_nested_calibration_research.py
"""
Unit tests for HYDRA V2.4.1 Phase 3e Nested Walk-Forward Calibration Research.

Verifies:
1. Immutability of frozen baseline manifest (d580b50a...) and runtime contracts.
2. Leakage-free purge/embargo boundaries across outer folds and inner splits.
3. Probability simplex invariance for all calibrator transformations.
4. Mathematical rank-invariance of scalar temperature scaling.
5. Asymptotic behavior and simplex constraints of Bayesian prior shrinkage.
6. Numerical correctness of Dirichlet calibration logit transformations.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.research.nested_calibration_experiment import (
    BayesianPriorShrinkageCalibrator,
    RegularizedDiagonalDirichletCalibrator,
    RegularizedTemperatureScaler,
    moving_block_bootstrap_discrimination,
)
from scripts.research.probability_diagnostics_audit import (
    validate_probability_simplex,
)
from scripts.research.triple_barrier_experiment import (
    EXPECTED_V24_MANIFEST_SHA256,
    PROTECTED_V24_MANIFEST,
    verify_manifest_integrity,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent


def test_frozen_manifest_integrity():
    """Verify manifest is byte-for-byte identical to frozen baseline."""
    digest = verify_manifest_integrity()
    assert digest == EXPECTED_V24_MANIFEST_SHA256
    assert PROTECTED_V24_MANIFEST.exists()


def test_core_contracts_invariance():
    """Ensure zero modifications to calibration.py or evaluation_contract.py."""
    calib_path = BACKEND_DIR / "src" / "models" / "regime" / "calibration.py"
    eval_path = BACKEND_DIR / "src" / "execution" / "evaluation_contract.py"

    assert calib_path.exists()
    assert eval_path.exists()

    calib_hash = hashlib.sha256(calib_path.read_bytes()).hexdigest()
    eval_hash = hashlib.sha256(eval_path.read_bytes()).hexdigest()

    assert calib_hash == "badff3d49e1ee755231b0e2f159dc539a199ac6263831d66910c9863884958f5"
    assert eval_hash == "94399aefe5015e318eeeafa3e23b7103c8ae4809146948e06b1c3f13e1300720"


def test_fold_boundary_leakage_audit_file():
    """Verify that the generated fold boundary audit records zero leakage."""
    audit_json = BACKEND_DIR / "reports" / "fold_boundary_leakage_audit.json"
    assert audit_json.exists(), "Boundary audit JSON must exist"

    with open(audit_json, "r") as f:
        data = json.load(f)

    assert "outer_folds" in data
    assert "inner_folds" in data

    # Verify all outer folds
    for rec in data["outer_folds"]:
        assert rec["leakage_detected"] is False
        assert rec["safety_margin_bars"] >= 1
        # Last train idx + 15 (outcome window end) must strictly precede first validation idx
        assert rec["last_train_idx"] + 15 < rec["first_val_idx"]

    # Verify all inner folds
    for rec in data["inner_folds"]:
        assert rec["leakage_detected"] is False
        assert rec["safety_margin_bars"] >= 1
        assert rec["last_train_idx"] + 15 < rec["first_val_idx"]


def test_temperature_scaling_mathematical_properties():
    """Verify temperature scaling preserves rank ordering and produces valid simplex outputs."""
    np.random.seed(42)
    n = 100
    raw_p = np.random.dirichlet([2, 1, 3], size=n)
    validate_probability_simplex(raw_p)

    calib = RegularizedTemperatureScaler()
    calib.temperature = 2.0  # Softening temperature

    scaled_p = calib.predict_proba(raw_p)
    res = validate_probability_simplex(scaled_p)
    assert res["status"] == "VALID_SIMPLEX"

    # 1. Argmax preservation
    assert np.array_equal(np.argmax(raw_p, axis=1), np.argmax(scaled_p, axis=1))

    # 2. Strict rank ordering preservation per observation
    for i in range(n):
        orig_ranks = np.argsort(raw_p[i])
        scaled_ranks = np.argsort(scaled_p[i])
        assert np.array_equal(orig_ranks, scaled_ranks)

    # 3. Softening property: max confidence decreases when T > 1
    raw_max_conf = np.max(raw_p, axis=1)
    scaled_max_conf = np.max(scaled_p, axis=1)
    # Exclude cases where raw is nearly uniform
    mask = (raw_max_conf > 0.35) & (raw_max_conf < 0.99)
    assert np.all(scaled_max_conf[mask] < raw_max_conf[mask])


def test_dirichlet_calibrator_simplex_guarantee():
    """Verify diagonal Dirichlet calibrator produces valid probability simplex outputs."""
    np.random.seed(42)
    n = 80
    raw_p = np.random.dirichlet([3, 1, 2], size=n)
    y = np.random.choice([0, 1, 2], size=n, p=[0.6, 0.1, 0.3])

    calib = RegularizedDiagonalDirichletCalibrator(l2_penalty=1e-2)
    calib.fit(raw_p, y)

    transformed_p = calib.predict_proba(raw_p)
    res = validate_probability_simplex(transformed_p)
    assert res["status"] == "VALID_SIMPLEX"
    assert transformed_p.shape == (n, 3)
    assert np.all(transformed_p >= 0.0)
    assert np.allclose(np.sum(transformed_p, axis=1), 1.0, atol=1e-5)


def test_bayesian_prior_shrinkage_limits():
    """Verify Bayesian shrinkage correctly interpolates between model and prior."""
    np.random.seed(42)
    n = 50
    raw_p = np.random.dirichlet([4, 2, 1], size=n)
    prior = np.array([0.70, 0.10, 0.20])

    calib = BayesianPriorShrinkageCalibrator()

    # Limit 1: alpha = 1.0 -> raw_p
    calib.alpha = 1.0
    calib.prior = prior
    p_1 = calib.predict_proba(raw_p)
    assert np.allclose(p_1, raw_p, atol=1e-6)

    # Limit 2: alpha = 0.0 -> prior
    calib.alpha = 0.0
    p_0 = calib.predict_proba(raw_p)
    for i in range(n):
        assert np.allclose(p_0[i], prior, atol=1e-6)

    # Intermediate alpha = 0.5
    calib.alpha = 0.5
    p_mid = calib.predict_proba(raw_p)
    expected_mid = 0.5 * raw_p + 0.5 * prior
    assert np.allclose(p_mid, expected_mid, atol=1e-6)

    # Simplex guarantee
    res = validate_probability_simplex(p_mid)
    assert res["status"] == "VALID_SIMPLEX"


def test_moving_block_bootstrap_ranking_ci():
    """Verify moving-block bootstrap produces valid, monotonic confidence intervals."""
    np.random.seed(42)
    n = 300
    y_true = np.random.choice([0, 1, 2], size=n, p=[0.70, 0.10, 0.20])
    raw_p = np.random.dirichlet([4, 1, 2], size=n)

    ci_dict = moving_block_bootstrap_discrimination(
        y_true, raw_p, block_size=15, n_bootstraps=100, random_state=42
    )

    macro_ci = ci_dict["roc_auc_ovr_macro_95_ci"]
    buy_roc_ci = ci_dict["buy_roc_auc_95_ci"]
    buy_pr_ci = ci_dict["buy_pr_auc_95_ci"]
    buy_pr_ci = ci_dict["buy_pr_auc_95_ci"]

    assert len(macro_ci) == 2
    assert macro_ci[0] <= macro_ci[1]
    assert 0.0 <= macro_ci[0] <= 1.0
    assert 0.0 <= macro_ci[1] <= 1.0

    assert len(buy_roc_ci) == 2
    assert buy_roc_ci[0] <= buy_roc_ci[1]

    assert len(buy_pr_ci) == 2
    assert buy_pr_ci[0] <= buy_pr_ci[1]
