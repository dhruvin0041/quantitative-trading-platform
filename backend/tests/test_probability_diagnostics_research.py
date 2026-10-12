# backend/tests/test_probability_diagnostics_research.py
"""
Unit tests for HYDRA V2.4.1 Phase 3d Probability Diagnostics Audit.

Verifies:
1. Immutability of frozen baseline manifest and runtime contracts.
2. Machine-precision exact mathematical decomposition of Brier score:
   - Brier_binned == Reliability - Resolution + Uncertainty (exact to 1e-10).
   - Brier_raw == Reliability - Resolution + Uncertainty + WithinBinResidual (exact to 1e-10).
3. Probability simplex validation (finite, non-negative, row sum == 1.0).
4. Theoretical calibration scores of Uniform and Training-Prior benchmarks.
5. Class-wise binary ECE calculation and partition mechanics.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.research.probability_diagnostics_audit import (
    compute_class_wise_ece,
    murphy_brier_decomposition,
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


def test_murphy_brier_exact_identities():
    """Verify that Murphy decomposition identities hold exactly to machine precision."""
    np.random.seed(42)
    n = 600
    y_true = np.random.choice([0, 1, 2], size=n, p=[0.70, 0.10, 0.20])
    raw = np.random.dirichlet([5, 1, 2], size=n)

    decomp = murphy_brier_decomposition(y_true, raw, n_bins=10)

    # 1. Binned identity: Rel - Res + Unc == Brier_binned (exact)
    assert decomp["exact_identity_binned_diff"] < 1e-10
    rel = decomp["reliability"]
    res = decomp["resolution"]
    unc = decomp["uncertainty"]
    binned_brier = decomp["brier_binned"]
    assert np.isclose(binned_brier, rel - res + unc, atol=1e-10)

    # 2. Continuous raw identity: Rel - Res + Unc + Within == Brier_raw (exact)
    assert decomp["exact_identity_raw_diff"] < 1e-10
    within = decomp["within_bin_residual"]
    raw_brier = decomp["brier_raw"]
    assert np.isclose(raw_brier, rel - res + unc + within, atol=1e-10)

    # 3. Class-wise decomposition verification (rounded to 5 decimal places in report)
    for c_name in ["SELL_0", "HOLD_1", "BUY_2"]:
        cd = decomp["class_decompositions"][c_name]
        assert np.isclose(
            cd["brier_binned"],
            cd["reliability"] - cd["resolution"] + cd["uncertainty"],
            atol=1e-4,
        )
        assert np.isclose(
            cd["brier_raw"],
            cd["reliability"] - cd["resolution"] + cd["uncertainty"] + cd["within_bin_residual"],
            atol=1e-4,
        )


def test_probability_simplex_validator():
    """Verify probability simplex validation rejects non-simplex arrays and passes valid ones."""
    # Valid
    valid_p = np.array([
        [0.7, 0.2, 0.1],
        [0.333333, 0.333333, 0.333334],
        [1.0, 0.0, 0.0],
    ])
    res = validate_probability_simplex(valid_p)
    assert res["status"] == "VALID_SIMPLEX"

    # Non-finite (NaN)
    with pytest.raises(ValueError, match="non-finite"):
        validate_probability_simplex(np.array([[np.nan, 0.5, 0.5]]))

    # Negative value
    with pytest.raises(ValueError, match="negative"):
        validate_probability_simplex(np.array([[-0.1, 0.6, 0.5]]))

    # Row does not sum to 1
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_probability_simplex(np.array([[0.5, 0.5, 0.5]]))


def test_theoretical_uniform_brier():
    """Uniform forecast on 3 classes must score exactly 2/3 (0.6667)."""
    n = 300
    y_true = np.array([0] * 100 + [1] * 100 + [2] * 100)
    p_uni = np.full((n, 3), 1.0 / 3.0)

    decomp = murphy_brier_decomposition(y_true, p_uni, n_bins=10)
    assert np.isclose(decomp["brier_raw"], 2.0 / 3.0, atol=1e-3)
    assert np.isclose(decomp["brier_binned"], 2.0 / 3.0, atol=1e-3)
    assert np.isclose(decomp["uncertainty"], 2.0 / 3.0, atol=1e-3)
    assert np.isclose(decomp["resolution"], 0.0, atol=1e-3)


def test_class_wise_ece_structure():
    """Verify class-wise ECE outputs dictionary with correct class keys."""
    y_true = np.array([0, 1, 2, 0, 0])
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.9, 0.05, 0.05],
    ])
    res = compute_class_wise_ece(y_true, probs, n_bins=5)
    assert "SELL_0" in res
    assert "HOLD_1" in res
    assert "BUY_2" in res
    assert 0.0 <= res["BUY_2"] <= 1.0


def test_phase_3d_artifact_reproduction():
    """Verify independent reproduction of Phase 3d metrics from persisted archive."""
    artifact_path = BACKEND_DIR / "artifacts" / "research" / "phase_3d_oof_predictions.joblib"
    assert artifact_path.exists(), "Phase 3d OOF predictions archive must exist"

    from scripts.research.verify_phase_3d_reproduction import reproduce_phase_3d_metrics

    res = reproduce_phase_3d_metrics(artifact_path)
    assert res["n_samples"] == 1656
    assert res["brier_scores"]["training_prior"] < res["brier_scores"]["control"]
    assert res["brier_skill_scores"]["control"] < 0.0
    assert res["murphy_decomposition_control"]["identity_diff_binned"] < 1e-10
    assert res["murphy_decomposition_control"]["identity_diff_raw"] < 1e-7
    assert 0.0 <= res["ranking_discrimination"]["control"]["roc_auc_ovr_macro"] <= 1.0
