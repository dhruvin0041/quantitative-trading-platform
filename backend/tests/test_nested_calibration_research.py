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


def test_phase_3e_exact_fold_brier_reconciliation():
    """Verify that arithmetic mean of fold prior Brier scores reproduces pooled score exactly."""
    import joblib

    from src.execution.evaluation_contract import canonical_multiclass_brier

    p3e_path = BACKEND_DIR / "artifacts" / "research" / "phase_3e_nested_calibrated_predictions.joblib"
    assert p3e_path.exists()

    data = joblib.load(p3e_path)
    y_val = data["y_val_pooled"]
    p_tp = data["p_tp_pooled"]
    p_raw = data["p_raw_pooled"]

    fold_briers_tp = []
    fold_briers_raw = []
    fold_briers_tp_exact = []
    fold_briers_raw_exact = []
    for f in range(4):
        yf = y_val[f * 414 : (f + 1) * 414]
        ptp_f = p_tp[f * 414 : (f + 1) * 414]
        praw_f = p_raw[f * 414 : (f + 1) * 414]
        yf_onehot = np.zeros_like(ptp_f)
        yf_onehot[np.arange(len(yf)), yf] = 1.0

        fold_briers_tp.append(float(canonical_multiclass_brier(yf, ptp_f)))
        fold_briers_raw.append(float(canonical_multiclass_brier(yf, praw_f)))
        fold_briers_tp_exact.append(float(np.mean(np.sum((ptp_f - yf_onehot) ** 2, axis=1))))
        fold_briers_raw_exact.append(float(np.mean(np.sum((praw_f - yf_onehot) ** 2, axis=1))))

    # Verify Fold 3 prior Brier is 0.3339 (reconciled from 0.3240)
    assert np.isclose(fold_briers_tp[2], 0.333938, atol=1e-4)

    # Verify exact unrounded arithmetic mean identity (machine precision)
    mean_tp_exact = np.mean(fold_briers_tp_exact)
    y_onehot = np.zeros_like(p_tp)
    y_onehot[np.arange(len(y_val)), y_val] = 1.0
    pooled_tp_exact = float(np.mean(np.sum((p_tp - y_onehot) ** 2, axis=1)))
    assert np.isclose(mean_tp_exact, pooled_tp_exact, atol=1e-12)

    # Verify rounded canonical metrics round to 0.4224
    assert round(mean_tp_exact, 4) == 0.4224
    assert canonical_multiclass_brier(y_val, p_tp) == 0.4224

    mean_raw_exact = np.mean(fold_briers_raw_exact)
    pooled_raw_exact = float(np.mean(np.sum((p_raw - y_onehot) ** 2, axis=1)))
    assert np.isclose(mean_raw_exact, pooled_raw_exact, atol=1e-12)
    assert canonical_multiclass_brier(y_val, p_raw) == 0.5964


def test_multiclass_temperature_scaling_cross_observation_swaps():
    """Verify that multiclass temperature scaling shifts cross-observation rankings while preserving argmax."""
    import joblib
    p3e_path = BACKEND_DIR / "artifacts" / "research" / "phase_3e_nested_calibrated_predictions.joblib"
    data = joblib.load(p3e_path)

    p_raw = data["p_raw_pooled"]
    p_temp = data["p_temp_pooled"]

    # 1. Within-observation argmax is strictly preserved
    assert np.array_equal(np.argmax(p_raw, axis=1), np.argmax(p_temp, axis=1))

    # 2. Cross-observation rankings for BUY class shift due to denominator variation
    ranks_raw = np.argsort(np.argsort(p_raw[:, 2]))
    ranks_temp = np.argsort(np.argsort(p_temp[:, 2]))
    swaps = np.sum(ranks_raw != ranks_temp)
    assert swaps > 1000, "Substantial cross-observation rank swaps must occur in multiclass setting"


def test_prior_baseline_within_fold_vs_pooled_ranking():
    """Verify that constant within-fold priors produce AUC 0.5 within-fold but deviate when pooled."""
    import joblib
    from sklearn.metrics import roc_auc_score

    p3e_path = BACKEND_DIR / "artifacts" / "research" / "phase_3e_nested_calibrated_predictions.joblib"
    data = joblib.load(p3e_path)

    y_val = data["y_val_pooled"]
    p_tp = data["p_tp_pooled"]

    # Pooled metrics reflect cross-fold prior variation
    y_buy = (y_val == 2).astype(int)
    pooled_buy_auc = roc_auc_score(y_buy, p_tp[:, 2])
    # Pooled AUC is NOT 0.5 because prior shifts across folds (0.175, 0.191, 0.214, 0.200)
    assert pooled_buy_auc < 0.49 or pooled_buy_auc > 0.51

    # Within each fold, prediction is strictly constant
    for f in range(4):
        ptp_f = p_tp[f * 414 : (f + 1) * 414, 2]
        assert np.all(ptp_f == ptp_f[0]), "Within each fold, prior prediction must be constant"


def test_canonical_reconciliation_artifacts_integrity():
    """Verify that canonical reconciliation artifacts exist, match hashes, and encode exact metrics."""
    json_path = BACKEND_DIR / "reports" / "canonical_phase_3e_reconciliation_report.json"
    md_path = BACKEND_DIR / "reports" / "canonical_phase_3e_reconciliation.md"
    csv_path = BACKEND_DIR / "reports" / "canonical_phase_3e_reconciliation_metrics.csv"

    assert json_path.exists(), "Canonical JSON report must exist"
    assert md_path.exists(), "Canonical Markdown report must exist"
    assert csv_path.exists(), "Canonical CSV metrics must exist"

    with open(json_path, "r") as f:
        data = json.load(f)

    # Check verified hashes
    hashes = data["authoritative_archive_hashes"]
    assert hashes["phase_3d_oof_predictions"]["sha256"] == "d85d5243743027c2e6849016acd79e89a4a70813ed3ff731c7e2ada97cee4c2b"
    assert hashes["phase_3e_nested_calibrated_predictions"]["sha256"] == "1e257def5ce0d89657b241d9a58c56efd8ec92ec985fd3c273a0b208d76b7d32"
    assert hashes["frozen_strategy_manifest_v2_4"]["sha256"] == "d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5"

    records = data["candidate_audit_records"]

    # Check Temperature scaling
    temp = records["Candidate_A_Temperature_Scaling"]
    assert temp["pooled"]["brier_canonical"] == 0.5872
    assert temp["pooled"]["canonical_ece"] == 0.0506
    assert temp["mean_of_folds"]["brier_canonical"] == 0.5873

    # Check Dirichlet
    diri = records["Candidate_B_Dirichlet_Calibration"]
    assert diri["pooled"]["brier_canonical"] == 0.4776
    assert diri["pooled"]["canonical_ece"] == 0.1336
    assert diri["mean_of_folds"]["brier_canonical"] == 0.4776

    # Check Prior
    prior = records["Training_Prior_Baseline"]
    assert prior["pooled"]["brier_canonical"] == 0.4224
    assert prior["pooled"]["canonical_ece"] == 0.0413
    assert prior["mean_of_folds"]["brier_canonical"] == 0.4224
    assert prior["ranking"]["within_fold"]["mean_macro_auc"] == 0.5000
    assert prior["ranking"]["within_fold"]["mean_buy_auc"] == 0.5000
    assert prior["ranking"]["within_fold"]["mean_buy_pr"] == 0.2035
    assert prior["ranking"]["pooled"]["macro_auc"] == 0.4838
    assert prior["ranking"]["pooled"]["buy_auc"] == 0.4558
    assert prior["ranking"]["pooled"]["buy_pr"] == 0.1870


def test_historical_provenance_reconciliation_artifact():
    """Verify that the historical provenance artifact records exact fold shifts and verifies the fold mean identity."""
    json_path = BACKEND_DIR / "reports" / "phase_3e_historical_provenance_reconciliation.json"
    md_path = BACKEND_DIR / "reports" / "phase_3e_historical_provenance_reconciliation.md"

    assert json_path.exists(), "Provenance JSON must exist"
    assert md_path.exists(), "Provenance Markdown must exist"

    with open(json_path, "r") as f:
        data = json.load(f)

    # 1. Verify Report 2 fold mean identity
    r2_scores = data["fold_and_pooled_brier_scores"]["Report_2_Intermediate"]
    r2_temp_folds = r2_scores["Temperature_Scaling"]["folds"]
    assert np.isclose(np.mean(r2_temp_folds), r2_scores["Temperature_Scaling"]["mean_of_folds"], atol=1e-5)
    assert round(np.mean(r2_temp_folds), 4) == 0.5850

    r2_diri_folds = r2_scores["Dirichlet"]["folds"]
    assert np.isclose(np.mean(r2_diri_folds), r2_scores["Dirichlet"]["mean_of_folds"], atol=1e-5)
    assert round(np.mean(r2_diri_folds), 4) == 0.4749

    # 2. Verify Report 3 fold mean identity
    r3_scores = data["fold_and_pooled_brier_scores"]["Report_3_Authoritative"]
    r3_temp_folds = r3_scores["Temperature_Scaling"]["folds"]
    assert np.isclose(np.mean(r3_temp_folds), r3_scores["Temperature_Scaling"]["mean_of_folds"], atol=1e-5)
    assert round(np.mean(r3_temp_folds), 4) == 0.5872

    r3_diri_folds = r3_scores["Dirichlet"]["folds"]
    assert np.isclose(np.mean(r3_diri_folds), r3_scores["Dirichlet"]["mean_of_folds"], atol=1e-5)
    assert round(np.mean(r3_diri_folds), 4) == 0.4776

    # 3. Verify that Fold 1 and Fold 2 scores are identical across Report 2 and Report 3
    assert np.isclose(r2_temp_folds[0], r3_temp_folds[0], atol=1e-4)  # Fold 1: 0.5471
    assert np.isclose(r2_temp_folds[1], r3_temp_folds[1], atol=1e-4)  # Fold 2: 0.6253

    # 4. Verify that Fold 3 and Fold 4 explain the full +0.0022 shift
    fold_diffs = np.array(r3_temp_folds) - np.array(r2_temp_folds)
    assert np.isclose(fold_diffs[0], 0.0, atol=1e-4)
    assert np.isclose(fold_diffs[1], 0.0, atol=1e-4)
    assert fold_diffs[2] > 0.0  # Fold 3 shifted +0.0025
    assert fold_diffs[3] > 0.0  # Fold 4 shifted +0.0064
    assert np.isclose(np.mean(fold_diffs), 0.587250 - 0.585025, atol=1e-4)

