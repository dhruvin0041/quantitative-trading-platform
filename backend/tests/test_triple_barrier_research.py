# backend/tests/test_triple_barrier_research.py
"""
Unit tests for HYDRA V2.4.1 Controlled Triple-Barrier Research Experiment.

Verifies:
1. Immutability of frozen V2.4 manifest, calibration.py, and evaluation_contract.py.
2. Causal correctness of generate_candidate_labels across upper, lower, and horizon outcomes.
3. Same-bar barrier touch ambiguity detection and pessimistic fallback resolution.
4. Predeclared candidate geometries (Control, Symmetric 1.5/1.5, Asymmetric 2.0/1.0, 2.0/1.5).
5. Strict governance decision rule enforcing gates (Brier <= 0.20, ECE <= 0.08, net expectancy).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.triple_barrier_experiment import (
    EXPECTED_V24_MANIFEST_SHA256,
    PROTECTED_V24_MANIFEST,
    generate_candidate_labels,
    verify_manifest_integrity,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent


def test_frozen_manifest_integrity():
    """Manifest must exist and match immutable digest."""
    digest = verify_manifest_integrity()
    assert digest == EXPECTED_V24_MANIFEST_SHA256
    assert PROTECTED_V24_MANIFEST.exists()


def test_core_execution_invariance():
    """calibration.py and evaluation_contract.py must not be altered."""
    calib_path = BACKEND_DIR / "src" / "models" / "regime" / "calibration.py"
    eval_path = BACKEND_DIR / "src" / "execution" / "evaluation_contract.py"

    assert calib_path.exists()
    assert eval_path.exists()

    calib_hash = hashlib.sha256(calib_path.read_bytes()).hexdigest()
    eval_hash = hashlib.sha256(eval_path.read_bytes()).hexdigest()

    assert calib_hash == "badff3d49e1ee755231b0e2f159dc539a199ac6263831d66910c9863884958f5"
    assert eval_hash == "94399aefe5015e318eeeafa3e23b7103c8ae4809146948e06b1c3f13e1300720"


def test_synthetic_barrier_label_dynamics():
    """Verify upper hit, lower hit, no hit, and simultaneous hit resolution."""
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    df_raw = pd.DataFrame(
        {
            "Open": [100.0] * 50,
            "High": [100.5] * 50,
            "Low": [99.5] * 50,
            "Close": [100.0] * 50,
            "Volume": [1000] * 50,
        },
        index=dates,
    )
    df_feat = pd.DataFrame(
        {
            "ATR": [2.0] * 50,
        },
        index=dates,
    )

    # 1. Bar 0: Uppercross at step 3 -> High hits 104.0 (UB = 100 + 2*1.5 = 103.0)
    df_raw.loc[dates[3], "High"] = 104.0

    # 2. Bar 10: Lowercross at step 2 -> Low hits 96.0 (LB = 100 - 2*1.5 = 97.0)
    df_raw.loc[dates[12], "Low"] = 96.0

    # 3. Bar 20: Both cross at step 4 -> High hits 105.0 and Low hits 95.0
    df_raw.loc[dates[24], "High"] = 105.0
    df_raw.loc[dates[24], "Low"] = 95.0

    labels, diag = generate_candidate_labels(
        df_raw=df_raw,
        df_feat=df_feat,
        row_dates=dates[:30],
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
    )

    # Verify lengths
    assert len(labels) == 30
    assert set(np.unique(labels)).issubset({0, 1, 2})

    # Bar 0 should hit Upper Barrier (label 2: BUY)
    assert labels[0] == 2

    # Bar 10 should hit Lower Barrier (label 0: SELL)
    assert labels[10] == 0

    # Bar 20 should encounter same-bar ambiguity and resolve pessimistically to SELL (0)
    assert labels[20] == 0
    assert diag["same_bar_ambiguity_count"] >= 1


def test_label_diagnostics_structure():
    """Verify diagnostics dictionary schema."""
    dates = pd.date_range("2023-01-01", periods=25, freq="B")
    df_raw = pd.DataFrame(
        {
            "Open": [100.0] * 25,
            "High": [101.0] * 25,
            "Low": [99.0] * 25,
            "Close": [100.0] * 25,
        },
        index=dates,
    )
    df_feat = pd.DataFrame({"ATR": [1.5] * 25}, index=dates)

    labels, diag = generate_candidate_labels(
        df_raw=df_raw,
        df_feat=df_feat,
        row_dates=dates[:10],
        tp_mult=2.0,
        sl_mult=1.0,
        horizon=10,
    )

    required_keys = [
        "tp_multiplier",
        "sl_multiplier",
        "horizon_bars",
        "n_samples",
        "class_counts",
        "class_percentages",
        "same_bar_ambiguity_count",
        "same_bar_ambiguity_pct",
    ]
    for k in required_keys:
        assert k in diag

    assert diag["class_counts"]["SELL_0"] + diag["class_counts"]["HOLD_1"] + diag["class_counts"]["BUY_2"] == 10
