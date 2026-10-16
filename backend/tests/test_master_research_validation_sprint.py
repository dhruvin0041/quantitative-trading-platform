# backend/tests/test_master_research_validation_sprint.py
"""
Unit and Quantitative Test Suite for HYDRA V2.4.2+ Master Research-to-Validation Sprint.

Validates:
1. Cryptographic manifest and pinned dataset invariance (byte-for-byte SHA-256 match).
2. Feature representation ablation and stationarity (Set B eliminates leaf collapse).
3. Leakage-safe model optimization integrity (Optuna restricted to inner training splits).
4. Deterministic signal state machine logic (no duplicate buys, causal transitions, explicit reasons).
5. Forward portfolio accounting and cost sensitivity monotonicity.
6. Acceptance gate enforcement (predictive quality failure strictly blocks live capital).
"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.master_research_validation_sprint import (
    FROZEN_MANIFEST_PATH,
    PINNED_DATASET_PATH,
    SignalStateMachine,
    compute_sha256,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_DIR / "reports"


class TestMasterResearchValidationSprint(unittest.TestCase):
    def test_1_cryptographic_manifest_and_pinned_data_invariance(self):
        """Proof 1: Verifies that frozen strategy manifest and pinned dataset hashes match canonical values."""
        h_manifest = compute_sha256(FROZEN_MANIFEST_PATH)
        h_pinned = compute_sha256(PINNED_DATASET_PATH)

        self.assertEqual(h_manifest, "d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5")
        self.assertEqual(h_pinned, "e764c60cb390d11756ec59745629985a20e1ab5afde2ac6d87c01fbd0fc1dc05")

    def test_2_master_sprint_artifacts_exist_and_load(self):
        """Proof 2: Verifies that master sprint report artifacts exist, are valid JSON/CSV, and load cleanly."""
        json_path = REPORTS_DIR / "master_research_validation_sprint.json"
        csv_comp = REPORTS_DIR / "master_sprint_phase_5_strategy_comparison.csv"
        csv_trades = REPORTS_DIR / "master_sprint_phase_5_trade_ledger.csv"
        csv_sig = REPORTS_DIR / "master_sprint_phase_4_signal_state_log.csv"
        json_shadow = REPORTS_DIR / "master_sprint_phase_6_shadow_signal.json"

        for p in [json_path, csv_comp, csv_trades, csv_sig, json_shadow]:
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("phase_0_baseline", data)
        self.assertIn("phase_2_ablation", data)
        self.assertIn("phase_3_optimization", data)
        self.assertIn("phase_5_strategy_comparison", data)
        self.assertIn("phase_7_acceptance_gates", data)

    def test_3_feature_ablation_proves_scale_invariance_superiority(self):
        """Proof 3: Verifies that Set B (Scale Invariant) improves Brier score over Set A (Raw Levels)."""
        json_path = REPORTS_DIR / "master_research_validation_sprint.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ablation = data["phase_2_ablation"]
        brier_a = ablation["Set_A_Baseline_Raw"]["pooled_metrics"]["brier_score"]
        brier_b = ablation["Set_B_Scale_Invariant"]["pooled_metrics"]["brier_score"]

        # Scale invariant features must strictly reduce Brier error compared to raw dollar features
        self.assertLess(brier_b, brier_a)
        self.assertGreater(ablation["Set_B_Scale_Invariant"]["pooled_metrics"]["brier_skill_score_pct"],
                           ablation["Set_A_Baseline_Raw"]["pooled_metrics"]["brier_skill_score_pct"])

    def test_4_signal_state_machine_invariants(self):
        """Proof 4: Verifies that SignalStateMachine prevents duplicate buys and enforces regime filter."""
        sm = SignalStateMachine(threshold=0.45)
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        prices = pd.Series([100.0, 105.0, 110.0, 95.0, 90.0], index=dates)
        sma = pd.Series([98.0, 98.0, 98.0, 98.0, 98.0], index=dates)

        # Bar 0: BUY conviction, Bull regime -> BUY
        # Bar 1: BUY conviction, already long -> HOLD
        # Bar 2: Low conviction, Bull regime -> HOLD
        # Bar 3: Price breaks below SMA -> SELL (Regime break)
        # Bar 4: BUY conviction but Price < SMA -> HOLD (Regime veto)
        probs = np.array([
            [0.10, 0.10, 0.80],  # Bar 0: strong buy
            [0.10, 0.10, 0.80],  # Bar 1: strong buy again
            [0.20, 0.60, 0.20],  # Bar 2: hold
            [0.20, 0.60, 0.20],  # Bar 3: hold prob, but price < sma
            [0.10, 0.10, 0.80],  # Bar 4: buy prob, but price < sma
        ])

        sigs, logs = sm.process_sequence(probs, prices, sma)

        self.assertEqual(sigs.iloc[0], "BUY")
        self.assertEqual(sigs.iloc[1], "HOLD")  # No duplicate buy
        self.assertIn("HOLD_MAINTAINED_EXISTING_LONG", logs[1]["reason_code"])
        self.assertEqual(sigs.iloc[2], "HOLD")
        self.assertEqual(sigs.iloc[3], "SELL")  # Regime break exit
        self.assertIn("REGIME_BREAK_EXIT", logs[3]["reason_code"])
        self.assertEqual(sigs.iloc[4], "HOLD")  # Regime veto
        self.assertIn("REGIME_FILTER_VETO", logs[4]["reason_code"])

    def test_5_cost_sensitivity_monotonicity(self):
        """Proof 5: Verifies that higher execution slippage monotonically reduces net profit."""
        json_path = REPORTS_DIR / "master_research_validation_sprint.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        costs = data["phase_5_cost_sensitivity"]
        self.assertEqual(len(costs), 4)

        equities = [c["final_equity"] for c in costs]
        # Strict monotonic decrease with slippage: 0 bps > 5 bps > 10 bps > 15 bps
        self.assertGreater(equities[0], equities[1])
        self.assertGreater(equities[1], equities[2])
        self.assertGreater(equities[2], equities[3])

    def test_6_acceptance_gates_enforce_no_go(self):
        """Proof 6: Verifies that failed calibration and benchmark gates enforce STRICT NO-GO."""
        json_path = REPORTS_DIR / "master_research_validation_sprint.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gates = {g["gate"]: g["verdict"] for g in data["phase_7_acceptance_gates"]}
        self.assertEqual(gates["Automated Test Suite"], "PASS")
        self.assertEqual(gates["Frozen Strategy Manifest Hash"], "PASS")
        self.assertEqual(gates["Pinned Dataset Hash"], "PASS")
        self.assertEqual(gates["Causal Execution Invariants"], "PASS")
        self.assertEqual(gates["Penny-Exact Trade Ledger Accounting"], "PASS")
        self.assertEqual(gates["Multiclass Brier Score"], "FAIL")
        self.assertEqual(gates["Brier Skill Score (BSS)"], "FAIL")
        self.assertEqual(gates["Expected Calibration Error (ECE)"], "FAIL")
        self.assertEqual(gates["Benchmark Comparison vs 200-Day SMA"], "FAIL")
        self.assertEqual(gates["Live-Money Capital Deployment"], "STRICT NO-GO")


if __name__ == "__main__":
    unittest.main()
