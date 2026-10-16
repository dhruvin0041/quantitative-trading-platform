# backend/tests/test_v2_4_3_hierarchical_regime_strategy.py
"""
Unit and Quantitative Regression Suite for HYDRA V2.4.3 Hierarchical Regime-Switching Strategy.

Validates:
1. Regime calculation point-in-time causality & perturbation invariance.
2. Prevention of duplicate entries, invalid exits, and illegal transitions.
3. Priority rules across Strong Uptrend, Neutral/Consolidation, Downtrend, and Elevated Volatility.
4. Missing / stale input handling (fails closed).
5. Bar-boundary execution timing (Close[t] -> Open[t+1]) and penny-exact accounting reconciliation.
6. Replay determinism (identical inputs yield identical signals and states).
7. Artifact provenance and cryptographic dataset/manifest immutability.
8. Strict institutional gate enforcement (calibration and benchmark failure -> STRICT NO-GO).
"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.v2_4_3_hierarchical_regime_strategy import (
    FROZEN_MANIFEST_PATH,
    PINNED_DATASET_PATH,
    HierarchicalSignalStateMachine,
    InstitutionalPortfolioSimulator,
    MacroRegimeClassifier,
    compute_sha256,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_DIR / "reports"


class TestHierarchicalRegimeStrategyV243(unittest.TestCase):
    def setUp(self):
        self.classifier = MacroRegimeClassifier(delta=0.07, vol_threshold=0.02)
        self.state_machine = HierarchicalSignalStateMachine(
            delta=0.07, vol_threshold=0.02, theta=0.45, overlay_mode=False
        )

    def test_1_regime_calculation_point_in_time_causality(self):
        """Proof 1: Verifies that future-bar perturbations do not change earlier regime classifications or signals."""
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        close_orig = [100.0, 102.0, 105.0, 108.0, 112.0, 115.0, 118.0, 120.0, 122.0, 125.0]
        sma_orig = [100.0] * 10
        slope_orig = [0.01] * 10
        vol_orig = [0.015] * 10

        df_orig = pd.DataFrame({
            "Close": close_orig,
            "SMA_200": sma_orig,
            "Normalized_SMA_Slope_20": slope_orig,
            "ATR_over_Close": vol_orig,
        }, index=dates)

        probs = np.array([[0.20, 0.20, 0.60]] * 10)
        sigs_orig, logs_orig = self.state_machine.process_sequence(probs, df_orig)

        # Perturb future bars (bars 6 to 9) dramatically
        df_perturbed = df_orig.copy()
        df_perturbed.iloc[6:, df_perturbed.columns.get_loc("Close")] = [50.0, 40.0, 30.0, 20.0]
        df_perturbed.iloc[6:, df_perturbed.columns.get_loc("ATR_over_Close")] = [0.10, 0.15, 0.20, 0.25]

        sigs_perturbed, logs_perturbed = self.state_machine.process_sequence(probs, df_perturbed)

        # Bars 0 to 5 MUST be strictly identical in signal and regime classification
        for i in range(6):
            self.assertEqual(sigs_orig.iloc[i], sigs_perturbed.iloc[i], f"Signal divergence at causal bar {i}")
            self.assertEqual(logs_orig[i]["regime"], logs_perturbed[i]["regime"], f"Regime divergence at causal bar {i}")
            self.assertEqual(logs_orig[i]["reason_code"], logs_perturbed[i]["reason_code"], f"Reason divergence at bar {i}")

    def test_2_regime_transitions_prevent_duplicate_entries_and_invalid_exits(self):
        """Proof 2: Verifies that regime transitions never generate duplicate entries or illegal short exits."""
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        # Bar 0: Strong Uptrend (Close 110, SMA 100) -> BUY
        # Bar 1: Strong Uptrend continues -> HOLD (prevent duplicate buy)
        # Bar 2: Neutral Consolidation (Close 104, SMA 100) -> LONG maintained (not silently liquidated)
        # Bar 3: Downtrend (Close 95, SMA 100) -> SELL (regime break exit)
        # Bar 4: Downtrend continues -> HOLD (flat, prevent illegal short exit)
        # Bar 5: Downtrend + Sell conviction -> HOLD (prevent illegal short exit)
        df = pd.DataFrame({
            "Close": [110.0, 112.0, 104.0, 95.0, 92.0, 90.0],
            "SMA_200": [100.0] * 6,
            "Normalized_SMA_Slope_20": [0.01, 0.01, 0.00, -0.01, -0.01, -0.01],
            "ATR_over_Close": [0.015] * 6,
        }, index=dates)

        probs = np.array([
            [0.10, 0.10, 0.80],  # Bar 0: BUY prob
            [0.10, 0.10, 0.80],  # Bar 1: BUY prob again
            [0.30, 0.40, 0.30],  # Bar 2: neutral prob
            [0.30, 0.40, 0.30],  # Bar 3: neutral prob
            [0.30, 0.40, 0.30],  # Bar 4: neutral prob
            [0.80, 0.10, 0.10],  # Bar 5: SELL prob while flat
        ])

        sigs, logs = self.state_machine.process_sequence(probs, df)

        self.assertEqual(sigs.iloc[0], "BUY")
        self.assertEqual(logs[0]["position_state_after_bar"], "IN_LONG")

        # Bar 1: BUY signal while long must emit HOLD (no duplicate buy)
        self.assertEqual(sigs.iloc[1], "HOLD")
        self.assertEqual(logs[1]["position_state_after_bar"], "IN_LONG")
        self.assertIn("TIER_1_UPTREND_HOLD_MAINTAINED", logs[1]["reason_code"])

        # Bar 2: Regime change to Neutral Consolidation does NOT silently liquidate
        self.assertEqual(sigs.iloc[2], "HOLD")
        self.assertEqual(logs[2]["position_state_after_bar"], "IN_LONG")

        # Bar 3: Downtrend triggers causal exit
        self.assertEqual(sigs.iloc[3], "SELL")
        self.assertEqual(logs[3]["position_state_after_bar"], "FLAT")
        self.assertIn("REGIME_BREAK_EXIT", logs[3]["reason_code"])

        # Bar 4 & 5: When FLAT in bear market, no illegal short sales
        self.assertEqual(sigs.iloc[4], "HOLD")
        self.assertEqual(logs[4]["position_state_after_bar"], "FLAT")
        self.assertEqual(sigs.iloc[5], "HOLD")
        self.assertEqual(logs[5]["position_state_after_bar"], "FLAT")

    def test_3_hierarchical_priority_rules(self):
        """Proof 3: Verifies that macro trend and ML tactical decisions follow documented priority rules."""
        # Strong Uptrend: Close >= 1.07 * SMA200 and Slope >= 0
        r_up = self.classifier.classify_bar(close=110.0, sma_200=100.0, slope_20=0.01, atr_over_close=0.015)
        self.assertEqual(r_up, "STRONG_UPTREND")

        # Neutral Consolidation: Close between 100 and 107
        r_neut = self.classifier.classify_bar(close=104.0, sma_200=100.0, slope_20=0.01, atr_over_close=0.015)
        self.assertEqual(r_neut, "NEUTRAL_CONSOLIDATION")

        # Downtrend: Close < 100
        r_down = self.classifier.classify_bar(close=98.0, sma_200=100.0, slope_20=0.01, atr_over_close=0.015)
        self.assertEqual(r_down, "DOWNTREND")

        # Elevated Volatility: ATR_over_Close > 0.02
        r_vol = self.classifier.classify_bar(close=110.0, sma_200=100.0, slope_20=0.01, atr_over_close=0.025)
        self.assertEqual(r_vol, "ELEVATED_VOLATILITY")

    def test_4_missing_or_stale_inputs_fail_closed(self):
        """Proof 4: Verifies that NaN or missing SMA200 fails closed safely without exceptions or rogue trades."""
        r_nan = self.classifier.classify_bar(close=100.0, sma_200=float("nan"), slope_20=0.0, atr_over_close=0.01)
        self.assertEqual(r_nan, "NEUTRAL_CONSOLIDATION")

        r_zero = self.classifier.classify_bar(close=100.0, sma_200=0.0, slope_20=0.0, atr_over_close=0.01)
        self.assertEqual(r_zero, "NEUTRAL_CONSOLIDATION")

    def test_5_execution_timing_and_penny_accounting_reconciliation(self):
        """Proof 5: Verifies that fills occur at Open[t+1] and portfolio equity reconciles to the exact cent."""
        sim = InstitutionalPortfolioSimulator(initial_capital=1000.0, slippage_bps=5.0)

        dates = pd.date_range("2026-01-01", periods=4, freq="D")
        df_prices = pd.DataFrame({
            "Open": [100.0, 102.0, 105.0, 108.0],
            "High": [101.0, 104.0, 107.0, 110.0],
            "Low": [99.0, 101.0, 103.0, 106.0],
            "Close": [100.5, 103.5, 106.5, 109.5],
        }, index=dates)

        # Signal emitted at Close[0]: BUY -> Executes at Open[1] (102.0)
        # Signal emitted at Close[2]: SELL -> Executes at Open[3] (108.0)
        signals = pd.Series(["BUY", "HOLD", "SELL", "HOLD"], index=dates)

        res = sim.simulate(df_prices, signals, "Test_Accounting")
        trades = res["trades"]
        summary = res["summary"]

        self.assertEqual(len(trades), 1)
        t = trades[0]

        # Verify entry fill price at Open[1] + 5 bps adverse slippage = 102.0 * 1.0005 = 102.051
        expected_entry_fill = 102.0 * 1.0005
        self.assertAlmostEqual(t["entry_fill_price"], expected_entry_fill, places=3)

        # Verify exit fill price at Open[3] - 5 bps adverse slippage = 108.0 * 0.9995 = 107.946
        expected_exit_fill = 108.0 * 0.9995
        self.assertAlmostEqual(t["exit_fill_price"], expected_exit_fill, places=3)

        # Verify penny reconciliation: Final Equity == Final Cash (since position closed)
        self.assertAlmostEqual(summary["final_equity"], summary["final_cash"], places=2)
        self.assertAlmostEqual(summary["final_equity"], 1000.0 + summary["net_profit_dollars"], places=2)

    def test_6_replaying_same_inputs_reproduces_identical_signals(self):
        """Proof 6: Verifies deterministic idempotency of signal generation."""
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "Close": [100.0, 108.0, 107.0, 95.0, 90.0],
            "SMA_200": [100.0] * 5,
            "Normalized_SMA_Slope_20": [0.01] * 5,
            "ATR_over_Close": [0.015] * 5,
        }, index=dates)
        probs = np.array([
            [0.10, 0.10, 0.80],
            [0.20, 0.20, 0.60],
            [0.30, 0.40, 0.30],
            [0.40, 0.30, 0.30],
            [0.10, 0.10, 0.80],
        ])

        sigs1, logs1 = self.state_machine.process_sequence(probs, df)
        sigs2, logs2 = self.state_machine.process_sequence(probs, df)

        pd.testing.assert_series_equal(sigs1, sigs2)
        self.assertEqual(len(logs1), len(logs2))
        for l1, l2 in zip(logs1, logs2):
            self.assertEqual(l1["emitted_signal"], l2["emitted_signal"])
            self.assertEqual(l1["reason_code"], l2["reason_code"])

    def test_7_v2_4_3_artifacts_and_cryptographic_immutability(self):
        """Proof 7: Verifies that frozen manifest and pinned dataset remain byte-for-byte unchanged."""
        h_manifest = compute_sha256(FROZEN_MANIFEST_PATH)
        h_pinned = compute_sha256(PINNED_DATASET_PATH)

        self.assertEqual(h_manifest, "d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5")
        self.assertEqual(h_pinned, "e764c60cb390d11756ec59745629985a20e1ab5afde2ac6d87c01fbd0fc1dc05")

        # Check newly created reports exist
        json_report = REPORTS_DIR / "v2_4_3_hierarchical_regime_strategy.json"
        csv_comp = REPORTS_DIR / "v2_4_3_hierarchical_strategy_comparison.csv"
        csv_trades = REPORTS_DIR / "v2_4_3_hierarchical_trade_ledger.csv"
        csv_sens = REPORTS_DIR / "v2_4_3_sensitivity_analysis.csv"
        csv_slip = REPORTS_DIR / "v2_4_3_slippage_sensitivity.csv"
        json_shadow = REPORTS_DIR / "v2_4_3_shadow_signal.json"

        for p in [json_report, csv_comp, csv_trades, csv_sens, csv_slip, json_shadow]:
            self.assertTrue(p.exists(), f"Missing required artifact: {p}")

        with open(json_report, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("reproduction_audit", data)
        self.assertIn("locked_architecture_rules", data)
        self.assertIn("pre_2025_sensitivity_analysis", data)
        self.assertIn("acceptance_gates", data)

    def test_8_strict_institutional_gate_enforcement(self):
        """Proof 8: Verifies that failing predictive quality and benchmark gates enforce STRICT NO-GO."""
        json_report = REPORTS_DIR / "v2_4_3_hierarchical_regime_strategy.json"
        with open(json_report, "r", encoding="utf-8") as f:
            data = json.load(f)

        gates = {g["gate"]: g["verdict"] for g in data["acceptance_gates"]}

        self.assertEqual(gates["Multiclass Brier Score"], "FAIL")
        self.assertEqual(gates["Brier Skill Score (BSS)"], "FAIL")
        self.assertEqual(gates["Expected Calibration Error (ECE)"], "FAIL")
        self.assertEqual(gates["Outperformance vs 200-Day SMA Benchmark"], "FAIL")
        self.assertEqual(gates["Live-Money Capital Deployment"], "STRICT NO-GO")


if __name__ == "__main__":
    unittest.main()
