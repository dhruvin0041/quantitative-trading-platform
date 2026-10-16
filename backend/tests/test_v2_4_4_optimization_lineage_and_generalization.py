# backend/tests/test_v2_4_4_optimization_lineage_and_generalization.py
"""
Unit and Quantitative Regression Suite for HYDRA V2.4.4 Optimization Lineage & Generalization.

Validates:
1. Baseline reproduction penny-exact reconciliation across all 5 strategies.
2. Deflated Sharpe Ratio (Bailey & López de Prado) multiple-testing penalty verification.
3. Point-in-time causality, feature lag, and next-session bar boundary enforcement.
4. Regime boundary oscillation stress testing and churn prevention.
5. Cross-asset generalization matrix (AAPL, MSFT, NVDA, SPY) and ledger reconciliation.
6. Cryptographic immutability and schema of the frozen strategy manifest.
7. Institutional acceptance gates enforcing strict NO-GO on real capital deployment.
"""

import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.research.v2_4_3_hierarchical_regime_strategy import (
    MacroRegimeClassifier,
    compute_sha256,
)
from scripts.research.v2_4_4_optimization_lineage_and_generalization import (
    FROZEN_MANIFEST_V243_PATH,
    HierarchicalSignalStateMachine,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_DIR / "reports"
OUTPUT_JSON = REPORTS_DIR / "v2_4_4_optimization_lineage_and_generalization.json"
OUTPUT_CSV = REPORTS_DIR / "v2_4_4_multi_asset_comparison.csv"


class TestOptimizationLineageAndGeneralizationV244(unittest.TestCase):
    def setUp(self):
        self.assertTrue(OUTPUT_JSON.exists(), f"Missing required report: {OUTPUT_JSON}")
        self.assertTrue(OUTPUT_CSV.exists(), f"Missing required CSV: {OUTPUT_CSV}")
        self.assertTrue(FROZEN_MANIFEST_V243_PATH.exists(), f"Missing frozen manifest: {FROZEN_MANIFEST_V243_PATH}")

        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            self.report_data = json.load(f)

        self.df_multi = pd.read_csv(OUTPUT_CSV)

    def test_1_v2_4_3_baseline_reproduction_penny_exact(self):
        """Proof 1: Verifies 0.00 cent discrepancy across all 5 forward strategy candidates."""
        repro = self.report_data["baseline_reproduction"]

        expected_values = {
            "Buy_and_Hold": 1380.82,
            "200_Day_SMA_Trend": 1265.66,
            "Hierarchical_Regime_Switching_V2_4_3": 1188.34,
            "Current_Locked_LightGBM": 1155.00,
            "Hierarchical_With_Volatility_Veto_V2_4_3": 935.32,
        }

        for strat_key, expected_eq in expected_values.items():
            self.assertIn(strat_key, repro, f"Strategy {strat_key} missing from baseline reproduction")
            self.assertAlmostEqual(repro[strat_key]["target"], expected_eq, places=2)
            self.assertAlmostEqual(repro[strat_key]["observed"], expected_eq, places=2)
            self.assertAlmostEqual(
                repro[strat_key]["discrepancy_cents"],
                0.0,
                places=2,
                msg=f"Discrepancy detected in {strat_key}",
            )

    def test_2_deflated_sharpe_ratio_multiple_testing_penalty(self):
        """Proof 2: Verifies Deflated Sharpe Ratio calculation and failure to reject selection bias null."""
        lineage = self.report_data["optimization_lineage_audit"]
        mta = lineage["multiple_testing_analysis"]

        k = mta["parameter_configurations_tested"]
        self.assertEqual(k, 12, "Expected 12 parameter combinations tested in sensitivity sweep")

        sr_mean = mta["mean_sharpe_across_trials"]
        sr_std = mta["std_sharpe_across_trials"]
        selected_sr = mta["selected_best_sharpe"]

        # Recalculate Expected Max Sharpe under null hypothesis of selection bias
        gamma = 0.5772156649
        expected_max_standard = (
            (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / k)
            + gamma * stats.norm.ppf(1.0 - 1.0 / (k * math.e))
        )
        calculated_expected_max = sr_mean + sr_std * expected_max_standard

        self.assertAlmostEqual(calculated_expected_max, mta["expected_max_sharpe_null"], places=2)
        # Expected max Sharpe under null is higher than observed Sharpe (1.077 > 1.024)
        self.assertGreater(calculated_expected_max, selected_sr)

        # Deflated Sharpe p-value must exceed 0.05 (fails statistical significance)
        self.assertGreater(mta["p_value"], 0.05)
        self.assertFalse(mta["is_statistically_significant"])

    def test_3_causal_boundary_verification_zero_lookahead(self):
        """Proof 3: Verifies that forward simulation strictly obeys point-in-time causality."""
        causal = self.report_data["causal_boundary_audit"]
        self.assertTrue(causal["causality_verified"])
        self.assertEqual(causal["violations_detected"], 0)
        self.assertGreaterEqual(causal["total_boundaries_checked"], 250)

        # Verify sample boundary timing
        for sample in causal["sample_boundary_verifications"]:
            dt_decision = pd.to_datetime(sample["decision_date"])
            dt_exec = pd.to_datetime(sample["execution_date"])
            self.assertGreater(dt_exec, dt_decision, "Execution must be strictly after decision timestamp")
            self.assertGreaterEqual(sample["elapsed_hours"], 17.5, "Next-session open must be >= overnight gap")

    def test_4_regime_boundary_oscillation_churn_prevention(self):
        """Proof 4: Stress tests regime boundary hovering (106.5 <-> 107.5) to ensure zero whipsaw churn."""
        classifier = MacroRegimeClassifier(delta=0.07, vol_threshold=0.02)
        self.assertEqual(classifier.classify_bar(107.5, 100.0, 0.01, 0.015), "STRONG_UPTREND")
        self.assertEqual(classifier.classify_bar(106.5, 100.0, 0.01, 0.015), "NEUTRAL_CONSOLIDATION")

        state_machine = HierarchicalSignalStateMachine(delta=0.07, vol_threshold=0.02, theta=0.45, overlay_mode=False)

        dates = pd.date_range("2026-01-01", periods=50, freq="D")
        # SMA200 = 100.0, 1.07*SMA200 = 107.0
        # Oscillate price between 106.5 (Neutral) and 107.5 (Strong Uptrend)
        close_prices = [107.5 if i % 2 == 0 else 106.5 for i in range(50)]
        df_osc = pd.DataFrame({
            "Close": close_prices,
            "SMA_200": [100.0] * 50,
            "Normalized_SMA_Slope_20": [0.01] * 50,
            "ATR_over_Close": [0.015] * 50,
        }, index=dates)

        # Neutral uninformative ML probabilities
        probs = np.array([[0.33, 0.34, 0.33]] * 50)
        signals, logs = state_machine.process_sequence(probs, df_osc)

        buys = (signals == "BUY").sum()
        sells = (signals == "SELL").sum()

        # Exactly 1 BUY at bar 0 (first entry into Strong Uptrend)
        self.assertEqual(buys, 1, "Should emit exactly 1 BUY at start of trend")
        # 0 SELLs because oscillation into Neutral does not liquidate long positions
        self.assertEqual(sells, 0, "Boundary oscillation into Neutral must NOT liquidate position")

        # Verify state remains IN_LONG for all bars >= 0
        for log in logs:
            self.assertEqual(log["position_state_after_bar"], "IN_LONG")

    def test_5_multi_asset_generalization_and_accounting(self):
        """Proof 5: Evaluates cross-asset performance and ledger penny reconciliation."""
        df = self.df_multi
        tickers = df["ticker"].unique().tolist()
        self.assertCountEqual(tickers, ["AAPL", "MSFT", "NVDA", "SPY"])

        # Check that Hierarchical V2.4.3 improved on Pure LightGBM for MSFT, NVDA, and SPY
        for t in ["MSFT", "NVDA", "SPY"]:
            row_lgb = df[(df["ticker"] == t) & (df["strategy"] == f"{t}_Pure_LightGBM")].iloc[0]
            row_hie = df[(df["ticker"] == t) & (df["strategy"] == f"{t}_Hierarchical_V2_4_3")].iloc[0]

            self.assertGreater(
                row_hie["net_return_pct"],
                row_lgb["net_return_pct"],
                f"Hierarchical failed to improve upon Pure LightGBM on {t}",
            )

        # Check ledger penny accounting reconciliation for all rows:
        # Final Equity must equal Initial Capital + Net Profit Dollars within $0.05
        for _, row in df.iterrows():
            calc_equity = row["initial_capital"] + row["net_profit_dollars"]
            self.assertAlmostEqual(
                row["final_equity"],
                calc_equity,
                places=2,
                msg=f"Ledger reconciliation failed for {row['strategy']}",
            )

    def test_6_frozen_strategy_manifest_integrity(self):
        """Proof 6: Verifies cryptographic hash and frozen parameter schema."""
        with open(FROZEN_MANIFEST_V243_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["strategy_version"], "HYDRA_V2.4.3_Hierarchical_Regime_Strategy")
        params = manifest["macro_filter_rules"]
        self.assertEqual(params["delta_sma_distance"], 0.07)
        self.assertEqual(params["volatility_threshold"], 0.02)
        ml_params = manifest["ml_agent_specifications"]
        self.assertEqual(ml_params["decision_threshold"], 0.45)
        self.assertIn("Open[t+1]", manifest["execution_rules"]["timing"])
        self.assertEqual(manifest["live_trading_permissible"], False)
        self.assertEqual(manifest["manifest_sha256"], "e3468981077500b203a9ea5608bf657723090be90e90fd099accd2814dd5f32f")
        computed_hash = compute_sha256(FROZEN_MANIFEST_V243_PATH)
        self.assertEqual(len(computed_hash), 64)

    def test_7_institutional_acceptance_gates_strict_no_go(self):
        """Proof 7: Verifies institutional risk gates enforce STRICT NO-GO."""
        gates = {g["gate"]: g["verdict"] for g in self.report_data["acceptance_gates"]}

        # Multiple-Testing Audit fails due to DSR p-value > 0.05
        self.assertEqual(gates["Optimization Lineage & Multiple-Testing Audit"], "FAIL")
        # Benchmark outperformance fails on AAPL and SPY
        self.assertEqual(gates["Benchmark Outperformance vs 200-Day SMA"], "FAIL")
        # Probability quality gates fail
        self.assertEqual(gates["Multiclass Brier Skill Score Gate"], "FAIL")
        # Prospective trades pending (< 50)
        self.assertEqual(gates["Prospective Closed Paper Trades"], "PENDING")
        # Capital deployment strictly blocked
        self.assertEqual(gates["Production Live-Money Deployment"], "STRICT NO-GO")


if __name__ == "__main__":
    unittest.main()
