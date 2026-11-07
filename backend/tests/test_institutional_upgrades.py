import unittest

import numpy as np
import pandas as pd

from src.execution.cross_sectional_ranker import CrossSectionalUniverseRanker
from src.execution.microstructure_execution import MicrostructureExecutionModel
from src.execution.paper_trading import PaperTradingEngine
from src.execution.statistical_engine import (
    StatisticalSufficiencyEngine,
    calculate_deflated_sharpe_ratio,
    calculate_expected_maximum_sharpe,
    calculate_probabilistic_sharpe_ratio,
)
from src.optimization.objective_functions import (
    deflated_sharpe_objective,
    sharpe_objective,
    simulate_strategy_returns,
)


class TestInstitutionalUpgrades(unittest.TestCase):
    """
    Unit test suite verifying the institutional upgrades:
    1. Marcos López de Prado's Deflated Sharpe Ratio (DSR) & Probabilistic Sharpe Ratio (PSR).
    2. Dynamic Microstructure Execution & Adverse Selection Model.
    3. Cross-Sectional Universe Ranking & Multi-Asset Gating Engine.
    4. Cost-Adjusted & Deflated Sharpe Optimization Objectives.
    """

    def setUp(self):
        np.random.seed(42)

    def test_probabilistic_sharpe_ratio_normal_and_fat_tails(self):
        # A positive Sharpe with 252 days should have PSR > 0.5 against benchmark 0.0
        psr_pos = calculate_probabilistic_sharpe_ratio(
            observed_sr=1.5, benchmark_sr=0.0, n_samples=252, skewness=0.0, kurtosis=3.0
        )
        self.assertGreater(psr_pos, 0.90)

        # Zero Sharpe should have PSR == 0.5 against benchmark 0.0
        psr_zero = calculate_probabilistic_sharpe_ratio(
            observed_sr=0.0, benchmark_sr=0.0, n_samples=252, skewness=0.0, kurtosis=3.0
        )
        self.assertAlmostEqual(psr_zero, 0.5, places=3)

        # Negative skewness and high kurtosis (fat tails) must reduce confidence (lower PSR)
        psr_normal = calculate_probabilistic_sharpe_ratio(
            observed_sr=1.2, benchmark_sr=0.0, n_samples=252, skewness=0.0, kurtosis=3.0
        )
        psr_skewed = calculate_probabilistic_sharpe_ratio(
            observed_sr=1.2, benchmark_sr=0.0, n_samples=252, skewness=-1.5, kurtosis=8.0
        )
        self.assertGreater(
            psr_normal, psr_skewed, "Fat tails and negative skew must penalize PSR"
        )

    def test_deflated_sharpe_ratio_multiple_testing_penalty(self):
        # When testing K=250 trials, the expected max Sharpe under null H0 must be positive
        expected_max_250 = calculate_expected_maximum_sharpe(n_trials=250, n_samples=252)
        expected_max_10 = calculate_expected_maximum_sharpe(n_trials=10, n_samples=252)
        self.assertGreater(
            expected_max_250,
            expected_max_10,
            "More trials must produce a higher false-positive Sharpe hurdle",
        )

        # A mediocre Sharpe (e.g. 1.0) with 250 trials should fail statistical significance
        dsr_mediocre = calculate_deflated_sharpe_ratio(
            observed_sr=1.0, n_trials=250, n_samples=252, skewness=0.0, kurtosis=3.0
        )
        self.assertFalse(
            dsr_mediocre["is_statistically_significant"],
            "Sharpe 1.0 after 250 trials must not be deemed statistically significant",
        )
        self.assertGreater(dsr_mediocre["p_value"], 0.05)

        # An institutional-grade Sharpe (e.g. 3.5) with 500 samples must pass significance
        dsr_elite = calculate_deflated_sharpe_ratio(
            observed_sr=3.5, n_trials=250, n_samples=500, skewness=0.0, kurtosis=3.0
        )
        self.assertTrue(
            dsr_elite["is_statistically_significant"],
            "Sharpe 3.5 across 500 samples must pass DSR 95% hurdle",
        )
        self.assertGreaterEqual(dsr_elite["dsr"], 0.95)

    def test_statistical_sufficiency_engine_dsr_integration(self):
        engine = StatisticalSufficiencyEngine()
        returns = list(np.random.normal(0.001, 0.015, 252))
        trades = [{"action": "SELL", "pnl": 50.0}] * 35
        summary = {
            "total_return": 18.5,
            "sharpe": 1.45,
            "sortino": 2.10,
            "calmar": 1.20,
            "win_rate": 58.0,
            "profit_factor": 1.65,
            "max_drawdown": -0.12,
            "trials_count": 250,
        }
        res = engine.validate_and_reconcile(summary, returns, trades)
        val = res["validated_metrics"]

        # Ensure PSR, DSR, and expected max Sharpe null are present and formatted
        self.assertIn("psr", val)
        self.assertIn("dsr", val)
        self.assertIn("expected_max_sharpe_null", val)
        self.assertIn("is_statistically_significant", val)
        self.assertIn("dsr_p_value", res["confidence_intervals"])
        self.assertIn("return_skewness", res["confidence_intervals"])

    def test_microstructure_execution_model_friction(self):
        model = MicrostructureExecutionModel(base_spread_bps=3.0, impact_eta=0.15)

        # Volatility scaling: higher volatility should yield higher spread
        low_vol_res = model.calculate_adverse_slippage(
            price=100.0, action="BUY", volatility_pct=0.01
        )
        high_vol_res = model.calculate_adverse_slippage(
            price=100.0, action="BUY", volatility_pct=0.04
        )
        self.assertGreater(high_vol_res["spread_bps"], low_vol_res["spread_bps"])

        # Market impact: larger order relative to bar volume must increase friction
        small_order = model.calculate_adverse_slippage(
            price=100.0,
            action="BUY",
            volatility_pct=0.015,
            order_shares=100,
            bar_volume=1_000_000,
        )
        large_order = model.calculate_adverse_slippage(
            price=100.0,
            action="BUY",
            volatility_pct=0.015,
            order_shares=20_000,
            bar_volume=1_000_000,
        )
        self.assertGreater(
            large_order["market_impact_bps"], small_order["market_impact_bps"]
        )

        # Adverse selection: buying during upward price surge costs extra
        neutral_order = model.calculate_adverse_slippage(
            price=100.0, action="BUY", recent_momentum_pct=0.0
        )
        surging_order = model.calculate_adverse_slippage(
            price=100.0, action="BUY", recent_momentum_pct=0.03
        )
        self.assertGreater(
            surging_order["adverse_selection_bps"],
            neutral_order["adverse_selection_bps"],
        )

        # Execution price: BUY executed price must be > nominal; SELL must be < nominal
        self.assertGreater(surging_order["executed_price"], 100.0)
        sell_order = model.calculate_adverse_slippage(
            price=100.0, action="SELL", recent_momentum_pct=-0.02
        )
        self.assertLess(sell_order["executed_price"], 100.0)

    def test_paper_trading_engine_microstructure_integration(self):
        engine = PaperTradingEngine(db_path="data/test_paper_trading_temp.json")
        try:
            # Verify _simulate_slippage executes without error and applies dynamic price
            buy_price = engine._simulate_slippage(150.0, "BUY")
            sell_price = engine._simulate_slippage(150.0, "SELL")
            hold_price = engine._simulate_slippage(150.0, "HOLD")

            self.assertGreater(buy_price, 150.0)
            self.assertLess(sell_price, 150.0)
            self.assertEqual(hold_price, 150.0)
        finally:
            import os

            if os.path.exists("data/test_paper_trading_temp.json"):
                os.remove("data/test_paper_trading_temp.json")

    def test_cross_sectional_universe_ranker(self):
        ranker = CrossSectionalUniverseRanker(benchmark_ticker="SPY")
        dates = pd.date_range("2024-01-01", periods=40, freq="D")

        # Create 5 distinct asset curves: AAPL outperforms strongly, TSLA plunges, SPY baseline
        prices = {
            "SPY": pd.Series(np.linspace(100, 105, 40), index=dates),
            "AAPL": pd.Series(np.linspace(100, 130, 40), index=dates),  # +30% leader
            "MSFT": pd.Series(np.linspace(100, 110, 40), index=dates),  # +10%
            "AMZN": pd.Series(np.linspace(100, 102, 40), index=dates),  # +2%
            "TSLA": pd.Series(np.linspace(100, 80, 40), index=dates),  # -20% laggard
        }

        ranks = ranker.compute_cross_sectional_ranks(prices, lookback_days=20)
        self.assertEqual(len(ranks), 5)

        # AAPL must be leader (top percentile)
        self.assertTrue(ranks["AAPL"]["is_leader"])
        self.assertGreaterEqual(ranks["AAPL"]["rank_percentile"], 0.75)

        # TSLA must be laggard (bottom percentile)
        self.assertTrue(ranks["TSLA"]["is_laggard"])
        self.assertLessEqual(ranks["TSLA"]["rank_percentile"], 0.25)

        # Signal gating tests
        # 1. BUY on AAPL (leader) gets boosted confidence
        gated_leader = ranker.gate_signal_with_cross_sectional_rank(
            "AAPL", "BUY", 75.0, ranks["AAPL"]
        )
        self.assertEqual(gated_leader["cross_sectional_status"], "LEADER_CONFIRMED")
        self.assertGreater(gated_leader["adjusted_confidence"], 75.0)

        # 2. BUY on TSLA (extreme laggard) is suppressed / vetoed
        gated_laggard = ranker.gate_signal_with_cross_sectional_rank(
            "TSLA", "BUY", 75.0, ranks["TSLA"]
        )
        self.assertIn(
            gated_laggard["cross_sectional_status"],
            ["LAGGARD_SUPPRESSION", "NEUTRAL"],
        )
        self.assertLessEqual(gated_laggard["adjusted_confidence"], 75.0)

    def test_cost_adjusted_and_deflated_sharpe_objective(self):
        # 50 periods with constant small positive returns
        true_returns = np.array([0.005] * 50)
        y_true = np.array([2] * 50)
        # Probabilities flipping between BUY and SELL every step (high turnover churn)
        y_pred_flip = np.array(
            [[0.1, 0.1, 0.8] if i % 2 == 0 else [0.8, 0.1, 0.1] for i in range(50)]
        )

        # Standard costless return would be positive
        costless_ret = simulate_strategy_returns(
            y_true, y_pred_flip, true_returns, threshold=0.5, friction_bps=0.0
        )
        # Cost-adjusted return must be lower due to friction deducted on every flip
        friction_ret = simulate_strategy_returns(
            y_true, y_pred_flip, true_returns, threshold=0.5, friction_bps=10.0
        )
        self.assertGreater(
            np.sum(costless_ret),
            np.sum(friction_ret),
            "Turnover friction must penalize strategy returns",
        )

        # Objective functions must execute cleanly and return valid floats
        cost_sharpe = sharpe_objective(
            y_true, y_pred_flip, true_returns, threshold=0.5, friction_bps=5.0
        )
        self.assertIsInstance(cost_sharpe, float)

        dsr_obj = deflated_sharpe_objective(
            y_true,
            y_pred_flip,
            true_returns,
            threshold=0.5,
            friction_bps=5.0,
            n_trials=250,
        )
        self.assertIsInstance(dsr_obj, float)


if __name__ == "__main__":
    unittest.main()
