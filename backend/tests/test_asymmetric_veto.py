import unittest

import numpy as np

from src.execution.asset_intelligence import (
    MODEL_REGISTRY,
    AdaptiveWeightingEngine,
    ModelRole,
)
from src.execution.consensus_engine import WeightedConsensusEngine
from src.optimization.objective_functions import calculate_calmar_ratio


class TestAsymmetricVetoAndCalmar(unittest.TestCase):
    def setUp(self):
        self.consensus_engine = WeightedConsensusEngine()
        self.weighting_engine = AdaptiveWeightingEngine()

    def test_calmar_sign_preservation_positive_returns(self):
        """Positive returns must yield a strictly positive Calmar ratio."""
        returns = np.array([0.02, 0.01, 0.03, -0.01, 0.02])
        calmar = calculate_calmar_ratio(returns, max_drawdown=0.10, sim_years=1.0)
        self.assertGreater(calmar, 0.0)
        self.assertAlmostEqual(calmar, 0.07 / 0.10, places=4)

    def test_calmar_sign_preservation_negative_returns(self):
        """Net-negative returns MUST yield a strictly negative Calmar ratio (no abs())."""
        returns = np.array([-0.02, -0.01, -0.03, 0.01, -0.02])
        calmar = calculate_calmar_ratio(returns, max_drawdown=0.20, sim_years=1.0)
        self.assertLess(calmar, 0.0)
        self.assertAlmostEqual(calmar, -0.07 / 0.20, places=4)

    def test_calmar_zero_returns_or_zero_drawdown(self):
        """Zero drawdown or zero returns should return 0.0 without divide-by-zero error."""
        returns_zero = np.array([0.0, 0.0, 0.0])
        self.assertEqual(calculate_calmar_ratio(returns_zero, max_drawdown=0.0), 0.0)

        returns_pos = np.array([0.01, 0.02])
        self.assertEqual(calculate_calmar_ratio(returns_pos, max_drawdown=0.0), 0.0)

    def test_calmar_percentage_max_drawdown_scaling(self):
        """If max_drawdown is passed as percentage (e.g. 20.0 for 20%), it should scale appropriately."""
        returns = np.array([0.05, 0.05])  # sum = 0.10
        calmar_pct = calculate_calmar_ratio(returns, max_drawdown=20.0, sim_years=1.0)
        calmar_dec = calculate_calmar_ratio(returns, max_drawdown=0.20, sim_years=1.0)
        self.assertAlmostEqual(calmar_pct, calmar_dec, places=4)
        self.assertAlmostEqual(calmar_pct, 0.10 / 0.20, places=4)

    def test_asymmetric_veto_pass_through_on_agreement(self):
        """When primary model (XGB) has high conviction and secondary agrees, signal passes through unmuted."""
        base_probs = {
            "XGB_AGENT": np.array([0.10, 0.15, 0.75]),  # Strong BUY (idx 2)
            "LGBM_AGENT": np.array([0.15, 0.20, 0.65]),  # Agreeing BUY
            "DQN_AGENT": np.array([0.20, 0.30, 0.50]),  # Neutral/Moderate
        }
        res = self.consensus_engine.compute_asymmetric_veto(
            base_probs,
            primary_key="XGB_AGENT",
            primary_threshold=0.60,
            veto_threshold=0.65,
        )
        self.assertFalse(res["is_vetoed"])
        self.assertEqual(res["dominant_direction"], "BUY")
        self.assertEqual(res["dominant_idx"], 2)
        self.assertAlmostEqual(res["agreement_score"], 75.0)

    def test_asymmetric_veto_blocks_buy_on_secondary_sell(self):
        """When primary model (XGB) signals BUY, but secondary model (LGBM or DQN) signals SELL (>=0.65), veto!"""
        base_probs = {
            "XGB_AGENT": np.array([0.10, 0.20, 0.70]),  # BUY (idx 2)
            "LGBM_AGENT": np.array([0.70, 0.15, 0.15]),  # Strong SELL (idx 0 >= 0.65)
            "DQN_AGENT": np.array([0.20, 0.40, 0.40]),
        }
        res = self.consensus_engine.compute_asymmetric_veto(
            base_probs,
            primary_key="XGB_AGENT",
            primary_threshold=0.60,
            veto_threshold=0.65,
        )
        self.assertTrue(res["is_vetoed"])
        self.assertEqual(res["dominant_direction"], "HOLD")
        self.assertEqual(res["dominant_idx"], 1)
        self.assertEqual(res["vetoed_by"], "LGBM_AGENT")

    def test_asymmetric_veto_blocks_short_on_secondary_buy(self):
        """When veto_short is True and primary model signals SELL, but secondary model signals BUY (>=0.65), veto!"""
        base_probs = {
            "XGB_AGENT": np.array([0.72, 0.18, 0.10]),  # SELL (idx 0)
            "LGBM_AGENT": np.array([0.45, 0.30, 0.25]),
            "DQN_AGENT": np.array([0.05, 0.15, 0.80]),  # Strong BUY (idx 2 >= 0.65)
        }
        res = self.consensus_engine.compute_asymmetric_veto(
            base_probs,
            primary_key="XGB_AGENT",
            primary_threshold=0.60,
            veto_threshold=0.65,
            veto_short=True,
        )
        self.assertTrue(res["is_vetoed"])
        self.assertEqual(res["dominant_direction"], "HOLD")
        self.assertEqual(res["dominant_idx"], 1)
        self.assertEqual(res["vetoed_by"], "DQN_AGENT")

    def test_quarantined_dl_fusion_cannot_veto(self):
        """DL_FUSION is quarantined and must never trigger a veto even with high conviction."""
        base_probs = {
            "XGB_AGENT": np.array([0.10, 0.20, 0.70]),  # BUY
            "DL_FUSION": np.array([0.95, 0.03, 0.02]),  # Quarantined model disagrees
            "LGBM_AGENT": np.array([0.20, 0.30, 0.50]),  # Neutral
            "DQN_AGENT": np.array([0.20, 0.40, 0.40]),  # Neutral
        }
        res = self.consensus_engine.compute_asymmetric_veto(
            base_probs,
            primary_key="XGB_AGENT",
            primary_threshold=0.60,
            veto_threshold=0.65,
        )
        self.assertFalse(res["is_vetoed"])
        self.assertEqual(res["dominant_direction"], "BUY")
        self.assertEqual(res["model_intelligence"]["DL_FUSION"]["role"], "QUARANTINED")

    def test_primary_conviction_gate(self):
        """If primary model conviction < 0.60, dominant direction is HOLD without veto trigger."""
        base_probs = {
            "XGB_AGENT": np.array([0.30, 0.55, 0.15]),  # Conviction < 0.60
            "LGBM_AGENT": np.array([0.10, 0.20, 0.70]),
        }
        res = self.consensus_engine.compute_asymmetric_veto(
            base_probs,
            primary_key="XGB_AGENT",
            primary_threshold=0.60,
            veto_threshold=0.65,
        )
        self.assertFalse(res["is_vetoed"])
        self.assertEqual(res["dominant_direction"], "HOLD")
        self.assertEqual(res["dominant_idx"], 1)

    def test_model_registry_hierarchy(self):
        """Verify model roles in MODEL_REGISTRY and AdaptiveWeightingEngine."""
        self.assertEqual(MODEL_REGISTRY["XGB_AGENT"]["role"], ModelRole.PRIMARY_ALPHA_DRIVER)
        self.assertEqual(MODEL_REGISTRY["XGB_AGENT"]["status"], "ACTIVE")
        self.assertEqual(MODEL_REGISTRY["LGBM_AGENT"]["role"], ModelRole.SECONDARY_VETO)
        self.assertEqual(MODEL_REGISTRY["DQN_AGENT"]["role"], ModelRole.SECONDARY_VETO)
        self.assertEqual(MODEL_REGISTRY["DL_FUSION"]["role"], ModelRole.QUARANTINED)
        self.assertEqual(MODEL_REGISTRY["DL_FUSION"]["status"], "QUARANTINED")

        weights = self.weighting_engine.calculate_weights("BULL_TREND", "EQUITY")
        self.assertEqual(weights["XGB_AGENT"]["weight"], 1.0)
        self.assertEqual(weights["DL_FUSION"]["weight"], 0.0)
        self.assertEqual(weights["DL_FUSION"]["status"], "QUARANTINED")


if __name__ == "__main__":
    unittest.main()
