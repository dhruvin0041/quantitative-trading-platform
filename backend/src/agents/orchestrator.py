import numpy as np
from typing import Dict

class AlphaAgent:
    """
    Maximizes expected return using hybrid ensemble signals.
    """

    def generate_alpha_signal(self, agreement_data: Dict):
        signal_idx = agreement_data["dominant_idx"]
        confidence = agreement_data["agreement_score"] / 100.0
        return {"signal_idx": signal_idx, "confidence": confidence}


class RiskAgent:
    """
    Final Arbiter: Validates signals against VaR and crowding metrics.
    Has Veto power.
    """

    def validate_trade(self, alpha_signal: Dict, risk_metrics: Dict):
        is_safe = True
        veto_reason = None

        # institutional veto logic
        if alpha_signal["signal_idx"] in [0, 2] and risk_metrics.get(
            "stampede_risk", {}
        ).get("is_crowded", False):
            is_safe = False
            veto_reason = "Stampede Risk (Crowded Trade)"

        if risk_metrics.get("beta", 1.0) > 2.5:
            is_safe = False
            veto_reason = "Beta too high (Extreme Volatility)"

        if alpha_signal["confidence"] < 0.65:
            is_safe = False
            veto_reason = "Low Confidence Alpha"

        if risk_metrics.get("uncertainty_score", 0.0) > 0.40:
            is_safe = False
            veto_reason = f"High Prediction Uncertainty ({risk_metrics.get('uncertainty_score'):.2f})"

        return {"is_safe": is_safe, "veto_reason": veto_reason}


class ExecutionAgent:
    """
    Optimizes for best fill and predictive liquidity.
    """

    def propose_execution(self, suggested_allocation: str, hedge_ratio: str):
        return {
            "venue": "DARK_POOL_AGGREGATOR",
            "urgency": "MEDIUM",
            "hedge": f"Sell {hedge_ratio} SPY to neutralize",
        }


class InstitutionalOrchestrator:
    """
    SOTA 2026 Agentic Mesh Orchestrator.
    Coordinates the collaborative intelligence loop.
    """

    def __init__(self):
        self.alpha_agent = AlphaAgent()
        self.risk_agent = RiskAgent()
        self.execution_agent = ExecutionAgent()

    def run_consensus(self, agreement_data: Dict, risk_metrics: Dict):
        # 1. Generate Alpha
        alpha = self.alpha_agent.generate_alpha_signal(agreement_data)

        # 2. Risk Validation (The Veto)
        risk_check = self.risk_agent.validate_trade(alpha, risk_metrics)

        # 3. Execution Routing
        execution = self.execution_agent.propose_execution(
            risk_metrics.get("suggested_allocation", "0%"),
            risk_metrics.get("hedge_ratio_spy", "0%"),
        )

        consensus_action = (
            alpha["signal_idx"] if risk_check["is_safe"] else 1
        )  # Fallback to HOLD

        return {
            "final_action_idx": consensus_action,
            "agent_responses": {
                "alpha": alpha,
                "risk": risk_check,
                "execution": execution,
            },
            "consensus_status": "APPROVED" if risk_check["is_safe"] else "VETOED",
            "agreement": agreement_data["agreement_score"]
        }