from typing import Dict
from src.execution.governance_engine import SignalGovernanceEngine


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
    Integrates with the Signal Governance Engine for explicit auditing.
    """

    def __init__(self):
        self.governance = SignalGovernanceEngine()

    def validate_trade(self, alpha_signal: Dict, risk_metrics: Dict, market_regime: str):
        # Delegate to institutional governance engine for transparent decision tree
        audit = self.governance.audit_signal(alpha_signal, risk_metrics, market_regime)
        return audit


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
    Coordinates the collaborative intelligence loop with full governance transparency.
    """

    def __init__(self):
        self.alpha_agent = AlphaAgent()
        self.risk_agent = RiskAgent()
        self.execution_agent = ExecutionAgent()

    def run_consensus(self, agreement_data: Dict, risk_metrics: Dict, market_regime: str):
        # 1. Generate Alpha
        alpha = self.alpha_agent.generate_alpha_signal(agreement_data)

        # 2. Risk Validation (The Institutional Governance Pass)
        governance_audit = self.risk_agent.validate_trade(alpha, risk_metrics, market_regime)

        # 3. Execution Routing
        execution = self.execution_agent.propose_execution(
            risk_metrics.get("suggested_allocation", "0%"),
            risk_metrics.get("hedge_ratio_spy", "0%"),
        )

        consensus_action = (
            alpha["signal_idx"] if governance_audit["is_safe"] else 1
        )  # Fallback to HOLD

        return {
            "final_action_idx": consensus_action,
            "agent_responses": {
                "alpha": alpha,
                "governance": governance_audit,
                "execution": execution,
            },
            "consensus_status": "APPROVED" if governance_audit["is_safe"] else "VETOED",
            "agreement": agreement_data["agreement_score"],
            "decision_tree": governance_audit["decision_tree"],
            "veto_reason": governance_audit["veto_reason"],
            "execution_state": governance_audit["execution_state"],
        }
