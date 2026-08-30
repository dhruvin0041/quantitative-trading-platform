from typing import Dict, List
import pandas as pd
from src.execution.governance_engine import SignalGovernanceEngine


class AlphaAgent:
    """
    Maximizes expected return using hybrid ensemble signals.
    """

    def generate_alpha_signal(self, agreement_data: Dict):
        signal_idx = agreement_data["dominant_idx"]
        confidence = agreement_data["agreement_score"] / 100.0
        return {
            "ticker": agreement_data.get("ticker", "UNKNOWN"),
            "signal_idx": signal_idx, 
            "confidence": confidence
        }


class RiskAgent:
    """
    Portfolio-Level Arbiter: Validates signals collectively across the 500-stock panel.
    Enforces VaR limits, detects sector crowding, and applies Volatility-Scaled Sizing (ATR).
    """

    def __init__(self, max_sector_exposure=0.15, max_portfolio_var=0.05):
        self.max_sector_exposure = max_sector_exposure  # e.g. 15% max capital per sector
        self.max_portfolio_var = max_portfolio_var      # e.g. 5% max portfolio VaR threshold
        self.governance = SignalGovernanceEngine()

    def evaluate_portfolio(self, alpha_signals: List[Dict], panel_data: pd.DataFrame, market_regime: str) -> Dict:
        """
        Takes independent signals from AlphaAgent for the panel and evaluates them collectively.
        """
        approved_trades = []
        rejected_trades = []
        
        # 1. Filter out non-actionable signals (Hold) and low confidence floor
        active_signals = [s for s in alpha_signals if s["signal_idx"] in [0, 2] and s["confidence"] >= 0.65]
        
        if not active_signals:
            return {"status": "NO_TRADES", "approved_allocations": {}, "rejected_trades": []}

        # 2. Volatility-Scaled Sizing (Inverse ATR)
        # We calculate the inverse of volatility (ATR) to allocate smaller positions to highly volatile stocks
        inv_vol_sum = 0
        for sig in active_signals:
            ticker = sig["ticker"]
            # Fallback to 1.0 if ATR missing to prevent crash
            atr = panel_data.loc[ticker, "ATR"] if "ATR" in panel_data.columns and ticker in panel_data.index else 1.0
            atr = max(atr, 1e-4) # Prevent division by zero
            sig["inv_atr"] = 1.0 / atr
            inv_vol_sum += sig["inv_atr"]
            
        for sig in active_signals:
            sig["target_weight"] = sig["inv_atr"] / inv_vol_sum

        # 3. Portfolio-Level Correlation & Crowding Check
        sector_exposure = {}
        for sig in active_signals:
            ticker = sig["ticker"]
            sector = panel_data.loc[ticker, "Sector"] if "Sector" in panel_data.columns and ticker in panel_data.index else "Unknown"
            sig["sector"] = sector
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + sig["target_weight"]

        # Veto lower-conviction trades if sector exposure exceeds cap
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_exposure:
                # Sort sector signals by lowest confidence first
                sector_sigs = sorted([s for s in active_signals if s["sector"] == sector], key=lambda x: x["confidence"])
                
                while exposure > self.max_sector_exposure and sector_sigs:
                    dropped_sig = sector_sigs.pop(0)
                    exposure -= dropped_sig["target_weight"]
                    active_signals.remove(dropped_sig)
                    rejected_trades.append({
                        "ticker": dropped_sig["ticker"], 
                        "veto_reason": f"Sector Crowding (Exceeded {self.max_sector_exposure*100}%)", 
                        "veto_code": "CROWDING_VETO"
                    })

        # Recalculate weights after crowding purge (Normalization)
        if not active_signals:
            return {"status": "NO_TRADES", "approved_allocations": {}, "rejected_trades": rejected_trades}

        inv_vol_sum_adj = sum(sig["inv_atr"] for sig in active_signals)
        for sig in active_signals:
            sig["final_weight"] = sig["inv_atr"] / inv_vol_sum_adj

        # 4. Value at Risk (VaR) Veto Check
        # Aggregate Portfolio VaR proxy
        portfolio_var = 0.0
        for sig in active_signals:
            ticker = sig["ticker"]
            asset_var = panel_data.loc[ticker, "VaR_95"] if "VaR_95" in panel_data.columns and ticker in panel_data.index else 0.02
            portfolio_var += sig["final_weight"] * asset_var

        if portfolio_var > self.max_portfolio_var:
            return {
                "status": "VETOED",
                "veto_reason": f"Portfolio VaR ({portfolio_var*100:.2f}%) exceeds absolute threshold ({self.max_portfolio_var*100:.2f}%)",
                "veto_code": "VAR_LIMIT_BREACH",
                "approved_allocations": {},
                "rejected_trades": active_signals + rejected_trades
            }

        approved_allocations = {
            sig["ticker"]: {
                "signal_idx": sig["signal_idx"],
                "confidence": sig["confidence"],
                "weight": sig["final_weight"],
                "sector": sig["sector"]
            } for sig in active_signals
        }

        return {
            "status": "APPROVED",
            "portfolio_var": portfolio_var,
            "approved_allocations": approved_allocations,
            "rejected_trades": rejected_trades
        }


class ExecutionAgent:
    """
    Optimizes for best fill and predictive liquidity.
    """

    def propose_execution(self, approved_allocations: Dict):
        # Now routes batch orders for the approved allocations
        return {
            "venue": "DARK_POOL_AGGREGATOR",
            "urgency": "MEDIUM",
            "orders": approved_allocations
        }


class InstitutionalOrchestrator:
    """
    SOTA 2026 Agentic Mesh Orchestrator.
    Now processes full 500-equity panel datasets at once to enable
    portfolio-level governance and dynamic execution routing.
    """

    def __init__(self):
        self.alpha_agent = AlphaAgent()
        self.risk_agent = RiskAgent()
        self.execution_agent = ExecutionAgent()

    def run_consensus(self, agreement_data: Dict, risk_data: Dict, market_regime: str):
        """Fallback for single-asset inference."""
        alpha = self.alpha_agent.generate_alpha_signal(agreement_data)
        
        gov_result = self.risk_agent.governance.audit_signal(
            alpha_signal=alpha,
            risk_metrics={"expected_value": risk_data},
            market_regime=market_regime
        )
        
        final_action_idx = alpha["signal_idx"] if gov_result["is_safe"] else 1
        
        return {
            "final_action_idx": final_action_idx,
            "consensus_status": gov_result["execution_state"],
            "veto_reason": gov_result["veto_reason"],
            "veto_code": gov_result["veto_code"],
            "decision_tree": gov_result["decision_tree"]
        }

    def run_panel_consensus(self, panel_agreement_data: List[Dict], panel_data: pd.DataFrame, market_regime: str):
        # 1. Generate Alpha across the entire panel
        alpha_signals = [self.alpha_agent.generate_alpha_signal(data) for data in panel_agreement_data]

        # 2. Risk Validation (The Institutional Governance Pass across the Portfolio)
        governance_audit = self.risk_agent.evaluate_portfolio(alpha_signals, panel_data, market_regime)

        # 3. Execution Routing
        if governance_audit["status"] == "APPROVED":
            execution = self.execution_agent.propose_execution(governance_audit["approved_allocations"])
        else:
            execution = {"venue": "NONE", "urgency": "NONE", "orders": {}}

        return {
            "consensus_status": governance_audit["status"],
            "portfolio_var": governance_audit.get("portfolio_var", 0.0),
            "approved_trades_count": len(governance_audit["approved_allocations"]),
            "rejected_trades": governance_audit["rejected_trades"],
            "execution": execution
        }
