import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class SignalGovernanceEngine:
    """
    Institutional Signal Governance Engine.
    Provides a Signal Decision Tree and explicit VETO reasoning for audit-grade transparency.
    """

    def audit_signal(self, alpha_signal: Dict, risk_metrics: Dict, market_regime: str) -> Dict[str, Any]:
        """
        Executes the Signal Decision Tree to determine final execution permission.
        """
        is_safe = True
        veto_reason = None
        veto_code = "NONE"
        severity = "STABLE"

        # 1. Confidence Check
        if alpha_signal["confidence"] < 0.65:
            is_safe = False
            veto_reason = "Alpha confidence below institutional floor (65%)"
            veto_code = "CONFIDENCE_DEFICIT"
            severity = "DEFENSIVE"

        # 2. EV Check
        ev = risk_metrics.get("expected_value", {}).get("ev_pct", 0.0)
        if is_safe and ev < 0:
            is_safe = False
            veto_reason = "Negative Expected Value (EV) detected"
            veto_code = "EV_DEFICIT"
            severity = "CRITICAL"

        # 3. Regime Conflict
        if is_safe:
            signal_type = "BUY" if alpha_signal["signal_idx"] == 2 else "SELL" if alpha_signal["signal_idx"] == 0 else "HOLD"
            if signal_type == "BUY" and "BEAR" in market_regime:
                is_safe = False
                veto_reason = f"Counter-trend entry blocked: {alpha_signal['confidence']*100:.0f}% BUY signal in {market_regime} regime"
                veto_code = "REGIME_CONFLICT"
                severity = "DEFENSIVE"
            elif signal_type == "SELL" and "BULL" in market_regime:
                is_safe = False
                veto_reason = f"Counter-trend exit blocked: {alpha_signal['confidence']*100:.0f}% SELL signal in {market_regime} regime"
                veto_code = "REGIME_CONFLICT"
                severity = "DEFENSIVE"

        # 4. Uncertainty & Entropy
        uncertainty = risk_metrics.get("uncertainty_score", 0.0)
        if is_safe and uncertainty > 0.35:  # Tightened floor
            is_safe = False
            veto_reason = f"Low Consensus / High Prediction Uncertainty ({uncertainty:.2f})"
            veto_code = "CONFIDENCE_DEFICIT"
            severity = "CRITICAL"

        # 5. Crowding & Liquidity
        if is_safe:
            liquidity = risk_metrics.get("liquidity_score", 1.0)
            if liquidity < 0.40:
                is_safe = False
                veto_reason = f"Insufficient Liquidity / High Slippage Risk ({liquidity:.2f})"
                veto_code = "LIQUIDITY_PENALTY"
                severity = "DEFENSIVE"
            elif risk_metrics.get("stampede_risk", {}).get("is_crowded", False):
                is_safe = False
                veto_reason = "Institutional crowding detected (Stampede Risk)"
                veto_code = "CROWDING_VETO"
                severity = "DEFENSIVE"

        decision_tree = [
            {"node": "Alpha Generation", "status": "PASS" if alpha_signal["confidence"] >= 0.65 else "FAIL", "detail": f"Conf: {alpha_signal['confidence']*100:.1f}%"},
            {"node": "Expected Value", "status": "PASS" if ev >= 0 else "FAIL", "detail": f"EV: {ev:.2f}%"},
            {"node": "Regime Filter", "status": "PASS" if veto_code != "REGIME_CONFLICT" else "FAIL", "detail": market_regime},
            {"node": "Consensus Auth", "status": "PASS" if uncertainty <= 0.35 else "FAIL", "detail": f"Ent: {uncertainty:.2f}"},
            {"node": "Execution Safety", "status": "PASS" if veto_code not in ["CROWDING_VETO", "LIQUIDITY_PENALTY"] else "FAIL", "detail": "Stable" if veto_code == "NONE" else "Restricted"},
        ]


        return {
            "is_safe": is_safe,
            "veto_reason": veto_reason,
            "veto_code": veto_code,
            "governance_severity": severity,
            "decision_tree": decision_tree,
            "execution_state": "APPROVED" if is_safe else "VETOED",
        }


class SignalGovernanceAnalytics:
    """
    Tracks veto rates, approval rates, and prevents signal starvation/paralysis.
    """

    def analyze_throughput(self, signal_journal: pd.DataFrame) -> Dict[str, Any]:
        if signal_journal.empty:
            return {
                "veto_rate": 0.0,
                "approval_rate": 0.0,
                "signal_density": 0.0,
                "governance_status": "STABLE",
            }

        total = len(signal_journal)
        vetoes = len(signal_journal[signal_journal["signal_type"] == "VETOED"])
        approvals = total - vetoes

        veto_rate = (vetoes / total) * 100
        approval_rate = (approvals / total) * 100

        # Determine governance status
        status = "STABLE"
        if veto_rate > 85:
            status = "PARALYSIS_RISK"
        elif veto_rate < 10:
            status = "CROWDING_RISK"

        recent_window = signal_journal.tail(20)
        recent_trades = len(
            recent_window[recent_window["signal_type"].isin(["BUY", "SELL"])]
        )

        starvation = False
        if recent_trades == 0 and total > 20:
            starvation = True
            status = "SIGNAL_STARVATION"

        return {
            "veto_rate": round(veto_rate, 1),
            "approval_rate": round(approval_rate, 1),
            "total_signals": total,
            "governance_status": status,
            "signal_starvation": starvation,
            "throughput_coherence": "HIGH"
            if not starvation and 20 <= veto_rate <= 60
            else "LOW",
        }
