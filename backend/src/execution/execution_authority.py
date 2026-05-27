from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExecutionAuthorityEngine:
    """
    Phase 2: Final Execution Authority Layer.
    Converts all lower-level telemetry (alpha, risk, regime, confidence)
    into a final institutional action state.
    """

    def __init__(self):
        # Confidence thresholds for execution states
        self.exec_threshold = 70.0  # Require 70% confidence for full execution
        self.observe_threshold = 40.0  # Below 40% is observe only

    def determine_execution_state(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final decision logic for institutional execution.
        """
        # Primary Inputs
        signal_note = signal_data.get("signal_note")
        ev_pct = signal_data.get("expected_value", {}).get("ev_pct", 0.0)
        confidence = signal_data.get("explainable_confidence", 0.0)
        agreement = signal_data.get("agreement", 0.0)
        risk_veto = signal_data.get("signal") == "VETOED"
        uncertainty = signal_data.get("uncertainty_score", 100.0)
        quality_score = signal_data.get("quality", {}).get("score", 0.0)

        # Action State Logic
        action_state = "BLOCKED"
        reasoning = "System default safety state."

        # 1. Hard Vetoes
        if risk_veto:
            action_state = "VETOED"
            reasoning = signal_note if signal_note else "Risk Agent Veto."
        elif ev_pct <= 0:
            action_state = "BLOCKED"
            reasoning = "Negative Expected Value (EV) detected."

        # 2. Quality & Confidence Gates
        elif quality_score < 40:
            action_state = "BLOCKED"
            reasoning = (
                f"Institutional Quality Score ({quality_score}) below safety threshold."
            )
        elif uncertainty > 50:
            action_state = "OBSERVE ONLY"
            reasoning = "Excessive prediction uncertainty (Noise > 50%)."
        elif confidence < self.observe_threshold:
            action_state = "OBSERVE ONLY"
            reasoning = "Insufficient confidence for deployment."

        # 3. Execution Levels
        elif confidence < self.exec_threshold:
            action_state = "REDUCED SIZE"
            reasoning = f"Confidence ({confidence:.1f}%) below full execution threshold ({self.exec_threshold}%)."
        elif agreement < 60:
            action_state = "REDUCED SIZE"
            reasoning = "Model ensemble disagreement prevents full sizing."
        else:
            action_state = "APPROVED"
            if signal_data.get("signal") == "BUY":
                action_state = "EXECUTE LONG"
            elif signal_data.get("signal") == "SELL":
                action_state = "EXECUTE SHORT"
            reasoning = "Institutional consensus and risk-reward profile verified."

        # Calibration Stability Check
        if signal_data.get("calibration", {}).get("is_calibrated") is False:
            action_state = "CALIBRATION UNSTABLE"
            reasoning = "Predictive calibration has drifted; observing only."

        return {
            "execution_state": action_state,
            "decision_reasoning": reasoning,
            "authority_verified": True,
        }
