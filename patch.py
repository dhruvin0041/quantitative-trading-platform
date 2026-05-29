import sys

with open('backend/src/execution/inference_service.py', 'r') as f:
    content = f.read()

content = content.replace('from src.execution.live_inference import (\n    fetch_live_data,\n    fetch_live_news,\n    get_meta_prediction,\n)', 'from src.execution.live_inference import (\n    fetch_live_data,\n    fetch_live_news,\n    get_meta_prediction,\n    compute_shap_explanation,\n    response_models_mapping,\n)')
content = content.replace('from src.execution.risk_manager import (\n    get_position_sizing,\n    calculate_beta,\n    detect_stampede_risk,\n)', 'from src.execution.risk_manager import (\n    get_position_sizing,\n    calculate_beta,\n    detect_stampede_risk,\n    calculate_institutional_risk_index,\n    get_risk_regime,\n    InstitutionalRiskArbitrator,\n)')
content = content.replace('self.governance_analytics = SignalGovernanceAnalytics()\n        self.paper_engine = paper_engine', 'self.governance_analytics = SignalGovernanceAnalytics()\n        self.risk_arbitrator = InstitutionalRiskArbitrator()\n        self.paper_engine = paper_engine')

old_auth_code = '''        if consensus_result["consensus_status"] == "VETOED":
            final_signal = "VETOED"
            signal_note = consensus_result["agent_responses"]["risk"]["veto_reason"]
        elif is_cooldown and final_signal in ["BUY", "SELL"]:
            final_signal = "HOLD"
            signal_note = "Signal Compressed: Cooldown window active"
        elif ev_metrics["ev_pct"] <= 0:
            final_signal = "HOLD"
            signal_note = "Suppressed: Negative Expected Value"'''

new_auth_code = '''        is_risk_vetoed = (consensus_result["consensus_status"] == "VETOED")
        quality_data = self.quality_engine.calculate_score(
            consensus_agreement=agreement_data["agreement_score"],
            calibrated_confidence=calibrated_prob,
            ev_metrics=ev_metrics,
            regime_v2=regime_detailed,
            risk_veto=is_risk_vetoed
        )
        
        portfolio_health_score = 100.0 - abs(mdd_pct)
        arbitration = self.risk_arbitrator.arbitrate(
            market_regime=regime_detailed,
            signal_quality=quality_data["score"],
            portfolio_health_score=portfolio_health_score,
            risk_index=risk_index
        )
        
        execution_state = arbitration["execution_state"]

        if arbitration["execution_state"] == "VETOED" or is_risk_vetoed:
            final_signal = "VETOED"
            execution_state = "VETOED"
            signal_note = arbitration.get("veto_reason") or consensus_result["agent_responses"].get("risk", {}).get("veto_reason", "Vetoed by Risk Arbitrator")
        elif ev_metrics["ev_pct"] <= 0:
            final_signal = "HOLD"
            signal_note = "Suppressed: Negative Expected Value"
            execution_state = "BLOCKED"
        elif is_cooldown and final_signal in ["BUY", "SELL"]:
            final_signal = "HOLD"
            signal_note = "Signal Compressed: Cooldown window active"
            execution_state = "COMPRESSED"
        elif arbitration["execution_state"] != "APPROVED":
            signal_note = arbitration["veto_reason"]'''

content = content.replace(old_auth_code, new_auth_code)
content = content.replace('"execution_state": consensus_result["consensus_status"],', '"execution_state": execution_state,')

old_proj = '''            "projections": {
                "floor": round(forecast_data["p10_price"], 2),
                "ceiling": round(forecast_data["p90_price"], 2),
                "confidence": round(forecast_data["forecast_confidence"], 1),
                "bias": forecast_data["forecast_bias"]
            },'''
new_proj = '''            "projections": {
                "p10": round(forecast_data["p10_price"], 2),
                "p50": round(forecast_data["p50_price"], 2),
                "p90": round(forecast_data["p90_price"], 2),
                "floor": round(forecast_data["p10_price"], 2),
                "median": round(forecast_data["p50_price"], 2),
                "ceiling": round(forecast_data["p90_price"], 2),
                "confidence": round(forecast_data["forecast_confidence"], 1),
                "reliability": forecast_data["forecast_reliability"],
                "bias": forecast_data["forecast_bias"],
                "drift": forecast_data["forecast_drift"],
                "expected_move": forecast_data["expected_move_10d"]
            },'''

content = content.replace(old_proj, new_proj)

with open('backend/src/execution/inference_service.py', 'w') as f:
    f.write(content)

print("Patch completed!")
