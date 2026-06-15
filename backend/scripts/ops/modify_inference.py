
with open('src/execution/inference_service.py', 'r') as f:
    content = f.read()

# Import the new engine
import_statement = 'from src.execution.consensus_engine import WeightedConsensusEngine\n'
content = content.replace('from src.models.monitoring.drift_monitor import DriftMonitor', import_statement + 'from src.models.monitoring.drift_monitor import DriftMonitor')

# Add engine to init
init_repl = 'self.report_gen = report_gen\n        self.consensus_engine = WeightedConsensusEngine()'
content = content.replace('self.report_gen = report_gen', init_repl)

# Replace ensemble_p logic
old_logic = '''        # Apply Dynamic Weights
        ensemble_p = np.zeros(3)
        for model_name, p in base_probs.items():
            w = model_weights_raw.get(model_name, {\
weight\: 0.25})[\weight\]
            ensemble_p += p * w

        final_prob_raw = float(np.max(ensemble_p))'''

new_logic = '''        # Apply Dynamic Weights via WeightedConsensusEngine
        extracted_weights = {k: v.get(\weight\, 0.25) for k, v in model_weights_raw.items()}
        agreement_data = self.consensus_engine.compute_agreement(base_probs, extracted_weights)
        
        final_prob_raw = agreement_data[\agreement_score\] / 100.0'''
        
content = content.replace(old_logic, new_logic)

# Replace run_consensus call
old_run = '''        consensus_result = self.orchestrator.run_consensus(
            ensemble_p, consensus_risk_input
        )'''
new_run = '''        consensus_result = self.orchestrator.run_consensus(
            agreement_data, consensus_risk_input
        )'''

content = content.replace(old_run, new_run)

# Replace agreement retrieval in inference_service
old_agree = 'consensus_agreement=float(consensus_result.get(\agreement\, 66.0)),'
new_agree = 'consensus_agreement=float(agreement_data.get(\agreement_score\, 66.0)),'
content = content.replace(old_agree, new_agree)

old_agree2 = '\agreement\: consensus_result.get(\agreement\, 0),'
new_agree2 = '\agreement\: agreement_data.get(\agreement_score\, 0),'
content = content.replace(old_agree2, new_agree2)

with open('src/execution/inference_service.py', 'w') as f:
    f.write(content)


