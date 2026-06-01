import re

with open('src/execution/inference_service.py', 'r') as f:
    content = f.read()

import_statement = 'from src.execution.timing_engine import PredictiveTimingEngine\nfrom src.execution.confidence_engine import ConfidenceBreakdownEngine\n'
content = content.replace('from src.execution.trade_engine import TradeConstructionEngine', import_statement + 'from src.execution.trade_engine import TradeConstructionEngine')

init_repl = 'self.trade_engine = TradeConstructionEngine()\n        self.timing_engine = PredictiveTimingEngine()\n        self.confidence_engine = ConfidenceBreakdownEngine()'
content = content.replace('self.trade_engine = TradeConstructionEngine()', init_repl)

# Add timing execution before meta prediction
old_meta = '''        # 5. Signal Selection (V2.0 Pipeline)
        regime_id_map = {\
BEAR\: 0, \NEUTRAL\: 1, \BULL\: 2}'''

new_meta = '''        # 5. Signal Selection (V2.0 Pipeline) & Predictive Timing
        timing_data = self.timing_engine.calculate_timing_features(ticker_df_risk)
        
        regime_id_map = {\BEAR\: 0, \NEUTRAL\: 1, \BULL\: 2}'''

content = content.replace(old_meta, new_meta)

# Inject ConfidenceBreakdownEngine before quality metrics
old_quality = '''        # Multi-Layer Quality Score
        quality_metrics = self.quality_engine.calculate_score('''

new_quality = '''        # Explainable Confidence Decomposition
        confidence_data = self.confidence_engine.decompose_confidence(
            regime=regime_detailed,
            volatility_ratio=vol_ratio,
            agreement_score=agreement_data[\agreement_score\],
            ev_pct=ev_metrics[\ev_pct\],
            timing_score=timing_data[\timing_score\],
            asset_class=asset_class
        )
        # Use explainable confidence instead of old calibrated probability if it makes sense, 
        # or expose it in the final payload alongside calibrated_prob.
        
        # Multi-Layer Quality Score
        quality_metrics = self.quality_engine.calculate_score('''

content = content.replace(old_quality, new_quality)

# Inject into response_data
old_resp = '\multi_timeframe_consensus\: mtf_consensus,'
new_resp = '\multi_timeframe_consensus\: mtf_consensus,\n            \timing_intelligence\: timing_data,\n            \confidence_breakdown\: confidence_data[\confidence_breakdown\],\n            \explainable_confidence\: confidence_data[\explainable_confidence\],'
content = content.replace(old_resp, new_resp)

with open('src/execution/inference_service.py', 'w') as f:
    f.write(content)


