# One-time migration helper script (historical reference)

with open('src/execution/inference_service.py', 'r') as f:
    content = f.read()

import_statement = (
    'from src.execution.timing_engine import PredictiveTimingEngine\n'
    'from src.execution.confidence_engine import ConfidenceBreakdownEngine\n'
)
content = content.replace(
    'from src.execution.trade_engine import TradeConstructionEngine',
    import_statement + 'from src.execution.trade_engine import TradeConstructionEngine'
)

init_repl = (
    'self.trade_engine = TradeConstructionEngine()\n'
    '        self.timing_engine = PredictiveTimingEngine()\n'
    '        self.confidence_engine = ConfidenceBreakdownEngine()'
)
content = content.replace('self.trade_engine = TradeConstructionEngine()', init_repl)

with open('src/execution/inference_service.py', 'w') as f:
    f.write(content)
