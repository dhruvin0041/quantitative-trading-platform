# One-time migration helper script (historical reference)

with open('src/execution/performance_analyzer.py', 'r') as f:
    content = f.read()

import_statement = 'from src.execution.statistical_engine import StatisticalValidityEngine\n'
content = content.replace(
    'from src.execution.signal_learning import SignalPerformanceResearch',
    import_statement + 'from src.execution.signal_learning import SignalPerformanceResearch'
)

init_repl = (
    'self.signal_research = SignalPerformanceResearch()\n'
    '        self.stat_engine = StatisticalValidityEngine()'
)
content = content.replace('self.signal_research = SignalPerformanceResearch()', init_repl)

with open('src/execution/performance_analyzer.py', 'w') as f:
    f.write(content)
