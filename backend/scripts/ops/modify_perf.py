import re

with open('src/execution/performance_analyzer.py', 'r') as f:
    content = f.read()

import_statement = 'from src.execution.statistical_engine import StatisticalValidityEngine\n'
content = content.replace('from src.execution.signal_learning import SignalPerformanceResearch', import_statement + 'from src.execution.signal_learning import SignalPerformanceResearch')

init_repl = 'self.signal_research = SignalPerformanceResearch()\n        self.stat_engine = StatisticalValidityEngine()'
content = content.replace('self.signal_research = SignalPerformanceResearch()', init_repl)

old_return = '''        return {
            \
summary\: {
                \total_return\: inception_return_pct,'''

new_return = '''        summary_metrics = {
            \total_return\: inception_return_pct,
            \sharpe\: sharpe,
            \sortino\: sortino,
            \calmar\: calmar,
            \max_drawdown\: max_dd * 100,
            \win_rate\: win_rate * 100,
            \profit_factor\: profit_factor,
            \today_pnl\: today_pnl,
            \mtd_pnl\: mtd_pnl,
            \ytd_pnl\: ytd_pnl,
            \inception_pnl\: inception_pnl,
            \realized_pnl\: realized_pnl,
            \unrealized_pnl\: unrealized_pnl,
            \total_trades\: total_trades,
            \open_trades\: total_trades - closed_trades_count,
            \closed_trades\: closed_trades_count,
            \winning_trades\: wins_count,
            \losing_trades\: losses_count,
            \initial_capital\: initial_capital,
            \peak_equity\: peak_info[\value\],
            \peak_date\: peak_info[\date\],
            \trough_equity\: trough_info[\value\],
            \trough_date\: trough_info[\date\],
            \expectancy\: expectancy,
        }
        
        stat_validity = self.stat_engine.validate_statistics(
            daily_returns.tolist() if not daily_returns.empty else [],
            trade_history if trade_history else [],
            summary_metrics
        )
        
        return {
            \summary\: stat_validity[\validated_metrics\],
            \confidence_intervals\: stat_validity[\confidence_intervals\],
            \sample_sizes\: stat_validity[\sample_sizes\],'''

content = content.replace(old_return, new_return)

with open('src/execution/performance_analyzer.py', 'w') as f:
    f.write(content)


