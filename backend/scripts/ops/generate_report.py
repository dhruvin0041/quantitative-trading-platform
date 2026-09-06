import os
from datetime import datetime

from src.execution.paper_trading import PaperTradingEngine
from src.execution.performance_analyzer import PerformanceAnalyzer


def generate_performance_report():
    """
    Generates a periodic performance report in Markdown format.
    """
    engine = PaperTradingEngine()
    analyzer = PerformanceAnalyzer()

    analysis = analyzer.analyze(
        engine.portfolio_snapshots, engine.history, engine.initial_capital
    )

    if not analysis:
        print("No performance data available.")
        return

    summary = analysis["summary"]

    report = f"""# HYDRA TERMINAL: Institutional Performance Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Executive Summary
- **Total Return**: {summary["total_return"]:.2f}%
- **Annualized Sharpe**: {summary["sharpe"]:.2f}
- **Annualized Sortino**: {summary["sortino"]:.2f}
- **Annualized Calmar**: {summary["calmar"]:.2f}
- **Maximum Drawdown**: {summary["max_drawdown"]:.2f}%
- **Win Rate**: {summary["win_rate"]:.2f}%
- **Profit Factor**: {summary["profit_factor"]:.2f}

## 2. Portfolio Attribution
### By Market Regime
"""
    for regime, pnl in analysis["attribution"]["by_regime"].items():
        report += f"- **{regime}**: ${pnl:,.2f} PnL\n"

    report += "\n### By Sector Exposure\n"
    for sector, pnl in analysis["attribution"]["by_sector"].items():
        report += f"- **{sector}**: ${pnl:,.2f} PnL\n"

    report += "\n## 3. Recent Trades\n"
    report += "| Time | Ticker | Action | Price | PnL |\n"
    report += "| :--- | :--- | :--- | :--- | :--- |\n"

    recent_trades = [t for t in engine.history if t["action"] == "SELL"][-10:]
    for trade in reversed(recent_trades):
        report += f"| {trade['time']} | {trade['ticker']} | {trade['action']} | ${trade['price']:.2f} | ${trade.get('pnl', 0):,.2f} |\n"

    report_path = f"reports/performance_{datetime.now().strftime('%Y%m%d')}.md"
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report generated successfully: {report_path}")


if __name__ == "__main__":
    generate_performance_report()
