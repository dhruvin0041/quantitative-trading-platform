import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class StatisticalSufficiencyEngine:
    """
    Phase 1: Institutional Statistical Sufficiency Engine.
    Enforces cascading metric dependency gates to prevent mathematical contradictions.
    Rules:
    - n_returns < 60: Sharpe, Sortino, Calmar = "Insufficient Sample"
    - n_trades < 30: Win Rate, Profit Factor, Expectancy = "Insufficient Sample"
    - If Sharpe is invalid, Calmar MUST be suppressed.
    """

    def __init__(self):
        self.min_trade_gate = 30
        self.min_return_gate = 60
        self.min_calmar_gate = 90  # Calmar requires more stability

    def validate_and_reconcile(
        self, summary: Dict[str, Any], returns: List[float], trades: List[Dict]
    ) -> Dict[str, Any]:
        """
        Performs a full cascading validation of all performance metrics.
        """
        n_trades = len(trades)
        n_returns = len(returns)
        total_return = summary.get("total_return", 0.0)
        
        valid_metrics = summary.copy()
        confidence_intervals = {}

        # 1. Trade-Related Cascading Gates
        if n_trades < self.min_trade_gate:
            valid_metrics["win_rate"] = "Insufficient Sample"
            valid_metrics["profit_factor"] = "Insufficient Sample"
            valid_metrics["expectancy"] = "Insufficient Sample"
            valid_metrics["avg_win"] = "Insufficient Sample"
            valid_metrics["avg_loss"] = "Insufficient Sample"
        else:
            # Calculate Confidence Interval for Win Rate
            wr = summary.get("win_rate", 0) / 100.0
            z = 1.96  # 95% confidence
            denominator = 1 + z**2 / n_trades
            centre_adjusted_probability = wr + z**2 / (2 * n_trades)
            adjusted_standard_deviation = np.sqrt(
                (wr * (1 - wr) + z**2 / (4 * n_trades)) / n_trades
            )
            lower = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
            upper = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
            confidence_intervals["win_rate_95ci"] = f"[{lower * 100:.1f}%, {upper * 100:.1f}%]"
            valid_metrics["win_rate"] = f"{summary.get('win_rate', 0):.1f}%"
            valid_metrics["profit_factor"] = f"{summary.get('profit_factor', 0.0):.2f}"

        # 2. Return-Related Cascading Gates (Sharpe/Sortino)
        if n_returns < self.min_return_gate:
            valid_metrics["sharpe"] = "Insufficient Sample"
            valid_metrics["sortino"] = "Insufficient Sample"
            valid_metrics["calmar"] = "Insufficient Sample"  # Dependency: Calmar follows Sharpe
        else:
            sharpe = summary.get("sharpe", 0.0)
            
            # Logic Consistency: Positive Sharpe requires positive total return
            if total_return <= 0 and sharpe > 0:
                sharpe = 0.0
                
            valid_metrics["sharpe"] = f"{sharpe:.2f}"
            valid_metrics["sortino"] = f"{summary.get('sortino', 0.0):.2f}"
            
            # Sharpe Confidence Interval
            se_sharpe = np.sqrt((1 + (sharpe**2) / 2) / n_returns)
            lower_s = sharpe - 1.96 * se_sharpe
            upper_s = sharpe + 1.96 * se_sharpe
            confidence_intervals["sharpe_95ci"] = f"[{lower_s:.2f}, {upper_s:.2f}]"

        # 3. Calmar Specific Gate
        if n_returns < self.min_calmar_gate or valid_metrics["sharpe"] == "Insufficient Sample":
            valid_metrics["calmar"] = "Insufficient Sample"
        else:
            max_dd = summary.get("max_drawdown", 0.0)
            if abs(max_dd) < 1e-6:
                valid_metrics["calmar"] = "N/A (Zero Drawdown)"
            else:
                valid_metrics["calmar"] = f"{summary.get('calmar', 0.0):.2f}"

        return {
            "validated_metrics": valid_metrics,
            "confidence_intervals": confidence_intervals,
            "sample_sizes": {"trades": n_trades, "returns": n_returns},
        }


class StatisticalValidityEngine:
    """
    Legacy wrapper for backward compatibility, now using the new Sufficiency Engine.
    """

    def __init__(self):
        self.engine = StatisticalSufficiencyEngine()

    def validate_statistics(
        self, returns: List[float], trades: List[Dict], metrics: Dict
    ) -> Dict:
        return self.engine.validate_and_reconcile(metrics, returns, trades)

