import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class RiskMetricValidator:
    """
    Phase 1: Institutional Risk Metric Validator.
    Ensures mathematical consistency and statistical significance of portfolio analytics.
    Prevents "Impossible Metrics" such as positive Sharpe with negative returns.
    """

    def __init__(self):
        self.min_trade_gate = 30
        self.min_return_gate = 60
        self.min_calmar_gate = 90

    def reconcile_metrics(
        self, summary: Dict[str, Any], returns: List[float], trades: List[Dict]
    ) -> Dict[str, Any]:
        """
        Performs a full institutional reconciliation of portfolio metrics.
        """
        reconciled = summary.copy()
        n_trades = len(trades)
        n_returns = len(returns)
        total_return = summary.get("total_return", 0.0)

        # 1. Logic Consistency Gate: Return vs. Ratios
        # It is mathematically impossible to have a positive Sharpe/Sortino with negative total return
        # (Assuming risk-free rate is non-negative and no extreme skew issues in tiny samples)
        if total_return <= 0:
            if (
                isinstance(reconciled.get("sharpe"), (int, float))
                and reconciled["sharpe"] > 0
            ):
                reconciled["sharpe"] = 0.0
                logger.warning(
                    "Corrected impossible positive Sharpe with negative total return."
                )

            if (
                isinstance(reconciled.get("sortino"), (int, float))
                and reconciled["sortino"] > 0
            ):
                reconciled["sortino"] = 0.0
                logger.warning(
                    "Corrected impossible positive Sortino with negative total return."
                )

            # Calmar must be zero or N/A if returns are negative
            reconciled["calmar"] = "N/A (Negative Return)"

        # 2. Sample Size Gating (Institutional Thresholds)
        if n_trades < self.min_trade_gate:
            reconciled["win_rate"] = "N/A (Small Sample)"
            reconciled["profit_factor"] = "N/A (Small Sample)"
            reconciled["expectancy"] = "N/A (Small Sample)"

        if n_returns < self.min_return_gate:
            reconciled["sharpe"] = "N/A (Insufficient Data)"
            reconciled["sortino"] = "N/A (Insufficient Data)"

        if n_returns < self.min_calmar_gate:
            reconciled["calmar"] = "N/A (Insufficient Data)"

        # 3. Drawdown Integrity
        max_dd = summary.get("max_drawdown", 0.0)
        if abs(max_dd) < 1e-6 and total_return != 0:
            # If there's return but 0 drawdown, it's likely a calculation error or single-trade artifact
            reconciled["calmar"] = "N/A (Zero Drawdown)"

        return reconciled


class StatisticalValidityEngine:
    """
    Enforces minimum sample sizes for performance metrics to prevent statistical noise.
    Calculates 95% confidence intervals for key metrics.
    """

    def __init__(self):
        self.min_samples = {
            "win_rate": 30,
            "sharpe": 60,
            "sortino": 60,
            "calmar": 90,
            "profit_factor": 30,
        }
        self.validator = RiskMetricValidator()

    def validate_statistics(
        self, returns: List[float], trades: List[Dict], metrics: Dict
    ) -> Dict:
        n_trades = len(trades)
        n_returns = len(returns)

        # First, reconcile raw metrics for mathematical sanity
        reconciled_metrics = self.validator.reconcile_metrics(metrics, returns, trades)

        valid_metrics = {}
        confidence_intervals = {}

        # 1. Apply Institutional Formatting/Gating
        # Win Rate CI
        if n_trades >= self.min_samples["win_rate"]:
            wr = metrics.get("win_rate", 0) / 100.0
            z = 1.96  # 95% confidence
            denominator = 1 + z**2 / n_trades
            centre_adjusted_probability = wr + z**2 / (2 * n_trades)
            adjusted_standard_deviation = np.sqrt(
                (wr * (1 - wr) + z**2 / (4 * n_trades)) / n_trades
            )
            lower = (
                centre_adjusted_probability - z * adjusted_standard_deviation
            ) / denominator
            upper = (
                centre_adjusted_probability + z * adjusted_standard_deviation
            ) / denominator
            confidence_intervals["win_rate_95ci"] = (
                f"[{lower * 100:.1f}%, {upper * 100:.1f}%]"
            )
            valid_metrics["win_rate"] = f"{metrics.get('win_rate', 0):.1f}%"
        else:
            valid_metrics["win_rate"] = "Insufficient Sample (< 30)"

        # Sharpe CI
        if n_returns >= self.min_samples["sharpe"] and isinstance(
            reconciled_metrics["sharpe"], float
        ):
            sharpe = reconciled_metrics["sharpe"]
            se_sharpe = np.sqrt((1 + (sharpe**2) / 2) / n_returns)
            lower = sharpe - 1.96 * se_sharpe
            upper = sharpe + 1.96 * se_sharpe
            confidence_intervals["sharpe_95ci"] = f"[{lower:.2f}, {upper:.2f}]"
            valid_metrics["sharpe"] = f"{sharpe:.2f}"
        else:
            valid_metrics["sharpe"] = reconciled_metrics.get(
                "sharpe", "Insufficient Sample"
            )

        # Sortino
        if (
            n_returns < self.min_samples["sortino"]
            or reconciled_metrics["sortino"] == "N/A (Insufficient Data)"
        ):
            valid_metrics["sortino"] = "Insufficient Sample (< 60)"
        else:
            valid_metrics["sortino"] = f"{reconciled_metrics.get('sortino', 0.0):.2f}"

        # Calmar
        if isinstance(reconciled_metrics["calmar"], str):
            valid_metrics["calmar"] = reconciled_metrics["calmar"]
        else:
            valid_metrics["calmar"] = f"{reconciled_metrics.get('calmar', 0.0):.2f}"

        # Profit Factor
        if n_trades < self.min_samples["profit_factor"]:
            valid_metrics["profit_factor"] = "Insufficient Sample (< 30)"
        else:
            pf = reconciled_metrics.get("profit_factor", 0.0)
            valid_metrics["profit_factor"] = f"{pf:.2f}"

            # Simple bootstrap approximation for PF CI
            try:
                gains = [t["pnl"] for t in trades if t["pnl"] > 0]
                losses = [abs(t["pnl"]) for t in trades if t["pnl"] <= 0]
                if sum(losses) > 0 and len(gains) > 0 and len(losses) > 0:
                    boot_pfs = []
                    for _ in range(1000):
                        b_gains = np.random.choice(gains, len(gains), replace=True)
                        b_losses = np.random.choice(losses, len(losses), replace=True)
                        b_pf = sum(b_gains) / sum(b_losses) if sum(b_losses) > 0 else pf
                        boot_pfs.append(b_pf)
                    lower = np.percentile(boot_pfs, 2.5)
                    upper = np.percentile(boot_pfs, 97.5)
                    confidence_intervals["profit_factor_95ci"] = (
                        f"[{lower:.2f}, {upper:.2f}]"
                    )
            except Exception:
                pass

        # Final Reconciled Metrics Update
        final_metrics = reconciled_metrics.copy()
        final_metrics.update(valid_metrics)

        return {
            "validated_metrics": final_metrics,
            "confidence_intervals": confidence_intervals,
            "sample_sizes": {"trades": n_trades, "returns": n_returns},
        }
