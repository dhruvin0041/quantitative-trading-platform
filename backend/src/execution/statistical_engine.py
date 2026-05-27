import numpy as np
from typing import Dict, List


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

    def validate_statistics(
        self, returns: List[float], trades: List[Dict], metrics: Dict
    ) -> Dict:
        n_trades = len(trades)
        n_returns = len(returns)

        valid_metrics = {}
        confidence_intervals = {}

        # 1. Enforce Minimum Sample Gates
        if n_trades < self.min_samples["win_rate"]:
            valid_metrics["win_rate"] = "Insufficient Sample (< 30)"
        else:
            valid_metrics["win_rate"] = f"{metrics.get('win_rate', 0):.1f}%"
            # Wilson score interval for binomial proportion
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

        if n_returns < self.min_samples["sharpe"]:
            valid_metrics["sharpe"] = "Insufficient Sample (< 60)"
        else:
            sharpe = metrics.get("sharpe", 0.0)
            valid_metrics["sharpe"] = f"{sharpe:.2f}"
            # Lo's approximation for Sharpe CI
            se_sharpe = np.sqrt((1 + (sharpe**2) / 2) / n_returns)
            lower = sharpe - 1.96 * se_sharpe
            upper = sharpe + 1.96 * se_sharpe
            confidence_intervals["sharpe_95ci"] = f"[{lower:.2f}, {upper:.2f}]"

        if n_returns < self.min_samples["sortino"]:
            valid_metrics["sortino"] = "Insufficient Sample (< 60)"
        else:
            valid_metrics["sortino"] = f"{metrics.get('sortino', 0.0):.2f}"

        if (
            n_returns < self.min_samples["calmar"]
            or metrics.get("total_return", 0.0) <= 0
        ):
            valid_metrics["calmar"] = "Insufficient Sample (< 90) or Negative Return"
        else:
            valid_metrics["calmar"] = f"{metrics.get('calmar', 0.0):.2f}"

        if n_trades < self.min_samples["profit_factor"]:
            valid_metrics["profit_factor"] = "Insufficient Sample (< 30)"
        else:
            pf = metrics.get("profit_factor", 0.0)
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

        metrics.update(valid_metrics)
        return {
            "validated_metrics": metrics,
            "confidence_intervals": confidence_intervals,
            "sample_sizes": {"trades": n_trades, "returns": n_returns},
        }
