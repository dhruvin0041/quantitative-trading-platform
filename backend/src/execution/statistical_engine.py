import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


def calculate_probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float = 0.0,
    n_samples: int = 252,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    annualization_factor: float = 252.0,
) -> float:
    """
    Computes Marcos López de Prado's Probabilistic Sharpe Ratio (PSR).
    Bailey & López de Prado (2012, 2014).
    Calculates the probability that the true Sharpe ratio exceeds benchmark_sr,
    accounting for sample size, skewness, and fat-tailed kurtosis.
    """
    if n_samples < 5:
        return 0.5

    # De-annualize to daily scale for standard error calculation
    scale = np.sqrt(annualization_factor) if annualization_factor > 0 else 1.0
    sr_daily = observed_sr / scale
    sr_bm_daily = benchmark_sr / scale

    # Asymptotic standard error of Sharpe ratio
    variance_term = 1.0 - (skewness * sr_daily) + (((kurtosis - 1.0) / 4.0) * (sr_daily**2))
    variance_term = max(variance_term, 1e-9)
    se_daily = np.sqrt(variance_term / max(n_samples - 1.0, 1.0))

    if se_daily < 1e-9:
        return 1.0 if sr_daily >= sr_bm_daily else 0.0

    z = (sr_daily - sr_bm_daily) / se_daily
    return float(stats.norm.cdf(z))


def calculate_expected_maximum_sharpe(
    n_trials: int = 250,
    sr_variance: Optional[float] = None,
    n_samples: int = 252,
    annualization_factor: float = 252.0,
) -> float:
    """
    Computes expected maximum Sharpe ratio under the null hypothesis (H0: true skill = 0)
    across K trials (Extreme Value Theory / Bailey & López de Prado 2014).
    """
    if n_trials <= 1:
        return 0.0

    gamma = 0.5772156649  # Euler-Mascheroni constant
    expected_max_standard = (
        (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )

    scale = np.sqrt(annualization_factor) if annualization_factor > 0 else 1.0
    if sr_variance is not None and sr_variance > 0:
        std_sr = np.sqrt(sr_variance)
    else:
        # Theoretical standard deviation of sample Sharpe ratio under null hypothesis
        std_sr = scale / np.sqrt(max(n_samples - 1, 1))

    return float(std_sr * expected_max_standard)


def calculate_deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int = 250,
    n_samples: int = 252,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_variance: Optional[float] = None,
    trials_sr_list: Optional[List[float]] = None,
    annualization_factor: float = 252.0,
) -> Dict[str, Any]:
    """
    Computes Marcos López de Prado's Deflated Sharpe Ratio (DSR).
    Discounts the observed Sharpe ratio for multiple testing bias across n_trials.
    """
    if trials_sr_list and len(trials_sr_list) > 1:
        n_trials = len(trials_sr_list)
        sr_variance = float(np.var(trials_sr_list, ddof=1))

    expected_max_sr = calculate_expected_maximum_sharpe(
        n_trials=n_trials,
        sr_variance=sr_variance,
        n_samples=n_samples,
        annualization_factor=annualization_factor,
    )

    psr_value = calculate_probabilistic_sharpe_ratio(
        observed_sr=observed_sr,
        benchmark_sr=0.0,
        n_samples=n_samples,
        skewness=skewness,
        kurtosis=kurtosis,
        annualization_factor=annualization_factor,
    )

    dsr_value = calculate_probabilistic_sharpe_ratio(
        observed_sr=observed_sr,
        benchmark_sr=expected_max_sr,
        n_samples=n_samples,
        skewness=skewness,
        kurtosis=kurtosis,
        annualization_factor=annualization_factor,
    )

    is_statistically_significant = bool(dsr_value >= 0.95)

    return {
        "observed_sharpe": round(float(observed_sr), 4),
        "expected_max_sharpe_null": round(float(expected_max_sr), 4),
        "psr": round(float(psr_value), 4),
        "dsr": round(float(dsr_value), 4),
        "p_value": round(float(1.0 - dsr_value), 4),
        "trials_penalized": int(n_trials),
        "skewness": round(float(skewness), 4),
        "kurtosis": round(float(kurtosis), 4),
        "is_statistically_significant": is_statistically_significant,
    }


class StatisticalSufficiencyEngine:
    """
    Phase 1: Institutional Statistical Sufficiency Engine.
    Enforces cascading metric dependency gates to prevent mathematical contradictions.
    Rules:
    - n_returns < 60: Sharpe, Sortino, Calmar = "Insufficient Sample"
    - n_trades < 30: Win Rate, Profit Factor, Expectancy = "Insufficient Sample"
    - If Sharpe is invalid, Calmar MUST be suppressed.
    - Multiple Testing & Non-Normality: Computes Probabilistic Sharpe Ratio (PSR)
      and Deflated Sharpe Ratio (DSR) discounting for Optuna hyperparameter trials.
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

        # 2. Return-Related Cascading Gates (Sharpe/Sortino/DSR/PSR)
        if n_returns < self.min_return_gate:
            valid_metrics["sharpe"] = "Insufficient Sample"
            valid_metrics["sortino"] = "Insufficient Sample"
            valid_metrics["calmar"] = "Insufficient Sample"  # Dependency: Calmar follows Sharpe
            valid_metrics["psr"] = "Insufficient Sample"
            valid_metrics["dsr"] = "Insufficient Sample"
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

            # Compute higher moments for López de Prado PSR and DSR
            ret_arr = np.array(returns, dtype=float)
            skew_val = float(stats.skew(ret_arr)) if len(ret_arr) >= 10 else 0.0
            kurt_val = float(stats.kurtosis(ret_arr, fisher=False)) if len(ret_arr) >= 10 else 3.0

            n_trials = summary.get("trials_count", 250)
            dsr_metrics = calculate_deflated_sharpe_ratio(
                observed_sr=sharpe,
                n_trials=n_trials,
                n_samples=n_returns,
                skewness=skew_val,
                kurtosis=kurt_val,
            )

            valid_metrics["psr"] = f"{dsr_metrics['psr'] * 100:.1f}%"
            valid_metrics["dsr"] = f"{dsr_metrics['dsr'] * 100:.1f}%"
            valid_metrics["expected_max_sharpe_null"] = f"{dsr_metrics['expected_max_sharpe_null']:.2f}"
            valid_metrics["is_statistically_significant"] = dsr_metrics["is_statistically_significant"]
            confidence_intervals["dsr_p_value"] = f"{dsr_metrics['p_value']:.4f}"
            confidence_intervals["return_skewness"] = f"{skew_val:.2f}"
            confidence_intervals["return_kurtosis"] = f"{kurt_val:.2f}"

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
