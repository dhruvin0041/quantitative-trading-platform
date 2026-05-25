import numpy as np
import pandas as pd


class PortfolioAnalytics:
    """
    Computes institutional-grade portfolio risk and performance metrics.
    """

    def __init__(self, risk_free_rate=0.04):
        self.rf_rate = risk_free_rate / 252  # Daily risk-free rate

    def compute_metrics(
        self, equity_curve: list, returns: list, market_returns: list = None
    ):
        if not equity_curve or len(equity_curve) < 2:
            return self._empty_metrics()

        eq_series = pd.Series(equity_curve)
        ret_series = pd.Series(returns) if returns else eq_series.pct_change().dropna()

        if len(ret_series) == 0:
            return self._empty_metrics()

        # Basic Stats
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        win_rate = (
            len(ret_series[ret_series > 0]) / len(ret_series)
            if len(ret_series) > 0
            else 0
        )

        gross_profit = ret_series[ret_series > 0].sum()
        gross_loss = abs(ret_series[ret_series < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

        # Risk Metrics
        std_dev = ret_series.std()
        downside_std = ret_series[ret_series < 0].std()

        sharpe = (
            (ret_series.mean() - self.rf_rate) / std_dev * np.sqrt(252)
            if std_dev != 0
            else 0
        )
        sortino = (
            (ret_series.mean() - self.rf_rate) / downside_std * np.sqrt(252)
            if downside_std != 0 and not np.isnan(downside_std)
            else sharpe
        )

        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_dd = np.min(drawdown)

        # Advanced Institutional Metrics
        beta = 1.0
        alpha = 0.0
        if market_returns is not None and len(market_returns) == len(ret_series):
            mkt_series = pd.Series(market_returns)
            cov_matrix = np.cov(ret_series, mkt_series)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0

            ann_ret = ret_series.mean() * 252
            ann_mkt_ret = mkt_series.mean() * 252
            alpha = ann_ret - (
                self.rf_rate * 252 + beta * (ann_mkt_ret - self.rf_rate * 252)
            )

        # Value at Risk (Historical 95%)
        var_95 = np.percentile(ret_series, 5)
        cvar_95 = (
            ret_series[ret_series <= var_95].mean()
            if len(ret_series[ret_series <= var_95]) > 0
            else var_95
        )

        return {
            "Total Return": float(total_return),
            "Win Rate": float(win_rate),
            "Average Return": float(ret_series.mean()),
            "Profit Factor": float(profit_factor),
            "Sharpe Ratio": float(sharpe),
            "Sortino Ratio": float(sortino),
            "Maximum Drawdown": float(max_dd),
            "Portfolio Beta": float(beta),
            "Jensen's Alpha": float(alpha),
            "VaR_95": float(var_95),
            "CVaR_95": float(cvar_95),
        }

    def _empty_metrics(self):
        return {
            "Total Return": 0.0,
            "Win Rate": 0.0,
            "Average Return": 0.0,
            "Profit Factor": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Maximum Drawdown": 0.0,
            "Portfolio Beta": 1.0,
            "Jensen's Alpha": 0.0,
            "VaR_95": 0.0,
            "CVaR_95": 0.0,
        }
