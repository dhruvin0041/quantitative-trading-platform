import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Institutional Portfolio Optimization using Mean-Variance (Markowitz)
    and Risk Parity approaches.
    """

    def __init__(self, risk_free_rate=0.04):
        self.risk_free_rate = risk_free_rate

    def mean_variance_optimization(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray, target_return=None
    ):
        """
        Maximizes Sharpe ratio or minimizes variance for a target return.
        """
        num_assets = len(expected_returns)
        args = (cov_matrix, expected_returns, self.risk_free_rate)
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))

        if target_return is not None:
            constraints = (
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {
                    "type": "eq",
                    "fun": lambda x: np.sum(x * expected_returns) - target_return,
                },
            )
            # Minimize variance
            result = minimize(
                self._portfolio_variance,
                num_assets
                * [
                    1.0 / num_assets,
                ],
                args=(cov_matrix,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
        else:
            # Maximize Sharpe
            result = minimize(
                self._negative_sharpe,
                num_assets
                * [
                    1.0 / num_assets,
                ],
                args=args,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

        return result.x

    def _portfolio_variance(self, weights, cov_matrix):
        return weights.T @ cov_matrix @ weights

    def _negative_sharpe(self, weights, cov_matrix, expected_returns, risk_free_rate):
        p_var = self._portfolio_variance(weights, cov_matrix)
        p_ret = np.sum(weights * expected_returns)
        return -(p_ret - risk_free_rate) / np.sqrt(p_var)

    def risk_parity_allocation(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates Risk Parity (Inverse Volatility) portfolio weights.
        Ensures low-volatility assets receive proportionally more capital
        while high-volatility assets are restrained, equalizing risk contribution.
        """
        variances = np.diag(cov_matrix)
        vols = np.sqrt(np.maximum(variances, 1e-8))
        inv_vols = 1.0 / vols
        weights = inv_vols / np.sum(inv_vols)
        return weights

    def fractional_kelly_portfolio(
        self,
        win_probs: np.ndarray,
        win_loss_ratios: np.ndarray,
        fraction: float = 0.5,
        max_single_asset_cap: float = 0.25,
    ) -> np.ndarray:
        """
        Computes Fractional Kelly position sizing vector across universe assets.
        Formula: K_i = max(0, W_i - (1 - W_i) / R_i) * fraction
        Constrained by institutional single-asset risk cap.
        """
        win_loss_clean = np.maximum(win_loss_ratios, 1e-4)
        kelly_full = win_probs - ((1.0 - win_probs) / win_loss_clean)
        kelly_fractional = np.maximum(0.0, kelly_full) * fraction
        clipped = np.clip(kelly_fractional, 0.0, max_single_asset_cap)
        total = np.sum(clipped)
        if total > 1.0:
            clipped = clipped / total
        return clipped

