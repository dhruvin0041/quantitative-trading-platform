import numpy as np
from scipy.optimize import minimize

class PortfolioOptimizer:
    """
    Institutional Portfolio Optimization using Mean-Variance (Markowitz) 
    and Risk Parity approaches.
    """
    
    def __init__(self, risk_free_rate=0.04):
        self.risk_free_rate = risk_free_rate

    def mean_variance_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray, target_return=None):
        """
        Maximizes Sharpe ratio or minimizes variance for a target return.
        """
        num_assets = len(expected_returns)
        args = (cov_matrix, expected_returns, self.risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
        
        if target_return is not None:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_return}
            )
            # Minimize variance
            result = minimize(self._portfolio_variance, num_assets * [1. / num_assets,], args=(cov_matrix,),
                              method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            # Maximize Sharpe
            result = minimize(self._negative_sharpe, num_assets * [1. / num_assets,], args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)
                              
        return result.x

    def _portfolio_variance(self, weights, cov_matrix):
        return weights.T @ cov_matrix @ weights

    def _negative_sharpe(self, weights, cov_matrix, expected_returns, risk_free_rate):
        p_var = self._portfolio_variance(weights, cov_matrix)
        p_ret = np.sum(weights * expected_returns)
        return -(p_ret - risk_free_rate) / np.sqrt(p_var)
