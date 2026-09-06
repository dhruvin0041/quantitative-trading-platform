import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate annualized Sharpe ratio from daily returns.
    """
    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    return np.sqrt(252) * (mean_return - risk_free_rate) / std_return


def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """
    Calculate annualized Sortino ratio (penalizes only downside volatility).
    """
    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return np.inf  # No downside volatility

    downside_std = np.std(downside_returns)

    if downside_std == 0:
        return 0.0

    return np.sqrt(252) * (mean_return - risk_free_rate) / downside_std


def calculate_profit_factor(returns):
    """
    Calculate Profit Factor (Gross Profit / Gross Loss).
    """
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_max_drawdown(returns):
    """
    Calculate Maximum Drawdown.
    """
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return np.min(drawdown)


def calculate_calmar_ratio(returns, max_drawdown=None, sim_years=None):
    """
    Calculate annualized Calmar ratio (Annualized Return / |Max Drawdown|).

    Strict sign preservation:
    - If annualized return is negative, Calmar is strictly negative.
    - No abs() is applied to the numerator or the final result.
    - If max drawdown is zero (< 1e-7), returns 0.0.

    Parameters:
    -----------
    returns : np.ndarray or list
        Array of period/trade returns.
    max_drawdown : float, optional
        Maximum drawdown (decimal or percentage). If None, computed from returns.
    sim_years : float, optional
        Simulation period length in years. If None, derived from len(returns)/252.
    """
    if len(returns) == 0:
        return 0.0

    if sim_years is None or sim_years <= 0:
        sim_years = max(1.0 / 252.0, len(returns) / 252.0)

    ann_return = float(np.sum(returns) / sim_years)

    if max_drawdown is None:
        max_dd = abs(float(calculate_max_drawdown(np.array(returns))))
    else:
        max_dd = abs(float(max_drawdown))
        # Handle percentage format (> 1.0) when returns are decimal (<= 1.0)
        max_ret_abs = float(np.max(np.abs(returns))) if len(returns) > 0 else 1.0
        if max_dd > 1.0 and max_ret_abs <= 1.0:
            max_dd = max_dd / 100.0

    if max_dd < 1e-7:
        return 0.0

    return float(ann_return / max_dd)


def simulate_strategy_returns(y_true, y_pred_probs, true_returns, threshold=0.6):
    """
    Simulate trading returns based on model predictions.
    y_pred_probs: array of shape (n_samples, 3) where columns are [Sell, Hold, Buy]
    true_returns: actual forward returns for each sample
    """
    strategy_returns = []

    for i in range(len(y_pred_probs)):
        probs = y_pred_probs[i]
        actual_ret = true_returns[i]

        # Action selection with confidence threshold
        action = np.argmax(probs)
        confidence = probs[action]

        if confidence < threshold:
            action = 1  # Force HOLD if not confident enough

        if action == 2:  # BUY
            strategy_returns.append(actual_ret)
        elif action == 0:  # SELL
            strategy_returns.append(-actual_ret)  # Assuming short selling
        else:  # HOLD
            strategy_returns.append(0.0)

    return np.array(strategy_returns)


def sharpe_objective(y_true, y_pred_probs, true_returns, threshold=0.6):
    """
    Objective function that optimizes for Sharpe Ratio.
    """
    strategy_returns = simulate_strategy_returns(
        y_true, y_pred_probs, true_returns, threshold
    )
    return calculate_sharpe_ratio(strategy_returns)
