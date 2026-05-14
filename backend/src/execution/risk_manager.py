import numpy as np
import pandas as pd


def calculate_beta(ticker_prices: pd.Series, spy_prices: pd.Series):
    """
    Calculates the Beta of a ticker relative to SPY.
    Beta > 1: More volatile than market.
    Beta < 1: Less volatile than market.
    """
    # Calculate returns
    ticker_returns = ticker_prices.pct_change().dropna()
    spy_returns = spy_prices.pct_change().dropna()

    # Align indices
    common_idx = ticker_returns.index.intersection(spy_returns.index)
    ticker_returns = ticker_returns.loc[common_idx]
    spy_returns = spy_returns.loc[common_idx]

    if len(spy_returns) < 2:
        return 1.0

    covariance = np.cov(ticker_returns, spy_returns)[0][1]
    variance = np.var(spy_returns)

    beta = covariance / variance if variance != 0 else 1.0
    return float(beta)


def calculate_jensens_alpha(ticker_returns, market_returns, beta, risk_free_rate=0.04):
    """
    Measures 'Skill' (Alpha) by subtracting expected market returns from actual returns.
    Alpha = R_i - [R_f + Beta * (R_m - R_f)]
    """
    actual_return = ticker_returns.mean() * 252  # Annualized
    market_return = market_returns.mean() * 252  # Annualized

    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    alpha = actual_return - expected_return
    return float(alpha)


def detect_stampede_risk(retail_sentiment_volatility, signal_confidence):
    """
    Elite Metric: Detects 'Crowding' or 'Stampede Risk'.
    If retail excitement is high and our confidence is high, the trade is 'Crowded'.
    """
    crowding_score = retail_sentiment_volatility * signal_confidence
    is_crowded = crowding_score > 0.8
    return {
        "crowding_score": round(float(crowding_score), 2),
        "is_crowded": is_crowded,
        "action": "SCALE_BACK" if is_crowded else "NORMAL",
    }


def calculate_full_kelly(win_prob, win_loss_ratio):
    """
    Calculates the Full Kelly Criterion for optimal position sizing.
    Formula: K% = W - [(1 - W) / R]
    W = Win Probability
    R = Win/Loss Ratio (Avg Win / Avg Loss)
    """
    if win_loss_ratio <= 0:
        return 0.0

    kelly = win_prob - ((1 - win_prob) / win_loss_ratio)

    # Constrain between 0% and 25% for safety, even if 'Full Kelly' is requested
    # Real-world institutional desks rarely exceed 20-25% on a single ticker
    return float(np.clip(kelly, 0.0, 0.25))


def get_position_sizing(model_confidence, historical_win_rate=0.55, avg_win_loss=1.2):
    """
    Determines the suggested capital allocation using a blend of confidence and Kelly.
    """
    kelly_fraction = calculate_full_kelly(historical_win_rate, avg_win_loss)

    # Scale Kelly fraction by model confidence (0.0 to 1.0)
    # If the model is only 70% sure, we only take 70% of the Kelly-suggested bet
    suggested_allocation = kelly_fraction * model_confidence

    return {
        "kelly_basis": f"{round(kelly_fraction * 100, 1)}%",
        "suggested_allocation": f"{round(suggested_allocation * 100, 1)}%",
        "raw_fraction": suggested_allocation,
    }
