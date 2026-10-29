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
    variance = np.var(spy_returns, ddof=1)

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
    # Softmax standard deviation maxes out at ~0.471. We normalize it to a 0-1 scale.
    normalized_volatility = min(1.0, retail_sentiment_volatility / 0.471)
    crowding_score = normalized_volatility * signal_confidence
    is_crowded = bool(crowding_score > 0.8)
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


def calculate_fractional_kelly(win_prob: float, win_loss_ratio: float, fraction: float = 0.5) -> float:
    """
    Calculates the Fractional Kelly Criterion (Half Kelly by default).
    Formula: K_f = max(0, W - [(1 - W) / R]) * fraction
    Mitigates parameter estimation risk and prevents over-leveraging.
    """
    if win_loss_ratio <= 0:
        return 0.0

    raw_kelly = win_prob - ((1.0 - win_prob) / win_loss_ratio)
    fractional = max(0.0, raw_kelly) * fraction
    return float(np.clip(fractional, 0.0, 0.25))


def get_position_sizing(model_confidence, trade_history=None):
    """
    Determines the suggested capital allocation using a blend of confidence and Kelly.
    Dynamically uses trade history if available.
    """
    if trade_history and len(trade_history) >= 10:
        wins = [t for t in trade_history if t.get("pnl", 0) > 0]
        losses = [t for t in trade_history if t.get("pnl", 0) < 0]
        win_rate = len(wins) / len(trade_history)

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0

        avg_win_loss = (avg_win / avg_loss) if avg_loss > 0 else 1.2
    else:
        win_rate = 0.55
        avg_win_loss = 1.2

    kelly_fraction = calculate_full_kelly(win_rate, avg_win_loss)

    # Scale Kelly fraction by model confidence (0.0 to 1.0)
    suggested_allocation = kelly_fraction * model_confidence

    return {
        "kelly_basis": f"{round(kelly_fraction * 100, 1)}%",
        "suggested_allocation": f"{round(suggested_allocation * 100, 1)}%",
        "raw_fraction": suggested_allocation,
    }


def calculate_institutional_risk_index(volatility_annualized, max_drawdown_pct, liquidity_score, exposure_ratio, market_correlation, signal_uncertainty=0.0):
    """
    Phase 9: Institutional Risk Index (0-100).
    Weights:
    - Volatility: 25%
    - Max Drawdown: 25%
    - Liquidity: 10%
    - Exposure: 15%
    - Market Correlation: 10%
    - Signal Uncertainty: 15%
    """
    # Normalize inputs to 0-1 range for calculation
    norm_vol = min(1.0, volatility_annualized / 0.45)   # 45% vol is high
    norm_dd = min(1.0, abs(max_drawdown_pct) / 25.0)    # 25% DD is high
    norm_liq = 1.0 - np.clip(liquidity_score, 0.0, 1.0) # Lower liquidity = higher risk
    norm_exp = np.clip(exposure_ratio / 0.25, 0.0, 1.0) # 25% exposure is high institutional
    norm_corr = abs(market_correlation)
    norm_uncert = np.clip(signal_uncertainty, 0.0, 1.0)

    risk_score = (
        (norm_vol * 25) +
        (norm_dd * 25) +
        (norm_liq * 10) +
        (norm_exp * 15) +
        (norm_corr * 10) +
        (norm_uncert * 15)
    )

    return float(np.clip(risk_score, 0, 100))


def get_risk_regime(risk_score):
    """
    Maps risk score to a semantic regime.
    [0-20] Stable, [21-40] Elevated, [41-60] Defensive, [61-80] Critical, [81-100] Panic
    """
    if risk_score <= 20:
        return "STABLE"
    elif risk_score <= 40:
        return "ELEVATED"
    elif risk_score <= 60:
        return "DEFENSIVE"
    elif risk_score <= 80:
        return "CRITICAL"
    else:
        return "PANIC"


class InstitutionalRiskArbitrator:
    """
    Phase 2: Institutional Risk State Arbitration.
    Defines hierarchy between Market, Signal, and Portfolio regimes.
    """

    def arbitrate(self, market_regime, signal_quality, portfolio_health_score, risk_index):
        """
        Determines the Execution Permission State based on all regimes.
        """
        risk_regime = get_risk_regime(risk_index)

        # Default state
        execution_state = "APPROVED"
        veto_reason = None

        if risk_regime in ["CRITICAL", "PANIC"]:
            execution_state = "VETOED"
            veto_reason = f"Institutional Risk Index ({risk_index:.1f}) in {risk_regime} territory"
        elif portfolio_health_score < 40:
            execution_state = "DEFENSIVE"
            veto_reason = "Portfolio drawdown / health score requires defensive sizing"
        elif "BEAR" in market_regime and signal_quality < 70:
            execution_state = "REDUCED_SIZE"
            veto_reason = "Bearish structural regime with sub-optimal signal quality"

        return {
            "risk_regime": risk_regime,
            "execution_state": execution_state,
            "veto_reason": veto_reason,
            "risk_layers": {
                "market": market_regime,
                "signal": "NOMINAL" if signal_quality >= 70 else "WEAK",
                "portfolio": "HEALTHY" if portfolio_health_score >= 70 else ("STRESSED" if portfolio_health_score >= 40 else "CRITICAL"),
                "authority": execution_state
            }
        }


def calculate_dynamic_volatility_sizing(
    vix: float,
    p_buy: float,
    is_dip_reversal: bool = False,
    is_macro_bull: bool = True,
    historical_win_rate: float = 0.58,
    historical_rr: float = 2.8,
) -> dict:
    """
    Dynamic Volatility Position Sizing (Half-Kelly & Volatility-Scaling Engine).
    Scales target capital allocation (0.0 to 1.0) dynamically between:
    - Calm, High-Conviction Bull Trend (VIX < 18, P(BUY) >= 0.60): 100% Capital (1.00x)
    - Moderate Volatility (18 <= VIX <= 22): 75% to 85% Capital (0.75x to 0.85x)
    - Turbulent Regime / Panic Dip Reversal (VIX > 22 or Capitulation Dip): 50% to 70% Capital (0.50x to 0.70x)
    - Extreme Crisis / Black Swan Turbulence (VIX > 28): 50% Defensive Floor (0.50x)
    """
    vix_val = float(vix) if pd.notna(vix) and vix > 0 else 18.0
    prob_buy = float(p_buy) if pd.notna(p_buy) else 0.50

    # Half-Kelly baseline reference
    full_kelly = historical_win_rate - ((1.0 - historical_win_rate) / max(0.5, historical_rr))
    half_kelly = max(0.25, full_kelly * 0.5)

    # Dynamic Volatility & Conviction Scaling
    if vix_val < 18.0 and prob_buy >= 0.60 and is_macro_bull:
        allocation = 1.00
        regime_label = "CALM_HIGH_CONVICTION"
        rationale = f"Calm market with high directional conviction (VIX={vix_val:.1f} < 18, P_buy={prob_buy:.2f} >= 0.60): 100% Full Sizing."
    elif vix_val > 28.0:
        allocation = 0.50
        regime_label = "HIGH_TURBULENCE_DEFENSIVE"
        rationale = f"Elevated market fear (VIX={vix_val:.1f} > 28): 50% Defensive Sizing to minimize dollar variance."
    elif vix_val > 22.0 or is_dip_reversal:
        # Scale between 50% and 70% based on conviction
        allocation = 0.70 if prob_buy >= 0.40 else 0.50
        regime_label = "TURBULENT_DIP_REVERSAL"
        rationale = f"Turbulent regime / Panic dip (VIX={vix_val:.1f}, Dip={is_dip_reversal}): {int(allocation * 100)}% Risk-Managed Sizing."
    elif 18.0 <= vix_val <= 22.0:
        allocation = 0.85 if prob_buy >= 0.60 else 0.75
        regime_label = "MODERATE_VOLATILITY_STANDARD"
        rationale = f"Moderate volatility (VIX={vix_val:.1f}): {int(allocation * 100)}% Standard Allocation."
    else:
        allocation = 1.00 if prob_buy >= 0.60 else 0.75
        regime_label = "NOMINAL_EQUITY"
        rationale = f"Nominal baseline allocation: {int(allocation * 100)}% Capital."

    return {
        "target_allocation": float(round(allocation, 2)),
        "target_exposure": float(round(allocation, 2)),
        "allocation_pct": int(allocation * 100),
        "vix_basis": float(round(vix_val, 2)),
        "half_kelly_fraction": float(round(half_kelly, 3)),
        "regime_label": regime_label,
        "sizing_rationale": rationale,
    }


