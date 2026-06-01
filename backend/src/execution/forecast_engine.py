import numpy as np
from typing import Dict, Any


class ForecastInterpretationEngine:
    """
    Phase 4: Forecast Interpretation Engine.
    Automatically interprets forecast distributions for institutional clarity.
    """

    def interpret(
        self, forecast_data: Dict[str, Any], volatility_state: str, regime: str
    ) -> Dict[str, Any]:
        """
        Derives semantic interpretation from the P10/P50/P90 distribution.
        """
        p10_ret = forecast_data.get("p10_return", 0.0)
        p50_ret = forecast_data.get("p50_return", 0.0)
        p90_ret = forecast_data.get("p90_return", 0.0)

        # 1. Skew & Bias Calculation
        skew = (p90_ret + p10_ret - 2 * p50_ret) / (p90_ret - p10_ret + 1e-9)
        spread = p90_ret - p10_ret

        # Semantic mapping
        interpretation = "Neutral Consolidation"
        explanation = "Forecast distribution shows balanced risk/reward."
        bias_state = "NEUTRAL"

        if p50_ret > 0.02 and skew > 0.1:
            interpretation = "Bullish Expansion"
            explanation = "Median drift is positive with significant upside skew."
            bias_state = "BULLISH"
        elif p50_ret < -0.02 and skew < -0.1:
            interpretation = "Bearish Distribution"
            explanation = "Median drift is negative with downside tail risk."
            bias_state = "BEARISH"
        elif spread < 0.03:
            interpretation = "Neutral Compression"
            explanation = (
                "Extremely tight forecast bounds suggest imminent volatility expansion."
            )
            bias_state = "NEUTRAL"
        elif volatility_state == "HIGH" and spread > 0.15:
            interpretation = "Volatility Spike Risk"
            explanation = (
                "Wide forecast spread indicates high institutional uncertainty."
            )
            bias_state = "VOLATILE"

        # Target Horizon Alignment Check
        if bias_state == "BULLISH" and "BEAR" in regime:
            interpretation = "Mean Reversion Probability"
            explanation = (
                "Counter-trend bullish distribution detected in a bearish structure."
            )
        elif bias_state == "BEARISH" and "BULL" in regime:
            interpretation = "Mean Reversion Probability"
            explanation = (
                "Counter-trend bearish distribution detected in a bullish structure."
            )

        # Stability Scoring
        instability_score = 0.0
        if forecast_data.get("forecast_confidence", 0.0) < 40:
            instability_score = 0.5
            interpretation = "Forecast Instability"
            explanation = "High model variance prevents reliable distribution mapping."

        return {
            "forecast_interpretation": interpretation,
            "forecast_bias": bias_state,
            "interpretation_explanation": explanation,
            "skew_coefficient": float(skew),
            "distribution_spread": float(spread),
            "instability_risk": instability_score,
        }


class ForecastCalibrationEngine:
    """
    Phase 6: Institutional Forecast Calibration Engine.
    Transforms point forecasts into volatility-adjusted confidence cones.
    """

    def __init__(self):
        self.asset_bounds = {
            "EQUITY": 0.08,
            "CRYPTO": 0.20,
            "COMMODITY": 0.05,
            "FOREX": 0.03,
            "INDEX": 0.05,
            "UNKNOWN": 0.10,
        }
        self.interpreter = ForecastInterpretationEngine()

    def calibrate_forecast(
        self,
        raw_forecasts: np.ndarray,
        current_price: float,
        atr: float,
        volatility: float,
        asset_class: str,
        regime: str,
        volatility_state: str = "MEDIUM",
    ) -> Dict:
        """
        Generates P10/P50/P90 percentile bands with forecast drift and implied bias.
        """
        # raw_forecasts are now percentage returns
        implied_move_pct = float(np.mean(raw_forecasts))

        # 1. Volatility Cone Parameters (Institutional Standard)
        # Use ATR-based standard deviation for the 10-day forecast horizon
        atr_pct = (atr / current_price) if current_price > 0 else 0.02
        # Sigma is roughly ATR * sqrt(horizon) / current_price
        sigma_10d = atr_pct * 3.16  # sqrt(10) days
        
        # 2. Sanity Validation & Clipping
        max_allowed = self.asset_bounds.get(asset_class, 0.10)
        if "TREND" in regime: max_allowed *= 1.25
        
        is_valid = abs(implied_move_pct) < max_allowed * 2.0
        safe_move = np.clip(implied_move_pct, -max_allowed, max_allowed)
        
        # 3. Percentile Band Generation (Confidence Cone)
        # P50 is the median expectation (calibrated point forecast)
        p50 = current_price * (1 + safe_move)
        
        # P10/P90 based on normal distribution approximation (1.28 z-score)
        p10 = p50 - (1.28 * sigma_10d * current_price)
        p90 = p50 + (1.28 * sigma_10d * current_price)
        
        # 4. Reliability & Bias
        spread_pct = (p90 - p10) / current_price
        # Reliability drops as volatility/spread increases
        reliability_score = max(0.0, min(100.0, 100.0 - (spread_pct * 150)))
        
        forecast_res = {
            "is_valid": is_valid,
            "p10_price": p10,
            "p50_price": p50,
            "p90_price": p90,
            "p10_return": (p10 - current_price) / current_price,
            "p50_return": (p50 - current_price) / current_price,
            "p90_return": (p90 - current_price) / current_price,
            "forecast_confidence": reliability_score,
            "forecast_drift": implied_move_pct,
            "expected_move_10d": spread_pct / 2.0,
            "forecast_reliability": "HIGH" if reliability_score > 75 else ("LOW" if reliability_score < 40 else "MEDIUM")
        }

        # Apply Semantic Interpretation
        interpretation = self.interpreter.interpret(forecast_res, volatility_state, regime)
        forecast_res.update(interpretation)

        return forecast_res

