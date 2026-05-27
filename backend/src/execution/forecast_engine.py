import numpy as np
from typing import Dict


class ForecastCalibrationEngine:
    """
    Calibrates point forecasts into realistic percentile bands based on volatility and asset class.
    Outputs: P10, P50, P90 forecasts, confidence metrics, and rejects unrealistic projections.
    """

    def __init__(self):
        # Base realistic bounds per asset class (max 10-day move percentage)
        self.asset_bounds = {
            "EQUITY": 0.08,
            "CRYPTO": 0.20,
            "COMMODITY": 0.05,  # Gold/Silver
            "FOREX": 0.03,
            "INDEX": 0.05,
            "UNKNOWN": 0.10,
        }

    def calibrate_forecast(
        self,
        raw_forecasts: np.ndarray,
        current_price: float,
        atr: float,
        volatility: float,
        asset_class: str,
        regime: str,
    ) -> Dict:
        # Convert raw TFT forecasts to pct returns
        avg_forecast = float(np.mean(raw_forecasts))
        implied_return = (
            (avg_forecast - current_price) / current_price if current_price > 0 else 0
        )

        # 1. Asset-Specific Volatility Envelope
        max_allowed_move = self.asset_bounds.get(asset_class, 0.10)

        # Expand bounds if volatility is exceptionally high or regime is trending
        if regime in ["BULL_TREND", "BEAR_TREND"]:
            max_allowed_move *= 1.2

        # Incorporate ATR (e.g. 10-day projection ~ sqrt(10)*ATR roughly)
        atr_pct = (atr / current_price) if current_price > 0 else 0.01
        vol_envelope = max(max_allowed_move, atr_pct * 3.16)  # ~10 days

        # 2. Sanity Validator
        is_valid = True
        reject_reason = None

        if abs(implied_return) > vol_envelope * 1.5:  # 1.5x buffer for extreme spikes
            is_valid = False
            reject_reason = f"Forecast move ({implied_return * 100:.1f}%) exceeds volatility envelope ({vol_envelope * 100:.1f}%)"

        # 3. Percentile Generation
        # Assume raw forecast is the mean, standard dev based on volatility
        sigma = max(atr_pct, volatility) * current_price

        # If forecast was valid, center around it, else center around 0 move or clipped move
        if not is_valid:
            # Clip to bounds
            safe_return = np.clip(implied_return, -vol_envelope, vol_envelope)
            center_price = current_price * (1 + safe_return)
        else:
            center_price = avg_forecast

        p10 = center_price - (1.28 * sigma)  # 10th percentile
        p50 = center_price  # 50th percentile (median)
        p90 = center_price + (1.28 * sigma)  # 90th percentile

        # 4. Forecast Confidence
        # Higher volatility/wider range = lower confidence
        spread_pct = (p90 - p10) / current_price
        forecast_confidence = max(0.0, min(100.0, 100.0 - (spread_pct * 100 * 2)))
        forecast_uncertainty = spread_pct

        return {
            "is_valid": is_valid,
            "reject_reason": reject_reason,
            "p10_price": p10,
            "p50_price": p50,
            "p90_price": p90,
            "p10_return": (p10 - current_price) / current_price,
            "p50_return": (p50 - current_price) / current_price,
            "p90_return": (p90 - current_price) / current_price,
            "forecast_confidence": forecast_confidence,
            "forecast_uncertainty": forecast_uncertainty,
            "forecast_reliability": "HIGH"
            if forecast_confidence > 70
            else "LOW"
            if forecast_confidence < 30
            else "MEDIUM",
        }
