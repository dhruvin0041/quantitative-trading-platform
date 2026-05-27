import pandas as pd
import numpy as np
from typing import Dict


class PredictiveTimingEngine:
    """
    Generates forward-looking features (Momentum Curvature, Volatility Expansion,
    Market Structure Proxies) rather than lagging indicators.
    """

    def calculate_timing_features(self, df: pd.DataFrame) -> Dict:
        if len(df) < 20:
            return {"timing_score": 0.0, "signal_lead_time": 0.0}

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values

        # 1. Momentum Curvature (2nd derivative of price)
        # Velocity
        v1 = close[-1] - close[-5]
        v2 = close[-5] - close[-9]
        # Acceleration (Curvature)
        momentum_accel = v1 - v2

        # 2. Volatility Expansion (ATR acceleration)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])),
        )
        atr_5 = np.mean(tr[-5:])
        atr_20 = np.mean(tr[-20:])
        vol_expansion = (atr_5 / atr_20) - 1.0 if atr_20 > 0 else 0.0

        # 3. Order Flow / Participation Surge
        avg_vol = np.mean(volume[-20:-1])
        vol_surge = (volume[-1] / avg_vol) - 1.0 if avg_vol > 0 else 0.0

        # 4. Market Structure Break Proxy
        # Simple higher-high or lower-low detection in recent window
        hh = high[-1] > max(high[-5:-1])
        ll = low[-1] < min(low[-5:-1])
        structure_break = 1 if hh else (-1 if ll else 0)

        # Composite Timing Score (-10 to +10)
        timing_score = 0.0
        if momentum_accel > 0:
            timing_score += 3
        else:
            timing_score -= 3

        if vol_expansion > 0.2:
            timing_score += 2 * np.sign(v1)  # Breakout
        elif vol_expansion < -0.2:
            timing_score -= 1 * np.sign(v1)  # Compression

        if vol_surge > 0.5:
            timing_score += 2 * np.sign(v1)

        timing_score += 3 * structure_break

        # Clip to bounds
        timing_score = max(-10.0, min(10.0, float(timing_score)))

        return {
            "timing_score": timing_score,
            "momentum_acceleration": float(momentum_accel),
            "volatility_expansion": float(vol_expansion),
            "volume_surge": float(vol_surge),
            "structure_break": structure_break,
            "signal_lead_time": 0.0,  # Placeholder for Phase 7 validation tracking
        }
