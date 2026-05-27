import numpy as np
from typing import Dict


class TradeConstructionEngine:
    """
    Dynamically constructs trade parameters (entry, stop loss, target) based on ATR,
    regime, and asset class. Outputs precise RR ratios and holding time estimates.
    """

    def __init__(self):
        self.asset_multipliers = {
            "EQUITY": {"stop": 1.5, "target": 3.0},
            "CRYPTO": {"stop": 2.5, "target": 5.0},
            "COMMODITY": {"stop": 1.2, "target": 2.5},
            "FOREX": {"stop": 1.0, "target": 2.0},
            "INDEX": {"stop": 1.2, "target": 2.5},
            "UNKNOWN": {"stop": 1.5, "target": 3.0},
        }

    def construct_trade(
        self,
        current_price: float,
        atr: float,
        direction: str,
        regime: str,
        asset_class: str,
        volatility: float,
    ) -> Dict:
        if direction not in ["BUY", "SELL"]:
            return {
                "entry_price": current_price,
                "stop_price": 0.0,
                "target_price": 0.0,
                "risk_distance": 0.0,
                "reward_distance": 0.0,
                "rr_ratio": 0.0,
                "holding_time_estimate": 0,
                "is_valid": False,
                "reject_reason": "Direction is HOLD",
            }

        base_mult = self.asset_multipliers.get(
            asset_class, self.asset_multipliers["UNKNOWN"]
        )
        stop_mult = base_mult["stop"]
        target_mult = base_mult["target"]

        # Adjust for regime and volatility
        if "TREND" in regime:
            target_mult *= 1.5  # Wider targets in trends
        elif "RANGE" in regime:
            target_mult *= 0.8  # Tighter targets in ranges

        if volatility > 0.03:  # High volatility
            stop_mult *= 1.2  # Wider stops to avoid chop

        # Calculate levels
        risk_distance = atr * stop_mult
        reward_distance = atr * target_mult

        if direction == "BUY":
            stop_price = current_price - risk_distance
            target_price = current_price + reward_distance
        else:  # SELL
            stop_price = current_price + risk_distance
            target_price = current_price - reward_distance

        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0

        # Estimate holding time based on expected move vs ATR
        # Assuming asset moves ~1 ATR per day directionally (highly simplified)
        holding_time_estimate = int(np.ceil(reward_distance / atr)) if atr > 0 else 5

        # Validation
        is_valid = True
        reject_reason = None
        if rr_ratio < 1.5:
            is_valid = False
            reject_reason = f"RR Ratio ({rr_ratio:.2f}) < 1.5"

        return {
            "entry_price": current_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "risk_distance": risk_distance,
            "reward_distance": reward_distance,
            "rr_ratio": rr_ratio,
            "holding_time_estimate": holding_time_estimate,
            "is_valid": is_valid,
            "reject_reason": reject_reason,
        }
