import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class MicrostructureExecutionModel:
    """
    Phase 4: Institutional Microstructure Execution Model.
    Replaces static/costless fills with dynamic execution mechanics:
    1. Volatility-scaled Bid-Ask Spread crossing.
    2. Almgren-Chriss square-root law Market Impact:
       Impact = eta * volatility * sqrt(Order_Size / Bar_Volume)
    3. Adverse Selection: Orders going with recent price momentum suffer liquidity dry-up.
    4. Volume Participation Limit: Hard ceiling on max allowable percentage of bar volume.
    """

    def __init__(
        self,
        base_spread_bps: float = 3.0,
        impact_eta: float = 0.15,
        max_participation_rate: float = 0.025,
        commission_per_share: float = 0.005,
    ):
        self.base_spread_bps = base_spread_bps
        self.impact_eta = impact_eta
        self.max_participation_rate = max_participation_rate
        self.commission_per_share = commission_per_share

    def calculate_adverse_slippage(
        self,
        price: float,
        action: str,
        volatility_pct: float = 0.015,
        order_shares: int = 100,
        bar_volume: float = 1_000_000.0,
        recent_momentum_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Computes dynamic microstructure execution price and friction breakdown.

        Parameters:
        -----------
        price : float
            Nominal price (e.g. bar Open or Close).
        action : str
            "BUY" or "SELL".
        volatility_pct : float
            Normalized ATR or daily return volatility (e.g., 0.015 = 1.5%).
        order_shares : int
            Quantity of shares in the order.
        bar_volume : float
            Total traded volume in the execution bar/session.
        recent_momentum_pct : float
            Preceding return direction (e.g., +0.02 if market jumped +2%).
        """
        price_val = float(price) if not np.isnan(price) else 0.0
        if price_val <= 0:
            return {
                "executed_price": price_val,
                "total_friction_bps": 0.0,
                "spread_bps": 0.0,
                "market_impact_bps": 0.0,
                "adverse_selection_bps": 0.0,
                "fill_ratio": 1.0,
            }

        act = str(action).upper().strip()
        vol = float(volatility_pct) if not np.isnan(volatility_pct) else 0.015
        vol = max(vol, 0.001)

        mom = float(recent_momentum_pct) if not np.isnan(recent_momentum_pct) else 0.0
        shares_safe = max(int(order_shares), 1)

        # 1. Spread Model: Base half-spread scaled by relative volatility
        # During high volatility regimes (vol > 2%), spreads widen significantly
        vol_scalar = max(vol / 0.015, 0.5)
        spread_bps = (self.base_spread_bps / 2.0) * vol_scalar

        # 2. Market Impact Model: Almgren-Chriss Square-Root Law
        # Impact = eta * sigma * sqrt(Order_Size / Bar_Volume)
        bar_volume_safe = max(float(bar_volume) if not np.isnan(bar_volume) else 10_000.0, 10_000.0)
        participation = min(shares_safe / bar_volume_safe, 1.0)

        # Check volume participation cap
        fill_ratio = 1.0
        if participation > self.max_participation_rate:
            fill_ratio = self.max_participation_rate / max(participation, 1e-6)

        market_impact_bps = (
            self.impact_eta * vol * np.sqrt(max(participation, 1e-6)) * 10_000.0
        )

        # 3. Adverse Selection: Aggressive execution with the momentum costs extra
        # Buying after an upward surge pays extra liquidity premium
        adverse_selection_bps = 0.0
        if act == "BUY" and mom > 0:
            adverse_selection_bps = min(mom * 10_000.0 * 0.05, 15.0)
        elif act == "SELL" and mom < 0:
            adverse_selection_bps = min(abs(mom) * 10_000.0 * 0.05, 15.0)

        # Total basis points of slippage
        total_friction_bps = spread_bps + market_impact_bps + adverse_selection_bps
        friction_fraction = min(total_friction_bps / 10_000.0, 0.20)  # Capped at 20% max slippage

        if act == "BUY":
            executed_price = price_val * (1.0 + friction_fraction)
        elif act == "SELL":
            executed_price = max(price_val * (1.0 - friction_fraction), 0.01)
        else:
            executed_price = price_val

        return {
            "executed_price": float(executed_price),
            "total_friction_bps": float(round(total_friction_bps, 2)),
            "spread_bps": float(round(spread_bps, 2)),
            "market_impact_bps": float(round(market_impact_bps, 2)),
            "adverse_selection_bps": float(round(adverse_selection_bps, 2)),
            "fill_ratio": float(round(fill_ratio, 4)),
        }

    def simulate_execution_price(
        self,
        price: float,
        action: str,
        volatility_pct: float = 0.015,
        order_shares: int = 100,
        bar_volume: float = 1_000_000.0,
        recent_momentum_pct: float = 0.0,
    ) -> float:
        """Convenience method returning the slippage-adjusted execution price."""
        res = self.calculate_adverse_slippage(
            price=price,
            action=action,
            volatility_pct=volatility_pct,
            order_shares=order_shares,
            bar_volume=bar_volume,
            recent_momentum_pct=recent_momentum_pct,
        )
        return res["executed_price"]

