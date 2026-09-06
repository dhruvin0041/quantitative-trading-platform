from typing import Dict


class ConfidenceBreakdownEngine:
    """
    Decomposes confidence into an explainable matrix:
    Trend, Regime, Volatility, Consensus, EV, Timing, Asset Intelligence.
    """

    def decompose_confidence(
        self,
        regime: str,
        volatility_ratio: float,
        agreement_score: float,
        ev_pct: float,
        timing_score: float,
        asset_class: str,
        direction: str = "BUY",
    ) -> Dict:

        # Base scoring out of 100 max composite

        # 1. Trend Score (-20 to +20): Direction-consistent trend-following reward
        trend_score = 0.0
        is_short = str(direction).upper() in ["SELL", "SHORT", "0"]
        if "BULL_TREND" in regime:
            trend_score = -20.0 if is_short else 20.0
        elif "BEAR_TREND" in regime:
            trend_score = 20.0 if is_short else -20.0
        elif "RANGE" in regime:
            trend_score = 0.0

        # 2. Regime Score (-15 to +15)
        # Represents cleanliness of regime
        regime_score = 10.0 if "TREND" in regime else -5.0

        # 3. Volatility Score (-10 to +10)
        # High vol ratio implies risk, low vol implies stability
        vol_score = (
            -10.0
            if volatility_ratio > 1.5
            else (10.0 if volatility_ratio < 0.8 else 0.0)
        )

        # 4. Consensus Score (-20 to +20)
        consensus_score = (agreement_score / 100.0) * 20.0
        if agreement_score < 40:
            consensus_score *= -1  # Lack of consensus is negative

        # 5. EV Score (-15 to +15)
        # ev_pct is in percentage units (e.g. 5.0 for 5% EV -> +10)
        ev_score = min(15.0, ev_pct * 2.0)
        if ev_pct <= 0:
            ev_score = -15.0

        # 6. Timing Score (-10 to +10)
        timing_score_mapped = float(timing_score)

        # 7. Asset Intelligence Score (-10 to +10)
        # Placeholder for structural asset biases
        asset_score = 5.0 if asset_class in ["EQUITY", "INDEX"] else 0.0

        total_score = (
            trend_score
            + regime_score
            + vol_score
            + consensus_score
            + ev_score
            + timing_score_mapped
            + asset_score
        )

        # Normalize to 0-100% confidence
        # A perfect score is ~100. A terrible score is ~-100.
        normalized_confidence = max(0.0, min(100.0, (total_score + 50) / 1.5))

        return {
            "confidence_breakdown": {
                "Trend": round(trend_score, 1),
                "Regime": round(regime_score, 1),
                "Volatility": round(vol_score, 1),
                "Consensus": round(consensus_score, 1),
                "Expected_Value": round(ev_score, 1),
                "Timing": round(timing_score_mapped, 1),
                "Asset_Intelligence": round(asset_score, 1),
                "Total_Raw_Score": round(total_score, 1),
            },
            "explainable_confidence": round(normalized_confidence, 1),
        }
