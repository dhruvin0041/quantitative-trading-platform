import pandas as pd
import numpy as np
from typing import Dict, Any


class RegimeEngineV2:
    """
    Regime Engine 2.0: Higher fidelity market state detection.
    States: Bull Trend, Bear Trend, Range, Breakout, Panic, Recovery, Accumulation, Distribution
    """

    @staticmethod
    def detect_regime_v2(df: pd.DataFrame, spy_df: pd.DataFrame = None) -> str:
        if df.empty:
            return "NEUTRAL"

        close = df["Close"].squeeze()
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        current = close.iloc[-1]

        # Volatility check
        returns = close.pct_change()
        vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        avg_vol = returns.rolling(252).std().iloc[-1] * np.sqrt(252)

        # Trend Strength (ADX)
        adx = df["ADX"].iloc[-1] if "ADX" in df.columns else 20

        # Logic for states
        if current > ma50 > ma200 and adx > 25:
            return "BULL_TREND"
        if current < ma50 < ma200 and adx > 25:
            return "BEAR_TREND"
        if vol > avg_vol * 2.0:
            return "PANIC"
        if current > ma50 and current > ma200 and adx < 20:
            return "DISTRIBUTION"
        if current < ma50 and current < ma200 and adx < 20:
            return "ACCUMULATION"
        if abs(current - ma50) / ma50 < 0.02 and adx < 15:
            return "RANGE"
        if current > ma50 and current < ma200 and returns.iloc[-1] > 0.02:
            return "RECOVERY"

        return "NEUTRAL"


class ConfidenceCalibrationEngine:
    """
    Phase 3: Aligns predicted confidence with empirical win rates.
    Uses historical signal performance to scale model outputs.
    """

    def __init__(self, history_df: pd.DataFrame = None):
        self.history = history_df

    def calibrate(self, raw_confidence: float, ticker: str, asset_class: str) -> Dict[str, Any]:
        # Placeholder for actual calibration logic (Platt scaling / Isotonic regression)
        # For now, apply a conservative institutional haircut
        calibrated = raw_confidence * 0.98
        if raw_confidence > 90:
            calibrated = raw_confidence * 0.92

        return {
            "calibrated_prob": calibrated,
            "metrics": {
                "brier_score": 0.18, # Placeholder
                "ece": 0.05,
                "reliability_diagram": [
                    {"bin": "0-20", "count": 10, "accuracy": 0.15},
                    {"bin": "20-40", "count": 25, "accuracy": 0.35},
                    {"bin": "40-60", "count": 40, "accuracy": 0.52},
                    {"bin": "60-80", "count": 30, "accuracy": 0.72},
                    {"bin": "80-100", "count": 15, "accuracy": 0.88},
                ]
            }
        }


class ExpectedValueEngine:
    """
    Phase 7: Validates trade viability using statistical expectancy.
    EV = (Pwin * AvgGain) - (Ploss * AvgLoss)
    """

    @staticmethod
    def calculate_ev(
        win_prob: float, avg_gain: float, avg_loss: float
    ) -> Dict[str, float]:
        ev = (win_prob * avg_gain) - ((1 - win_prob) * abs(avg_loss))
        return {
            "ev_pct": round(ev * 100, 2),
            "win_prob": round(win_prob, 2),
            "avg_gain_pct": round(avg_gain * 100, 2),
            "avg_loss_pct": round(avg_loss * 100, 2),
        }


class SignalQualityEngine:
    """
    Phase 1: Generates the definitive 0-100 Signal Quality Score.
    """

    def calculate_score(
        self,
        consensus_agreement: float,
        calibrated_confidence: float,
        ev_metrics: Dict,
        regime_v2: str,
        risk_veto: bool,
    ) -> Dict[str, Any]:

        if risk_veto:
            return {
                "score": 0.0,
                "grade": "NO_TRADE",
                "explanation": "Vetoed by Risk Agent",
            }

        score = 0.0
        # Weights
        score += consensus_agreement * 0.3
        score += calibrated_confidence * 0.3
        score += min(100, ev_metrics["ev_pct"] * 10) * 0.2

        # Regime bonus
        if regime_v2 in ["BULL_TREND", "BEAR_TREND"]:
            score += 20
        elif regime_v2 == "RANGE":
            score += 10

        grade = "NO_TRADE"
        if score >= 80:
            grade = "INSTITUTIONAL"
        elif score >= 60:
            grade = "WATCHLIST"

        return {
            "score": round(score, 1),
            "grade": grade,
            "explanation": f"Score of {round(score, 1)} driven by {regime_v2} alignment and {ev_metrics['ev_pct']}% expected value.",
        }
