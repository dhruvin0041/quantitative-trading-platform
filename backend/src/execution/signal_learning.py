import pandas as pd
from typing import Dict, Any


class SignalLearningEngine:
    """
    Phase 8: Signal Learning System.
    Analyzes historical signal outcomes to discover optimal trading conditions.
    """

    def __init__(self, journal):
        self.journal = journal

    def discover_optimal_conditions(self) -> Dict[str, Any]:
        df = self.journal.get_all_signals()
        if df.empty or "outcome" not in df.columns:
            return {"status": "INSUFFICIENT_DATA"}

        # Filter for closed signals with outcomes
        closed = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
        if len(closed) < 10:
            return {"status": "INSUFFICIENT_DATA", "count": len(closed)}

        closed["is_win"] = (closed["outcome"] == "WIN").astype(int)

        # Analyze by Regime
        regime_stats = (
            closed.groupby("market_regime_v2")["is_win"]
            .agg(["mean", "count"])
            .to_dict("index")
        )

        # Analyze by Quality Grade
        grade_stats = (
            closed.groupby("quality_grade")["is_win"]
            .agg(["mean", "count"])
            .to_dict("index")
        )

        # Discover "Golden" combination
        # Group by multiple factors
        combo_stats = closed.groupby(
            ["market_regime_v2", "asset_class", "quality_grade"]
        )["is_win"].agg(["mean", "count"])
        best_combo = (
            combo_stats[combo_stats["count"] >= 3]
            .sort_values("mean", ascending=False)
            .head(1)
        )

        golden_condition = "UNKNOWN"
        if not best_combo.empty:
            idx = best_combo.index[0]
            golden_condition = f"{idx[1]} in {idx[0]} ({idx[2]} grade)"

        return {
            "status": "SUCCESS",
            "total_signals_analyzed": len(closed),
            "regime_performance": regime_stats,
            "grade_performance": grade_stats,
            "golden_condition": golden_condition,
            "win_rate": round(closed["is_win"].mean() * 100, 1),
        }


class SignalPerformanceResearch:
    """
    Phase 12: Signal Performance Research.
    Validates if higher Quality Scores correlate with better performance.
    """

    def analyze_quality_buckets(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty or "quality_score" not in df.columns:
            return {}

        # Define buckets
        buckets = [
            (90, 100, "ELITE"),
            (80, 90, "INSTITUTIONAL"),
            (70, 80, "HIGH_CONVICTION"),
            (60, 70, "WATCHLIST"),
            (0, 60, "LOW_QUALITY"),
        ]

        results = {}
        df["is_win"] = (df["outcome"] == "WIN").astype(int)

        for low, high, label in buckets:
            mask = (df["quality_score"] >= low) & (df["quality_score"] < high)
            subset = df[mask]

            if not subset.empty:
                win_rate = subset["is_win"].mean()
                avg_pnl = subset["realized_pnl"].mean()
                results[label] = {
                    "count": len(subset),
                    "win_rate": round(win_rate * 100, 1),
                    "avg_pnl": round(avg_pnl, 4),
                    "score_range": f"{low}-{high}",
                }

        return results
