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
    Phase 9: Signal Quality Validation Research.
    Validates if higher Quality Scores correlate with better performance.
    """

    def analyze_quality_buckets(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty or "quality_score" not in df.columns:
            return {}

        # Define buckets as requested
        buckets = [
            (90, 100, "90-100"),
            (80, 90, "80-90"),
            (70, 80, "70-80"),
            (60, 70, "60-70"),
        ]

        results = {}
        # Ensure we only check closed trades for accuracy
        valid_trades = (
            df[df["outcome"].isin(["WIN", "LOSS"])].copy()
            if "outcome" in df.columns
            else pd.DataFrame()
        )

        if not valid_trades.empty:
            valid_trades["is_win"] = (valid_trades["outcome"] == "WIN").astype(int)

            for low, high, label in buckets:
                mask = (valid_trades["quality_score"] >= low) & (
                    valid_trades["quality_score"] <= high
                )
                subset = valid_trades[mask]

                if not subset.empty:
                    win_rate = subset["is_win"].mean()

                    # Compute simplified Sharpe and Profit Factor for the bucket
                    # Note: These are rough approximations per signal bucket
                    avg_pnl = (
                        subset["realized_pnl"].mean()
                        if "realized_pnl" in subset.columns
                        else 0.0
                    )
                    std_pnl = (
                        subset["realized_pnl"].std()
                        if "realized_pnl" in subset.columns
                        else 0.0
                    )
                    sharpe = (avg_pnl / std_pnl) if std_pnl > 0 else 0.0

                    gross_profit = (
                        subset[subset["realized_pnl"] > 0]["realized_pnl"].sum()
                        if "realized_pnl" in subset.columns
                        else 0.0
                    )
                    gross_loss = (
                        abs(subset[subset["realized_pnl"] < 0]["realized_pnl"].sum())
                        if "realized_pnl" in subset.columns
                        else 0.0
                    )
                    pf = (
                        (gross_profit / gross_loss)
                        if gross_loss > 0
                        else (10.0 if gross_profit > 0 else 0.0)
                    )

                    expectancy = (
                        (
                            subset[subset["is_win"] == 1]["realized_pnl"].mean()
                            * win_rate
                        )
                        - (
                            abs(subset[subset["is_win"] == 0]["realized_pnl"].mean())
                            * (1 - win_rate)
                        )
                        if "realized_pnl" in subset.columns
                        else 0.0
                    )

                    drawdown = (
                        subset["realized_pnl"].min()
                        if "realized_pnl" in subset.columns
                        else 0.0
                    )

                    results[label] = {
                        "count": len(subset),
                        "win_rate": round(win_rate * 100, 1),
                        "sharpe": round(sharpe, 2),
                        "profit_factor": round(pf, 2),
                        "expectancy": round(expectancy, 4),
                        "avg_return": round(avg_pnl, 4),
                        "drawdown": round(drawdown, 4),
                    }

        return results
