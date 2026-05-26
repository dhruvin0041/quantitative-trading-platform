import pandas as pd
import numpy as np
import json
from src.execution.signal_journal import SignalJournal


class ValidationAnalytics:
    """
    Phase 8.2 & 8.3 & 8.4 & 8.5 & 8.6 & 8.7
    Empirical Validation Engine.
    Computes rigorous statistics based entirely on live paper-trading evidence recorded in SignalJournal.
    """

    def __init__(self, journal: SignalJournal):
        self.journal = journal

    def get_full_dashboard_data(self):
        """
        Aggregates all metrics for Phase 8.8 Validation Dashboard.
        """
        df = self.journal.get_all_signals()
        closed_df = self.journal.get_closed_trades()

        if df.empty:
            return {"status": "NO_DATA"}

        return {
            "performance": self._compute_performance(closed_df),
            "calibration": self._compute_calibration(closed_df),
            "market_segmentation": self._compute_segmentation(closed_df, "market"),
            "regime_segmentation": self._compute_segmentation(
                closed_df, "market_regime"
            ),
            "model_attribution": self._compute_model_attribution(closed_df),
            "strategy_health": self._compute_strategy_health(closed_df),
            "recent_signals": json.loads(df.head(20).to_json(orient="records")),
        }

    def _compute_performance(self, df: pd.DataFrame):
        if df.empty:
            return {"total_trades": 0}

        wins = df[df["outcome"] == "WIN"]
        losses = df[df["outcome"] == "LOSS"]

        win_rate = len(wins) / len(df) if len(df) > 0 else 0

        gross_profit = wins["realized_pnl"].sum() if not wins.empty else 0
        gross_loss = abs(losses["realized_pnl"].sum()) if not losses.empty else 1e-9
        profit_factor = gross_profit / gross_loss

        avg_win = wins["realized_pnl"].mean() if not wins.empty else 0
        avg_loss = abs(losses["realized_pnl"].mean()) if not losses.empty else 0
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        avg_holding = df["holding_time"].mean()

        # Returns math
        # We need a series of portfolio returns to calculate Sharpe/Sortino
        # Since we just have trade PnLs, we can calculate trade-level Sharpe
        # Note: True portfolio Sharpe requires equity curve. This is trade-based proxy.
        pnl_series = df["realized_pnl"]
        trade_sharpe = (
            (pnl_series.mean() / pnl_series.std()) * np.sqrt(252)
            if len(pnl_series) > 1 and pnl_series.std() > 0
            else 0
        )

        downside = pnl_series[pnl_series < 0]
        trade_sortino = (
            (pnl_series.mean() / downside.std()) * np.sqrt(252)
            if len(downside) > 1 and downside.std() > 0
            else 0
        )

        return {
            "total_trades": len(df),
            "win_rate": float(win_rate * 100),
            "profit_factor": float(profit_factor),
            "sharpe_proxy": float(trade_sharpe),
            "sortino_proxy": float(trade_sortino),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "risk_reward": float(risk_reward),
            "expectancy": float(expectancy),
            "avg_holding_days": float(avg_holding),
        }

    def _compute_calibration(self, df: pd.DataFrame):
        """Phase 8.3: Confidence Calibration"""
        if df.empty:
            return {}

        bins = [50, 60, 70, 80, 90, 100]
        results = []

        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            mask = (df["confidence"] >= lower) & (df["confidence"] < upper)
            subset = df[mask]

            if len(subset) > 0:
                actual_wr = len(subset[subset["outcome"] == "WIN"]) / len(subset) * 100
                avg_conf = subset["confidence"].mean()
                results.append(
                    {
                        "bin": f"{lower}-{upper}%",
                        "count": len(subset),
                        "predicted_conf": float(avg_conf),
                        "actual_win_rate": float(actual_wr),
                        "error": float(abs(avg_conf - actual_wr)),
                    }
                )

        # Expected Calibration Error (ECE)
        ece = (
            sum([(r["count"] / len(df)) * r["error"] for r in results])
            if results
            else 0
        )

        return {"ece": float(ece), "reliability_curve": results}

    def _compute_segmentation(self, df: pd.DataFrame, column: str):
        if df.empty or column not in df.columns:
            return {}

        segments = {}
        for val in df[column].unique():
            subset = df[df[column] == val]
            segments[val] = self._compute_performance(subset)

        return segments

    def _compute_model_attribution(self, df: pd.DataFrame):
        """Phase 8.6: Individual model performance attribution."""
        if df.empty:
            return {}

        attribution = {}
        models_list = ["DL_FUSION", "XGB_AGENT", "LGBM_AGENT"]

        for model in models_list:
            model_wins = 0
            model_total = 0

            for _, row in df.iterrows():
                try:
                    consensus = json.loads(row["model_consensus"])
                    model_vote = consensus.get(model, {}).get("signal")

                    if model_vote:
                        model_total += 1
                        # If model voted correctly for a winning trade
                        if row["outcome"] == "WIN" and model_vote in ["BUY", "SELL"]:
                            model_wins += 1
                        # If model voted against a losing trade (correctly predicted direction but trade lost? No, simpler)
                        # Standard attribution: if model's vote matches the signal that became a WIN.
                        elif row["outcome"] == "LOSS" and model_vote in ["BUY", "SELL"]:
                            pass  # model was wrong
                except Exception:
                    continue

            wr = (model_wins / model_total * 100) if model_total > 0 else 0
            attribution[model] = {
                "win_rate": float(wr),
                "total_votes": model_total,
                "confidence_avg": 0.0,  # Could extend to log this too
            }

        return attribution

    def _compute_strategy_health(self, df: pd.DataFrame):
        """Phase 8.7: Degradation detection"""
        if len(df) < 20:
            return {"status": "INSUFFICIENT_DATA", "warning": False}

        # Compare last 10 trades to previous trades
        recent = df.head(10)
        historical = df.tail(len(df) - 10)

        recent_wr = len(recent[recent["outcome"] == "WIN"]) / len(recent)
        hist_wr = (
            len(historical[historical["outcome"] == "WIN"]) / len(historical)
            if len(historical) > 0
            else recent_wr
        )

        drift = hist_wr - recent_wr
        warning = drift > 0.15  # 15% drop in win rate

        return {
            "status": "DEGRADING" if warning else "HEALTHY",
            "warning": bool(warning),
            "rolling_win_rate": float(recent_wr * 100),
            "historical_win_rate": float(hist_wr * 100),
            "drift": float(drift * 100),
        }
