import numpy as np
import pandas as pd

from src.execution.forensic_reset import ForensicPortfolioResetEngine
from src.execution.signal_learning import SignalPerformanceResearch
from src.execution.statistical_engine import StatisticalValidityEngine


class PerformanceAnalyzer:
    """
    Institutional Performance Analysis Engine.
    Computes Sharpe, Sortino, Calmar, Drawdowns, and Sector/Regime attribution.
    """

    def __init__(self, risk_free_rate=0.04):
        self.risk_free_rate = risk_free_rate
        self.signal_research = SignalPerformanceResearch()
        self.stat_engine = StatisticalValidityEngine()
        self.forensic_reset = ForensicPortfolioResetEngine()

    def analyze(self, snapshots, trade_history, initial_capital, signal_data=None):
        if not snapshots:
            return {
                "summary": {
                    "total_return": 0.0,
                    "sharpe": 0.0,
                    "sortino": 0.0,
                    "calmar": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "today_pnl": 0.0,
                    "mtd_pnl": 0.0,
                    "ytd_pnl": 0.0,
                    "inception_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "total_trades": 0,
                    "open_trades": 0,
                    "closed_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "initial_capital": initial_capital,
                    "expectancy": 0.0,
                },
                "returns": {"daily": {}, "monthly": {}},
                "attribution": {"by_regime": {}, "by_sector": {}},
                "signal_research": {},
            }

        # Segmentation Logic (Phase 7)
        segmentation = self.forensic_reset.segment_telemetry(snapshots, trade_history)
        active_snapshots = segmentation["validated_snapshots"] if segmentation["validated_snapshots"] else snapshots
        active_history = segmentation["validated_history"] if segmentation["validated_history"] else trade_history

        df_snapshots = pd.DataFrame(active_snapshots)
        df_snapshots["time"] = pd.to_datetime(df_snapshots["time"])
        df_snapshots = df_snapshots.set_index("time")

        # Daily Returns & Rolling Metrics (Phase 10)
        daily_equity = df_snapshots["equity"].resample("D").last().ffill()
        daily_returns = daily_equity.pct_change().dropna()

        rolling_sharpe = daily_returns.rolling(window=60).apply(lambda x: self.calculate_sharpe(x)) if len(daily_returns) >= 60 else pd.Series()
        rolling_sortino = daily_returns.rolling(window=60).apply(lambda x: self.calculate_sortino(x)) if len(daily_returns) >= 60 else pd.Series()

        # Institutional PnL Breakdown
        today_equity = daily_equity.iloc[-1] if not daily_equity.empty else initial_capital
        inception_return_pct = ((today_equity / initial_capital) - 1) * 100 if initial_capital != 0 else 0.0

        # Monthly Returns
        monthly_equity = df_snapshots["equity"].resample("ME").last().ffill()
        monthly_returns = monthly_equity.pct_change().dropna()

        # Portfolio Level Metrics
        sharpe = self.calculate_sharpe(daily_returns)
        sortino = self.calculate_sortino(daily_returns)
        max_dd, calmar, peak_info, trough_info = self.calculate_drawdown_metrics(daily_equity, daily_returns)

        # Advanced Trade Analytics (Phase 10)
        trades_df = pd.DataFrame(active_history) if active_history else pd.DataFrame()
        win_rate = 0.0
        profit_factor = 0.0
        wins_count = 0
        closed_trades_count = 0
        realized_pnl = 0.0
        expectancy = 0.0
        mae_avg = 0.0
        mfe_avg = 0.0
        avg_hold_duration = 0.0

        if not trades_df.empty:
            sells = trades_df[trades_df["action"] == "SELL"]
            if not sells.empty:
                closed_trades_count = len(sells)
                wins = sells[sells["pnl"] > 0]
                losses = sells[sells["pnl"] < 0]
                wins_count = len(wins)
                win_rate = wins_count / closed_trades_count
                realized_pnl = sells["pnl"].sum()

                gross_profit = wins["pnl"].sum()
                gross_loss = abs(losses["pnl"].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if wins_count > 0 else 0.0)
                expectancy = self.calculate_expectancy(win_rate, wins, losses)

                # MAE / MFE Calculation (Simulation or Paper History)
                if "mae" in sells.columns:
                    mae_avg = sells["mae"].mean()
                if "mfe" in sells.columns:
                    mfe_avg = sells["mfe"].mean()

                # Holding Duration
                if "entry_time" in sells.columns and "exit_time" in sells.columns:
                    durations = pd.to_datetime(sells["exit_time"]) - pd.to_datetime(sells["entry_time"])
                    avg_hold_duration = durations.mean().total_seconds() / 3600 # hours

        summary_metrics = {
            "total_return": inception_return_pct,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd * 100,
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "realized_pnl": realized_pnl,
            "total_trades": len(active_history),
            "closed_trades": closed_trades_count,
            "expectancy": expectancy,
            "mae_avg": mae_avg,
            "mfe_avg": mfe_avg,
            "avg_hold_duration": avg_hold_duration,
        }

        # Unified Statistical Gating (Phase 1)
        stat_validity = self.stat_engine.validate_statistics(
            daily_returns.tolist(),
            active_history,
            summary_metrics,
        )

        return {
            "summary": stat_validity["validated_metrics"],
            "confidence_intervals": stat_validity["confidence_intervals"],
            "sample_sizes": stat_validity["sample_sizes"],
            "rolling": {
                "sharpe": rolling_sharpe.tail(30).to_dict(),
                "sortino": rolling_sortino.tail(30).to_dict()
            },
            "returns": {
                "daily": daily_returns.tail(30).to_dict(),
                "monthly": monthly_returns.to_dict(),
            },
            "forensic_audit": {
                "validated_start_date": segmentation["validated_start_date"],
                "trusted_era_active": segmentation["trusted_era_active"],
            },
        }


    def calculate_sharpe(self, returns):
        """Standardized Sharpe: (Mean - RiskFree) / StdDev * sqrt(252)"""
        if len(returns) < 5 or returns.std() < 1e-7:
            return 0.0
        adj_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1
        excess_returns = returns - adj_rf
        return np.sqrt(252) * excess_returns.mean() / (returns.std() + 1e-9)

    def calculate_sortino(self, returns):
        """Standardized Sortino: (Mean - RiskFree) / DownsideDev * sqrt(252)"""
        if len(returns) < 5:
            return 0.0
        adj_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1
        excess_returns = returns - adj_rf
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 2:
            return 0.0
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / (downside_std + 1e-9)

    def calculate_drawdown_metrics(self, equity, returns):
        if len(equity) < 2:
            return 0.0, 0.0, {"value": 0, "date": ""}, {"value": 0, "date": ""}
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / (rolling_max + 1e-9)
        max_dd = drawdowns.min()
        trough_idx = drawdowns.idxmin()
        peak_idx = (
            equity[:trough_idx].idxmax()
            if not equity[:trough_idx].empty
            else equity.index[0]
        )
        peak_info = {"value": float(equity.loc[peak_idx]), "date": str(peak_idx.date())}
        trough_info = {
            "value": float(equity.loc[trough_idx]),
            "date": str(trough_idx.date()),
        }
        annual_return = returns.mean() * 252
        calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-7 else 0.0
        return max_dd, calmar, peak_info, trough_info

    def calculate_expectancy(self, win_rate, wins, losses):
        """Avg Profit * Win% - Avg Loss * Loss%"""
        avg_win = wins["pnl"].mean() if not wins.empty else 0.0
        avg_loss = abs(losses["pnl"].mean()) if not losses.empty else 0.0
        return (avg_win * win_rate) - (avg_loss * (1 - win_rate))
