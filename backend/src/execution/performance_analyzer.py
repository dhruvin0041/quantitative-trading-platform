import pandas as pd
import numpy as np
from datetime import datetime
from src.execution.signal_learning import SignalPerformanceResearch


class PerformanceAnalyzer:
    """
    Institutional Performance Analysis Engine.
    Computes Sharpe, Sortino, Calmar, Drawdowns, and Sector/Regime attribution.
    """

    def __init__(self, risk_free_rate=0.04):
        self.risk_free_rate = risk_free_rate
        self.signal_research = SignalPerformanceResearch()

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

        df_snapshots = pd.DataFrame(snapshots)
        df_snapshots["time"] = pd.to_datetime(df_snapshots["time"])
        df_snapshots = df_snapshots.set_index("time")

        # Daily Returns
        daily_equity = df_snapshots["equity"].resample("D").last().ffill()
        daily_returns = daily_equity.pct_change().dropna()

        # Institutional PnL Breakdown
        today = datetime.now().date()
        mtd_start = today.replace(day=1)
        ytd_start = today.replace(month=1, day=1)

        today_equity = (
            daily_equity.iloc[-1] if not daily_equity.empty else initial_capital
        )
        yesterday_equity = (
            daily_equity.iloc[-2] if len(daily_equity) > 1 else initial_capital
        )

        mtd_equity_start = (
            daily_equity.loc[daily_equity.index >= pd.Timestamp(mtd_start)].iloc[0]
            if not daily_equity.loc[daily_equity.index >= pd.Timestamp(mtd_start)].empty
            else initial_capital
        )
        ytd_equity_start = (
            daily_equity.loc[daily_equity.index >= pd.Timestamp(ytd_start)].iloc[0]
            if not daily_equity.loc[daily_equity.index >= pd.Timestamp(ytd_start)].empty
            else initial_capital
        )

        today_pnl = today_equity - yesterday_equity
        mtd_pnl = today_equity - mtd_equity_start
        ytd_pnl = today_equity - ytd_equity_start
        inception_pnl = today_equity - initial_capital
        inception_return_pct = (
            ((today_equity / initial_capital) - 1) * 100
            if initial_capital != 0
            else 0.0
        )

        # Monthly Returns
        monthly_equity = df_snapshots["equity"].resample("ME").last().ffill()
        monthly_returns = monthly_equity.pct_change().dropna()

        # Risk Metrics (Portfolio Level)
        sharpe = self.calculate_sharpe(daily_returns)
        sortino = self.calculate_sortino(daily_returns)
        max_dd, calmar, peak_info, trough_info = self.calculate_drawdown_metrics(
            daily_equity, daily_returns
        )

        # Consistent Trade Metrics (Asset Level)
        trades_df = pd.DataFrame(trade_history) if trade_history else pd.DataFrame()
        win_rate = 0.0
        profit_factor = 0.0
        wins_count = 0
        losses_count = 0
        closed_trades_count = 0
        realized_pnl = 0.0
        expectancy = 0.0

        if not trades_df.empty:
            if "pnl" in trades_df.columns:
                realized_pnl = trades_df[trades_df["action"] == "SELL"]["pnl"].sum()

            sells = trades_df[trades_df["action"] == "SELL"]
            if not sells.empty:
                closed_trades_count = len(sells)
                wins = sells[sells["pnl"] > 0]
                losses = sells[sells["pnl"] < 0]
                wins_count = len(wins)
                losses_count = len(losses)
                win_rate = wins_count / closed_trades_count

                gross_profit = wins["pnl"].sum()
                gross_loss = abs(losses["pnl"].sum())

                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss
                else:
                    profit_factor = 10.0 if wins_count > 0 else 0.0

                expectancy = self.calculate_expectancy(win_rate, wins, losses)

        total_trades = len(trade_history) if trade_history else 0
        unrealized_pnl = today_equity - (initial_capital + realized_pnl)

        # Phase 12: Signal Quality Research
        quality_research = {}
        if signal_data is not None:
            quality_research = self.signal_research.analyze_quality_buckets(signal_data)

        summary_metrics = {
            "total_return": inception_return_pct,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd * 100,
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "today_pnl": today_pnl,
            "mtd_pnl": mtd_pnl,
            "ytd_pnl": ytd_pnl,
            "inception_pnl": inception_pnl,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_trades": total_trades,
            "open_trades": total_trades - closed_trades_count,
            "closed_trades": closed_trades_count,
            "winning_trades": wins_count,
            "losing_trades": losses_count,
            "initial_capital": initial_capital,
            "peak_equity": peak_info["value"],
            "peak_date": peak_info["date"],
            "trough_equity": trough_info["value"],
            "trough_date": trough_info["date"],
            "expectancy": expectancy,
        }
        
        stat_validity = self.stat_engine.validate_statistics(
            daily_returns.tolist() if not daily_returns.empty else [],
            trade_history if trade_history else [],
            summary_metrics
        )

        return {
            "summary": stat_validity["validated_metrics"],
            "confidence_intervals": stat_validity["confidence_intervals"],
            "sample_sizes": stat_validity["sample_sizes"],
            "returns": {
                "daily": daily_returns.tail(30).to_dict(),
                "monthly": monthly_returns.to_dict(),
            },
            "attribution": {"by_regime": {}, "by_sector": {}},
            "signal_research": quality_research,
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
