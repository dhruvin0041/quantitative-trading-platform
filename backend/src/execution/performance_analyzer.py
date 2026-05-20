import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PerformanceAnalyzer:
    """
    Institutional Performance Analysis Engine.
    Computes Sharpe, Sortino, Calmar, Drawdowns, and Sector/Regime attribution.
    """
    
    def __init__(self, risk_free_rate=0.04):
        self.risk_free_rate = risk_free_rate

    def analyze(self, snapshots, trade_history, initial_capital):
        if not snapshots:
            return {}

        df_snapshots = pd.DataFrame(snapshots)
        df_snapshots['time'] = pd.to_datetime(df_snapshots['time'])
        df_snapshots = df_snapshots.set_index('time')
        
        # Daily Returns
        daily_equity = df_snapshots['equity'].resample('D').last().ffill()
        daily_returns = daily_equity.pct_change().dropna()
        
        # Monthly Returns
        monthly_equity = df_snapshots['equity'].resample('M').last().ffill()
        monthly_returns = monthly_equity.pct_change().dropna()

        # Risk Metrics
        sharpe = self.calculate_sharpe(daily_returns)
        sortino = self.calculate_sortino(daily_returns)
        max_dd, calmar = self.calculate_drawdown_metrics(daily_equity, daily_returns)
        
        # Trade Metrics
        trades_df = pd.DataFrame(trade_history) if trade_history else pd.DataFrame()
        win_rate = 0.0
        profit_factor = 0.0
        if not trades_df.empty and 'pnl' in trades_df.columns:
            sells = trades_df[trades_df['action'] == 'SELL']
            if not sells.empty:
                wins = sells[sells['pnl'] > 0]
                losses = sells[sells['pnl'] <= 0]
                win_rate = len(wins) / len(sells)
                
                gross_profit = wins['pnl'].sum()
                gross_loss = abs(losses['pnl'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Attribution
        regime_perf = {}
        sector_perf = {}
        if not trades_df.empty and 'pnl' in trades_df.columns:
            sells = trades_df[trades_df['action'] == 'SELL']
            if not sells.empty:
                regime_perf = sells.groupby('regime')['pnl'].sum().to_dict()
                sector_perf = sells.groupby('sector')['pnl'].sum().to_dict()

        return {
            "summary": {
                "total_return": ((daily_equity[-1] / initial_capital) - 1) * 100,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "max_drawdown": max_dd * 100,
                "win_rate": win_rate * 100,
                "profit_factor": profit_factor
            },
            "returns": {
                "daily": daily_returns.tail(30).to_dict(),
                "monthly": monthly_returns.to_dict()
            },
            "attribution": {
                "by_regime": regime_perf,
                "by_sector": sector_perf
            }
        }

    def calculate_sharpe(self, returns):
        if len(returns) < 2: return 0.0
        adj_rf = (1 + self.risk_free_rate)**(1/252) - 1
        excess_returns = returns - adj_rf
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0.0

    def calculate_sortino(self, returns):
        if len(returns) < 2: return 0.0
        adj_rf = (1 + self.risk_free_rate)**(1/252) - 1
        excess_returns = returns - adj_rf
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std != 0 else 0.0

    def calculate_drawdown_metrics(self, equity, returns):
        if len(equity) < 2: return 0.0, 0.0
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        annual_return = returns.mean() * 252
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0
        return max_dd, calmar
