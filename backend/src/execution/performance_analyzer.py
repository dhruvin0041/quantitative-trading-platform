import pandas as pd
import numpy as np
from datetime import datetime

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
        
        # Institutional PnL Breakdown
        today = datetime.now().date()
        mtd_start = today.replace(day=1)
        ytd_start = today.replace(month=1, day=1)
        
        today_equity = daily_equity.iloc[-1] if not daily_equity.empty else initial_capital
        yesterday_equity = daily_equity.iloc[-2] if len(daily_equity) > 1 else initial_capital
        
        mtd_equity_start = daily_equity.loc[daily_equity.index >= pd.Timestamp(mtd_start)].iloc[0] if not daily_equity.loc[daily_equity.index >= pd.Timestamp(mtd_start)].empty else initial_capital
        ytd_equity_start = daily_equity.loc[daily_equity.index >= pd.Timestamp(ytd_start)].iloc[0] if not daily_equity.loc[daily_equity.index >= pd.Timestamp(ytd_start)].empty else initial_capital

        today_pnl = today_equity - yesterday_equity
        mtd_pnl = today_equity - mtd_equity_start
        ytd_pnl = today_equity - ytd_equity_start
        inception_pnl = today_equity - initial_capital

        # Monthly Returns
        monthly_equity = df_snapshots['equity'].resample('M').last().ffill()
        monthly_returns = monthly_equity.pct_change().dropna()

        # Risk Metrics
        sharpe = self.calculate_sharpe(daily_returns)
        sortino = self.calculate_sortino(daily_returns)
        max_dd, calmar = self.calculate_drawdown_metrics(daily_equity, daily_returns)
        
        # Trade Counts
        total_trades = len(trade_history) if trade_history else 0
        
        # In this system, paper_trading engine positions are separate from history.
        # We'll use a better heuristic for open trades: history items with action=BUY that don't have a matching SELL yet.
        buy_counts = {}
        if trade_history:
            for t in trade_history:
                ticker = t['ticker']
                if t['action'] == 'BUY':
                    buy_counts[ticker] = buy_counts.get(ticker, 0) + t['shares']
                elif t['action'] == 'SELL':
                    buy_counts[ticker] = buy_counts.get(ticker, 0) - t['shares']
            
            open_trades_count = len([ticker for ticker, shares in buy_counts.items() if shares > 0])
        else:
            open_trades_count = 0

        # Trade Metrics
        trades_df = pd.DataFrame(trade_history) if trade_history else pd.DataFrame()

        win_rate = 0.0
        profit_factor = 0.0
        wins_count = 0
        losses_count = 0
        closed_trades_count = 0
        
        if not trades_df.empty:
            if 'pnl' in trades_df.columns:
                realized_pnl = trades_df[trades_df['action'] == 'SELL']['pnl'].sum()
            
            # Simple unrealized PnL proxy: Current Equity - (Cash + Realized PnL from initial)
            unrealized_pnl = today_equity - (initial_capital + realized_pnl)

            sells = trades_df[trades_df['action'] == 'SELL']
            if not sells.empty:
                closed_trades_count = len(sells)
                wins = sells[sells['pnl'] > 0]
                losses = sells[sells['pnl'] <= 0]
                wins_count = len(wins)
                losses_count = len(losses)
                win_rate = wins_count / closed_trades_count
                
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
                "total_return": ((today_equity / initial_capital) - 1) * 100,
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
                "open_trades": open_trades_count,
                "closed_trades": len(sells) if not trades_df.empty and 'action' in trades_df.columns else 0,
                "winning_trades": wins_count,
                "losing_trades": losses_count
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
