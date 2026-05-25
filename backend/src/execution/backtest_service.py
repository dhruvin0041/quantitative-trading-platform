import os
import pandas as pd
import numpy as np
from fastapi import HTTPException
from src.schemas import BacktestSummary, BacktestSignal

class BacktestService:
    @staticmethod
    def get_summary(ticker: str, period: str) -> BacktestSummary:
        try:
            trades_path = "backtest_results/backtest_trades.csv"
            if not os.path.exists(trades_path):
                raise HTTPException(status_code=404, detail="Backtest results not found.")
            
            df = pd.read_csv(trades_path)
            # Ensure ticker matches (it might be AAPL or AAPL.NS depending on how it was saved)
            ticker_upper = ticker.upper()
            df_ticker = df[df['ticker'].str.upper() == ticker_upper]
            
            if df_ticker.empty:
                return BacktestSummary(
                    ticker=ticker, period=period, win_rate=0.0, 
                    profit_factor=0.0, sharpe_ratio=0.0, max_drawdown=0.0, 
                    vetoed_rate=0.0, coverage=0.0
                )

            df_ticker['date_parsed'] = pd.to_datetime(df_ticker['date'])
            end_date = df_ticker['date_parsed'].max()
            
            if period == '3m': start_date = end_date - pd.DateOffset(months=3)
            elif period == '6m': start_date = end_date - pd.DateOffset(months=6)
            elif period == '1y': start_date = end_date - pd.DateOffset(years=1)
            elif period == '2y': start_date = end_date - pd.DateOffset(years=2)
            else: start_date = end_date - pd.DateOffset(years=1)
            
            df_ticker = df_ticker[df_ticker['date_parsed'] >= start_date]
            
            if df_ticker.empty:
                return BacktestSummary(
                    ticker=ticker, period=period, win_rate=0.0, 
                    profit_factor=0.0, sharpe_ratio=0.0, max_drawdown=0.0, 
                    vetoed_rate=0.0, coverage=0.0
                )

            correct = df_ticker[df_ticker['was_correct'] == True]
            win_rate = len(correct) / len(df_ticker) * 100
            
            profits = df_ticker[df_ticker['actual_5day_return'] > 0]['actual_5day_return'].sum()
            losses = abs(df_ticker[df_ticker['actual_5day_return'] < 0]['actual_5day_return'].sum())
            pf = profits / losses if losses > 0 else float('inf')
            
            returns = df_ticker['actual_5day_return'] / 100
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252/5) if returns.std() > 0 else 0.0
            
            cum_ret = (1 + returns).cumprod()
            rolling_max = cum_ret.cummax()
            drawdowns = (cum_ret - rolling_max) / rolling_max
            max_dd = drawdowns.min() * 100
            
            best_row = df_ticker.loc[df_ticker['actual_5day_return'].idxmax()]
            worst_row = df_ticker.loc[df_ticker['actual_5day_return'].idxmin()]
            
            return BacktestSummary(
                ticker=ticker,
                period=period,
                win_rate=round(win_rate, 1),
                profit_factor=round(pf, 2),
                sharpe_ratio=round(sharpe, 2),
                max_drawdown=round(max_dd, 1),
                vetoed_rate=0.0,
                coverage=100.0,
                best_signal=BacktestSignal(
                    date=str(best_row['date']), ticker=ticker, 
                    signal=best_row['signal'], confidence=best_row['confidence'], 
                    actual_return=best_row['actual_5day_return']
                ),
                worst_signal=BacktestSignal(
                    date=str(worst_row['date']), ticker=ticker, 
                    signal=worst_row['signal'], confidence=worst_row['confidence'], 
                    actual_return=worst_row['actual_5day_return']
                )
            )
        except HTTPException:
            raise
        except Exception as e:
            raise Exception(f"Backtest retrieval error: {str(e)}")
