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
                raise HTTPException(
                    status_code=404, detail="Backtest results not found."
                )

            df = pd.read_csv(trades_path)
            # Ensure ticker matches (it might be AAPL or AAPL.NS depending on how it was saved)
            ticker_upper = ticker.upper()
            df_ticker = df[df["ticker"].str.upper() == ticker_upper].copy()

            if df_ticker.empty:
                return BacktestSummary(
                    ticker=ticker,
                    period=period,
                    win_rate=0.0,
                    profit_factor=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    vetoed_rate=0.0,
                    coverage=0.0,
                )

            # Defensive Column Check
            if "was_correct" not in df_ticker.columns:
                df_ticker["was_correct"] = False
            
            # Ensure was_correct is boolean
            df_ticker["was_correct"] = df_ticker["was_correct"].astype(bool)

            df_ticker["date_parsed"] = pd.to_datetime(df_ticker["date"])
            end_date = df_ticker["date_parsed"].max()

            if period == "3m":
                start_date = end_date - pd.DateOffset(months=3)
            elif period == "6m":
                start_date = end_date - pd.DateOffset(months=6)
            elif period == "1y":
                start_date = end_date - pd.DateOffset(years=1)
            elif period == "2y":
                start_date = end_date - pd.DateOffset(years=2)
            else:
                start_date = end_date - pd.DateOffset(years=1)

            df_ticker = df_ticker[df_ticker["date_parsed"] >= start_date]

            if df_ticker.empty:
                return BacktestSummary(
                    ticker=ticker,
                    period=period,
                    win_rate=0.0,
                    profit_factor=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    vetoed_rate=0.0,
                    coverage=0.0,
                )

            all_active = df_ticker[df_ticker["signal"].isin(["BUY", "SELL", "VETOED"])]
            vetoed = df_ticker[df_ticker["signal"] == "VETOED"]
            
            vetoed_rate = (len(vetoed) / len(all_active) * 100) if not all_active.empty else 0.0
            coverage = (len(all_active) / len(df_ticker) * 100) if not df_ticker.empty else 0.0

            df_trades = df_ticker[df_ticker["signal"].isin(["BUY", "SELL"])]
            if df_trades.empty:
                 return BacktestSummary(
                    ticker=ticker,
                    period=period,
                    win_rate=0.0,
                    profit_factor=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    vetoed_rate=round(vetoed_rate, 1),
                    coverage=round(coverage, 1),
                )

            correct = df_trades[df_trades["was_correct"]]
            win_rate = len(correct) / len(df_trades) * 100

            profits = df_trades[df_trades["actual_5day_return"] > 0][
                "actual_5day_return"
            ].sum()
            losses = abs(
                df_trades[df_trades["actual_5day_return"] < 0][
                    "actual_5day_return"
                ].sum()
            )
            pf = profits / losses if losses > 0 else float("inf")

            returns = df_trades["actual_5day_return"] / 100
            sharpe = (
                (returns.mean() / returns.std()) * np.sqrt(252 / 5)
                if returns.std() > 0
                else 0.0
            )

            cum_ret = (1 + returns).cumprod()
            rolling_max = cum_ret.cummax()
            drawdowns = (cum_ret - rolling_max) / rolling_max
            max_dd = drawdowns.min() * 100

            best_row = df_trades.loc[df_trades["actual_5day_return"].idxmax()]
            worst_row = df_trades.loc[df_trades["actual_5day_return"].idxmin()]

            return BacktestSummary(
                ticker=ticker,
                period=period,
                win_rate=round(win_rate, 1),
                profit_factor=round(pf, 2),
                sharpe_ratio=round(sharpe, 2),
                max_drawdown=round(max_dd, 1),
                vetoed_rate=round(vetoed_rate, 1),
                coverage=round(coverage, 1),
                best_signal=BacktestSignal(
                    date=str(best_row["date"]),
                    ticker=ticker,
                    signal=best_row["signal"],
                    confidence=best_row["confidence"],
                    actual_return=best_row["actual_5day_return"],
                ),
                worst_signal=BacktestSignal(
                    date=str(worst_row["date"]),
                    ticker=ticker,
                    signal=worst_row["signal"],
                    confidence=worst_row["confidence"],
                    actual_return=worst_row["actual_5day_return"],
                ),
            )
        except HTTPException:
            raise
        except Exception as e:
            raise Exception(f"Backtest retrieval error: {str(e)}")
