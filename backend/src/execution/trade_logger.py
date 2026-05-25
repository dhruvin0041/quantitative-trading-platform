import sqlite3
import pandas as pd
from datetime import datetime
import os

class TradeLogger:
    """
    Institutional-grade trade and signal logger.
    Persists all model outputs, consensus metrics, and execution results to SQLite.
    """
    def __init__(self, db_path="data/trading_log.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    ticker TEXT,
                    signal TEXT,
                    confidence_score REAL,
                    market_regime TEXT,
                    predicted_direction TEXT,
                    actual_direction TEXT,
                    holding_period INTEGER,
                    trade_return REAL,
                    agreement_score REAL
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_metrics (
                    timestamp TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    win_rate REAL,
                    average_return REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    max_drawdown REAL,
                    portfolio_beta REAL,
                    jensens_alpha REAL,
                    kelly_exposure REAL
                )
            ''')

    def log_signal(self, data: dict):
        with self.conn:
            self.conn.execute('''
                INSERT INTO signals (
                    timestamp, ticker, signal, confidence_score, market_regime,
                    predicted_direction, actual_direction, holding_period, trade_return, agreement_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get("timestamp", datetime.now().isoformat()),
                data.get("ticker", "UNKNOWN"),
                data.get("signal", "HOLD"),
                data.get("confidence_score", 0.0),
                data.get("market_regime", "NEUTRAL"),
                data.get("predicted_direction", "NONE"),
                data.get("actual_direction", "PENDING"),
                data.get("holding_period", 0),
                data.get("return", 0.0),
                data.get("agreement_score", 0.0)
            ))

    def log_portfolio_metrics(self, metrics: dict):
        with self.conn:
            self.conn.execute('''
                INSERT OR REPLACE INTO portfolio_metrics (
                    timestamp, total_trades, win_rate, average_return, profit_factor,
                    sharpe_ratio, sortino_ratio, max_drawdown, portfolio_beta,
                    jensens_alpha, kelly_exposure
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                metrics.get("Total Trades", 0),
                metrics.get("Win Rate", 0.0),
                metrics.get("Average Return", 0.0),
                metrics.get("Profit Factor", 0.0),
                metrics.get("Sharpe Ratio", 0.0),
                metrics.get("Sortino Ratio", 0.0),
                metrics.get("Maximum Drawdown", 0.0),
                metrics.get("Portfolio Beta", 1.0),
                metrics.get("Jensen's Alpha", 0.0),
                metrics.get("Kelly Exposure", 0.0)
            ))

    def get_signals_df(self):
        return pd.read_sql_query("SELECT * FROM signals", self.conn)

    def get_metrics_df(self):
        return pd.read_sql_query("SELECT * FROM portfolio_metrics ORDER BY timestamp DESC LIMIT 100", self.conn)
