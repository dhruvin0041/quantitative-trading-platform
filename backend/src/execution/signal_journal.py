import sqlite3
import pandas as pd
from datetime import datetime
import os

class SignalJournal:
    """
    Phase 8.1 - Signal Journal System
    Permanently records every generated signal for empirical validation.
    """
    def __init__(self, db_path="data/empirical_validation.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    asset TEXT,
                    market TEXT,
                    exchange TEXT,
                    signal_type TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    position_size REAL,
                    confidence REAL,
                    uncertainty REAL,
                    agreement REAL,
                    market_regime TEXT,
                    volatility_regime TEXT,
                    model_consensus TEXT, -- JSON string of model predictions
                    holding_time INTEGER,
                    realized_pnl REAL,
                    unrealized_pnl REAL,
                    outcome TEXT -- WIN, LOSS, PENDING, VETOED, HOLD
                )
            ''')
            
    def log_signal(self, data: dict):
        with self.conn:
            self.conn.execute('''
                INSERT OR REPLACE INTO signals (
                    signal_id, timestamp, asset, market, exchange, signal_type,
                    entry_price, exit_price, position_size, confidence, uncertainty,
                    agreement, market_regime, volatility_regime, model_consensus,
                    holding_time, realized_pnl, unrealized_pnl, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get("signal_id"),
                data.get("timestamp", datetime.now().isoformat()),
                data.get("asset"),
                data.get("market", "UNKNOWN"),
                data.get("exchange", "UNKNOWN"),
                data.get("signal_type"),
                data.get("entry_price", 0.0),
                data.get("exit_price", 0.0),
                data.get("position_size", 0.0),
                data.get("confidence", 0.0),
                data.get("uncertainty", 0.0),
                data.get("agreement", 0.0),
                data.get("market_regime", "UNKNOWN"),
                data.get("volatility_regime", "UNKNOWN"),
                data.get("model_consensus", "{}"),
                data.get("holding_time", 0),
                data.get("realized_pnl", 0.0),
                data.get("unrealized_pnl", 0.0),
                data.get("outcome", "PENDING")
            ))

    def update_signal_exit(self, signal_id: str, exit_price: float, realized_pnl: float, holding_time: int):
        outcome = "WIN" if realized_pnl > 0 else ("LOSS" if realized_pnl < 0 else "FLAT")
        with self.conn:
            self.conn.execute('''
                UPDATE signals 
                SET exit_price = ?, realized_pnl = ?, outcome = ?, holding_time = ?
                WHERE signal_id = ?
            ''', (exit_price, realized_pnl, outcome, holding_time, signal_id))

    def get_all_signals(self):
        return pd.read_sql_query("SELECT * FROM signals ORDER BY timestamp DESC", self.conn)
        
    def get_closed_trades(self):
        return pd.read_sql_query("SELECT * FROM signals WHERE outcome IN ('WIN', 'LOSS', 'FLAT')", self.conn)

