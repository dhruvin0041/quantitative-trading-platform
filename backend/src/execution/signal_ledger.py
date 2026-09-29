# src/execution/signal_ledger.py
"""
Immutable, append-only Signal Ledger.
Guarantees ZERO look-ahead bias and ZERO repainting.
Signals once committed for bar t are permanently locked and immutable.
"""
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("SignalLedger")


class SignalLedger:
    """
    Append-only persistent SQLite ledger for live trading signals.
    Once a signal is committed for a specific symbol and bar_timestamp,
    it can NEVER be overwritten, modified, or deleted.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to artifacts directory
            base_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(base_dir / "signal_ledger.db")
        else:
            self.db_path = db_path
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency between web server and background tasks
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Initializes the append-only ledger schema."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    bar_timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    execution_target_bar TEXT NOT NULL,
                    execution_price REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    raw_features_hash TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    UNIQUE(symbol, bar_timestamp)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_symbol_bar
                ON signal_ledger(symbol, bar_timestamp);
                """
            )
            conn.commit()

    @staticmethod
    def compute_features_hash(features_dict_or_array: Any) -> str:
        """Computes a SHA-256 fingerprint for input features to ensure reproducibility."""
        if isinstance(features_dict_or_array, np.ndarray):
            raw_bytes = features_dict_or_array.tobytes()
        elif isinstance(features_dict_or_array, (dict, list)):
            raw_bytes = json.dumps(features_dict_or_array, sort_keys=True).encode("utf-8")
        elif isinstance(features_dict_or_array, pd.Series):
            raw_bytes = features_dict_or_array.to_json().encode("utf-8")
        else:
            raw_bytes = str(features_dict_or_array).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def record_signal(
        self,
        symbol: str,
        bar_timestamp: str,
        signal: str,
        confidence: float,
        execution_target_bar: str,
        execution_price: float,
        model_version: str = "v1.0.0",
        raw_features_hash: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Appends a newly confirmed signal to the ledger.
        If a signal already exists for (symbol, bar_timestamp), this insert is IGNORED.
        Returns True if a new signal was inserted, False if it was already recorded (immutability preserved).
        """
        symbol = symbol.upper().strip()
        now_utc = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata or {}, sort_keys=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO signal_ledger (
                    timestamp, bar_timestamp, symbol, signal, confidence,
                    execution_target_bar, execution_price, model_version,
                    raw_features_hash, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    now_utc,
                    bar_timestamp,
                    symbol,
                    signal,
                    float(confidence),
                    execution_target_bar,
                    float(execution_price),
                    model_version,
                    raw_features_hash,
                    metadata_json,
                ),
            )
            conn.commit()
            inserted = cursor.rowcount > 0

        if inserted:
            logger.info(
                f"[LEDGER COMMITTED] {symbol} @ {bar_timestamp}: {signal} (Conf: {confidence:.2f}, Exec Price: ${execution_price:.2f})"
            )
        else:
            logger.debug(
                f"[LEDGER IMMUTABLE] {symbol} @ {bar_timestamp} already exists. Record preserved without modification."
            )
        return inserted

    def get_signals(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        actions_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves committed signals in chronological order.
        """
        symbol = symbol.upper().strip()
        query = "SELECT * FROM signal_ledger WHERE symbol = ?"
        params: List[Any] = [symbol]

        if start_date:
            query += " AND bar_timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND bar_timestamp <= ?"
            params.append(end_date)
        if actions_filter:
            placeholders = ",".join("?" for _ in actions_filter)
            query += f" AND signal IN ({placeholders})"
            params.extend(actions_filter)

        query += " ORDER BY bar_timestamp ASC;"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        signals = []
        for r in rows:
            signals.append(
                {
                    "timestamp": r["timestamp"],
                    "bar_timestamp": r["bar_timestamp"],
                    "time": r["bar_timestamp"],  # For lightweight-charts compatibility
                    "symbol": r["symbol"],
                    "signal": r["signal"],
                    "action": r["signal"],  # For chart compatibility
                    "confidence": r["confidence"],
                    "probability": round(float(r["confidence"]) * 100, 1),
                    "execution_target_bar": r["execution_target_bar"],
                    "execution_price": r["execution_price"],
                    "price": r["execution_price"],  # For chart compatibility
                    "model_version": r["model_version"],
                    "raw_features_hash": r["raw_features_hash"],
                    "metadata": json.loads(r["metadata"]),
                }
            )
        return signals

    def count_signals(self, symbol: str) -> int:
        symbol = symbol.upper().strip()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM signal_ledger WHERE symbol = ?;", (symbol,)
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def get_latest_bar_timestamp(self, symbol: str) -> Optional[str]:
        """Returns the most recent bar_timestamp recorded for the symbol, or None if none exist."""
        symbol = symbol.upper().strip()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT MAX(bar_timestamp) as latest FROM signal_ledger WHERE symbol = ?;",
                (symbol,),
            ).fetchone()
            if row and row["latest"]:
                return str(row["latest"])
            return None
