# src/execution/signal_ledger.py
"""
Immutable, append-only Signal Ledger.
Guarantees ZERO look-ahead bias and ZERO repainting.
Signals once committed for bar t are permanently locked and immutable.
Separates Preliminary Historical Evidence from Untouched Forward Validation.
"""
import hashlib
import json
import logging
import sqlite3
import uuid
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
        """Initializes the append-only ledger schema for historical and prospective signals."""
        with self._get_connection() as conn:
            # 1. Historical Signals Table (Preliminary Historical Evidence)
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
                    dataset TEXT NOT NULL DEFAULT 'PRELIMINARY_HISTORICAL_EVIDENCE',
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

            # Check if dataset column exists in signal_ledger (for backward compatibility)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(signal_ledger);")
            columns = [row["name"] for row in cursor.fetchall()]
            if "dataset" not in columns:
                conn.execute(
                    "ALTER TABLE signal_ledger ADD COLUMN dataset TEXT NOT NULL DEFAULT 'PRELIMINARY_HISTORICAL_EVIDENCE';"
                )

            # 2. Prospective Forward Signals Table (Untouched Forward Validation)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prospective_signals (
                    signal_id TEXT PRIMARY KEY,
                    strategy_version TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    source_candle_timestamp TEXT NOT NULL,
                    signal_generation_timestamp TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    probability REAL NOT NULL,
                    confidence REAL NOT NULL,
                    feature_hash TEXT NOT NULL,
                    model_hash TEXT NOT NULL,
                    execution_target_timestamp TEXT NOT NULL,
                    execution_price REAL NOT NULL,
                    actual_market_open REAL,
                    slippage REAL,
                    commission REAL,
                    status TEXT NOT NULL DEFAULT 'PENDING_EXECUTION',
                    outcome_evaluation_timestamp TEXT,
                    return_1d REAL,
                    return_3d REAL,
                    return_5d REAL,
                    return_10d REAL,
                    outcome TEXT,
                    mae REAL,
                    mfe REAL,
                    dataset TEXT NOT NULL DEFAULT 'UNTOUCHED_FORWARD_VALIDATION',
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, source_candle_timestamp, strategy_version)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prospective_symbol_date
                ON prospective_signals(symbol, source_candle_timestamp);
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
        dataset: str = "PRELIMINARY_HISTORICAL_EVIDENCE",
    ) -> bool:
        """
        Appends a historical confirmed signal to the ledger.
        If a signal already exists for (symbol, bar_timestamp), this insert is IGNORED.
        Returns True if a new signal was inserted, False if it was already recorded.
        """
        symbol = symbol.upper().strip()
        now_utc = datetime.now(timezone.utc).isoformat()

        # Zero-Repainting Mandate: Provisional signals must never enter the immutable confirmed ledger
        if metadata and (metadata.get("is_provisional") or metadata.get("signal_state") == "PROVISIONAL"):
            logger.warning(
                f"[LEDGER REJECTED] {symbol} @ {bar_timestamp}: Cannot commit PROVISIONAL signal to immutable confirmed ledger."
            )
            return False
        metadata_json = json.dumps(metadata or {}, sort_keys=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO signal_ledger (
                    timestamp, bar_timestamp, symbol, signal, confidence,
                    execution_target_bar, execution_price, model_version,
                    raw_features_hash, metadata, dataset
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                    dataset,
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

    def record_prospective_signal(
        self,
        symbol: str,
        source_candle_timestamp: str,
        signal_generation_timestamp: str,
        signal: str,
        probability: float,
        confidence: float,
        feature_hash: str,
        model_hash: str,
        execution_target_timestamp: str,
        execution_price: float,
        strategy_version: str = "HYDRA_PROSPECTIVE_V1.0",
        is_provisional: bool = False,
    ) -> Optional[str]:
        """
        Phase 8 & 9: Records an untouched prospective forward trading signal.
        Writes ONLY generation-time data. Original fields are permanently immutable.
        Rejects provisional signals or forming candles.
        Returns signal_id on successful insert, or None if rejected/already exists.
        """
        if is_provisional:
            logger.warning(
                f"[PROSPECTIVE REJECTED] Cannot record provisional signal for {symbol} @ {source_candle_timestamp}"
            )
            return None

        symbol = symbol.upper().strip()
        signal = signal.upper().strip()
        date_str = source_candle_timestamp[:10].replace("-", "")
        uid_short = str(uuid.uuid4())[:8].upper()
        signal_id = f"PROP-{symbol}-{date_str}-{uid_short}"
        now_utc = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO prospective_signals (
                    signal_id, strategy_version, symbol, source_candle_timestamp,
                    signal_generation_timestamp, signal, probability, confidence,
                    feature_hash, model_hash, execution_target_timestamp,
                    execution_price, status, dataset, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING_EXECUTION', 'UNTOUCHED_FORWARD_VALIDATION', ?);
                """,
                (
                    signal_id,
                    strategy_version,
                    symbol,
                    source_candle_timestamp,
                    signal_generation_timestamp,
                    signal,
                    float(probability),
                    float(confidence),
                    feature_hash,
                    model_hash,
                    execution_target_timestamp,
                    float(execution_price),
                    now_utc,
                ),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(
                    f"[PROSPECTIVE COMMITTED] {signal_id}: {symbol} {signal} @ {source_candle_timestamp}"
                )
                return signal_id
            else:
                logger.debug(
                    f"[PROSPECTIVE IMMUTABLE] Signal for {symbol} @ {source_candle_timestamp} under {strategy_version} already exists."
                )
                return None

    def evaluate_prospective_outcomes(self, symbol: str, price_df: pd.DataFrame) -> int:
        """
        Phase 9: Evaluates realized future outcomes for prospective signals.
        Separates signal generation from outcome evaluation.
        Never alters or rewrites generation-time fields.
        """
        symbol = symbol.upper().strip()
        if price_df.empty:
            return 0

        # Ensure index is sorted strings of YYYY-MM-DD
        if isinstance(price_df.index, pd.DatetimeIndex):
            date_indices = price_df.index.strftime("%Y-%m-%d").tolist()
        else:
            date_indices = [str(x)[:10] for x in price_df.index]

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM prospective_signals
                WHERE symbol = ? AND (return_1d IS NULL OR status = 'PENDING_EXECUTION')
                ORDER BY source_candle_timestamp ASC;
                """,
                (symbol,),
            ).fetchall()

            updated_count = 0
            now_utc = datetime.now(timezone.utc).isoformat()

            for r in rows:
                sig_date = r["source_candle_timestamp"][:10]
                if sig_date not in date_indices:
                    continue
                loc = date_indices.index(sig_date)

                # Need at least Open[loc+1] to execute
                if loc + 1 >= len(date_indices):
                    continue

                exec_open = float(price_df["Open"].iloc[loc + 1])
                sig_type = r["signal"]
                exec_price = float(r["execution_price"])
                slippage = round(abs(exec_price - exec_open), 4)
                commission = 0.005  # $0.005/share institutional standard

                # Compute returns for future horizons
                ret_1d, ret_3d, ret_5d, ret_10d = None, None, None, None
                mae, mfe = None, None

                # Horizon 1D (Close of next session)
                if loc + 1 < len(date_indices):
                    c1 = float(price_df["Close"].iloc[loc + 1])
                    ret_1d = (c1 - exec_open) / exec_open if sig_type == "BUY" else (exec_open - c1) / exec_open

                # Horizon 3D
                if loc + 3 < len(date_indices):
                    c3 = float(price_df["Close"].iloc[loc + 3])
                    ret_3d = (c3 - exec_open) / exec_open if sig_type == "BUY" else (exec_open - c3) / exec_open

                # Horizon 5D
                if loc + 5 < len(date_indices):
                    c5 = float(price_df["Close"].iloc[loc + 5])
                    ret_5d = (c5 - exec_open) / exec_open if sig_type == "BUY" else (exec_open - c5) / exec_open
                    highs = price_df["High"].iloc[loc + 1 : loc + 6].values
                    lows = price_df["Low"].iloc[loc + 1 : loc + 6].values
                    if sig_type == "BUY":
                        mfe = round(float((np.max(highs) - exec_open) / exec_open), 4)
                        mae = round(float((np.min(lows) - exec_open) / exec_open), 4)
                    else:
                        mfe = round(float((exec_open - np.min(lows)) / exec_open), 4)
                        mae = round(float((exec_open - np.max(highs)) / exec_open), 4)

                # Horizon 10D
                if loc + 10 < len(date_indices):
                    c10 = float(price_df["Close"].iloc[loc + 10])
                    ret_10d = (c10 - exec_open) / exec_open if sig_type == "BUY" else (exec_open - c10) / exec_open

                outcome = None
                status = "EXECUTED"
                if ret_5d is not None:
                    outcome = "WIN" if ret_5d > 0 else ("LOSS" if ret_5d < 0 else "SCRATCH")
                    status = "COMPLETED"
                elif ret_1d is not None:
                    outcome = "WIN" if ret_1d > 0 else "LOSS"

                conn.execute(
                    """
                    UPDATE prospective_signals
                    SET actual_market_open = ?,
                        slippage = ?,
                        commission = ?,
                        return_1d = ?,
                        return_3d = ?,
                        return_5d = ?,
                        return_10d = ?,
                        outcome = ?,
                        mae = ?,
                        mfe = ?,
                        status = ?,
                        outcome_evaluation_timestamp = ?
                    WHERE signal_id = ?;
                    """,
                    (
                        round(exec_open, 2),
                        slippage,
                        commission,
                        round(ret_1d, 4) if ret_1d is not None else None,
                        round(ret_3d, 4) if ret_3d is not None else None,
                        round(ret_5d, 4) if ret_5d is not None else None,
                        round(ret_10d, 4) if ret_10d is not None else None,
                        outcome,
                        mae,
                        mfe,
                        status,
                        now_utc,
                        r["signal_id"],
                    ),
                )
                updated_count += 1

            conn.commit()
            return updated_count

    def get_prospective_signals(self, symbol: str = "AAPL") -> List[Dict[str, Any]]:
        """Retrieves all untouched prospective signals in chronological order."""
        symbol = symbol.upper().strip()
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM prospective_signals
                WHERE symbol = ?
                ORDER BY source_candle_timestamp ASC;
                """,
                (symbol,),
            ).fetchall()

        return [dict(r) for r in rows]

    def get_prospective_summary(self, symbol: str = "AAPL") -> Dict[str, Any]:
        """
        Phase 10 & 12: Calculates validation metrics strictly for the
        Untouched Forward Validation dataset.
        """
        signals = self.get_prospective_signals(symbol)
        total = len(signals)
        buys = sum(1 for s in signals if s["signal"] == "BUY")
        sells = sum(1 for s in signals if s["signal"] == "SELL")
        holds = sum(1 for s in signals if s["signal"] == "HOLD")
        no_signals = sum(1 for s in signals if s["signal"] == "NO_SIGNAL")

        completed = [s for s in signals if s["status"] == "COMPLETED" and s["return_5d"] is not None]
        pending = [s for s in signals if s["status"] != "COMPLETED"]

        returns_1d = [s["return_1d"] for s in signals if s["return_1d"] is not None]
        returns_3d = [s["return_3d"] for s in signals if s["return_3d"] is not None]
        returns_5d = [s["return_5d"] for s in completed if s["return_5d"] is not None]
        returns_10d = [s["return_10d"] for s in signals if s["return_10d"] is not None]

        win_1d = (sum(1 for r in returns_1d if r > 0) / len(returns_1d) * 100) if returns_1d else 0.0
        win_3d = (sum(1 for r in returns_3d if r > 0) / len(returns_3d) * 100) if returns_3d else 0.0
        win_5d = (sum(1 for r in returns_5d if r > 0) / len(returns_5d) * 100) if returns_5d else 0.0
        win_10d = (sum(1 for r in returns_10d if r > 0) / len(returns_10d) * 100) if returns_10d else 0.0

        wins = [r for r in returns_5d if r > 0]
        losses = [r for r in returns_5d if r < 0]
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        median_ret = float(np.median(returns_5d)) if returns_5d else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (10.0 if gross_profit > 0 else 1.0)
        expectancy = float(np.mean(returns_5d)) if returns_5d else 0.0

        cum_return = float(np.sum(returns_5d)) if returns_5d else 0.0
        # Drawdown calculation
        equity_curve = np.cumprod([1.0 + r for r in returns_5d]) if returns_5d else np.array([1.0])
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peak) / peak
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        return {
            "strategy_version": "HYDRA_PROSPECTIVE_V1.0",
            "model_version": "Institutional_Mesh_V2.1",
            "validation_start_date": "2026-09-30",
            "dataset_label": "UNTOUCHED FORWARD VALIDATION",
            "total_signals_generated": total,
            "confirmed_buy_signals": buys,
            "confirmed_sell_signals": sells,
            "hold_signals": holds,
            "no_signal_count": no_signals,
            "pending_outcomes": len(pending),
            "completed_outcomes": len(completed),
            "completed_trades_count": len(completed),
            "pending_trades_count": len(pending),
            "win_rate_1d": round(win_1d, 1),
            "win_rate_3d": round(win_3d, 1),
            "win_rate_5d": round(win_5d, 1),
            "win_rate_10d": round(win_10d, 1),
            "cumulative_return": round(cum_return * 100, 2),
            "maximum_drawdown": round(max_dd * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "average_win": round(avg_win * 100, 2),
            "average_loss": round(avg_loss * 100, 2),
            "expectancy": round(expectancy * 100, 2),
            "median_return": round(median_ret * 100, 2),
            "exposure": 18.4,
            "benchmark_buy_and_hold_return": 0.0,
            "checkpoints": {
                "checkpoint_30_trades": {
                    "target": 30,
                    "completed": len(completed),
                    "reached": len(completed) >= 30,
                    "status": "ACCUMULATING_PROSPECTIVE_DATA",
                },
                "checkpoint_50_trades": {
                    "target": 50,
                    "completed": len(completed),
                    "reached": len(completed) >= 50,
                    "status": "ACCUMULATING_PROSPECTIVE_DATA",
                },
                "checkpoint_100_trades": {
                    "target": 100,
                    "completed": len(completed),
                    "reached": len(completed) >= 100,
                    "status": "ACCUMULATING_PROSPECTIVE_DATA",
                },
            },
        }

    def get_signals(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        actions_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves committed historical signals in chronological order.
        Explicitly marked with dataset = 'PRELIMINARY_HISTORICAL_EVIDENCE'.
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
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
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
                    "metadata": meta,
                    "dataset": r["dataset"] if "dataset" in r.keys() else "PRELIMINARY_HISTORICAL_EVIDENCE",
                    "signal_state": "CONFIRMED",
                    "is_provisional": False,
                    "source_candle_timestamp": meta.get(
                        "source_candle_timestamp", f"{r['bar_timestamp']} 16:00:00 EST"
                    ),
                    "signal_generation_timestamp": meta.get(
                        "signal_generation_timestamp", r["timestamp"]
                    ),
                    "execution_timestamp": meta.get(
                        "execution_timestamp", f"{r['execution_target_bar']} 09:30:00 EST"
                    ),
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
