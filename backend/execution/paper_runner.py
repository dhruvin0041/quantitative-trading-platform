import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from execution.broker_interface import MockPaperBroker
from src.execution.asset_intelligence import AssetExpectancyFilter
from src.execution.consensus_engine import WeightedConsensusEngine

logger = logging.getLogger("DailyPaperRunner")


class DailyPaperRunner:
    """
    Institutional Automated Daily Execution Runner.
    Orchestrates the complete quantitative systematic execution cycle:
    1. Market Data Fetch: Daily OHLCV for Universe (AAPL, MSFT, NVDA) and Macro (SPY, ^VIX, ^TNX).
    2. Macro Regime Filter: Gating long (Close >= SMA200 & SPY >= SMA50) and short (Close < SMA200 | SPY < SMA50).
    3. Model Consensus & Asymmetric Veto: Evaluates primary alpha and secondary risk vetoes.
    4. Asset Expectancy Filter: Enforces rolling 90-day PF >= 1.15 hysteresis hurdle.
    5. Volatility-Adaptive Trailing Stops: Adjusts stop distance based on sigma_asset / sigma_SPY.
    6. Position Sizing: ATR volatility targeting (1% risk target, clipped 0.1 to 1.0).
    7. Order Execution & State Management: Dispatches orders to MockPaperBroker and logs state to SQLite.
    """

    def __init__(
        self,
        broker: Optional[MockPaperBroker] = None,
        state_db_path: Optional[str] = None,
        universe: Optional[List[str]] = None,
        benchmark_tickers: Optional[List[str]] = None,
        target_risk_pct: float = 0.01,
        initial_capital: float = 100000.0,
    ):
        self.universe = universe or ["AAPL", "MSFT", "NVDA"]
        self.benchmark_tickers = benchmark_tickers or ["SPY", "^VIX", "^TNX"]
        self.target_risk_pct = target_risk_pct

        if state_db_path is None:
            artifacts_dir = BACKEND_DIR / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            self.state_db_path = str(artifacts_dir / "paper_execution_state.db")
        else:
            self.state_db_path = state_db_path

        self.broker = broker or MockPaperBroker(
            initial_capital=initial_capital,
            state_file=self.state_db_path,
        )

        self.consensus_engine = WeightedConsensusEngine()
        self.expectancy_filter = AssetExpectancyFilter(
            suspension_threshold=1.15,
            recovery_threshold=1.20,
            lookback_days=90,
            min_trades=3,
        )

        self._init_runner_db()

    def _init_runner_db(self) -> None:
        """Initializes SQLite schema for execution runs and audit logs."""
        conn = None
        try:
            conn = sqlite3.connect(self.state_db_path)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_execution_runs (
                    run_id TEXT PRIMARY KEY,
                    run_date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    positions_count INTEGER NOT NULL,
                    orders_count INTEGER NOT NULL,
                    summary_json TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trailing_stops (
                    symbol TEXT PRIMARY KEY,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    peak_trough_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    ts_mult REAL NOT NULL,
                    atr REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        except Exception as e:
            logger.error("Failed to initialize runner SQLite schema: %s", e)
        finally:
            if conn:
                conn.close()

    def fetch_market_data(self, lookback_days: int = 300) -> Dict[str, pd.DataFrame]:
        """
        Fetches daily OHLCV for universe and benchmarks from Yahoo Finance.
        Computes SMAs, ATR, and rolling volatility metrics.
        """
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=lookback_days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        data: Dict[str, pd.DataFrame] = {}
        all_symbols = list(set(self.universe + self.benchmark_tickers))

        for sym in all_symbols:
            try:
                df = yf.download(sym, start=start_str, end=end_str, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                if df.empty:
                    logger.warning("Empty data returned for symbol: %s", sym)
                    continue

                # Ensure numeric and drop NaNs
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna()

                # ATR 14 computation
                high_low = df["High"] - df["Low"]
                high_close = (df["High"] - df["Close"].shift(1)).abs()
                low_close = (df["Low"] - df["Close"].shift(1)).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df["ATR"] = tr.rolling(14, min_periods=5).mean()

                # SMA 200 on underlying
                df["SMA_200"] = df["Close"].rolling(200, min_periods=20).mean()

                # Rolling 20-day volatility
                df["vol_20d"] = df["Close"].pct_change().rolling(20, min_periods=5).std()

                data[sym] = df
            except Exception as e:
                logger.error("Error fetching market data for %s: %s", sym, e)

        # Re-compute relative volatility and adaptive trailing stop multiplier against SPY
        if "SPY" in data and not data["SPY"].empty:
            spy_df = data["SPY"]
            spy_df["SPY_SMA_50"] = spy_df["Close"].rolling(50, min_periods=10).mean()
            vol_spy = spy_df["vol_20d"]

            for sym in self.universe:
                if sym in data and not data[sym].empty:
                    sym_df = data[sym]
                    # Align SPY volatility
                    aligned_spy_vol = vol_spy.reindex(sym_df.index).ffill()
                    vol_ratio = sym_df["vol_20d"] / (aligned_spy_vol + 1e-9)
                    # Adaptive Multiplier: min(4.0, max(2.5, 2.5 * vol_ratio))
                    sym_df["ts_mult"] = np.clip(2.5 * vol_ratio.fillna(1.0), 2.5, 4.0)

        return data

    def evaluate_macro_filters(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates Symmetric Macro Regime Filters:
        long_allowed: (Close >= SMA_200) and (SPY_Close >= SPY_SMA_50)
        short_allowed: (Close < SMA_200) or (SPY_Close < SPY_SMA_50)
        """
        filters: Dict[str, Dict[str, Any]] = {}
        spy_df = data.get("SPY")

        if spy_df is None or spy_df.empty:
            logger.warning("SPY benchmark missing; defaulting macro filters to allowed.")
            for sym in self.universe:
                filters[sym] = {
                    "long_allowed": True,
                    "short_allowed": True,
                    "underlying_close": 0.0,
                    "underlying_sma_200": 0.0,
                    "spy_close": 0.0,
                    "spy_sma_50": 0.0,
                }
            return filters

        curr_spy_close = float(spy_df["Close"].iloc[-1])
        spy_sma_50 = float(spy_df["SPY_SMA_50"].iloc[-1]) if "SPY_SMA_50" in spy_df.columns else curr_spy_close

        for sym in self.universe:
            df = data.get(sym)
            if df is None or df.empty:
                continue

            curr_close = float(df["Close"].iloc[-1])
            sma_200 = float(df["SMA_200"].iloc[-1]) if "SMA_200" in df.columns else curr_close

            long_allowed = bool((curr_close >= sma_200) and (curr_spy_close >= spy_sma_50))
            short_allowed = bool((curr_close < sma_200) or (curr_spy_close < spy_sma_50))

            filters[sym] = {
                "underlying_close": round(curr_close, 2),
                "underlying_sma_200": round(sma_200, 2),
                "spy_close": round(curr_spy_close, 2),
                "spy_sma_50": round(spy_sma_50, 2),
                "long_allowed": long_allowed,
                "short_allowed": short_allowed,
            }

        return filters

    def update_existing_positions(
        self, data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        Inspects active positions, ratchets volatility-adaptive trailing stops,
        and triggers stop exits if the stop price is breached.
        """
        closed_orders: List[Dict[str, Any]] = []
        positions = self.broker.get_positions()
        if not positions:
            return closed_orders

        for pos in positions:
            sym = pos["symbol"]
            qty = pos["qty"]
            side = pos["side"]
            df = data.get(sym)
            if df is None or df.empty:
                continue

            curr_close = float(df["Close"].iloc[-1])
            curr_high = float(df["High"].iloc[-1])
            curr_low = float(df["Low"].iloc[-1])
            atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else curr_close * 0.02
            m = float(df["ts_mult"].iloc[-1]) if "ts_mult" in df.columns else 2.5

            stop_dist = m * atr
            stop_state = self._get_trailing_stop_state(sym)

            if stop_state is None:
                # Initialize trailing stop state
                peak_trough = pos["avg_entry_price"]
                stop_price = (
                    peak_trough - stop_dist if side == "LONG" else peak_trough + stop_dist
                )
                self._save_trailing_stop_state(
                    sym, side, pos["avg_entry_price"], peak_trough, stop_price, m, atr
                )
            else:
                peak_trough = stop_state["peak_trough_price"]
                stop_price = stop_state["stop_price"]

                # Ratchet trailing stops
                if side == "LONG":
                    if curr_high > peak_trough:
                        peak_trough = curr_high
                        stop_price = max(stop_price, peak_trough - stop_dist)
                    # Check stop breach
                    if curr_low <= stop_price:
                        fill_p = min(curr_close, stop_price)
                        logger.info(
                            "Trailing Stop Hit for %s LONG at %.2f (Stop: %.2f)",
                            sym,
                            fill_p,
                            stop_price,
                        )
                        order = self.broker.submit_order(
                            symbol=sym,
                            qty=qty,
                            side="SELL",
                            current_price=fill_p,
                        )
                        closed_orders.append(order)
                        self._delete_trailing_stop_state(sym)
                        # Record trade into expectancy filter
                        pnl_ret = (fill_p - pos["avg_entry_price"]) / pos["avg_entry_price"]
                        self.expectancy_filter.record_trade(sym, datetime.now(), pnl_ret)
                        continue
                    else:
                        self._save_trailing_stop_state(
                            sym, side, pos["avg_entry_price"], peak_trough, stop_price, m, atr
                        )

                elif side == "SHORT":
                    if curr_low < peak_trough:
                        peak_trough = curr_low
                        stop_price = min(stop_price, peak_trough + stop_dist)
                    # Check stop breach
                    if curr_high >= stop_price:
                        fill_p = max(curr_close, stop_price)
                        logger.info(
                            "Trailing Stop Hit for %s SHORT at %.2f (Stop: %.2f)",
                            sym,
                            fill_p,
                            stop_price,
                        )
                        order = self.broker.submit_order(
                            symbol=sym,
                            qty=qty,
                            side="BUY",
                            current_price=fill_p,
                        )
                        closed_orders.append(order)
                        self._delete_trailing_stop_state(sym)
                        # Record trade into expectancy filter
                        pnl_ret = (pos["avg_entry_price"] - fill_p) / pos["avg_entry_price"]
                        self.expectancy_filter.record_trade(sym, datetime.now(), pnl_ret)
                        continue
                    else:
                        self._save_trailing_stop_state(
                            sym, side, pos["avg_entry_price"], peak_trough, stop_price, m, atr
                        )

        return closed_orders

    def generate_and_filter_signals(
        self,
        data: Dict[str, pd.DataFrame],
        macro_filters: Dict[str, Dict[str, Any]],
        mock_predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executes Asymmetric Veto consensus and filters candidate signals through:
        1. Macro Regime Gate (SMA200 / SPY SMA50)
        2. Asset Expectancy Filter (Trailing 90-day PF >= 1.15 Hysteresis)
        """
        signals: Dict[str, Dict[str, Any]] = {}
        now_ts = datetime.now()

        for sym in self.universe:
            df = data.get(sym)
            if df is None or df.empty:
                continue

            curr_close = float(df["Close"].iloc[-1])
            m_filter = macro_filters.get(sym, {})
            long_allowed = m_filter.get("long_allowed", True)
            short_allowed = m_filter.get("short_allowed", True)

            # Model Probabilities: [P(SELL), P(HOLD), P(BUY)]
            if mock_predictions and sym in mock_predictions:
                base_probs = mock_predictions[sym]
            else:
                # Default baseline neutral if live model server is not connected
                base_probs = {
                    "XGB_AGENT": np.array([0.15, 0.70, 0.15]),
                    "LGBM_AGENT": np.array([0.20, 0.60, 0.20]),
                    "DQN_AGENT": np.array([0.25, 0.50, 0.25]),
                }

            cons = self.consensus_engine.compute_asymmetric_veto(
                base_probs,
                primary_key="XGB_AGENT",
                primary_threshold=0.60,
                veto_threshold=0.65,
                veto_short=True,
            )

            raw_direction = cons["dominant_direction"]
            agreement_score = cons["agreement_score"]
            is_vetoed = cons["is_vetoed"]
            veto_reason = cons.get("veto_reason")

            final_signal = "HOLD"
            signal_note = "Neutral"

            if is_vetoed:
                final_signal = "HOLD"
                signal_note = f"Vetoed by secondary risk gate: {veto_reason}"
            elif agreement_score >= 60.0 and raw_direction in ["BUY", "SELL"]:
                # Macro Regime Filter Gating
                if raw_direction == "BUY" and not long_allowed:
                    final_signal = "HOLD"
                    signal_note = (
                        f"Suppressed by Macro Filter: Close {curr_close:.2f} < SMA200 "
                        f"or SPY < SMA50"
                    )
                elif raw_direction == "SELL" and not short_allowed:
                    final_signal = "HOLD"
                    signal_note = (
                        "Suppressed by Macro Filter: Counter-trend SHORT forbidden in confirmed bull regime"
                    )
                else:
                    # Asset-Level Expectancy Gating
                    exp_allowed, exp_reason = self.expectancy_filter.is_entry_allowed(sym, now_ts)
                    if not exp_allowed:
                        final_signal = "HOLD"
                        signal_note = exp_reason
                    else:
                        final_signal = raw_direction
                        signal_note = f"Approved by Asymmetric Consensus ({agreement_score:.1f}%)"

            signals[sym] = {
                "signal": final_signal,
                "raw_direction": raw_direction,
                "agreement_score": agreement_score,
                "signal_note": signal_note,
                "current_price": curr_close,
                "atr": float(df["ATR"].iloc[-1]) if "ATR" in df.columns else curr_close * 0.02,
                "ts_mult": float(df["ts_mult"].iloc[-1]) if "ts_mult" in df.columns else 2.5,
            }

        return signals

    def size_and_execute_orders(
        self,
        signals: Dict[str, Dict[str, Any]],
        data: Dict[str, pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """
        Computes ATR volatility position sizing and submits new orders to MockPaperBroker.
        """
        executed_orders: List[Dict[str, Any]] = []
        open_positions = {p["symbol"]: p for p in self.broker.get_positions()}
        balance = self.broker.get_account_balance()
        equity = balance["equity"]

        for sym, sig_info in signals.items():
            sig = sig_info["signal"]
            curr_p = sig_info["current_price"]
            atr = sig_info["atr"]
            ts_mult = sig_info["ts_mult"]

            if sig not in ["BUY", "SELL"]:
                continue

            # Position Sizing: Target Risk 1% via normalized ATR
            norm_atr = atr / (curr_p + 1e-9)
            vol_size = float(np.clip(self.target_risk_pct / (norm_atr + 1e-9), 0.1, 1.0))
            allocated_capital = equity * vol_size
            qty = max(1, int(allocated_capital / (curr_p + 1e-9)))

            # If already holding this position, avoid pyramiding
            existing_pos = open_positions.get(sym)
            if existing_pos:
                if (sig == "BUY" and existing_pos["side"] == "LONG") or (
                    sig == "SELL" and existing_pos["side"] == "SHORT"
                ):
                    logger.info("Already holding %s position in %s; skipping new entry.", sig, sym)
                    continue
                else:
                    # Close conflicting position first
                    opp_side = "SELL" if existing_pos["side"] == "LONG" else "BUY"
                    close_order = self.broker.submit_order(
                        symbol=sym,
                        qty=existing_pos["qty"],
                        side=opp_side,
                        current_price=curr_p,
                    )
                    executed_orders.append(close_order)
                    self._delete_trailing_stop_state(sym)

            # Trailing stop parameters
            stop_dist = ts_mult * atr
            if sig == "BUY":
                stop_loss = curr_p - stop_dist
                take_profit = curr_p + 2.0 * stop_dist
                order = self.broker.submit_order(
                    symbol=sym,
                    qty=qty,
                    side="BUY",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    current_price=curr_p,
                )
                executed_orders.append(order)
                if order["status"] == "FILLED":
                    fill_p = order["fill_price"]
                    self._save_trailing_stop_state(
                        sym, "LONG", fill_p, fill_p, fill_p - stop_dist, ts_mult, atr
                    )
            elif sig == "SELL":
                stop_loss = curr_p + stop_dist
                take_profit = curr_p - 2.0 * stop_dist
                order = self.broker.submit_order(
                    symbol=sym,
                    qty=qty,
                    side="SELL",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    current_price=curr_p,
                )
                executed_orders.append(order)
                if order["status"] == "FILLED":
                    fill_p = order["fill_price"]
                    self._save_trailing_stop_state(
                        sym, "SHORT", fill_p, fill_p, fill_p + stop_dist, ts_mult, atr
                    )

        return executed_orders

    def run_daily_cycle(
        self,
        mock_data: Optional[Dict[str, pd.DataFrame]] = None,
        mock_predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes one complete institutional daily execution cycle.
        """
        run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info("Executing Daily Paper Cycle: %s", run_id)

        # 1. Fetch market data
        data = mock_data or self.fetch_market_data()

        # 2. Evaluate Macro Filters
        macro_filters = self.evaluate_macro_filters(data)

        # 3. Update Existing Positions & Trailing Stops
        closed_orders = self.update_existing_positions(data)

        # 4. Generate & Filter Signals
        signals = self.generate_and_filter_signals(data, macro_filters, mock_predictions)

        # 5. Position Sizing & Order Execution
        new_orders = self.size_and_execute_orders(signals, data)

        # 6. Build EOD Report & Persist
        balance = self.broker.get_account_balance()
        positions = self.broker.get_positions()

        summary = {
            "run_id": run_id,
            "run_date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "macro_filters": macro_filters,
            "signals": signals,
            "closed_orders": closed_orders,
            "new_orders": new_orders,
            "balance": balance,
            "positions": positions,
        }

        self._persist_run_to_sqlite(summary)
        return summary

    def _persist_run_to_sqlite(self, summary: Dict[str, Any]) -> None:
        """Persists the daily run audit log to SQLite."""
        conn = None
        try:
            conn = sqlite3.connect(self.state_db_path)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO daily_execution_runs (
                    run_id, run_date, timestamp, cash, equity, positions_count, orders_count, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary["run_id"],
                    summary["run_date"],
                    summary["timestamp"],
                    summary["balance"]["cash"],
                    summary["balance"]["equity"],
                    len(summary["positions"]),
                    len(summary["closed_orders"]) + len(summary["new_orders"]),
                    json.dumps(summary),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error("Error logging daily run to SQLite: %s", e)
        finally:
            if conn:
                conn.close()

    def _get_trailing_stop_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        conn = None
        try:
            conn = sqlite3.connect(self.state_db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT side, entry_price, peak_trough_price, stop_price, ts_mult, atr FROM trailing_stops WHERE symbol = ?",
                (symbol,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "side": row[0],
                    "entry_price": row[1],
                    "peak_trough_price": row[2],
                    "stop_price": row[3],
                    "ts_mult": row[4],
                    "atr": row[5],
                }
            return None
        except Exception as e:
            logger.error("Error loading trailing stop state for %s: %s", symbol, e)
            return None
        finally:
            if conn:
                conn.close()

    def _save_trailing_stop_state(
        self,
        symbol: str,
        side: str,
        entry_p: float,
        peak_trough_p: float,
        stop_p: float,
        ts_mult: float,
        atr: float,
    ) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self.state_db_path)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO trailing_stops (
                    symbol, side, entry_price, peak_trough_price, stop_price, ts_mult, atr, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    side,
                    entry_p,
                    peak_trough_p,
                    stop_p,
                    ts_mult,
                    atr,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error("Error saving trailing stop state for %s: %s", symbol, e)
        finally:
            if conn:
                conn.close()

    def _delete_trailing_stop_state(self, symbol: str) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self.state_db_path)
            cur = conn.cursor()
            cur.execute("DELETE FROM trailing_stops WHERE symbol = ?", (symbol,))
            conn.commit()
        except Exception as e:
            logger.error("Error deleting trailing stop state for %s: %s", symbol, e)
        finally:
            if conn:
                conn.close()

    def print_eod_summary(self, summary: Dict[str, Any]) -> None:
        """Prints formatted institutional execution summary to console."""
        print("\n" + "=" * 95)
        print(f"=== END-OF-DAY EXECUTION REPORT: {summary['run_id']} ({summary['run_date']}) ===")
        print("=" * 95)

        bal = summary["balance"]
        print(f"Cash: ${bal['cash']:,.2f} | Equity: ${bal['equity']:,.2f} | Buying Power: ${bal['buying_power']:,.2f}")
        print("-" * 95)

        print("MACRO REGIME GATING:")
        print(f"{'Ticker':<8} | {'Close':<10} | {'SMA200':<10} | {'Long Allowed':<14} | {'Short Allowed':<14}")
        print("-" * 65)
        for sym, mf in summary["macro_filters"].items():
            print(
                f"{sym:<8} | ${mf['underlying_close']:<9.2f} | ${mf['underlying_sma_200']:<9.2f} | "
                f"{str(mf['long_allowed']):<14} | {str(mf['short_allowed']):<14}"
            )

        print("\nSIGNAL GENERATION & EXPECTANCY GATING:")
        print(f"{'Ticker':<8} | {'Signal':<8} | {'Score':<8} | {'Stop Mult':<10} | {'Signal Note'}")
        print("-" * 95)
        for sym, sig in summary["signals"].items():
            print(
                f"{sym:<8} | {sig['signal']:<8} | {sig['agreement_score']:<7.1f}% | "
                f"{sig['ts_mult']:<9.2f}x | {sig['signal_note']}"
            )

        orders = summary["closed_orders"] + summary["new_orders"]
        print(f"\nEXECUTED ORDERS ({len(orders)}):")
        if orders:
            print(f"{'Order ID':<16} | {'Symbol':<8} | {'Side':<6} | {'Qty':<6} | {'Fill Price':<12} | {'Status'}")
            print("-" * 70)
            for o in orders:
                print(
                    f"{o['order_id']:<16} | {o['symbol']:<8} | {o['side']:<6} | {o['qty']:<6} | "
                    f"${o['fill_price']:<11.2f} | {o['status']}"
                )
        else:
            print("No orders executed in this cycle.")

        positions = summary["positions"]
        print(f"\nACTIVE POSITIONS ({len(positions)}):")
        if positions:
            print(f"{'Symbol':<8} | {'Side':<6} | {'Qty':<6} | {'Entry Price':<12} | {'Current Price':<14} | {'Unrealized PnL'}")
            print("-" * 75)
            for p in positions:
                pnl_str = f"${p['unrealized_pnl']:+,.2f} ({p['unrealized_pnl_pct']:+.2f}%)"
                print(
                    f"{p['symbol']:<8} | {p['side']:<6} | {p['qty']:<6} | "
                    f"${p['avg_entry_price']:<11.2f} | ${p['current_price']:<13.2f} | {pnl_str}"
                )
        else:
            print("Portfolio 100% in cash (no active open positions).")
        print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Automated Daily Execution Runner")
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite state persistence database",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital for paper broker (default: 100000.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    runner = DailyPaperRunner(state_db_path=args.db_path, initial_capital=args.capital)
    summary = runner.run_daily_cycle()
    runner.print_eod_summary(summary)


if __name__ == "__main__":
    main()
