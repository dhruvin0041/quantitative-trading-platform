import json
import logging
import os
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseBrokerAdapter(ABC):
    """
    Institutional Abstract Broker Execution Adapter.
    Standardized execution interface for paper or live orders across broker venues
    (e.g., Alpaca, Interactive Brokers, or Mock Paper Broker).
    """

    @abstractmethod
    def get_account_balance(self) -> dict:
        """
        Retrieves current account balance and liquidity metrics.
        Returns:
            dict: {
                "cash": float,
                "equity": float,
                "buying_power": float,
                "currency": str,
                "initial_capital": float,
            }
        """
        pass

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """
        Retrieves all currently open positions.
        Returns:
            list[dict]: [
                {
                    "symbol": str,
                    "qty": float,
                    "side": str,  # 'LONG' or 'SHORT'
                    "avg_entry_price": float,
                    "current_price": float,
                    "market_value": float,
                    "unrealized_pnl": float,
                    "unrealized_pnl_pct": float,
                }, ...
            ]
        """
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "MARKET",
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        current_price: float = 0.0,
    ) -> dict:
        """
        Submits an order for simulated or live broker execution.
        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            qty: Quantity in shares / contracts (must be > 0)
            side: 'BUY' or 'SELL'
            order_type: 'MARKET', 'LIMIT', etc.
            stop_loss: Protective stop loss price (0.0 if omitted)
            take_profit: Take profit target price (0.0 if omitted)
            current_price: Benchmark market price for fill calculation
        Returns:
            dict: Order execution confirmation record.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancels an active or pending order by its unique order ID.
        Returns:
            bool: True if order was active and successfully cancelled, False otherwise.
        """
        pass


class MockPaperBroker(BaseBrokerAdapter):
    """
    Production-grade Mock Execution Broker Adapter.
    Simulates real-world execution with 10 bps slippage per side, manages cash and positions,
    and provides persistent order tracking to either SQLite or JSON state files.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        currency: str = "USD",
        slippage_bps: float = 10.0,
        state_file: Optional[str] = "artifacts/paper_broker_state.json",
    ):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.currency = currency.upper()
        self.slippage_bps = float(slippage_bps)
        self.state_file = state_file

        # In-memory stores
        # positions: {symbol: {"symbol": str, "qty": float, "side": str, "avg_entry_price": float, "current_price": float, "stop_loss": float, "take_profit": float}}
        self.positions: Dict[str, Dict] = {}
        # orders: {order_id: dict}
        self.orders: Dict[str, Dict] = {}

        self.db_type = "memory"
        if self.state_file and self.state_file != ":memory:":
            if self.state_file.endswith(".db") or self.state_file.endswith(".sqlite"):
                self.db_type = "sqlite"
                self._init_sqlite()
            else:
                self.db_type = "json"
                self._init_json()
        self._load_state()

    # --- Persistence Helpers ---

    def _init_sqlite(self):
        if self.state_file is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
        conn = sqlite3.connect(self.state_file)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS broker_account (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS broker_positions (
                    symbol TEXT PRIMARY KEY,
                    qty REAL,
                    side TEXT,
                    avg_entry_price REAL,
                    current_price REAL,
                    stop_loss REAL,
                    take_profit REAL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS broker_orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    qty REAL,
                    side TEXT,
                    order_type TEXT,
                    status TEXT,
                    fill_price REAL,
                    slippage_bps REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    timestamp TEXT,
                    raw_payload TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _init_json(self):
        if self.state_file is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)

    def _load_state(self):
        if self.db_type == "json" and self.state_file and os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.cash = float(data.get("cash", self.initial_capital))
                    self.initial_capital = float(data.get("initial_capital", self.initial_capital))
                    self.currency = data.get("currency", self.currency)
                    self.positions = data.get("positions", {})
                    self.orders = data.get("orders", {})
            except Exception as e:
                logger.warning(f"Failed to load JSON broker state from {self.state_file}: {e}")
        elif self.db_type == "sqlite" and self.state_file and os.path.exists(self.state_file):
            conn = None
            try:
                conn = sqlite3.connect(self.state_file)
                cursor = conn.cursor()
                cursor.execute("SELECT key, value FROM broker_account")
                acc_rows = dict(cursor.fetchall())
                if "cash" in acc_rows:
                    self.cash = float(acc_rows["cash"])
                if "initial_capital" in acc_rows:
                    self.initial_capital = float(acc_rows["initial_capital"])
                if "currency" in acc_rows:
                    self.currency = acc_rows["currency"]

                cursor.execute("SELECT symbol, qty, side, avg_entry_price, current_price, stop_loss, take_profit FROM broker_positions")
                pos_rows = cursor.fetchall()
                self.positions = {}
                for row in pos_rows:
                    self.positions[row[0]] = {
                        "symbol": row[0],
                        "qty": float(row[1]),
                        "side": row[2],
                        "avg_entry_price": float(row[3]),
                        "current_price": float(row[4]),
                        "stop_loss": float(row[5]),
                        "take_profit": float(row[6]),
                    }

                cursor.execute("SELECT order_id, raw_payload FROM broker_orders")
                order_rows = cursor.fetchall()
                self.orders = {row[0]: json.loads(row[1]) for row in order_rows}
            except Exception as e:
                logger.warning(f"Failed to load SQLite broker state from {self.state_file}: {e}")
            finally:
                if conn is not None:
                    conn.close()

    def _persist_state(self):
        if self.db_type == "json" and self.state_file:
            temp_path = f"{self.state_file}.tmp"
            try:
                with open(temp_path, "w") as f:
                    json.dump(
                        {
                            "cash": self.cash,
                            "initial_capital": self.initial_capital,
                            "currency": self.currency,
                            "positions": self.positions,
                            "orders": self.orders,
                        },
                        f,
                        indent=2,
                    )
                os.replace(temp_path, self.state_file)
            except Exception as e:
                logger.error(f"Failed to persist JSON broker state: {e}")
        elif self.db_type == "sqlite" and self.state_file:
            conn = None
            try:
                conn = sqlite3.connect(self.state_file)
                cursor = conn.cursor()
                cursor.execute("INSERT OR REPLACE INTO broker_account (key, value) VALUES ('cash', ?)", (str(self.cash),))
                cursor.execute("INSERT OR REPLACE INTO broker_account (key, value) VALUES ('initial_capital', ?)", (str(self.initial_capital),))
                cursor.execute("INSERT OR REPLACE INTO broker_account (key, value) VALUES ('currency', ?)", (self.currency,))

                cursor.execute("DELETE FROM broker_positions")
                for p in self.positions.values():
                    cursor.execute(
                        """
                        INSERT INTO broker_positions (symbol, qty, side, avg_entry_price, current_price, stop_loss, take_profit)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            p["symbol"],
                            p["qty"],
                            p["side"],
                            p["avg_entry_price"],
                            p["current_price"],
                            p.get("stop_loss", 0.0),
                            p.get("take_profit", 0.0),
                        ),
                    )

                for o in self.orders.values():
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO broker_orders (order_id, symbol, qty, side, order_type, status, fill_price, slippage_bps, stop_loss, take_profit, timestamp, raw_payload)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            o["order_id"],
                            o["symbol"],
                            o["qty"],
                            o["side"],
                            o["order_type"],
                            o["status"],
                            o.get("fill_price", 0.0),
                            o.get("slippage_bps", self.slippage_bps),
                            o.get("stop_loss", 0.0),
                            o.get("take_profit", 0.0),
                            o.get("timestamp", datetime.now().isoformat()),
                            json.dumps(o),
                        ),
                    )
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to persist SQLite broker state: {e}")
            finally:
                if conn is not None:
                    conn.close()

    # --- Public Adapter Interface Methods ---

    def get_account_balance(self) -> dict:
        """Computes equity as cash plus current market value of all open positions."""
        total_market_value = 0.0
        for pos in self.positions.values():
            curr_p = pos.get("current_price", pos.get("avg_entry_price", 0.0))
            if pos.get("side") == "SHORT":
                # For short positions, equity change is: (entry - current) * qty
                unrealized = (pos["avg_entry_price"] - curr_p) * pos["qty"]
                total_market_value += unrealized
            else:
                total_market_value += curr_p * pos["qty"]

        total_equity = self.cash + total_market_value
        return {
            "cash": round(self.cash, 2),
            "equity": round(total_equity, 2),
            "buying_power": round(max(0.0, self.cash), 2),
            "currency": self.currency,
            "initial_capital": round(self.initial_capital, 2),
        }

    def get_positions(self) -> List[dict]:
        """Returns detailed summary of all open positions with unrealized PnL."""
        res = []
        for pos in self.positions.values():
            curr_p = pos.get("current_price", pos.get("avg_entry_price", 0.0))
            qty = pos["qty"]
            avg_p = pos["avg_entry_price"]
            side = pos.get("side", "LONG")

            if side == "SHORT":
                mkt_val = -(curr_p * qty)
                unrealized_pnl = (avg_p - curr_p) * qty
            else:
                mkt_val = curr_p * qty
                unrealized_pnl = (curr_p - avg_p) * qty

            cost_basis = avg_p * qty
            unrealized_pct = (unrealized_pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0

            res.append(
                {
                    "symbol": pos["symbol"],
                    "qty": round(qty, 4),
                    "side": side,
                    "avg_entry_price": round(avg_p, 4),
                    "current_price": round(curr_p, 4),
                    "market_value": round(mkt_val, 2),
                    "unrealized_pnl": round(unrealized_pnl, 2),
                    "unrealized_pnl_pct": round(unrealized_pct, 2),
                    "stop_loss": round(pos.get("stop_loss", 0.0), 4),
                    "take_profit": round(pos.get("take_profit", 0.0), 4),
                }
            )
        return res

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "MARKET",
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        current_price: float = 0.0,
    ) -> dict:
        """
        Executes order simulation with 10 bps slippage model.
        Fills immediately for MARKET orders if buying power allows.
        """
        symbol = symbol.upper()
        side = side.upper()
        order_type = order_type.upper()
        qty = float(qty)

        if qty <= 0:
            raise ValueError(f"Order quantity must be positive. Received {qty}")
        if side not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid order side: {side}. Must be 'BUY' or 'SELL'")
        if current_price <= 0:
            raise ValueError(f"current_price must be > 0 for fill calculation. Received {current_price}")

        order_id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"
        timestamp = datetime.now().isoformat()

        # Simulate execution friction: 10 bps slippage
        # BUY orders slip up: price * (1 + slippage)
        # SELL orders slip down: price * (1 - slippage)
        slip_mult = 1.0 + (self.slippage_bps / 10000.0) if side == "BUY" else 1.0 - (self.slippage_bps / 10000.0)
        fill_price = current_price * slip_mult
        gross_notional = qty * current_price
        net_notional = qty * fill_price

        # Check buying power on opening long orders
        if side == "BUY":
            existing_pos = self.positions.get(symbol)
            # If closing a short, buying covers short and doesn't require extra cash beyond coverage
            if existing_pos and existing_pos.get("side") == "SHORT":
                pass
            elif net_notional > self.cash:
                order_rec = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "qty": qty,
                    "side": side,
                    "order_type": order_type,
                    "status": "REJECTED",
                    "reason": f"Insufficient buying power: Required ${net_notional:.2f}, Available ${self.cash:.2f}",
                    "fill_price": 0.0,
                    "slippage_bps": self.slippage_bps,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "timestamp": timestamp,
                }
                self.orders[order_id] = order_rec
                self._persist_state()
                return order_rec

        # Apply fills to positions & cash
        if side == "BUY":
            if symbol in self.positions:
                pos = self.positions[symbol]
                if pos["side"] == "LONG":
                    # Increase long position
                    total_qty = pos["qty"] + qty
                    new_avg = ((pos["avg_entry_price"] * pos["qty"]) + (fill_price * qty)) / total_qty
                    pos["qty"] = total_qty
                    pos["avg_entry_price"] = new_avg
                    pos["current_price"] = current_price
                    pos["stop_loss"] = stop_loss or pos.get("stop_loss", 0.0)
                    pos["take_profit"] = take_profit or pos.get("take_profit", 0.0)
                    self.cash -= net_notional
                else:
                    # Covering a short position
                    short_qty = pos["qty"]
                    if qty >= short_qty:
                        # Fully cover short
                        realized = (pos["avg_entry_price"] - fill_price) * short_qty
                        self.cash += realized
                        remaining = qty - short_qty
                        if remaining > 0:
                            self.positions[symbol] = {
                                "symbol": symbol,
                                "qty": remaining,
                                "side": "LONG",
                                "avg_entry_price": fill_price,
                                "current_price": current_price,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                            }
                            self.cash -= remaining * fill_price
                        else:
                            del self.positions[symbol]
                    else:
                        # Partial cover
                        realized = (pos["avg_entry_price"] - fill_price) * qty
                        pos["qty"] -= qty
                        self.cash += realized
            else:
                # Open fresh long position
                self.positions[symbol] = {
                    "symbol": symbol,
                    "qty": qty,
                    "side": "LONG",
                    "avg_entry_price": fill_price,
                    "current_price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }
                self.cash -= net_notional

        elif side == "SELL":
            if symbol in self.positions:
                pos = self.positions[symbol]
                if pos["side"] == "LONG":
                    # Selling out of long position
                    long_qty = pos["qty"]
                    if qty >= long_qty:
                        # Fully close long
                        self.cash += long_qty * fill_price
                        remaining = qty - long_qty
                        if remaining > 0:
                            # Flip into short
                            self.positions[symbol] = {
                                "symbol": symbol,
                                "qty": remaining,
                                "side": "SHORT",
                                "avg_entry_price": fill_price,
                                "current_price": current_price,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                            }
                        else:
                            del self.positions[symbol]
                    else:
                        # Partial close
                        pos["qty"] -= qty
                        self.cash += qty * fill_price
                else:
                    # Expand short position
                    total_qty = pos["qty"] + qty
                    new_avg = ((pos["avg_entry_price"] * pos["qty"]) + (fill_price * qty)) / total_qty
                    pos["qty"] = total_qty
                    pos["avg_entry_price"] = new_avg
                    pos["current_price"] = current_price
                    pos["stop_loss"] = stop_loss or pos.get("stop_loss", 0.0)
                    pos["take_profit"] = take_profit or pos.get("take_profit", 0.0)
            else:
                # Open fresh short position
                self.positions[symbol] = {
                    "symbol": symbol,
                    "qty": qty,
                    "side": "SHORT",
                    "avg_entry_price": fill_price,
                    "current_price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }

        order_rec = {
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": "FILLED",
            "fill_price": round(fill_price, 4),
            "benchmark_price": round(current_price, 4),
            "slippage_bps": self.slippage_bps,
            "slippage_cost": round(abs(net_notional - gross_notional), 4),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "timestamp": timestamp,
        }
        self.orders[order_id] = order_rec
        self._persist_state()
        return order_rec

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancels an open order. If the order was already filled or does not exist, returns False.
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.get("status") == "OPEN":
            order["status"] = "CANCELLED"
            order["cancelled_at"] = datetime.now().isoformat()
            self._persist_state()
            return True
        return False
