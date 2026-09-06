import json
import os
from datetime import datetime

import numpy as np

from src.execution.fx_engine import FXEngine


class PaperTradingEngine:
    def __init__(
        self,
        initial_capital=1000000.0,
        db_path="data/paper_trading.json",
        base_currency="USD",
        fx_engine=None,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.base_currency = base_currency
        self.positions = {}  # ticker -> {shares, avg_price, sector, currency, market}
        self.history = []
        self.portfolio_snapshots = []  # List of {time, equity, cash, base_currency}
        self.db_path = db_path
        self.fx_engine = fx_engine if fx_engine else FXEngine()
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    self.initial_capital = data.get(
                        "initial_capital", self.initial_capital
                    )
                    self.capital = data.get("capital", self.initial_capital)
                    self.base_currency = data.get("base_currency", "USD")
                    self.positions = data.get("positions", {})
                    self.history = data.get("history", [])
                    self.portfolio_snapshots = data.get("portfolio_snapshots", [])
            except Exception:
                pass

    def _save(self):
        temp_path = f"{self.db_path}.tmp"
        with open(temp_path, "w") as f:
            json.dump(
                {
                    "initial_capital": self.initial_capital,
                    "capital": self.capital,
                    "base_currency": self.base_currency,
                    "positions": self.positions,
                    "history": self.history,
                    "portfolio_snapshots": self.portfolio_snapshots,
                },
                f,
                indent=2,
            )
        os.replace(temp_path, self.db_path)

    def set_base_currency(self, new_base: str):
        """Institutional Base Currency Switch: Re-normalizes the entire account and history."""
        if new_base == self.base_currency:
            return

        old_base = self.base_currency
        # 1. Convert capital and initial_capital
        self.capital = self.fx_engine.convert_to_base(self.capital, old_base, new_base)
        self.initial_capital = self.fx_engine.convert_to_base(
            self.initial_capital, old_base, new_base
        )

        # 2. Convert portfolio snapshots (Historical equity curve)
        for snapshot in self.portfolio_snapshots:
            snapshot["equity"] = self.fx_engine.convert_to_base(
                snapshot["equity"], old_base, new_base
            )
            snapshot["cash"] = self.fx_engine.convert_to_base(
                snapshot["cash"], old_base, new_base
            )
            snapshot["base_currency"] = new_base

        # 3. Convert trade history (Realized metrics)
        for trade in self.history:
            if "cost_base" in trade:
                trade["cost_base"] = self.fx_engine.convert_to_base(
                    trade["cost_base"], old_base, new_base
                )
            if "revenue_base" in trade:
                trade["revenue_base"] = self.fx_engine.convert_to_base(
                    trade["revenue_base"], old_base, new_base
                )
            if "pnl" in trade:
                trade["pnl"] = self.fx_engine.convert_to_base(
                    trade["pnl"], old_base, new_base
                )

        self.base_currency = new_base
        self._save()

    def _simulate_slippage(self, price, action):
        """Institutional Slippage Simulation: 0.05% for Buy, -0.05% for Sell."""
        slippage = 0.0005
        if action == "BUY":
            return price * (1 + slippage)
        elif action == "SELL":
            return price * (1 - slippage)
        return price

    def execute_trade(
        self,
        ticker,
        action,
        price,
        confidence_fraction,
        regime="NORMAL",
        sector="Unknown",
        stop_loss=None,
        take_profit=None,
        currency="USD",
        market="USA",
        signal_id=None,
    ):
        if action == "HOLD" or action == "VETOED" or "SCALE_BACK" in action:
            return None

        executed_price = self._simulate_slippage(price, action)

        # Calculate target allocation in BASE CURRENCY
        target_allocation_base = (
            self.capital + self._get_positions_value_base()
        ) * confidence_fraction
        # Convert target allocation to ASSET CURRENCY
        target_allocation_asset = self.fx_engine.convert_to_base(
            target_allocation_base, self.base_currency, currency
        )

        trade_record = None

        if action == "BUY":
            if ticker in self.positions and self.positions[ticker]["shares"] > 0:
                return None

            shares_to_buy = int(target_allocation_asset / executed_price)
            cost_asset = shares_to_buy * executed_price
            cost_base = self.fx_engine.convert_to_base(
                cost_asset, currency, self.base_currency
            )

            if self.capital >= cost_base and shares_to_buy > 0:
                self.capital -= cost_base
                self.positions[ticker] = {
                    "signal_id": signal_id,
                    "shares": shares_to_buy,
                    "avg_price": executed_price,
                    "sector": sector,
                    "currency": currency,
                    "market": market,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": datetime.now().isoformat(),
                }

                trade_record = {
                    "time": datetime.now().isoformat(),
                    "ticker": ticker,
                    "action": "BUY",
                    "shares": shares_to_buy,
                    "price": executed_price,
                    "currency": currency,
                    "cost_base": cost_base,
                    "regime": regime,
                    "sector": sector,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "signal_id": signal_id,
                }
                self.history.append(trade_record)

        elif action == "SELL":
            if ticker in self.positions and self.positions[ticker]["shares"] > 0:
                shares_to_sell = self.positions[ticker]["shares"]
                revenue_asset = shares_to_sell * executed_price
                revenue_base = self.fx_engine.convert_to_base(
                    revenue_asset, currency, self.base_currency
                )

                self.capital += revenue_base

                cost_base = self.fx_engine.convert_to_base(
                    shares_to_sell * self.positions[ticker]["avg_price"],
                    self.positions[ticker]["currency"],
                    self.base_currency,
                )
                pnl_base = revenue_base - cost_base

                trade_record = {
                    "time": datetime.now().isoformat(),
                    "ticker": ticker,
                    "action": "SELL",
                    "shares": shares_to_sell,
                    "price": executed_price,
                    "currency": currency,
                    "revenue_base": revenue_base,
                    "pnl": pnl_base,
                    "regime": regime,
                    "sector": self.positions[ticker].get("sector", "Unknown"),
                    "signal_id": self.positions[ticker].get("signal_id"),
                    "entry_time": self.positions[ticker].get("entry_time"),
                }
                self.positions.pop(ticker, None)
                self.history.append(trade_record)

        self._save()
        return trade_record

    def _get_positions_value_base(self, current_prices=None):
        val = 0.0
        for ticker, pos in self.positions.items():
            if pos["shares"] > 0:
                px = pos["avg_price"]
                if current_prices and ticker in current_prices:
                    px = current_prices[ticker]

                asset_val = pos["shares"] * px
                val += self.fx_engine.convert_to_base(
                    asset_val, pos.get("currency", "USD"), self.base_currency
                )
        return val

    def update_positions(self, current_prices):
        closed_trades = []
        for ticker, pos in list(self.positions.items()):
            if pos["shares"] <= 0 or ticker not in current_prices:
                continue

            curr_price = current_prices[ticker]
            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")

            action = None
            reason = None

            if sl and curr_price <= sl:
                action = "SELL"
                reason = "STOP_LOSS"
            elif tp and curr_price >= tp:
                action = "SELL"
                reason = "TAKE_PROFIT"

            if action:
                executed_price = self._simulate_slippage(curr_price, action)
                shares = pos["shares"]
                revenue_asset = shares * executed_price
                revenue_base = self.fx_engine.convert_to_base(
                    revenue_asset, pos["currency"], self.base_currency
                )

                self.capital += revenue_base

                cost_base = self.fx_engine.convert_to_base(
                    shares * pos["avg_price"], pos["currency"], self.base_currency
                )
                pnl_base = revenue_base - cost_base

                trade_record = {
                    "time": datetime.now().isoformat(),
                    "ticker": ticker,
                    "action": action,
                    "reason": reason,
                    "shares": shares,
                    "price": executed_price,
                    "currency": pos["currency"],
                    "revenue_base": revenue_base,
                    "pnl": pnl_base,
                    "regime": "AUTO",
                    "sector": pos.get("sector", "Unknown"),
                    "signal_id": pos.get("signal_id"),
                    "entry_time": pos.get("entry_time"),
                }
                self.history.append(trade_record)
                self.positions.pop(ticker)
                closed_trades.append(trade_record)

        if closed_trades:
            self._save()
        return closed_trades

    def record_snapshot(self, total_equity):
        snapshot = {
            "time": datetime.now().isoformat(),
            "equity": total_equity,
            "cash": self.capital,
            "base_currency": self.base_currency,
        }
        self.portfolio_snapshots.append(snapshot)
        if len(self.portfolio_snapshots) > 1000:
            self.portfolio_snapshots = self.portfolio_snapshots[-1000:]
        self._save()

    def get_portfolio_summary(self, current_prices):
        total_equity = self.capital + self._get_positions_value_base(current_prices)
        self.record_snapshot(total_equity)

        realized_pnl = sum(
            [t.get("pnl", 0) for t in self.history if t["action"] == "SELL"]
        )
        unrealized_pnl = total_equity - (self.initial_capital + realized_pnl)

        today_pnl = 0.0
        if len(self.portfolio_snapshots) > 1:
            today_pnl = total_equity - self.portfolio_snapshots[-2]["equity"]

        # Normalize positions for display
        display_positions = {}
        for ticker, pos in self.positions.items():
            if pos["shares"] > 0:
                # Convert average price to base currency
                base_avg_price = self.fx_engine.convert_to_base(
                    pos["avg_price"], pos.get("currency", "USD"), self.base_currency
                )
                display_positions[ticker] = {
                    **pos,
                    "avg_price": base_avg_price,
                    "original_currency": pos.get("currency", "USD"),
                }

        return {
            "cash": round(self.capital, 2),
            "equity": round(total_equity, 2),
            "base_currency": self.base_currency,
            "initial_capital": self.initial_capital,
            "return_pct": round(((total_equity / self.initial_capital) - 1) * 100, 2),
            "today_pnl": round(today_pnl, 2),
            "mtd_pnl": round(total_equity - self.initial_capital, 2),
            "ytd_pnl": round(total_equity - self.initial_capital, 2),
            "inception_pnl": round(total_equity - self.initial_capital, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "positions": display_positions,
            "fx_rates": self.fx_engine.rates,
        }

    def calculate_var(self, returns_history, confidence_level=0.95):
        """Value at Risk (Historical)"""
        if len(returns_history) < 5:
            return 0.0
        return np.percentile(returns_history, (1 - confidence_level) * 100)

    def calculate_expected_shortfall(self, returns_history, confidence_level=0.95):
        """Expected Shortfall (CVaR)"""
        if len(returns_history) < 5:
            return 0.0
        var = self.calculate_var(returns_history, confidence_level)
        tail_returns = [r for r in returns_history if r <= var]
        return np.mean(tail_returns) if tail_returns else 0.0
