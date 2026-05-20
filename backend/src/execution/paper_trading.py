import json
import os
from datetime import datetime
import pandas as pd
import numpy as np

class PaperTradingEngine:
    def __init__(self, initial_capital=1000000.0, db_path="paper_trading.json"):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {} # ticker -> {shares, avg_price}
        self.history = []
        self.db_path = db_path
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    self.capital = data.get("capital", self.initial_capital)
                    self.positions = data.get("positions", {})
                    self.history = data.get("history", [])
            except Exception:
                pass

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump({
                "capital": self.capital,
                "positions": self.positions,
                "history": self.history
            }, f, indent=2)
            
    def _simulate_slippage(self, price, action):
        # 5 bps slippage
        slippage = 0.0005
        if action == "BUY":
            return price * (1 + slippage)
        else:
            return price * (1 - slippage)

    def execute_trade(self, ticker, action, price, confidence_fraction):
        if action == "HOLD" or "SCALE_BACK" in action:
            return None
            
        executed_price = self._simulate_slippage(price, action)
        
        # Calculate target allocation based on Kelly/Confidence
        target_allocation = self.capital * confidence_fraction
        
        trade_record = None

        if action == "BUY":
            shares_to_buy = int(target_allocation / executed_price)
            cost = shares_to_buy * executed_price
            if self.capital >= cost and shares_to_buy > 0:
                self.capital -= cost
                if ticker not in self.positions:
                    self.positions[ticker] = {"shares": 0, "avg_price": 0.0}
                
                prev_shares = self.positions[ticker]["shares"]
                prev_avg = self.positions[ticker]["avg_price"]
                new_shares = prev_shares + shares_to_buy
                new_avg = ((prev_shares * prev_avg) + cost) / new_shares
                
                self.positions[ticker] = {"shares": new_shares, "avg_price": new_avg}
                trade_record = {
                    "time": datetime.now().isoformat(),
                    "ticker": ticker,
                    "action": "BUY",
                    "shares": shares_to_buy,
                    "price": executed_price,
                    "cost": cost
                }
                self.history.append(trade_record)
                
        elif action == "SELL":
            if ticker in self.positions and self.positions[ticker]["shares"] > 0:
                shares_to_sell = self.positions[ticker]["shares"]
                revenue = shares_to_sell * executed_price
                self.capital += revenue
                
                pnl = revenue - (shares_to_sell * self.positions[ticker]["avg_price"])
                
                self.positions[ticker] = {"shares": 0, "avg_price": 0.0}
                trade_record = {
                    "time": datetime.now().isoformat(),
                    "ticker": ticker,
                    "action": "SELL",
                    "shares": shares_to_sell,
                    "price": executed_price,
                    "revenue": revenue,
                    "pnl": pnl
                }
                self.history.append(trade_record)

        self._save()
        return trade_record

    def get_portfolio_summary(self, current_prices):
        total_equity = self.capital
        for ticker, pos in self.positions.items():
            if pos["shares"] > 0 and ticker in current_prices:
                total_equity += pos["shares"] * current_prices[ticker]
                
        return {
            "cash": round(self.capital, 2),
            "equity": round(total_equity, 2),
            "return_pct": round(((total_equity / self.initial_capital) - 1) * 100, 2),
            "positions": self.positions
        }

    def calculate_var(self, returns_history, confidence_level=0.95):
        """Value at Risk (Historical)"""
        if len(returns_history) < 20:
            return 0.0
        return np.percentile(returns_history, (1 - confidence_level) * 100)
        
    def calculate_expected_shortfall(self, returns_history, confidence_level=0.95):
        """Expected Shortfall (CVaR)"""
        if len(returns_history) < 20:
            return 0.0
        var = self.calculate_var(returns_history, confidence_level)
        return np.mean([r for r in returns_history if r <= var])
