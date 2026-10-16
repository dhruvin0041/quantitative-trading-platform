# backend/scripts/research/workstream_3_forward_simulation.py
"""
HYDRA V2.4.2+ — Workstream 3: Forward $1,000 Portfolio Simulation
Period: 2025-10-13 to 2026-10-13  (One Calendar Year)

This script answers the user's question:
  "How much would I have made from 13-10-2025 to 13-10-2026 following
   each and every signal, starting from $1,000?"

Architecture:
1.  Fetch AAPL OHLCV from Yahoo Finance (with 300-day warmup for SMA200).
2.  Compute all required technical indicators (SMA200, RSI14, BB, ATR14, Donchian).
3.  Generate signals for 6 strategies across the evaluation window.
4.  Run the InstitutionalPortfolioSimulator on $1,000 initial capital
    with the standard 5-bps execution model.
5.  Export complete trade ledger, equity curves, and summary report.

ML Strategy Logic:
- The Frozen V2.4 ML and Remediated ML strategies are trained ONLY on
  historical data from the pinned development dataset (ending before
  the forward period). The trained model then predicts on the forward
  period's engineered features. There is NO look-ahead bias — all
  training data strictly precedes the evaluation window.
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Path Setup
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Workstream3Forward")

REPORTS_DIR = BACKEND_DIR / "reports"
PINNED_DATASET_PATH = BACKEND_DIR / "artifacts" / "research" / "workstream_1_pinned_development_dataset.joblib"
PHASE_3D_OOF_PATH = BACKEND_DIR / "artifacts" / "research" / "phase_3d_oof_predictions.joblib"

# Evaluation window
EVAL_START = "2025-10-13"
EVAL_END = "2026-10-14"
# Warmup window for indicator computation and ML training: start at 2015-01-01 for full history
WARMUP_START = "2015-01-01"


# ──────────────────────────────────────────────────────────────────────────────
# 1. PORTFOLIO SIMULATOR (identical to Workstream 2 for reproducibility)
# ──────────────────────────────────────────────────────────────────────────────

class InstitutionalPortfolioSimulator:
    """
    Deterministic chronological portfolio simulator.
    - Initial cash: $1,000.00
    - Fixed costs: 5 bps adverse slippage + $0.005/share commission ($1.00 min per ticket)
    - Whole-share purchases only
    - Non-negative cash constraint
    - Mark-to-market equity: Equity_t = Cash_t + Q_t * Close_t
    - Causal timing: Decision at Close[t] -> Fill at Open[t+1]
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.slippage_rate = slippage_bps / 10000.0
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

    def compute_commission(self, shares: int) -> float:
        if shares <= 0:
            return 0.0
        return max(self.min_commission, shares * self.commission_per_share)

    def simulate(
        self,
        df_prices: pd.DataFrame,
        signals_series: pd.Series,
        strategy_name: str,
    ) -> Dict[str, Any]:
        """
        Executes a chronological simulation given a series of signals ('BUY', 'SELL', 'HOLD').
        df_prices must contain 'Open', 'High', 'Low', 'Close'.
        signals_series index must match df_prices index.
        """
        cash = float(self.initial_capital)
        position = 0
        entry_price = 0.0
        entry_fees = 0.0
        entry_bar_idx = 0
        entry_timestamp = None
        entry_signal_id = 0

        trades: List[Dict[str, Any]] = []
        equity_curve: List[Dict[str, Any]] = []
        signals_log: List[Dict[str, Any]] = []

        n_bars = len(df_prices)
        dates = df_prices.index

        pending_action: Optional[str] = None
        pending_signal_meta: Dict[str, Any] = {}

        for i in range(n_bars):
            dt = dates[i]
            open_p = float(df_prices["Open"].iloc[i])
            high_p = float(df_prices["High"].iloc[i])
            low_p = float(df_prices["Low"].iloc[i])
            close_p = float(df_prices["Close"].iloc[i])

            # 1. Execute pending orders from previous session's close at Open[t]
            if pending_action == "BUY" and position == 0:
                fill_price = open_p * (1.0 + self.slippage_rate)
                max_alloc_cash = cash - self.min_commission
                if max_alloc_cash > fill_price:
                    q = int(math.floor(max_alloc_cash / fill_price))
                    comm = self.compute_commission(q)
                    total_cost = (q * fill_price) + comm
                    if total_cost <= cash and q > 0:
                        cash -= total_cost
                        position = q
                        entry_price = fill_price
                        entry_fees = comm
                        entry_bar_idx = i
                        entry_timestamp = str(dt)
                        entry_signal_id = pending_signal_meta.get("signal_id", i)

                        pending_signal_meta["status"] = "FILLED"
                        pending_signal_meta["fill_price"] = fill_price
                        pending_signal_meta["fill_shares"] = q
                        pending_signal_meta["fill_fees"] = comm
                        pending_signal_meta["fill_timestamp"] = str(dt)
                    else:
                        pending_signal_meta["status"] = "REJECTED_INSUFFICIENT_CASH"
                else:
                    pending_signal_meta["status"] = "REJECTED_CASH_BELOW_SHARE_PRICE"

            elif pending_action == "SELL" and position > 0:
                fill_price = open_p * (1.0 - self.slippage_rate)
                comm = self.compute_commission(position)
                gross_proceeds = position * fill_price
                net_proceeds = gross_proceeds - comm
                gross_entry_cost = position * entry_price
                total_entry_cost = gross_entry_cost + entry_fees

                gross_pnl = gross_proceeds - gross_entry_cost
                net_pnl = net_proceeds - total_entry_cost
                ret_pct = (net_pnl / total_entry_cost) * 100.0 if total_entry_cost > 0 else 0.0

                pre_equity = cash + (position * open_p)
                cash += net_proceeds
                post_equity = cash

                trade_record = {
                    "trade_id": len(trades) + 1,
                    "strategy": strategy_name,
                    "symbol": "AAPL",
                    "entry_signal_id": entry_signal_id,
                    "entry_timestamp": entry_timestamp,
                    "entry_fill_price": round(entry_price, 4),
                    "entry_shares": position,
                    "entry_fees": round(entry_fees, 4),
                    "entry_total_cost": round(total_entry_cost, 4),
                    "exit_timestamp": str(dt),
                    "exit_fill_price": round(fill_price, 4),
                    "exit_shares": position,
                    "exit_fees": round(comm, 4),
                    "exit_net_proceeds": round(net_proceeds, 4),
                    "gross_pnl": round(gross_pnl, 4),
                    "net_pnl": round(net_pnl, 4),
                    "net_return_pct": round(ret_pct, 4),
                    "holding_period_bars": i - entry_bar_idx,
                    "exit_reason": pending_signal_meta.get("exit_reason", "SIGNAL"),
                    "pre_trade_equity": round(pre_equity, 2),
                    "post_trade_equity": round(post_equity, 2),
                    "is_winner": bool(net_pnl > 0),
                }
                trades.append(trade_record)

                pending_signal_meta["status"] = "FILLED"
                pending_signal_meta["fill_price"] = fill_price
                pending_signal_meta["fill_shares"] = position
                pending_signal_meta["fill_fees"] = comm
                pending_signal_meta["fill_timestamp"] = str(dt)

                position = 0
                entry_price = 0.0
                entry_fees = 0.0

            pending_action = None
            pending_signal_meta = {}

            # 2. Mark to market daily equity at Close[t]
            market_value = position * close_p
            daily_equity = cash + market_value

            equity_curve.append({
                "bar_index": i,
                "timestamp": str(dt),
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "cash": round(cash, 2),
                "shares": position,
                "market_value": round(market_value, 2),
                "equity": round(daily_equity, 2),
                "position_state": "LONG" if position > 0 else "FLAT",
            })

            # 3. Process new strategy decision at Close[t] for next session Open[t+1]
            raw_sig = signals_series.iloc[i] if i < len(signals_series) else "HOLD"
            sig_action = str(raw_sig).upper()

            signal_id = i + 1
            sig_entry = {
                "signal_id": signal_id,
                "timestamp": str(dt),
                "strategy": strategy_name,
                "symbol": "AAPL",
                "raw_signal": sig_action,
                "current_position": "LONG" if position > 0 else "FLAT",
                "reference_close_price": close_p,
                "action": "HOLD",
                "target_execution": "NEXT_SESSION_OPEN",
                "status": "EVALUATED",
            }

            if sig_action == "BUY" and position == 0 and i < n_bars - 1:
                pending_action = "BUY"
                sig_entry["action"] = "BUY"
                sig_entry["status"] = "PENDING_EXECUTION"
                pending_signal_meta = {
                    "signal_id": signal_id,
                    "action": "BUY",
                }
            elif sig_action == "SELL" and position > 0 and i < n_bars - 1:
                pending_action = "SELL"
                sig_entry["action"] = "SELL"
                sig_entry["status"] = "PENDING_EXECUTION"
                pending_signal_meta = {
                    "signal_id": signal_id,
                    "action": "SELL",
                    "exit_reason": "STRATEGY_SIGNAL",
                }
            elif sig_action == "SELL" and position == 0:
                sig_entry["action"] = "HOLD"
                sig_entry["status"] = "SUPPRESSED_NAKED_SHORT"

            signals_log.append(sig_entry)

        # Mark-to-market at conclusion
        final_equity = equity_curve[-1]["equity"] if equity_curve else cash
        final_cash = cash
        net_profit = final_equity - self.initial_capital
        net_return_pct = (net_profit / self.initial_capital) * 100.0

        # Performance summary metrics
        eq_values = np.array([x["equity"] for x in equity_curve])
        peak = np.maximum.accumulate(eq_values)
        dd = eq_values - peak
        dd_pct = np.where(peak > 0, dd / peak, 0.0)
        max_dd_dollars = float(np.min(dd)) if len(dd) > 0 else 0.0
        max_dd_pct = float(np.min(dd_pct) * 100.0) if len(dd_pct) > 0 else 0.0

        # Trade analytics
        n_trades = len(trades)
        wins = [t for t in trades if t["is_winner"]]
        losses = [t for t in trades if not t["is_winner"]]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / n_trades * 100.0) if n_trades > 0 else 0.0

        gross_profit = sum(t["gross_pnl"] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t["gross_pnl"] for t in losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        avg_trade_pnl = float(np.mean([t["net_pnl"] for t in trades])) if n_trades > 0 else 0.0
        med_trade_pnl = float(np.median([t["net_pnl"] for t in trades])) if n_trades > 0 else 0.0

        # Daily returns for Sharpe / Sortino
        daily_returns = np.diff(eq_values) / eq_values[:-1] if len(eq_values) > 1 else np.array([0.0])
        mean_ret = float(np.mean(daily_returns)) if len(daily_returns) > 0 else 0.0
        std_ret = float(np.std(daily_returns)) if len(daily_returns) > 0 else 0.0
        downside_std = float(np.std(daily_returns[daily_returns < 0])) if np.any(daily_returns < 0) else 1e-6

        sharpe_ratio = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0
        sortino_ratio = (mean_ret / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

        # Exposure
        bars_in_market = sum(1 for x in equity_curve if x["shares"] > 0)
        exposure_pct = (bars_in_market / n_bars * 100.0) if n_bars > 0 else 0.0

        # CAGR
        years = n_bars / 252.0
        cagr_pct = (((final_equity / self.initial_capital) ** (1.0 / years) - 1.0) * 100.0) if (years > 0 and final_equity > 0) else 0.0

        # Open position (if still holding at end)
        open_position_info = None
        if position > 0:
            unrealized_pnl = (position * equity_curve[-1]["close"]) - (position * entry_price) - entry_fees
            open_position_info = {
                "shares": position,
                "entry_price": round(entry_price, 4),
                "current_price": equity_curve[-1]["close"],
                "unrealized_pnl": round(unrealized_pnl, 2),
                "entry_date": entry_timestamp,
            }

        summary = {
            "strategy": strategy_name,
            "evaluation_period": f"{EVAL_START} to {EVAL_END}",
            "initial_capital": self.initial_capital,
            "final_cash": round(final_cash, 2),
            "final_equity": round(final_equity, 2),
            "net_profit_dollars": round(net_profit, 2),
            "net_return_pct": round(net_return_pct, 2),
            "cagr_pct": round(cagr_pct, 2),
            "total_trades": n_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != 999.0 else "INF_NO_LOSSES",
            "avg_trade_pnl": round(avg_trade_pnl, 2),
            "median_trade_pnl": round(med_trade_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "max_drawdown_dollars": round(max_dd_dollars, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "sortino_ratio": round(sortino_ratio, 3),
            "exposure_pct": round(exposure_pct, 2),
            "total_bars": n_bars,
            "slippage_bps": self.slippage_rate * 10000.0,
            "commission_model": f"${self.commission_per_share}/share (min ${self.min_commission})",
            "open_position": open_position_info,
        }

        return {
            "summary": summary,
            "trades": trades,
            "equity_curve": equity_curve,
            "signals_log": signals_log,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA FETCHING & INDICATOR COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def fetch_aapl_data() -> pd.DataFrame:
    """Fetch AAPL OHLCV from Yahoo Finance with warmup buffer."""
    import yfinance as yf
    logger.info(f"Fetching AAPL data from {WARMUP_START} to {EVAL_END}...")
    df = yf.download("AAPL", start=WARMUP_START, end=EVAL_END, progress=False)
    if df.empty:
        raise ValueError("No AAPL data returned from Yahoo Finance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    logger.info(f"Fetched {len(df)} bars: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all required technical indicators on the full dataframe (including warmup)."""
    import ta

    out = df.copy()

    # SMA 200
    out["SMA_200"] = out["Close"].rolling(window=200, min_periods=200).mean()

    # RSI 14
    rsi_indicator = ta.momentum.RSIIndicator(close=out["Close"], window=14)
    out["RSI"] = rsi_indicator.rsi()

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(close=out["Close"], window=20, window_dev=2)
    out["BB_Upper"] = bb.bollinger_hband()
    out["BB_Middle"] = bb.bollinger_mavg()
    out["BB_Lower"] = bb.bollinger_lband()

    # ATR 14
    atr = ta.volatility.AverageTrueRange(high=out["High"], low=out["Low"], close=out["Close"], window=14)
    out["ATR"] = atr.average_true_range()

    # ADX 14
    adx = ta.trend.ADXIndicator(high=out["High"], low=out["Low"], close=out["Close"], window=14)
    out["ADX"] = adx.adx()

    # MACD
    macd = ta.trend.MACD(close=out["Close"])
    out["MACD"] = macd.macd()
    out["MACD_Signal"] = macd.macd_signal()
    out["MACD_Diff"] = macd.macd_diff()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high=out["High"], low=out["Low"], close=out["Close"])
    out["Stoch_K"] = stoch.stoch()
    out["Stoch_D"] = stoch.stoch_signal()

    # OBV
    out["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=out["Close"], volume=out["Volume"]).on_balance_volume()

    # Donchian channels (20-day, lagged by 1 for causality)
    out["Donchian_High_20"] = out["High"].shift(1).rolling(window=20, min_periods=20).max()
    out["Donchian_Low_20"] = out["Low"].shift(1).rolling(window=20, min_periods=20).min()

    # Volatility ratio
    out["Volatility_Ratio"] = out["ATR"] / out["Close"]

    # Returns
    out["Return_1d"] = out["Close"].pct_change(1)
    out["Return_5d"] = out["Close"].pct_change(5)
    out["Return_20d"] = out["Close"].pct_change(20)

    # Volume moving averages
    out["Volume_SMA_20"] = out["Volume"].rolling(window=20).mean()
    out["Volume_Ratio"] = out["Volume"] / out["Volume_SMA_20"]

    logger.info(f"Computed {len([c for c in out.columns if c not in df.columns])} indicators")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3. SIGNAL GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def generate_buy_and_hold_signals(eval_dates: pd.DatetimeIndex) -> pd.Series:
    """Buy on day 1, hold forever."""
    sig = pd.Series("HOLD", index=eval_dates)
    sig.iloc[0] = "BUY"
    return sig


def generate_sma200_signals(df_full: pd.DataFrame, eval_dates: pd.DatetimeIndex) -> pd.Series:
    """200-day SMA trend following: BUY when Close >= SMA200, SELL when Close < SMA200."""
    sma = df_full["SMA_200"].loc[eval_dates]
    close = df_full["Close"].loc[eval_dates]
    sig = pd.Series("HOLD", index=eval_dates)
    for i in range(len(eval_dates)):
        c = float(close.iloc[i])
        s = float(sma.iloc[i])
        if not np.isnan(s):
            if c >= s:
                sig.iloc[i] = "BUY"
            else:
                sig.iloc[i] = "SELL"
    return sig


def generate_donchian_signals(df_full: pd.DataFrame, eval_dates: pd.DatetimeIndex) -> pd.Series:
    """20-day Donchian breakout: BUY on upper breakout, SELL on lower breakout."""
    dh = df_full["Donchian_High_20"].loc[eval_dates]
    dl = df_full["Donchian_Low_20"].loc[eval_dates]
    close = df_full["Close"].loc[eval_dates]
    sig = pd.Series("HOLD", index=eval_dates)
    for i in range(len(eval_dates)):
        c = float(close.iloc[i])
        h = float(dh.iloc[i])
        l_val = float(dl.iloc[i])
        if not np.isnan(h) and not np.isnan(l_val):
            if c > h:
                sig.iloc[i] = "BUY"
            elif c < l_val:
                sig.iloc[i] = "SELL"
    return sig


def generate_dip_entry_signals(df_full: pd.DataFrame, eval_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Dip-Entry in Uptrend:
    - Entry: Close >= SMA200 AND (RSI < 42 OR Close <= BB_Lower)
    - Exit: RSI > 70 OR Close < SMA200 OR 15-day timeout
    """
    sma = df_full["SMA_200"].loc[eval_dates]
    close = df_full["Close"].loc[eval_dates]
    rsi = df_full["RSI"].loc[eval_dates]
    bbl = df_full["BB_Lower"].loc[eval_dates]

    sig = pd.Series("HOLD", index=eval_dates)
    in_pos = False
    entry_idx = 0

    for i in range(len(eval_dates)):
        c = float(close.iloc[i])
        s = float(sma.iloc[i])
        r = float(rsi.iloc[i])
        b = float(bbl.iloc[i])

        if np.isnan(s):
            continue

        uptrend = c >= s
        pullback = (r < 42.0) or (c <= b)

        if not in_pos:
            if uptrend and pullback:
                sig.iloc[i] = "BUY"
                in_pos = True
                entry_idx = i
        else:
            trend_break = c < s
            overbought = r > 70.0
            timeout = (i - entry_idx >= 15)
            if trend_break or overbought or timeout:
                sig.iloc[i] = "SELL"
                in_pos = False

    return sig


def generate_ml_signals_from_retrained_model(
    df_full: pd.DataFrame,
    eval_dates: pd.DatetimeIndex,
    strategy_name: str,
    conviction_threshold: float = 0.50,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    horizon: int = 15,
) -> pd.Series:
    """
    Train a walk-forward XGBoost model on ALL historical data BEFORE the evaluation
    window (with 15-day embargo), then predict on the forward period.

    This avoids look-ahead bias: the model only sees data before eval_start.
    """
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    logger.info(f"Training ML model ({strategy_name}, TP={tp_mult}x, SL={sl_mult}x, H={horizon}) on pre-{EVAL_START} data...")

    # Feature columns — use the same indicators we computed
    feature_cols = [
        "RSI", "ATR", "ADX", "MACD", "MACD_Signal", "MACD_Diff",
        "Stoch_K", "Stoch_D", "BB_Upper", "BB_Middle", "BB_Lower",
        "Volatility_Ratio", "Return_1d", "Return_5d", "Return_20d",
        "Volume_Ratio", "SMA_200",
    ]

    # Verify all features exist
    missing = [c for c in feature_cols if c not in df_full.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Split: train on everything BEFORE eval_start minus 15-day embargo
    eval_start_dt = pd.Timestamp(EVAL_START)
    embargo_start = eval_start_dt - pd.Timedelta(days=20)  # 15 trading days ~ 20 calendar days

    train_df = df_full.loc[:embargo_start].dropna(subset=feature_cols)
    eval_df = df_full.loc[eval_dates].copy()

    if len(train_df) < 200:
        logger.warning(f"Insufficient training data ({len(train_df)} bars), ML signals will be HOLD")
        return pd.Series("HOLD", index=eval_dates)

    # Generate triple-barrier labels for training data
    y_train = []
    train_close = train_df["Close"]
    train_high = train_df["High"]
    train_low = train_df["Low"]
    train_atr = train_df["ATR"]

    future_highs_list = []
    future_lows_list = []
    for h in range(1, horizon + 1):
        future_highs_list.append(train_high.shift(-h))
        future_lows_list.append(train_low.shift(-h))

    fh_df = pd.concat(future_highs_list, axis=1)
    fl_df = pd.concat(future_lows_list, axis=1)

    valid_mask = []
    for idx, dt in enumerate(train_df.index):
        c_p = float(train_close.loc[dt])
        atr_p = float(train_atr.loc[dt])
        ub = c_p + (atr_p * tp_mult)
        lb = c_p - (atr_p * sl_mult)

        fh_row = fh_df.loc[dt].values
        fl_row = fl_df.loc[dt].values

        if np.any(np.isnan(fh_row)) or np.any(np.isnan(fl_row)):
            y_train.append(-1)  # Invalid, will be filtered
            valid_mask.append(False)
            continue

        u_hit = np.where(fh_row >= ub)[0]
        l_hit = np.where(fl_row <= lb)[0]
        u_step = u_hit[0] + 1 if len(u_hit) > 0 else horizon + 1
        l_step = l_hit[0] + 1 if len(l_hit) > 0 else horizon + 1

        if u_step > horizon and l_step > horizon:
            y_train.append(1)  # HOLD
        elif u_step < l_step:
            y_train.append(2)  # BUY
        else:
            y_train.append(0)  # SELL
        valid_mask.append(True)

    y_train = np.array(y_train, dtype=int)
    valid_mask = np.array(valid_mask)

    X_train = train_df[feature_cols].values[valid_mask]
    y_train = y_train[valid_mask]

    logger.info(f"Training set: {len(X_train)} samples, label distribution: "
                f"SELL={np.sum(y_train == 0)}, HOLD={np.sum(y_train == 1)}, BUY={np.sum(y_train == 2)}")

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Fill NaN in eval features with forward-fill then zero
    X_eval_raw = eval_df[feature_cols].copy()
    X_eval_raw = X_eval_raw.ffill().fillna(0.0)
    X_eval = X_eval_raw.values
    X_eval_s = scaler.transform(X_eval)

    # Train XGBoost
    xgb_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.70,
        "min_child_weight": 2.0,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": 1,
    }

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_eval_s)

    logger.info(f"ML prediction shape: {probs.shape}, mean P(BUY)={probs[:, 2].mean():.4f}")

    # Generate signals with macro trend filter
    sma = df_full["SMA_200"].loc[eval_dates]
    close = df_full["Close"].loc[eval_dates]

    sig = pd.Series("HOLD", index=eval_dates)
    in_pos = False
    for i in range(len(eval_dates)):
        c = float(close.iloc[i])
        s = float(sma.iloc[i])
        pb = probs[i, 2]  # P(BUY)
        ps = probs[i, 0]  # P(SELL)
        bull_regime = (c >= s) if not np.isnan(s) else True

        if not in_pos:
            if pb >= conviction_threshold and bull_regime:
                sig.iloc[i] = "BUY"
                in_pos = True
        else:
            if ps >= conviction_threshold or (not bull_regime):
                sig.iloc[i] = "SELL"
                in_pos = False

    return sig


# ──────────────────────────────────────────────────────────────────────────────
# 4. MASTER EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def run_forward_simulation() -> Dict[str, Any]:
    """Main entry point: fetch data, compute indicators, run all 6 strategies, export results."""
    logger.info("=" * 80)
    logger.info("HYDRA V2.4.2+ — Workstream 3: Forward $1,000 Portfolio Simulation")
    logger.info(f"Evaluation Period: {EVAL_START} to {EVAL_END}")
    logger.info("=" * 80)

    # 1. Fetch data
    df_raw = fetch_aapl_data()

    # 2. Compute indicators
    df_full = compute_indicators(df_raw)

    # 3. Define evaluation window
    eval_mask = (df_full.index >= pd.Timestamp(EVAL_START)) & (df_full.index <= pd.Timestamp(EVAL_END))
    eval_dates = df_full.index[eval_mask]

    if len(eval_dates) == 0:
        raise ValueError(f"No data found in evaluation window {EVAL_START} to {EVAL_END}")

    eval_prices = df_raw.loc[eval_dates].copy()
    logger.info(f"Evaluation window: {len(eval_dates)} bars from {eval_dates[0].strftime('%Y-%m-%d')} to {eval_dates[-1].strftime('%Y-%m-%d')}")

    # 4. Generate signals for all strategies
    logger.info("Generating strategy signals...")
    strategies = {}

    # 4a. Buy and Hold
    strategies["Buy_and_Hold"] = generate_buy_and_hold_signals(eval_dates)

    # 4b. 200-day SMA Trend Following
    strategies["200_Day_SMA_Trend"] = generate_sma200_signals(df_full, eval_dates)

    # 4c. 20-day Donchian Breakout
    strategies["20_Day_Donchian_Breakout"] = generate_donchian_signals(df_full, eval_dates)

    # 4d. Dip-Entry in Uptrend
    strategies["Dip_Entry_In_Uptrend"] = generate_dip_entry_signals(df_full, eval_dates)

    # 4e. HYDRA Frozen V2.4 ML (retrained on historical, applied forward with original control geometry: TP=3.0, SL=0.5)
    strategies["HYDRA_Frozen_V2.4_ML"] = generate_ml_signals_from_retrained_model(
        df_full, eval_dates, "HYDRA_Frozen_V2.4_ML", conviction_threshold=0.50, tp_mult=3.0, sl_mult=0.5, horizon=15
    )

    # 4f. HYDRA Remediated ML Candidate (retrained with remediated candidate geometry: TP=2.0, SL=1.5)
    strategies["HYDRA_Remediated_ML_Candidate"] = generate_ml_signals_from_retrained_model(
        df_full, eval_dates, "HYDRA_Remediated_ML_Candidate", conviction_threshold=0.50, tp_mult=2.0, sl_mult=1.5, horizon=15
    )

    # 5. Run portfolio simulations
    logger.info("Running $1,000 portfolio simulations with 5-bps execution model...")
    sim = InstitutionalPortfolioSimulator(
        initial_capital=1000.0,
        slippage_bps=5.0,
        commission_per_share=0.005,
        min_commission=1.00,
    )

    sim_results: Dict[str, Dict[str, Any]] = {}
    for s_name, s_sigs in strategies.items():
        res = sim.simulate(eval_prices, s_sigs, s_name)
        sim_results[s_name] = res
        s = res["summary"]
        open_note = ""
        if s.get("open_position"):
            op = s["open_position"]
            open_note = f" [OPEN POS: {op['shares']} shares, unrealized {op['unrealized_pnl']:+.2f}]"
        logger.info(
            f"  [{s_name}]: Final Equity=${s['final_equity']:.2f} "
            f"({s['net_return_pct']:+.2f}%), Trades={s['total_trades']}, "
            f"WinRate={s['win_rate_pct']}%, MaxDD={s['max_drawdown_pct']:.2f}%, "
            f"Sharpe={s['sharpe_ratio']}{open_note}"
        )

    # 6. Export results
    logger.info("Exporting forward simulation artifacts...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 6a. Summary comparison table
    summary_rows = [res["summary"] for res in sim_results.values()]
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(REPORTS_DIR / "workstream_3_forward_strategy_comparison.csv", index=False)
    with open(REPORTS_DIR / "workstream_3_forward_strategy_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, default=str)

    # 6b. Complete trade ledger
    all_trades: List[Dict[str, Any]] = []
    for res in sim_results.values():
        all_trades.extend(res["trades"])
    df_trades = pd.DataFrame(all_trades)
    df_trades.to_csv(REPORTS_DIR / "workstream_3_forward_trade_ledger.csv", index=False)
    with open(REPORTS_DIR / "workstream_3_forward_trade_ledger.json", "w", encoding="utf-8") as f:
        json.dump(all_trades, f, indent=2, default=str)

    # 6c. Equity curves
    eq_matrix = pd.DataFrame(index=[x["timestamp"] for x in sim_results["Buy_and_Hold"]["equity_curve"]])
    for s_name, res in sim_results.items():
        eq_matrix[s_name] = [x["equity"] for x in res["equity_curve"]]
    eq_matrix.to_csv(REPORTS_DIR / "workstream_3_forward_equity_curves.csv", index_label="timestamp")

    # 6d. Per-strategy signals
    all_signals: List[Dict[str, Any]] = []
    for res in sim_results.values():
        all_signals.extend(res["signals_log"])
    df_signals = pd.DataFrame(all_signals)
    df_signals.to_csv(REPORTS_DIR / "workstream_3_forward_signals.csv", index=False)

    # 6e. Print final report
    print("\n" + "=" * 100)
    print("HYDRA V2.4.2+ — FORWARD PORTFOLIO SIMULATION RESULTS")
    print(f"Period: {eval_dates[0].strftime('%Y-%m-%d')} to {eval_dates[-1].strftime('%Y-%m-%d')} ({len(eval_dates)} trading days)")
    print("Initial Capital: $1,000.00")
    print("Execution: 5 bps adverse slippage + $0.005/share commission ($1.00 min)")
    print("=" * 100)
    print(f"\n{'Strategy':<35} {'Final Equity':>14} {'Net P&L':>12} {'Return':>10} {'Trades':>8} "
          f"{'Win Rate':>10} {'Max DD':>10} {'Sharpe':>8} {'Open Pos':>10}")
    print("-" * 127)

    for s in summary_rows:
        open_str = "None"
        if s.get("open_position"):
            op = s["open_position"]
            open_str = f"{op['shares']}sh"
        print(f"{s['strategy']:<35} ${s['final_equity']:>12,.2f} ${s['net_profit_dollars']:>10,.2f} "
              f"{s['net_return_pct']:>8.2f}% {s['total_trades']:>7} "
              f"{s['win_rate_pct']:>8.1f}% {s['max_drawdown_pct']:>8.2f}% "
              f"{s['sharpe_ratio']:>7.3f} {open_str:>10}")

    print("-" * 127)

    # Print individual trade details for each strategy
    for s_name, res in sim_results.items():
        trades = res["trades"]
        if trades:
            print(f"\n{'-' * 100}")
            print(f"TRADE LEDGER: {s_name} ({len(trades)} closed trades)")
            print(f"{'-' * 100}")
            print(f"{'#':>4} {'Entry Date':>12} {'Entry $':>10} {'Qty':>5} {'Exit Date':>12} "
                  f"{'Exit $':>10} {'Net P&L':>12} {'Return':>9} {'Days':>5} {'Result':>8}")
            print(f"{'-' * 88}")
            for t in trades:
                result = "WIN" if t["is_winner"] else "LOSS"
                print(f"{t['trade_id']:>4} {t['entry_timestamp'][:10]:>12} ${t['entry_fill_price']:>8.2f} "
                      f"{t['entry_shares']:>5} {t['exit_timestamp'][:10]:>12} ${t['exit_fill_price']:>8.2f} "
                      f"${t['net_pnl']:>10.2f} {t['net_return_pct']:>7.2f}% {t['holding_period_bars']:>5} "
                      f"{result:>8}")

        # Report open position if any
        s = res["summary"]
        if s.get("open_position"):
            op = s["open_position"]
            print(f"\n  ** OPEN POSITION: {op['shares']} shares @ ${op['entry_price']:.2f} "
                  f"(entry {op['entry_date'][:10]}), current ${op['current_price']:.2f}, "
                  f"unrealized P&L: ${op['unrealized_pnl']:+.2f} **")

    print("\n" + "=" * 100)
    print("GOVERNANCE: These are HISTORICAL SIMULATION results, NOT live trading.")
    print("PRODUCTION INTEGRATION REMAINS BLOCKED. NO LIVE CAPITAL.")
    print("=" * 100)

    return {
        "summary": summary_rows,
        "trades_count": len(all_trades),
        "eval_bars": len(eval_dates),
        "eval_start": eval_dates[0].strftime('%Y-%m-%d'),
        "eval_end": eval_dates[-1].strftime('%Y-%m-%d'),
    }


if __name__ == "__main__":
    run_forward_simulation()
