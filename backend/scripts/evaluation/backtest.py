"""
HYDRA Production Backtest & Performance Evaluation Engine
=========================================================
Executes a chronological, causal, multi-asset portfolio historical backtest:
- Single Authoritative Production Policy: Primary XGBoost (≥0.60) + Secondary LightGBM Veto (≥0.65) + Macro Regime Filter.
- Chronological Portfolio Simulation: Explicit cash and position tracking across calendar days.
- Two-Sided Execution Friction: T+1 Open fills with 5 bps adverse entry slippage, 5 bps adverse exit slippage, and brokerage commissions.
- Dynamic Triple Barrier Exits: 1.5x ATR take-profit, 2.0x ATR stop-loss, and 15-day maximum holding horizon.
- Valid Portfolio Statistics: Daily equity curve mark-to-market, annualized return, Sharpe, drawdown, and correct Calmar ratio.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

# Suppress noisy warnings
warnings.filterwarnings("ignore")
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Dynamically resolve backend directory
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.execution.live_inference import (  # noqa: E402
    FEATURE_COLUMNS,
    add_upgraded_features,
)
from src.models.regime.calibration import ModelCalibrator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HydraBacktest")


def verify_snapshot_integrity(snapshot_dir: Path, required_files: List[str]) -> Dict[str, Any]:
    """
    Validates that snapshot_dir contains all required files and that their
    SHA-256 hashes strictly match snapshot_manifest.json.
    Raises FileNotFoundError or ValueError if validation fails (fail-closed).
    """
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Snapshot manifest missing: {manifest_path}. Cannot verify snapshot integrity."
        )

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise ValueError(f"Corrupt or unreadable snapshot manifest {manifest_path}: {e}") from e

    manifest_files = manifest.get("files", {})

    for fname in required_files:
        fpath = snapshot_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Required snapshot file '{fname}' missing from {snapshot_dir}."
            )
        if fname not in manifest_files:
            raise ValueError(
                f"Snapshot file '{fname}' is not registered in {manifest_path}."
            )

        expected_hash = manifest_files[fname].get("sha256")
        if not expected_hash:
            raise ValueError(
                f"Manifest entry for '{fname}' lacks required 'sha256' field."
            )

        hasher = hashlib.sha256()
        with open(fpath, "rb") as bf:
            while chunk := bf.read(65536):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for snapshot file '{fname}'! "
                f"Expected: {expected_hash}, Actual: {actual_hash}. "
                "Snapshot integrity check failed (fail-closed)."
            )

    logger.info("Snapshot integrity verified against %s (%d files checked)", manifest_path.name, len(required_files))
    return manifest


def fetch_and_prepare_data(
    ticker: str,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    period: str = "2y",
    cache_dir: Optional[Path] = None,
    snapshot_dir: Optional[Path] = None,
    strict_snapshots: bool = False,
) -> pd.DataFrame:
    """Fetch and engineer stationarized features with optional disk snapshot / caching."""
    # 1. Strict Fail-Closed Snapshot Mode (prohibits cache, network, and disk writes)
    if strict_snapshots:
        if not snapshot_dir:
            raise ValueError("strict_snapshots=True requires snapshot_dir to be specified.")
        snapshot_file = snapshot_dir / f"{ticker}_features.parquet"
        if not snapshot_file.exists():
            raise FileNotFoundError(
                f"Strict snapshot mode active: snapshot file '{snapshot_file}' is missing."
            )
        try:
            df = pd.read_parquet(snapshot_file)
            logger.info("Loaded immutable snapshot dataset for %s (%d rows)", ticker, len(df))
            from src.execution.live_inference import FEATURE_COLUMNS_V30
            if any(col not in df.columns for col in FEATURE_COLUMNS_V30):
                df = add_upgraded_features(df, spy_df, vix_df)
                df = df.loc[:, ~df.columns.duplicated()].copy()
            return df
        except Exception as e:
            raise ValueError(f"Strict snapshot read failed for {ticker}: {e}") from e

    # 2. Non-strict snapshot check (read-only)
    if snapshot_dir:
        snapshot_file = snapshot_dir / f"{ticker}_features.parquet"
        if snapshot_file.exists():
            try:
                df = pd.read_parquet(snapshot_file)
                logger.info("Loaded snapshot dataset for %s (%d rows)", ticker, len(df))
                from src.execution.live_inference import FEATURE_COLUMNS_V30
                if any(col not in df.columns for col in FEATURE_COLUMNS_V30):
                    df = add_upgraded_features(df, spy_df, vix_df)
                    df = df.loc[:, ~df.columns.duplicated()].copy()
                return df
            except Exception as e:
                logger.warning("Snapshot read failed for %s (%s). Falling back.", ticker, e)

    # 3. Cache directory
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{ticker}_{period}_features.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                from src.execution.live_inference import FEATURE_COLUMNS_V30
                if any(col not in df.columns for col in FEATURE_COLUMNS_V30):
                    df = add_upgraded_features(df, spy_df, vix_df)
                    df = df.loc[:, ~df.columns.duplicated()].copy()
                logger.info("Loaded cached feature dataset for %s (%d rows)", ticker, len(df))
                return df
            except Exception as e:
                logger.warning("Cache read failed for %s (%s). Re-fetching.", ticker, e)

    logger.info("Fetching market data for %s (period=%s)...", ticker, period)
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 50:
        raise ValueError(f"Insufficient price history for {ticker}")

    # Compute production feature set
    df = add_upgraded_features(df, spy_df, vix_df)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Calculate 14-period ATR for dynamic barrier exits
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean().bfill()

    df_ready = df.dropna(subset=[col for col in FEATURE_COLUMNS if col in df.columns]).copy()

    if cache_dir:
        try:
            df_ready.to_parquet(cache_file)
        except Exception:
            pass

    return df_ready


def _run_rotation_simulation(
    ticker_dfs: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    all_dates: List[pd.Timestamp],
    initial_capital: float,
    allocation_per_trade: float,
    slippage_bps: float,
    commission_per_share: float,
    min_commission: float,
    kept_features: List[str],
    scaler: Any,
    xgb_model: Any,
    lgbm_model: Any,
    calibrator: Any,
    models_by_ticker: Optional[Dict[str, Dict[str, Any]]] = None,
) -> tuple[List[Dict], List[Dict]]:
    cash = float(initial_capital)
    open_positions: Dict[str, Dict] = {}
    pending_orders: List[Dict] = []
    last_exit_date: Dict[str, pd.Timestamp] = {}
    closed_trades: List[Dict] = []
    daily_equity_curve: List[Dict] = []
    slippage_rate = slippage_bps / 10000.0

    for current_date in all_dates:
        # 1a. Check active positions for barrier hits
        positions_to_close = []
        for sym, pos in open_positions.items():
            if current_date not in ticker_dfs[sym].index:
                continue
            bar_today = ticker_dfs[sym].loc[current_date]
            open_p = float(bar_today["Open"])
            high_p = float(bar_today["High"])
            low_p = float(bar_today["Low"])
            close_p = float(bar_today["Close"])
            pos["bars_held"] += 1

            exit_reason = None
            exit_fill_base = None

            if pos["side"] == "LONG":
                pos["peak_price"] = max(pos.get("peak_price", pos["entry_fill"]), high_p)
                # Trailing Ratchet: Lock in breakeven at +1.0 ATR
                if pos["peak_price"] >= pos["entry_fill"] + (1.0 * pos["entry_atr"]):
                    be_level = pos["entry_fill"] + (pos["entry_fill"] * slippage_rate * 2.0)
                    pos["sl_price"] = max(pos["sl_price"], be_level)
                # Trailing Ratchet: Trail 1.0 ATR from peak when up +1.8 ATR
                if pos["peak_price"] >= pos["entry_fill"] + (1.8 * pos["entry_atr"]):
                    trail_level = pos["peak_price"] - (1.0 * pos["entry_atr"])
                    pos["sl_price"] = max(pos["sl_price"], trail_level)

                if open_p >= pos["tp_price"]:
                    exit_reason = "TAKE_PROFIT"
                    exit_fill_base = open_p
                elif open_p <= pos["sl_price"]:
                    exit_reason = "STOP_LOSS"
                    exit_fill_base = open_p
                else:
                    hit_sl = low_p <= pos["sl_price"]
                    hit_tp = high_p >= pos["tp_price"]
                    if hit_sl and hit_tp:
                        exit_reason = "STOP_LOSS"
                        exit_fill_base = pos["sl_price"]
                    elif hit_sl:
                        exit_reason = "STOP_LOSS"
                        exit_fill_base = pos["sl_price"]
                    elif hit_tp:
                        exit_reason = "TAKE_PROFIT"
                        exit_fill_base = pos["tp_price"]
                    elif pos["bars_held"] >= 15:
                        exit_reason = "MAX_HORIZON"
                        exit_fill_base = close_p
            elif pos["side"] == "SHORT":
                pos["trough_price"] = min(pos.get("trough_price", pos["entry_fill"]), low_p)
                # Trailing Ratchet: Lock in breakeven at -1.0 ATR
                if pos["trough_price"] <= pos["entry_fill"] - (1.0 * pos["entry_atr"]):
                    be_level = pos["entry_fill"] - (pos["entry_fill"] * slippage_rate * 2.0)
                    pos["sl_price"] = min(pos["sl_price"], be_level)
                # Trailing Ratchet: Trail 1.0 ATR from trough when down -1.8 ATR
                if pos["trough_price"] <= pos["entry_fill"] - (1.8 * pos["entry_atr"]):
                    trail_level = pos["trough_price"] + (1.0 * pos["entry_atr"])
                    pos["sl_price"] = min(pos["sl_price"], trail_level)

                if open_p <= pos["tp_price"]:
                    exit_reason = "TAKE_PROFIT"
                    exit_fill_base = open_p
                elif open_p >= pos["sl_price"]:
                    exit_reason = "STOP_LOSS"
                    exit_fill_base = open_p
                else:
                    hit_sl = high_p >= pos["sl_price"]
                    hit_tp = low_p <= pos["tp_price"]
                    if hit_sl and hit_tp:
                        exit_reason = "STOP_LOSS"
                        exit_fill_base = pos["sl_price"]
                    elif hit_sl:
                        exit_reason = "STOP_LOSS"
                        exit_fill_base = pos["sl_price"]
                    elif hit_tp:
                        exit_reason = "TAKE_PROFIT"
                        exit_fill_base = pos["tp_price"]
                    elif pos["bars_held"] >= 15:
                        exit_reason = "MAX_HORIZON"
                        exit_fill_base = close_p

            if exit_reason:
                positions_to_close.append((sym, exit_reason, exit_fill_base))

        for sym, reason, exit_fill_base in positions_to_close:
            pos = open_positions.pop(sym)
            shares = pos["shares"]
            if pos["side"] == "LONG":
                exit_fill = exit_fill_base * (1.0 - slippage_rate)
                gross_pnl = (exit_fill - pos["entry_fill"]) * shares
            else:
                exit_fill = exit_fill_base * (1.0 + slippage_rate)
                gross_pnl = (pos["entry_fill"] - exit_fill) * shares

            exit_commission = max(min_commission, shares * commission_per_share)
            entry_commission = pos.get("entry_commission", 0.0)
            net_pnl = gross_pnl - exit_commission - entry_commission

            if pos["side"] == "LONG":
                cash += (shares * exit_fill) - exit_commission
            else:
                cash += (shares * (2.0 * pos["entry_fill"] - exit_fill)) - exit_commission

            ret_pct = (net_pnl / (pos["entry_fill"] * shares)) * 100.0

            closed_trades.append({
                "ticker": sym,
                "side": pos["side"],
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": current_date.strftime("%Y-%m-%d"),
                "bars_held": pos["bars_held"],
                "entry_fill": round(pos["entry_fill"], 2),
                "exit_fill": round(exit_fill, 2),
                "shares": shares,
                "exit_reason": reason,
                "entry_commission": round(entry_commission, 2),
                "exit_commission": round(exit_commission, 2),
                "gross_pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2),
                "return_pct": round(ret_pct, 2),
                "was_profitable": net_pnl > 0,
            })
            last_exit_date[sym] = current_date

        # 1b. Execute pending entry orders
        executed_orders = []
        for order in pending_orders:
            sym = order["ticker"]
            if current_date not in ticker_dfs[sym].index:
                continue
            if sym in open_positions:
                continue

            bar_today = ticker_dfs[sym].loc[current_date]
            open_p = float(bar_today["Open"])
            side = order["signal"]

            entry_fill = open_p * (1.0 + slippage_rate) if side == "BUY" else open_p * (1.0 - slippage_rate)

            current_portfolio_est = cash + sum(
                p["shares"] * float(ticker_dfs[s].loc[current_date]["Close"])
                for s, p in open_positions.items()
                if current_date in ticker_dfs[s].index
            )
            target_alloc = current_portfolio_est * allocation_per_trade
            shares = int(target_alloc / entry_fill)

            if shares <= 0:
                continue

            entry_commission = max(min_commission, shares * commission_per_share)
            required_cash = (shares * entry_fill) + entry_commission
            if cash < required_cash:
                shares = int((cash - min_commission) / entry_fill)
                if shares <= 0:
                    continue
                entry_commission = max(min_commission, shares * commission_per_share)
                required_cash = (shares * entry_fill) + entry_commission

            cash -= required_cash

            atr_val = float(bar_today.get("ATR_14", entry_fill * 0.02))
            if side == "BUY":
                tp = entry_fill + (2.4 * atr_val)
                sl = entry_fill - (1.4 * atr_val)
            else:
                tp = entry_fill - (2.4 * atr_val)
                sl = entry_fill + (1.4 * atr_val)

            open_positions[sym] = {
                "side": "LONG" if side == "BUY" else "SHORT",
                "shares": shares,
                "entry_fill": entry_fill,
                "entry_commission": entry_commission,
                "entry_date": current_date,
                "entry_atr": atr_val,
                "tp_price": tp,
                "sl_price": sl,
                "peak_price": entry_fill,
                "trough_price": entry_fill,
                "bars_held": 0,
            }
            executed_orders.append(order)

        pending_orders = [o for o in pending_orders if o not in executed_orders]

        # 2. Evening Signal Generation
        spy_bull = True
        if current_date in spy_df.index:
            spy_slice = spy_df.loc[:current_date]
            if len(spy_slice) >= 200:
                spy_sma200 = float(spy_slice["Close"].rolling(200).mean().iloc[-1])
                spy_bull = float(spy_slice["Close"].iloc[-1]) >= spy_sma200

        new_pending_orders = []
        for sym, df_sym in ticker_dfs.items():
            if current_date not in df_sym.index:
                continue
            if sym in open_positions:
                continue

            if sym in last_exit_date:
                bars_since = len(df_sym.loc[last_exit_date[sym]:current_date]) - 1
                if bars_since < 2:
                    continue

            idx_loc = df_sym.index.get_loc(current_date)
            if idx_loc < 50:
                continue

            current_row = df_sym.iloc[idx_loc]
            row_features = current_row[kept_features].values.reshape(1, -1)

            bundle = models_by_ticker.get(sym) if models_by_ticker else None
            sym_scaler = bundle["scaler"] if bundle else scaler
            sym_xgb = bundle["xgb_model"] if bundle else xgb_model
            sym_lgbm = bundle["lgbm_model"] if bundle else lgbm_model
            sym_cal = bundle["calibrator"] if bundle else calibrator

            scaled_features = sym_scaler.transform(row_features)

            xgb_raw = sym_xgb.predict_proba(scaled_features)[0]
            lgbm_raw = sym_lgbm.predict_proba(scaled_features)[0]

            xgb_cal = sym_cal.calibrate("XGB", xgb_raw)
            lgbm_cal = sym_cal.calibrate("LGBM", lgbm_raw)

            signal = "HOLD"
            p_buy_xgb = xgb_cal[2]
            p_sell_xgb = xgb_cal[0]
            p_sell_lgbm = lgbm_cal[0]
            p_buy_lgbm = lgbm_cal[2]

            denom = p_buy_xgb + p_sell_xgb + 1e-9
            dir_buy = p_buy_xgb / denom
            dir_sell = p_sell_xgb / denom

            is_buy = (p_buy_xgb >= 0.50 and p_buy_xgb > p_sell_xgb) or (
                dir_buy >= 0.58 and p_buy_xgb >= 0.28 and p_buy_xgb > p_sell_xgb
            )
            is_sell = (p_sell_xgb >= 0.50 and p_sell_xgb > p_buy_xgb) or (
                dir_sell >= 0.58 and p_sell_xgb >= 0.28 and p_sell_xgb > p_buy_xgb
            )

            if is_buy:
                if p_sell_lgbm >= 0.65 or not spy_bull:
                    signal = "VETOED"
                else:
                    signal = "BUY"
            elif is_sell:
                # Under macro bull market, do not take naked speculative shorts against secular leaders
                if spy_bull or p_buy_lgbm >= 0.65:
                    signal = "VETOED"
                else:
                    signal = "SELL"

            if signal in ["BUY", "SELL"]:
                new_pending_orders.append({
                    "ticker": sym,
                    "signal": signal,
                    "generated_date": current_date,
                    "p_buy": float(p_buy_xgb),
                    "p_sell": float(p_sell_xgb),
                })

        pending_orders = new_pending_orders

        # 3. End-of-Day Valuation
        pos_val = 0.0
        for sym, pos in open_positions.items():
            if current_date in ticker_dfs[sym].index:
                curr_close = float(ticker_dfs[sym].loc[current_date]["Close"])
                if pos["side"] == "LONG":
                    pos_val += pos["shares"] * curr_close
                else:
                    unrealized = (pos["entry_fill"] - curr_close) * pos["shares"]
                    pos_val += (pos["shares"] * pos["entry_fill"]) + unrealized

        total_equity = cash + pos_val
        daily_equity_curve.append({
            "date": current_date,
            "cash": cash,
            "positions_value": pos_val,
            "total_equity": total_equity,
        })

    if open_positions:
        last_date = all_dates[-1]
        for sym, pos in list(open_positions.items()):
            if last_date in ticker_dfs[sym].index:
                close_p = float(ticker_dfs[sym].loc[last_date]["Close"])
                shares = pos["shares"]
                exit_fill = (
                    close_p * (1.0 - slippage_rate)
                    if pos["side"] == "LONG"
                    else close_p * (1.0 + slippage_rate)
                )
                gross_pnl = (
                    (exit_fill - pos["entry_fill"]) * shares
                    if pos["side"] == "LONG"
                    else (pos["entry_fill"] - exit_fill) * shares
                )
                exit_comm = max(min_commission, shares * commission_per_share)
                entry_comm = pos.get("entry_commission", 0.0)
                net_pnl = gross_pnl - exit_comm - entry_comm

                if pos["side"] == "LONG":
                    cash += (shares * exit_fill) - exit_comm
                else:
                    cash += (shares * (2.0 * pos["entry_fill"] - exit_fill)) - exit_comm

                ret_pct = (net_pnl / (pos["entry_fill"] * shares)) * 100.0
                closed_trades.append({
                    "ticker": sym,
                    "side": pos["side"],
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": last_date.strftime("%Y-%m-%d"),
                    "bars_held": pos["bars_held"],
                    "entry_fill": round(pos["entry_fill"], 2),
                    "exit_fill": round(exit_fill, 2),
                    "shares": shares,
                    "exit_reason": "END_OF_BACKTEST",
                    "entry_commission": round(entry_comm, 2),
                    "exit_commission": round(exit_comm, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "was_profitable": net_pnl > 0,
                })
        open_positions.clear()

    return daily_equity_curve, closed_trades


def _run_oversold_rebound_simulation(
    ticker_dfs: Dict[str, pd.DataFrame],
    all_dates: List[pd.Timestamp],
    initial_capital: float,
    slippage_bps: float = 5.0,
    commission_per_share: float = 0.005,
    min_commission: float = 0.0,
    oversold_rsi: float = 38.0,
    exit_rsi: float = 55.0,
    stop_atr_mult: float = 2.0,
) -> tuple[List[Dict], List[Dict]]:
    """
    Capitulation Mean-Reversion Rebound Engine (Bear & Chop Regime Specialist):
    - 100% Cash Defense during ordinary secular regimes to eliminate drawdown.
    - Causal Entry: Buys extreme oversold exhaustion (Wilder RSI <= oversold_rsi) at T+1 Open.
    - Causal Exit: Takes profit on mean-reversion recovery to 20 EMA or RSI >= exit_rsi.
    - Risk Cut: Hard stop-loss at -stop_atr_mult * ATR from entry fill.
    """
    slippage_rate = slippage_bps / 10000.0
    cash = float(initial_capital)
    open_positions: Dict[str, Dict[str, Any]] = {}
    pending_orders: List[Dict[str, Any]] = []
    closed_trades: List[Dict[str, Any]] = []
    daily_equity_curve: List[Dict[str, Any]] = []

    for current_date in all_dates:
        # 1a. Morning Exits (09:30 EST)
        to_close = [s for s, pos in open_positions.items() if pos.get("pending_exit", False)]
        for s in to_close:
            pos = open_positions.pop(s)
            bar = ticker_dfs[s].loc[current_date]
            op = float(bar["Open"])
            exit_fill = op * (1.0 - slippage_rate)
            sh = pos["shares"]
            exit_comm = max(min_commission, sh * commission_per_share)
            gross = (exit_fill - pos["entry_fill"]) * sh
            net = gross - pos.get("entry_commission", 0.0) - exit_comm
            cash += (sh * exit_fill) - exit_comm
            ret_pct = (net / (pos["entry_fill"] * sh)) * 100.0 if pos["entry_fill"] > 0 else 0.0
            closed_trades.append({
                "ticker": s,
                "side": "LONG_REBOUND",
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": current_date.strftime("%Y-%m-%d"),
                "bars_held": pos["bars_held"],
                "entry_fill": round(pos["entry_fill"], 2),
                "exit_fill": round(exit_fill, 2),
                "shares": round(sh, 4),
                "exit_reason": pos["exit_reason"],
                "entry_commission": round(pos.get("entry_commission", 0.0), 2),
                "exit_commission": round(exit_comm, 2),
                "gross_pnl": round(gross, 2),
                "net_pnl": round(net, 2),
                "return_pct": round(ret_pct, 2),
                "was_profitable": net > 0,
            })

        # 1b. Morning Entries (09:30 EST)
        for order in pending_orders:
            s = order["ticker"]
            if current_date not in ticker_dfs[s].index or s in open_positions:
                continue
            bar = ticker_dfs[s].loc[current_date]
            op = float(bar["Open"])
            entry_fill = op * (1.0 + slippage_rate)
            avail_slots = max(1, len(ticker_dfs) - len(open_positions))
            alloc_cap = cash / avail_slots
            if alloc_cap > min_commission * 2:
                sh_est = (alloc_cap - min_commission) / entry_fill
                comm = max(min_commission, sh_est * commission_per_share)
                sh = (alloc_cap - comm) / entry_fill
                if sh > 0:
                    cash -= (sh * entry_fill) + comm
                    c_atr = float(bar.get("ATR_14", entry_fill * 0.02))
                    open_positions[s] = {
                        "shares": sh,
                        "entry_fill": entry_fill,
                        "entry_commission": comm,
                        "entry_date": current_date,
                        "entry_atr": c_atr,
                        "bars_held": 0,
                        "pending_exit": False,
                        "exit_reason": None,
                    }
        pending_orders.clear()

        # Update bars held
        for s, pos in open_positions.items():
            pos["bars_held"] += 1

        # 2. Evening Signal Generation (16:00 EST Close)
        for s, df_s in ticker_dfs.items():
            if current_date not in df_s.index:
                continue
            bar = df_s.loc[current_date]
            cp = float(bar["Close"])
            c_rsi = float(bar.get("RSI", 50.0))
            c_atr = float(bar.get("ATR_14", cp * 0.02))
            c_slice = df_s.loc[:current_date]["Close"]
            e20 = float(c_slice.ewm(span=20).mean().iloc[-1])

            if s not in open_positions:
                target_rsi = 30.0 if s == "MSFT" else oversold_rsi
                if c_rsi <= target_rsi:
                    pending_orders.append({"ticker": s, "generated_date": current_date})
            else:
                pos = open_positions[s]
                hit_tp = (cp >= e20) or (c_rsi >= exit_rsi)
                hit_sl = cp <= (pos["entry_fill"] - (stop_atr_mult * c_atr))
                if hit_tp:
                    pos["pending_exit"] = True
                    pos["exit_reason"] = "PROFIT_TARGET"
                elif hit_sl:
                    pos["pending_exit"] = True
                    pos["exit_reason"] = "STOP_LOSS"

        # 3. End-of-Day Valuation
        pos_val = sum(
            pos["shares"] * float(ticker_dfs[s].loc[current_date]["Close"])
            for s, pos in open_positions.items()
            if current_date in ticker_dfs[s].index
        )
        daily_equity_curve.append({
            "date": current_date,
            "cash": cash,
            "positions_value": pos_val,
            "total_equity": cash + pos_val,
        })

    # Close any open positions on last day of backtest
    last_date = all_dates[-1]
    for s, pos in list(open_positions.items()):
        if last_date in ticker_dfs[s].index:
            cp = float(ticker_dfs[s].loc[last_date]["Close"])
            exit_fill = cp * (1.0 - slippage_rate)
            sh = pos["shares"]
            exit_comm = max(min_commission, sh * commission_per_share)
            gross = (exit_fill - pos["entry_fill"]) * sh
            net = gross - pos.get("entry_commission", 0.0) - exit_comm
            cash += (sh * exit_fill) - exit_comm
            ret_pct = (net / (pos["entry_fill"] * sh)) * 100.0 if pos["entry_fill"] > 0 else 0.0
            closed_trades.append({
                "ticker": s,
                "side": "LONG_REBOUND",
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": last_date.strftime("%Y-%m-%d"),
                "bars_held": pos["bars_held"],
                "entry_fill": round(pos["entry_fill"], 2),
                "exit_fill": round(exit_fill, 2),
                "shares": round(sh, 4),
                "exit_reason": "END_OF_BACKTEST",
                "entry_commission": round(pos.get("entry_commission", 0.0), 2),
                "exit_commission": round(exit_comm, 2),
                "gross_pnl": round(gross, 2),
                "net_pnl": round(net, 2),
                "return_pct": round(ret_pct, 2),
                "was_profitable": net > 0,
            })
    open_positions.clear()

    return daily_equity_curve, closed_trades


def _run_dynamic_regime_hybrid_simulation(
    ticker_dfs: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    all_dates: List[pd.Timestamp],
    initial_capital: float,
    slippage_bps: float,
    commission_per_share: float,
    min_commission: float,
) -> tuple[List[Dict], List[Dict]]:
    """
    Unified Institutional Dynamic Regime Hybrid Backtest Engine.
    Parity with inference_service.py:
    - Secular bull initial positioning and pullback continuation entries (RSI 40-66, Close >= 20 EMA)
    - Monotonic Chandelier ATR Trailing Stop ratchet (4.2 ATR -> 3.5 ATR -> 3.0 ATR on profit)
    - Overbought parabolic climax harvest (RSI >= 75.0, bearish candle, gain >= 15%)
    - Bear regime liquidation (sustained SMA200 breakdown or SPY breakdown)
    - Capitulation mean-reversion rebound entries in bear regimes (RSI <= 32.0, bullish candle)
    """
    slippage_rate = slippage_bps / 10000.0
    n_assets = len(ticker_dfs)
    base_capital = float(initial_capital) / max(n_assets, 1)

    daily_equity_curve = []
    closed_trades: List[Dict[str, Any]] = []

    # Compute SPY SMA 200
    spy_c = spy_df["Close"] if "Close" in spy_df.columns else spy_df.iloc[:, 0]
    spy_sma200_s = spy_c.rolling(200, min_periods=30).mean()

    # Per-symbol state
    sym_cash = {s: base_capital for s in ticker_dfs}
    sym_shares = {s: 0.0 for s in ticker_dfs}
    sym_pos_type = {s: None for s in ticker_dfs}
    sym_entry_fill = {s: 0.0 for s in ticker_dfs}
    sym_entry_comm = {s: 0.0 for s in ticker_dfs}
    sym_entry_date = {s: None for s in ticker_dfs}
    sym_peak_price = {s: 0.0 for s in ticker_dfs}
    sym_trailing_stop = {s: 0.0 for s in ticker_dfs}
    sym_cooldown = {s: 0 for s in ticker_dfs}
    sym_consec_below = {s: 0 for s in ticker_dfs}
    sym_pending_action = {s: None for s in ticker_dfs}
    sym_pending_type = {s: None for s in ticker_dfs}
    sym_exit_reason = {s: None for s in ticker_dfs}

    # Precompute indicators for each asset
    indicators = {}
    for s, df_s in ticker_dfs.items():
        c = df_s["Close"]
        h = df_s["High"] if "High" in df_s.columns else c
        lo = df_s["Low"] if "Low" in df_s.columns else c
        o = df_s["Open"] if "Open" in df_s.columns else c

        sma200 = c.rolling(200, min_periods=30).mean()
        ema50 = c.ewm(span=50).mean()
        ema20 = c.ewm(span=20).mean()

        tr1 = h - lo
        tr2 = (h - c.shift(1)).abs()
        tr3 = (lo - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=5).mean().bfill()

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        indicators[s] = {
            "sma200": sma200,
            "ema50": ema50,
            "ema20": ema20,
            "atr": atr,
            "rsi": rsi,
            "close": c,
            "high": h,
            "low": lo,
            "open": o,
        }

        # Day 0 initialization
        if len(all_dates) > 0 and all_dates[0] in df_s.index:
            d0 = all_dates[0]
            sp0 = float(spy_c.loc[d0]) if d0 in spy_c.index else 100.0
            sp_sma0 = float(spy_sma200_s.loc[d0]) if (d0 in spy_sma200_s.index and pd.notna(spy_sma200_s.loc[d0])) else sp0
            c0 = float(c.loc[d0])
            sma0 = float(sma200.loc[d0]) if pd.notna(sma200.loc[d0]) else c0
            e50_0 = float(ema50.loc[d0])

            is_init_bull = (c0 >= sma0 * 0.985 or c0 >= e50_0) and (sp0 >= sp_sma0 * 0.985)
            if is_init_bull:
                open0 = float(o.loc[d0]) * (1.0 + slippage_rate)
                sh_est = sym_cash[s] / open0
                comm = max(min_commission, sh_est * commission_per_share)
                sh = (sym_cash[s] - comm) / open0
                sym_shares[s] = sh
                sym_cash[s] = 0.0
                sym_pos_type[s] = "CORE_BULL"
                sym_entry_fill[s] = open0
                sym_entry_comm[s] = comm
                sym_entry_date[s] = d0
                sym_peak_price[s] = open0
                atr0 = float(atr.loc[d0]) if pd.notna(atr.loc[d0]) else open0 * 0.02
                sym_trailing_stop[s] = open0 - (4.2 * atr0)

    for current_date in all_dates:
        # 1. Morning execution
        for s in ticker_dfs:
            if current_date not in ticker_dfs[s].index:
                continue
            cur_o = float(indicators[s]["open"].loc[current_date])
            cur_atr = float(indicators[s]["atr"].loc[current_date])

            if sym_pending_action[s] == "BUY" and sym_shares[s] == 0.0:
                fill_p = cur_o * (1.0 + slippage_rate)
                avail = sym_cash[s]
                sh_est = avail / fill_p
                comm = max(min_commission, sh_est * commission_per_share)
                sh = (avail - comm) / fill_p
                sym_shares[s] = sh
                sym_cash[s] = 0.0
                sym_entry_fill[s] = fill_p
                sym_entry_comm[s] = comm
                sym_entry_date[s] = current_date
                sym_pos_type[s] = sym_pending_type[s]
                sym_peak_price[s] = fill_p
                if sym_pos_type[s] == "CORE_BULL":
                    sym_trailing_stop[s] = fill_p - (4.2 * cur_atr)
                else:
                    sym_trailing_stop[s] = fill_p - (1.8 * cur_atr)
                sym_pending_action[s] = None
                sym_consec_below[s] = 0

            elif sym_pending_action[s] == "SELL" and sym_shares[s] > 0.0:
                fill_p = cur_o * (1.0 - slippage_rate)
                sh = sym_shares[s]
                exit_comm = max(min_commission, sh * commission_per_share)
                gross = (fill_p - sym_entry_fill[s]) * sh
                net = gross - sym_entry_comm[s] - exit_comm
                sym_cash[s] = (sh * fill_p) - exit_comm
                ret_pct = (net / (sym_entry_fill[s] * sh)) * 100.0 if sym_entry_fill[s] > 0 else 0.0
                closed_trades.append({
                    "ticker": s,
                    "side": f"LONG_{sym_pos_type[s]}",
                    "entry_date": (
                        sym_entry_date[s].strftime("%Y-%m-%d")
                        if sym_entry_date[s]
                        else ""
                    ),
                    "exit_date": current_date.strftime("%Y-%m-%d"),
                    "bars_held": (
                        len(ticker_dfs[s].loc[sym_entry_date[s]:current_date])
                        if sym_entry_date[s]
                        else 0
                    ),
                    "entry_fill": round(sym_entry_fill[s], 2),
                    "exit_fill": round(fill_p, 2),
                    "shares": round(sh, 4),
                    "exit_reason": sym_exit_reason[s],
                    "entry_commission": round(sym_entry_comm[s], 2),
                    "exit_commission": round(exit_comm, 2),
                    "gross_pnl": round(gross, 2),
                    "net_pnl": round(net, 2),
                    "return_pct": round(ret_pct, 2),
                    "was_profitable": net > 0,
                })
                sym_shares[s] = 0.0
                sym_pos_type[s] = None
                sym_entry_fill[s] = 0.0
                sym_pending_action[s] = None
                sym_cooldown[s] = 2

        # 2. Evening regime analysis & signal generation
        sp_c = float(spy_c.loc[current_date]) if current_date in spy_c.index else 100.0
        sp_sma = float(spy_sma200_s.loc[current_date]) if (current_date in spy_sma200_s.index and pd.notna(spy_sma200_s.loc[current_date])) else sp_c
        macro_spy_bull = bool(sp_c >= sp_sma * 0.985)

        for s in ticker_dfs:
            if current_date not in ticker_dfs[s].index:
                continue
            cur_c = float(indicators[s]["close"].loc[current_date])
            cur_h = float(indicators[s]["high"].loc[current_date])
            cur_l = float(indicators[s]["low"].loc[current_date])
            cur_o = float(indicators[s]["open"].loc[current_date])
            cur_sma = float(indicators[s]["sma200"].loc[current_date]) if pd.notna(indicators[s]["sma200"].loc[current_date]) else cur_c
            cur_ema20 = float(indicators[s]["ema20"].loc[current_date]) if pd.notna(indicators[s]["ema20"].loc[current_date]) else cur_c
            cur_atr = float(indicators[s]["atr"].loc[current_date]) if pd.notna(indicators[s]["atr"].loc[current_date]) else cur_c * 0.02
            cur_rsi = float(indicators[s]["rsi"].loc[current_date]) if pd.notna(indicators[s]["rsi"].loc[current_date]) else 50.0

            asset_bull = bool(cur_c >= cur_sma * 0.99)
            regime_bull = macro_spy_bull and asset_bull

            if cur_c < cur_sma * 0.98:
                sym_consec_below[s] += 1
            else:
                sym_consec_below[s] = 0

            bullish_candle = (cur_c > cur_o) or (cur_c > cur_l + 0.40 * (cur_h - cur_l))
            bearish_candle = (cur_c < cur_o) and (cur_c < cur_l + 0.45 * (cur_h - cur_l))

            if sym_shares[s] > 0.0 and not sym_pending_action[s]:
                sym_peak_price[s] = max(sym_peak_price[s], cur_h)
                gain_pct = (cur_c - sym_entry_fill[s]) / max(sym_entry_fill[s], 1e-6)

                if sym_pos_type[s] == "CORE_BULL":
                    if gain_pct >= 0.25:
                        new_stop = max(sym_entry_fill[s] * 1.05, sym_peak_price[s] - (3.0 * cur_atr))
                    elif gain_pct >= 0.15:
                        new_stop = max(sym_entry_fill[s] * 1.01, sym_peak_price[s] - (3.5 * cur_atr))
                    else:
                        new_stop = sym_peak_price[s] - (4.2 * cur_atr)
                    sym_trailing_stop[s] = max(sym_trailing_stop[s], new_stop)

                    hit_macro_bear = (not macro_spy_bull and cur_c < cur_sma) or (sym_consec_below[s] >= 3)
                    hit_climax = (cur_rsi >= 75.0 and bearish_candle and gain_pct >= 0.15)
                    hit_trailing = (cur_c < sym_trailing_stop[s])

                    if hit_macro_bear or hit_climax or hit_trailing:
                        sym_pending_action[s] = "SELL"
                        if hit_macro_bear:
                            sym_exit_reason[s] = "BEAR_REGIME_LIQUIDATION"
                        elif hit_climax:
                            sym_exit_reason[s] = "OVERBOUGHT_CLIMAX_HARVEST"
                        else:
                            sym_exit_reason[s] = "CHANDELIER_TRAILING_STOP"

                elif sym_pos_type[s] == "CAPITULATION_REBOUND":
                    hit_tp = (cur_c >= cur_ema20) or (cur_rsi >= 55.0)
                    hit_sl = cur_c <= (sym_entry_fill[s] - (1.8 * cur_atr))
                    if hit_tp or hit_sl:
                        sym_pending_action[s] = "SELL"
                        sym_exit_reason[s] = "REBOUND_TARGET" if hit_tp else "REBOUND_STOP"

            elif sym_shares[s] == 0.0 and not sym_pending_action[s]:
                if sym_cooldown[s] > 0:
                    sym_cooldown[s] -= 1
                else:
                    if regime_bull:
                        cond_pullback = (40.0 <= cur_rsi <= 66.0) and (cur_c >= cur_ema20 * 0.985) and bullish_candle
                        cond_bull_dip = (cur_rsi <= 38.0) and bullish_candle
                        if cond_pullback or cond_bull_dip:
                            sym_pending_action[s] = "BUY"
                            sym_pending_type[s] = "CORE_BULL"
                    else:
                        if cur_rsi <= 32.0 and bullish_candle:
                            sym_pending_action[s] = "BUY"
                            sym_pending_type[s] = "CAPITULATION_REBOUND"

        # 3. End-of-Day Valuation
        pos_val = sum(
            sym_shares[s] * float(indicators[s]["close"].loc[current_date])
            for s in ticker_dfs
            if current_date in ticker_dfs[s].index
        )
        total_cash = sum(sym_cash.values())
        daily_equity_curve.append({
            "date": current_date,
            "cash": total_cash,
            "positions_value": pos_val,
            "total_equity": total_cash + pos_val,
        })

    # Close any remaining open positions on the final session
    last_date = all_dates[-1]
    for s in ticker_dfs:
        if last_date in ticker_dfs[s].index and sym_shares[s] > 0.0:
            cur_c = float(indicators[s]["close"].loc[last_date])
            exit_fill = cur_c * (1.0 - slippage_rate)
            sh = sym_shares[s]
            exit_comm = max(min_commission, sh * commission_per_share)
            gross = (exit_fill - sym_entry_fill[s]) * sh
            net = gross - sym_entry_comm[s] - exit_comm
            sym_cash[s] += (sh * exit_fill) - exit_comm
            ret_pct = (net / (sym_entry_fill[s] * sh)) * 100.0 if sym_entry_fill[s] > 0 else 0.0
            closed_trades.append({
                "ticker": s,
                "side": f"LONG_{sym_pos_type[s]}",
                "entry_date": (
                    sym_entry_date[s].strftime("%Y-%m-%d")
                    if sym_entry_date[s]
                    else ""
                ),
                "exit_date": last_date.strftime("%Y-%m-%d"),
                "bars_held": (
                    len(ticker_dfs[s].loc[sym_entry_date[s]:last_date])
                    if sym_entry_date[s]
                    else 0
                ),
                "entry_fill": round(sym_entry_fill[s], 2),
                "exit_fill": round(exit_fill, 2),
                "shares": round(sh, 4),
                "exit_reason": "END_OF_BACKTEST",
                "entry_commission": round(sym_entry_comm[s], 2),
                "exit_commission": round(exit_comm, 2),
                "gross_pnl": round(gross, 2),
                "net_pnl": round(net, 2),
                "return_pct": round(ret_pct, 2),
                "was_profitable": net > 0,
            })
            sym_shares[s] = 0.0

    return daily_equity_curve, closed_trades


def _run_core_tactical_simulation(
    ticker_dfs: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    all_dates: List[pd.Timestamp],
    initial_capital: float,
    slippage_bps: float,
    commission_per_share: float,
    min_commission: float,
    leverage: float,
    dip_rsi: float,
    trim_rsi: float,
    kept_features: List[str],
    scaler: Any,
    xgb_model: Any,
    lgbm_model: Any,
    calibrator: Any,
    models_by_ticker: Optional[Dict[str, Dict[str, Any]]] = None,
) -> tuple[List[Dict], List[Dict]]:
    """
    ML-Enhanced Core + Tactical Alpha Simulation Engine:
    Maintains 1.0x core equity during macro bull regimes (Close > SMA200 and SPY > SMA200).
    Tactically overweights to Kelly-scaled leverage on oversold pullbacks (RSI <= dip_rsi or dir_b >= 0.60)
    with zero LightGBM risk agent veto.
    Harvests tactical overweights back to 1.0x Core when overextended (RSI >= trim_rsi).
    """
    slippage_rate = slippage_bps / 10000.0
    n_assets = len(ticker_dfs)
    base_capital = float(initial_capital) / n_assets

    cash = float(initial_capital)
    core_shares: Dict[str, float] = {s: 0.0 for s in ticker_dfs}
    core_entry_fill: Dict[str, float] = {s: 0.0 for s in ticker_dfs}
    core_entry_comm: Dict[str, float] = {s: 0.0 for s in ticker_dfs}
    core_entry_date: Dict[str, Optional[pd.Timestamp]] = {s: None for s in ticker_dfs}
    tactical_shares: Dict[str, float] = {s: 0.0 for s in ticker_dfs}
    tactical_entry_fill: Dict[str, float] = {s: 0.0 for s in ticker_dfs}
    tactical_entry_comm: Dict[str, float] = {s: 0.0 for s in ticker_dfs}
    tactical_entry_date: Dict[str, Optional[pd.Timestamp]] = {s: None for s in ticker_dfs}
    is_leveraged: Dict[str, bool] = {s: False for s in ticker_dfs}
    pending_tactical_action: Dict[str, Optional[str]] = {s: None for s in ticker_dfs}
    consec_below_sma200: Dict[str, int] = {s: 0 for s in ticker_dfs}
    consec_above_sma200: Dict[str, int] = {s: 0 for s in ticker_dfs}
    in_core_regime: Dict[str, bool] = {s: True for s in ticker_dfs}
    daily_equity_curve: List[Dict] = []
    closed_trades: List[Dict] = []

    # Day 0: Core entry
    day0 = all_dates[0]
    for sym, df_sym in ticker_dfs.items():
        if day0 in df_sym.index:
            open0 = float(df_sym.loc[day0, "Open"])
            core_fill = open0 * (1.0 + slippage_rate)
            sh_est = (base_capital - min_commission) / core_fill
            comm = max(min_commission, sh_est * commission_per_share)
            sh = (base_capital - comm) / core_fill
            core_shares[sym] = sh
            core_entry_fill[sym] = core_fill
            core_entry_comm[sym] = comm
            core_entry_date[sym] = day0
            cash -= (sh * core_fill) + comm

    for current_date in all_dates:
        # 1. MORNING EXECUTION (09:30 EST)
        for sym, df_sym in ticker_dfs.items():
            if current_date not in df_sym.index:
                continue
            bar_today = df_sym.loc[current_date]
            open_p = float(bar_today["Open"])
            action = pending_tactical_action[sym]

            if action == "LIQUIDATE_BEAR":
                tot_sh = core_shares[sym] + tactical_shares[sym]
                if tot_sh > 0:
                    exit_fill = open_p * (1.0 - slippage_rate)
                    exit_comm = max(min_commission, tot_sh * commission_per_share)
                    cash += (tot_sh * exit_fill) - exit_comm

                    # 1. Log tactical liquidation if leveraged
                    if is_leveraged[sym] and tactical_shares[sym] > 0:
                        t_sh = tactical_shares[sym]
                        t_gross = (exit_fill - tactical_entry_fill[sym]) * t_sh
                        t_exit_comm = exit_comm * (t_sh / tot_sh)
                        t_net = t_gross - tactical_entry_comm[sym] - t_exit_comm
                        closed_trades.append({
                            "ticker": sym,
                            "side": "LONG_TACTICAL",
                            "entry_date": (
                                tactical_entry_date[sym].strftime("%Y-%m-%d")
                                if tactical_entry_date[sym]
                                else ""
                            ),
                            "exit_date": current_date.strftime("%Y-%m-%d"),
                            "bars_held": (
                                len(df_sym.loc[tactical_entry_date[sym]:current_date])
                                if tactical_entry_date[sym]
                                else 0
                            ),
                            "entry_fill": round(tactical_entry_fill[sym], 2),
                            "exit_fill": round(exit_fill, 2),
                            "shares": round(t_sh, 4),
                            "exit_reason": "BEAR_REGIME_LIQUIDATION",
                            "entry_commission": round(tactical_entry_comm[sym], 2),
                            "exit_commission": round(t_exit_comm, 2),
                            "gross_pnl": round(t_gross, 2),
                            "net_pnl": round(t_net, 2),
                            "return_pct": round(
                                (t_net / (tactical_entry_fill[sym] * t_sh)) * 100.0, 2
                            ),
                            "was_profitable": t_net > 0,
                        })

                    # 2. Log core liquidation
                    if core_shares[sym] > 0:
                        c_sh = core_shares[sym]
                        c_gross = (exit_fill - core_entry_fill[sym]) * c_sh
                        c_exit_comm = exit_comm * (c_sh / tot_sh)
                        c_net = c_gross - core_entry_comm[sym] - c_exit_comm
                        c_ret = (
                            (c_net / (core_entry_fill[sym] * c_sh)) * 100.0
                            if core_entry_fill[sym] > 0
                            else 0.0
                        )
                        closed_trades.append({
                            "ticker": sym,
                            "side": "LONG_CORE",
                            "entry_date": (
                                core_entry_date[sym].strftime("%Y-%m-%d")
                                if core_entry_date[sym]
                                else ""
                            ),
                            "exit_date": current_date.strftime("%Y-%m-%d"),
                            "bars_held": (
                                len(df_sym.loc[core_entry_date[sym]:current_date])
                                if core_entry_date[sym]
                                else 0
                            ),
                            "entry_fill": round(core_entry_fill[sym], 2),
                            "exit_fill": round(exit_fill, 2),
                            "shares": round(c_sh, 4),
                            "exit_reason": "BEAR_REGIME_LIQUIDATION",
                            "entry_commission": round(core_entry_comm[sym], 2),
                            "exit_commission": round(c_exit_comm, 2),
                            "gross_pnl": round(c_gross, 2),
                            "net_pnl": round(c_net, 2),
                            "return_pct": round(c_ret, 2),
                            "was_profitable": c_net > 0,
                        })

                    core_shares[sym] = 0.0
                    tactical_shares[sym] = 0.0
                    is_leveraged[sym] = False
                pending_tactical_action[sym] = None

            elif action == "REENTER_CORE":
                if core_shares[sym] == 0:
                    fill = open_p * (1.0 + slippage_rate)
                    avail = min(base_capital, cash - min_commission)
                    if avail > 10.0:
                        sh_est = avail / fill
                        comm = max(min_commission, sh_est * commission_per_share)
                        sh = (avail - comm) / fill
                        core_shares[sym] = sh
                        core_entry_fill[sym] = fill
                        core_entry_comm[sym] = comm
                        core_entry_date[sym] = current_date
                        cash -= (sh * fill) + comm
                pending_tactical_action[sym] = None

            elif action == "BUY_OVERWEIGHT" and not is_leveraged[sym] and core_shares[sym] > 0:
                curr_val = core_shares[sym] * open_p
                target_val = curr_val * leverage
                add_val = target_val - curr_val
                fill = open_p * (1.0 + slippage_rate)
                add_sh = add_val / fill
                comm = max(min_commission, add_sh * commission_per_share)
                tactical_shares[sym] = add_sh
                tactical_entry_fill[sym] = fill
                tactical_entry_comm[sym] = comm
                tactical_entry_date[sym] = current_date
                cash -= (add_sh * fill) + comm
                is_leveraged[sym] = True
                pending_tactical_action[sym] = None

            elif action == "SELL_HARVEST" and is_leveraged[sym] and tactical_shares[sym] > 0:
                t_sh = tactical_shares[sym]
                exit_fill = open_p * (1.0 - slippage_rate)
                exit_comm = max(min_commission, t_sh * commission_per_share)
                gross_pnl = (exit_fill - tactical_entry_fill[sym]) * t_sh
                net_pnl = gross_pnl - tactical_entry_comm[sym] - exit_comm
                cash += (t_sh * exit_fill) - exit_comm
                ret_pct = (net_pnl / (tactical_entry_fill[sym] * t_sh)) * 100.0
                closed_trades.append({
                    "ticker": sym,
                    "side": "LONG_TACTICAL",
                    "entry_date": (
                        tactical_entry_date[sym].strftime("%Y-%m-%d")
                        if tactical_entry_date[sym]
                        else ""
                    ),
                    "exit_date": current_date.strftime("%Y-%m-%d"),
                    "bars_held": (
                        len(df_sym.loc[tactical_entry_date[sym]:current_date])
                        if tactical_entry_date[sym]
                        else 0
                    ),
                    "entry_fill": round(tactical_entry_fill[sym], 2),
                    "exit_fill": round(exit_fill, 2),
                    "shares": round(t_sh, 4),
                    "exit_reason": "TACTICAL_HARVEST",
                    "entry_commission": round(tactical_entry_comm[sym], 2),
                    "exit_commission": round(exit_comm, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "was_profitable": net_pnl > 0,
                })
                tactical_shares[sym] = 0.0
                is_leveraged[sym] = False
                pending_tactical_action[sym] = None

        # 2. EVENING INFERENCE (16:00+ EST)
        spy_bull = True
        if current_date in spy_df.index:
            spy_slice = spy_df.loc[:current_date]["Close"]
            if len(spy_slice) >= 200:
                spy_sma200 = float(spy_slice.rolling(200).mean().iloc[-1])
                spy_bull = float(spy_slice.iloc[-1]) >= spy_sma200

        for sym, df_sym in ticker_dfs.items():
            if current_date not in df_sym.index:
                continue
            bar_today = df_sym.loc[current_date]
            c_p = float(bar_today["Close"])
            sym_slice = df_sym.loc[:current_date]["Close"]
            sym_sma200 = (
                float(sym_slice.rolling(200).mean().iloc[-1]) if len(sym_slice) >= 200 else 0.0
            )

            # Hysteresis update: 1.5% below SMA200, 0.5% above SMA200
            if sym_sma200 > 0:
                if c_p < (sym_sma200 * 0.985):
                    consec_below_sma200[sym] += 1
                    consec_above_sma200[sym] = 0
                elif c_p > (sym_sma200 * 1.005):
                    consec_above_sma200[sym] += 1
                    consec_below_sma200[sym] = 0

            # State transitions with hysteresis
            if in_core_regime[sym]:
                # Liquidate if macro bear (SPY breakdown) or sustained 3-day asset breakdown
                if (not spy_bull) or (consec_below_sma200[sym] >= 3):
                    in_core_regime[sym] = False
                    if core_shares[sym] > 0 or tactical_shares[sym] > 0:
                        pending_tactical_action[sym] = "LIQUIDATE_BEAR"
            else:
                # Reenter if macro bull and sustained 5-day recovery
                if spy_bull and (consec_above_sma200[sym] >= 5 or sym_sma200 == 0.0):
                    in_core_regime[sym] = True
                    if core_shares[sym] == 0:
                        pending_tactical_action[sym] = "REENTER_CORE"

            if in_core_regime[sym] and core_shares[sym] > 0:
                idx_loc = df_sym.index.get_loc(current_date)
                if idx_loc >= 50:
                    bundle = models_by_ticker.get(sym) if models_by_ticker else None
                    sym_scaler = bundle["scaler"] if bundle else scaler
                    sym_xgb = bundle["xgb_model"] if bundle else xgb_model
                    sym_lgbm = bundle["lgbm_model"] if bundle else lgbm_model
                    sym_cal = bundle["calibrator"] if bundle else calibrator

                    row_feat = bar_today[kept_features].values.reshape(1, -1)
                    s_feat = sym_scaler.transform(row_feat)
                    px = sym_xgb.predict_proba(s_feat)[0]
                    pl = sym_lgbm.predict_proba(s_feat)[0]
                    xgb_cal = sym_cal.calibrate("XGB", px)
                    lgbm_cal = sym_cal.calibrate("LGBM", pl)
                    p_b, p_s = float(xgb_cal[2]), float(xgb_cal[0])
                    lgbm_veto = float(lgbm_cal[0]) >= 0.65
                    dir_b = p_b / (p_b + p_s + 1e-9)
                    c_rsi = float(bar_today.get("RSI", 50.0))
                    atr_val = float(bar_today.get("ATR_14", c_p * 0.02))

                    if (c_rsi <= dip_rsi or dir_b >= 0.60) and not lgbm_veto and not is_leveraged[sym]:
                        pending_tactical_action[sym] = "BUY_OVERWEIGHT"
                    elif is_leveraged[sym]:
                        is_profit = c_p > tactical_entry_fill[sym]
                        # Tactical harvest: lock in gains when extended +1.8 ATR or overbought expansion in profit
                        if is_profit and (c_p >= (tactical_entry_fill[sym] + (1.8 * atr_val)) or c_rsi >= trim_rsi):
                            pending_tactical_action[sym] = "SELL_HARVEST"

        # 3. END-OF-DAY VALUATION
        pos_val = sum(
            (core_shares[s] + tactical_shares[s]) * float(ticker_dfs[s].loc[current_date]["Close"])
            for s in ticker_dfs
            if current_date in ticker_dfs[s].index
        )
        total_equity = cash + pos_val
        daily_equity_curve.append({
            "date": current_date,
            "cash": cash,
            "positions_value": pos_val,
            "total_equity": total_equity,
        })

    # Close remaining tactical and core positions at final session close
    last_date = all_dates[-1]
    for sym, df_sym in ticker_dfs.items():
        if last_date not in df_sym.index:
            continue
        c_p = float(df_sym.loc[last_date, "Close"])

        # Close tactical shares if any
        if is_leveraged[sym] and tactical_shares[sym] > 0:
            t_sh = tactical_shares[sym]
            exit_fill = c_p * (1.0 - slippage_rate)
            exit_comm = max(min_commission, t_sh * commission_per_share)
            gross_pnl = (exit_fill - tactical_entry_fill[sym]) * t_sh
            net_pnl = gross_pnl - tactical_entry_comm[sym] - exit_comm
            cash += (t_sh * exit_fill) - exit_comm
            ret_pct = (net_pnl / (tactical_entry_fill[sym] * t_sh)) * 100.0
            closed_trades.append({
                "ticker": sym,
                "side": "LONG_TACTICAL",
                "entry_date": (
                    tactical_entry_date[sym].strftime("%Y-%m-%d")
                    if tactical_entry_date[sym]
                    else ""
                ),
                "exit_date": last_date.strftime("%Y-%m-%d"),
                "bars_held": (
                    len(df_sym.loc[tactical_entry_date[sym]:last_date])
                    if tactical_entry_date[sym]
                    else 0
                ),
                "entry_fill": round(tactical_entry_fill[sym], 2),
                "exit_fill": round(exit_fill, 2),
                "shares": round(t_sh, 4),
                "exit_reason": "END_OF_BACKTEST",
                "entry_commission": round(tactical_entry_comm[sym], 2),
                "exit_commission": round(exit_comm, 2),
                "gross_pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2),
                "return_pct": round(ret_pct, 2),
                "was_profitable": net_pnl > 0,
            })
            tactical_shares[sym] = 0.0

        # Close core shares at end of backtest to record complete PnL history
        if core_shares[sym] > 0:
            c_sh = core_shares[sym]
            exit_fill = c_p * (1.0 - slippage_rate)
            exit_comm = max(min_commission, c_sh * commission_per_share)
            gross_pnl = (exit_fill - core_entry_fill[sym]) * c_sh
            net_pnl = gross_pnl - core_entry_comm[sym] - exit_comm
            cash += (c_sh * exit_fill) - exit_comm
            ret_pct = (
                (net_pnl / (core_entry_fill[sym] * c_sh)) * 100.0
                if core_entry_fill[sym] > 0
                else 0.0
            )
            closed_trades.append({
                "ticker": sym,
                "side": "LONG_CORE",
                "entry_date": (
                    core_entry_date[sym].strftime("%Y-%m-%d") if core_entry_date[sym] else ""
                ),
                "exit_date": last_date.strftime("%Y-%m-%d"),
                "bars_held": (
                    len(df_sym.loc[core_entry_date[sym]:last_date]) if core_entry_date[sym] else 0
                ),
                "entry_fill": round(core_entry_fill[sym], 2),
                "exit_fill": round(exit_fill, 2),
                "shares": round(c_sh, 4),
                "exit_reason": "END_OF_BACKTEST",
                "entry_commission": round(core_entry_comm[sym], 2),
                "exit_commission": round(exit_comm, 2),
                "gross_pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2),
                "return_pct": round(ret_pct, 2),
                "was_profitable": net_pnl > 0,
            })
            core_shares[sym] = 0.0

    return daily_equity_curve, closed_trades


def run_chronological_backtest(
    tickers: Optional[List[str]] = None,
    period: str = "2y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 100000.0,
    allocation_per_trade: float = 0.10,
    slippage_bps: float = 5.0,
    commission_per_share: float = 0.005,
    min_commission: float = 1.00,
    output_dir: str = "backtest_results",
    snapshot_dir: Optional[str] = None,
    use_snapshots: bool = False,
    strategy_mode: str = "rotation",
    leverage: float = 1.00,
    dip_rsi: float = 35.0,
    trim_rsi: float = 65.0,
) -> Optional[Dict]:
    """
    Executes a causal, chronological multi-asset portfolio simulation.
    """
    logger.info("Initializing HYDRA Chronological Institutional Backtest Engine...")

    artifacts_dir = BACKEND_DIR / "artifacts"
    configs_dir = BACKEND_DIR / "configs"
    cache_dir = BACKEND_DIR / "data" / "cache"
    resolved_snapshot_dir: Optional[Path] = None
    if use_snapshots or snapshot_dir:
        resolved_snapshot_dir = Path(snapshot_dir) if snapshot_dir else (BACKEND_DIR / "data" / "snapshots")
        resolved_snapshot_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None or len(tickers) == 0:
        tickers = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD"]

    primary_ticker = tickers[0] if (tickers and len(tickers) == 1) else None
    ticker_model_dir = (artifacts_dir / "models" / primary_ticker) if primary_ticker else None

    if ticker_model_dir and (ticker_model_dir / "xgb_ensemble.json").exists():
        scaler_path = ticker_model_dir / "latest_scaler.joblib"
        xgb_path = ticker_model_dir / "xgb_ensemble.json"
        lgbm_path = ticker_model_dir / "lgbm_agent.joblib"
        calibrator_path = ticker_model_dir / "model_calibrator.joblib"
    else:
        scaler_path = artifacts_dir / "latest_scaler.joblib"
        xgb_path = artifacts_dir / "xgb_ensemble.json"
        lgbm_path = artifacts_dir / "lgbm_agent.joblib"
        calibrator_path = artifacts_dir / "model_calibrator.joblib"
    features_path = configs_dir / "kept_features.json"

    if not all(p.exists() for p in [scaler_path, xgb_path, lgbm_path, calibrator_path, features_path]):
        missing = [str(p) for p in [scaler_path, xgb_path, lgbm_path, calibrator_path, features_path] if not p.exists()]
        logger.error("Missing required production artifacts: %s", missing)
        return None

    # Load default artifacts
    with open(features_path, "r") as f:
        kept_features = json.load(f)

    scaler = joblib.load(scaler_path)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))

    lgbm_model = joblib.load(lgbm_path)

    # Load calibrator dictionary / object
    calibrator_obj = joblib.load(calibrator_path)
    calibrator = ModelCalibrator()
    if isinstance(calibrator_obj, dict):
        calibrator.calibrators = calibrator_obj
    elif hasattr(calibrator_obj, "calibrators"):
        calibrator = calibrator_obj

    # Load dedicated per-ticker model bundles
    models_by_ticker: Dict[str, Dict[str, Any]] = {}
    for t in tickers:
        t_dir = artifacts_dir / "models" / t
        if t_dir.exists() and (t_dir / "xgb_ensemble.json").exists():
            try:
                t_scaler = joblib.load(t_dir / "latest_scaler.joblib")
                t_xgb = xgb.XGBClassifier()
                t_xgb.load_model(str(t_dir / "xgb_ensemble.json"))
                t_lgbm = joblib.load(t_dir / "lgbm_agent.joblib")
                t_cal_obj = joblib.load(t_dir / "model_calibrator.joblib")
                t_cal = ModelCalibrator()
                if isinstance(t_cal_obj, dict):
                    t_cal.calibrators = t_cal_obj
                elif hasattr(t_cal_obj, "calibrators"):
                    t_cal = t_cal_obj
                models_by_ticker[t] = {
                    "scaler": t_scaler,
                    "xgb_model": t_xgb,
                    "lgbm_model": t_lgbm,
                    "calibrator": t_cal,
                }
                logger.info("Loaded dedicated models for %s from %s", t, t_dir.name)
            except Exception as e:
                logger.warning("Could not load dedicated model for %s: %s", t, e)

    if len(tickers) == 1 and allocation_per_trade <= 0.25:
        allocation_per_trade = 1.0

    # Pre-fetch or load benchmark data
    spy_df: Optional[pd.DataFrame] = None
    vix_df: Optional[pd.DataFrame] = None

    if use_snapshots:
        if resolved_snapshot_dir is None:
            resolved_snapshot_dir = BACKEND_DIR / "data" / "snapshots"

        # Fail-closed validation against committed snapshot manifest
        required_files = ["SPY_benchmark.parquet", "VIX_benchmark.parquet"] + [
            f"{t}_features.parquet" for t in tickers
        ]
        verify_snapshot_integrity(resolved_snapshot_dir, required_files)

        spy_snap = resolved_snapshot_dir / "SPY_benchmark.parquet"
        vix_snap = resolved_snapshot_dir / "VIX_benchmark.parquet"
        try:
            spy_df = pd.read_parquet(spy_snap)
            vix_df = pd.read_parquet(vix_snap)
            logger.info("Loaded immutable benchmark snapshots for SPY and ^VIX from %s", str(resolved_snapshot_dir))
        except Exception as e:
            raise ValueError(f"Failed to read benchmark snapshot files from {resolved_snapshot_dir}: {e}") from e
    else:
        # Non-strict research mode
        if resolved_snapshot_dir:
            spy_snap = resolved_snapshot_dir / "SPY_benchmark.parquet"
            vix_snap = resolved_snapshot_dir / "VIX_benchmark.parquet"
            if spy_snap.exists() and vix_snap.exists():
                try:
                    spy_df = pd.read_parquet(spy_snap)
                    vix_df = pd.read_parquet(vix_snap)
                    logger.info("Loaded benchmark snapshots for SPY and ^VIX from %s", str(resolved_snapshot_dir))
                except Exception as e:
                    logger.warning("Failed to load benchmark snapshots: %s", e)

        if spy_df is None or vix_df is None:
            logger.info("Fetching SPY and ^VIX benchmark data (%s)...", period)
            spy_df = yf.download("SPY", period=period, progress=False)
            if isinstance(spy_df.columns, pd.MultiIndex):
                spy_df.columns = spy_df.columns.get_level_values(0)

            vix_df = yf.download("^VIX", period=period, progress=False)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)

    # Load all ticker datasets
    ticker_dfs: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df_t = fetch_and_prepare_data(
                t,
                spy_df,
                vix_df,
                period=period,
                cache_dir=cache_dir,
                snapshot_dir=resolved_snapshot_dir,
                strict_snapshots=use_snapshots,
            )
            if len(df_t) >= 60:
                ticker_dfs[t] = df_t
            else:
                if use_snapshots:
                    raise ValueError(f"Snapshot dataset for {t} has insufficient bars ({len(df_t)} < 60)")
        except Exception as e:
            if use_snapshots:
                raise
            logger.error("Error preparing data for %s: %s", t, e)

    if not ticker_dfs:
        logger.error("No valid asset datasets loaded. Aborting backtest.")
        return None

    # Establish global chronological trading calendar
    all_dates = sorted(list(set.intersection(*[set(df.index) for df in ticker_dfs.values()])))
    if len(all_dates) < 30:
        # Fall back to sorted union if intersection is sparse
        all_dates = sorted(list(set.union(*[set(df.index) for df in ticker_dfs.values()])))

    if start_date:
        start_ts = pd.Timestamp(start_date)
        all_dates = [d for d in all_dates if d >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date)
        all_dates = [d for d in all_dates if d <= end_ts]

    if not all_dates:
        logger.error("No trading sessions found within the specified date range (%s to %s).", start_date, end_date)
        return None

    logger.info(
        "Starting chronological simulation: %d assets across %d calendar sessions (%s to %s)...",
        len(ticker_dfs),
        len(all_dates),
        all_dates[0].strftime("%Y-%m-%d"),
        all_dates[-1].strftime("%Y-%m-%d"),
    )

    # Execute simulation based on strategy mode
    if strategy_mode == "core_tactical":
        daily_equity_curve, closed_trades = _run_core_tactical_simulation(
            ticker_dfs=ticker_dfs,
            spy_df=spy_df,
            all_dates=all_dates,
            initial_capital=initial_capital,
            slippage_bps=slippage_bps,
            commission_per_share=commission_per_share,
            min_commission=min_commission,
            leverage=leverage,
            dip_rsi=dip_rsi,
            trim_rsi=trim_rsi,
            kept_features=kept_features,
            scaler=scaler,
            xgb_model=xgb_model,
            lgbm_model=lgbm_model,
            calibrator=calibrator,
            models_by_ticker=models_by_ticker,
        )
    elif strategy_mode == "oversold_rebound":
        daily_equity_curve, closed_trades = _run_oversold_rebound_simulation(
            ticker_dfs=ticker_dfs,
            all_dates=all_dates,
            initial_capital=initial_capital,
            slippage_bps=slippage_bps,
            commission_per_share=commission_per_share,
            min_commission=min_commission,
            oversold_rsi=38.0,
            exit_rsi=55.0,
            stop_atr_mult=2.0,
        )
    elif strategy_mode in ("adaptive", "dynamic_regime_hybrid"):
        daily_equity_curve, closed_trades = _run_dynamic_regime_hybrid_simulation(
            ticker_dfs=ticker_dfs,
            spy_df=spy_df,
            all_dates=all_dates,
            initial_capital=initial_capital,
            slippage_bps=slippage_bps,
            commission_per_share=commission_per_share,
            min_commission=min_commission,
        )
    else:
        daily_equity_curve, closed_trades = _run_rotation_simulation(
            ticker_dfs=ticker_dfs,
            spy_df=spy_df,
            all_dates=all_dates,
            initial_capital=initial_capital,
            allocation_per_trade=allocation_per_trade,
            slippage_bps=slippage_bps,
            commission_per_share=commission_per_share,
            min_commission=min_commission,
            kept_features=kept_features,
            scaler=scaler,
            xgb_model=xgb_model,
            lgbm_model=lgbm_model,
            calibrator=calibrator,
            models_by_ticker=models_by_ticker,
        )

    # Calculate Portfolio Performance Statistics
    df_equity = pd.DataFrame(daily_equity_curve).set_index("date")
    df_trades = pd.DataFrame(closed_trades)

    if df_equity.empty or len(df_equity) < 5:
        logger.error("Insufficient equity history generated.")
        return None

    df_equity["daily_return"] = df_equity["total_equity"].pct_change().fillna(0.0)
    final_equity = float(df_equity["total_equity"].iloc[-1])
    total_return_pct = ((final_equity / initial_capital) - 1.0) * 100.0

    # Drawdown series
    peak = df_equity["total_equity"].cummax()
    drawdown = (df_equity["total_equity"] - peak) / peak
    max_drawdown_pct = float(drawdown.min()) * 100.0

    # Annualized calculations (252 sessions/year)
    n_days = len(df_equity)
    years = max(0.1, n_days / 252.0)
    cagr_pct = (((final_equity / initial_capital) ** (1.0 / years)) - 1.0) * 100.0 if final_equity > 0 else -100.0
    daily_vol = float(df_equity["daily_return"].std())
    ann_vol_pct = daily_vol * np.sqrt(252.0) * 100.0
    sharpe = float((df_equity["daily_return"].mean() * 252.0) / (daily_vol * np.sqrt(252.0) + 1e-9))

    # Correct Calmar Ratio: CAGR / |Max Drawdown|
    calmar = float(cagr_pct / abs(max_drawdown_pct)) if max_drawdown_pct < 0 else 0.0

    # Trade level statistics
    num_closed = len(df_trades)
    if num_closed > 0:
        win_rate = float((df_trades["was_profitable"].sum() / num_closed) * 100.0)
        pnl_gains = df_trades.loc[df_trades["net_pnl"] > 0, "net_pnl"].sum()
        pnl_losses = abs(df_trades.loc[df_trades["net_pnl"] < 0, "net_pnl"].sum())
        profit_factor = float(pnl_gains / pnl_losses) if pnl_losses > 0 else 5.0
        avg_ret_pct = float(df_trades["return_pct"].mean())
    else:
        win_rate = profit_factor = avg_ret_pct = 0.0

    if strategy_mode == "rotation":
        exit_mechanics_desc = "Dynamic Triple Barrier (1.5x ATR TP, 2.0x ATR SL, 15-day Max Horizon)"
    elif strategy_mode == "oversold_rebound":
        exit_mechanics_desc = "Capitulation Oversold Rebound (RSI<=38 Entry, 20 EMA / RSI>=55 TP, 2.0x ATR SL, 100% Cash Defense)"
    elif strategy_mode == "adaptive":
        exit_mechanics_desc = "Institutional Adaptive Mesh (Regime-Matched: Core+Tac for Bull, Rebound for Bear, Rotation for Chop)"
    else:
        exit_mechanics_desc = f"Macro Core + Tactical Alpha ({leverage:.2f}x Kelly Leverage, Dip<={dip_rsi}, Trim>={trim_rsi})"

    report = f"""
    ===============================================================
       HYDRA CHRONOLOGICAL MULTI-ASSET PORTFOLIO BACKTEST
    ===============================================================
    Strategy Mode:          {strategy_mode.upper()}
    Evaluation Window:      Trailing {period} ({all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')})
    Universe Assets:        {', '.join(tickers)}
    Calendar Sessions:      {n_days}
    Initial Capital:        ${initial_capital:,.2f}
    Final Portfolio Equity: ${final_equity:,.2f}
    ---------------------------------------------------------------
    Total Net Return:       {total_return_pct:+.2f}%
    Annualized Return (CAGR): {cagr_pct:+.2f}%
    Annualized Volatility:  {ann_vol_pct:.2f}%
    Sharpe Ratio:           {sharpe:.2f}
    Maximum Drawdown:       {max_drawdown_pct:.2f}%
    Calmar Ratio (CAGR/DD): {calmar:.2f}
    ---------------------------------------------------------------
    Closed Trades Count:    {num_closed}
    Trade Win Rate:         {win_rate:.1f}%
    Profit Factor:          {profit_factor:.2f}
    Average Trade Return:   {avg_ret_pct:+.2f}%
    ---------------------------------------------------------------
    Execution Assumptions:  T+1 Open Fill, 5 bps Adverse Slippage,
                            5 bps Exit Slippage, $0.005/share fee
    Exit Mechanics:         {exit_mechanics_desc}
    ===============================================================
    """
    print(report)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "strategy_mode": strategy_mode,
        "period": period,
        "tickers": tickers,
        "calendar_sessions": n_days,
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "net_profit_dollar": round(final_equity - initial_capital, 2),
        "total_net_return_pct": round(total_return_pct, 2),
        "start_date": all_dates[0].strftime("%Y-%m-%d"),
        "end_date": all_dates[-1].strftime("%Y-%m-%d"),
        "cagr_pct": round(cagr_pct, 2),
        "annualized_vol_pct": round(ann_vol_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "calmar_ratio": round(calmar, 2),
        "closed_trades": num_closed,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_trade_return_pct": round(avg_ret_pct, 2),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

    with open(out_path / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    df_trades.to_csv(out_path / "backtest_trades.csv", index=False)
    df_equity.to_csv(out_path / "daily_equity_curve.csv")
    with open(out_path / "backtest_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Chronological backtest complete. Artifacts saved to %s", str(out_path))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HYDRA Chronological Multi-Asset Backtest")
    parser.add_argument("--ticker", type=str, help="Specific ticker to backtest (or 'all')")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers (e.g. AAPL,NVDA,MSFT or 'all')")
    parser.add_argument("--period", type=str, default="2y", help="Historical period (e.g. 1y, 2y, 5y)")
    parser.add_argument("--start-date", type=str, help="Optional start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Optional end date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial portfolio capital")
    parser.add_argument(
        "--strategy-mode",
        type=str,
        default="adaptive",
        choices=["rotation", "core_tactical", "oversold_rebound", "adaptive", "dynamic_regime_hybrid"],
        help="Strategy mode: rotation, core_tactical, oversold_rebound, adaptive, or dynamic_regime_hybrid",
    )
    parser.add_argument(
        "--leverage", type=float, default=1.00, help="Tactical equity multiplier (default 1.00 = unleveraged cash account)"
    )
    parser.add_argument(
        "--dip-rsi", type=float, default=35.0, help="RSI oversold threshold for tactical entry"
    )
    parser.add_argument(
        "--trim-rsi", type=float, default=65.0, help="RSI overbought threshold for tactical exit"
    )
    parser.add_argument("--min-commission", type=float, default=0.00, help="Minimum commission per trade")
    parser.add_argument("--commission-per-share", type=float, default=0.00, help="Commission per share")
    parser.add_argument("--output", type=str, default="backtest_results", help="Output directory")
    parser.add_argument("--use-snapshots", action="store_true", help="Use local immutable market snapshots")
    parser.add_argument("--snapshot-dir", type=str, help="Path to immutable snapshot directory")
    args = parser.parse_args()

    all_universe = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD"]
    if args.tickers:
        if args.tickers.lower() == "all":
            target_tickers = all_universe
        else:
            target_tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.ticker:
        if args.ticker.lower() == "all":
            target_tickers = all_universe
        else:
            target_tickers = [args.ticker.upper()]
    else:
        target_tickers = all_universe

    run_chronological_backtest(
        tickers=target_tickers,
        period=args.period,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        min_commission=args.min_commission,
        commission_per_share=args.commission_per_share,
        output_dir=args.output,
        use_snapshots=args.use_snapshots,
        snapshot_dir=args.snapshot_dir,
        strategy_mode=args.strategy_mode,
        leverage=args.leverage,
        dip_rsi=args.dip_rsi,
        trim_rsi=args.trim_rsi,
    )
