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


def run_chronological_backtest(
    tickers: Optional[List[str]] = None,
    period: str = "2y",
    initial_capital: float = 100000.0,
    allocation_per_trade: float = 0.10,
    slippage_bps: float = 5.0,
    commission_per_share: float = 0.005,
    min_commission: float = 1.00,
    output_dir: str = "backtest_results",
    snapshot_dir: Optional[str] = None,
    use_snapshots: bool = False,
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

    scaler_path = artifacts_dir / "latest_scaler.joblib"
    xgb_path = artifacts_dir / "xgb_ensemble.json"
    lgbm_path = artifacts_dir / "lgbm_agent.joblib"
    calibrator_path = artifacts_dir / "model_calibrator.joblib"
    features_path = configs_dir / "kept_features.json"

    if not all(p.exists() for p in [scaler_path, xgb_path, lgbm_path, calibrator_path, features_path]):
        missing = [str(p) for p in [scaler_path, xgb_path, lgbm_path, calibrator_path, features_path] if not p.exists()]
        logger.error("Missing required production artifacts: %s", missing)
        return None

    # Load artifacts
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

    if tickers is None or len(tickers) == 0:
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN"]

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

    logger.info(
        "Starting chronological simulation: %d assets across %d calendar sessions (%s to %s)...",
        len(ticker_dfs),
        len(all_dates),
        all_dates[0].strftime("%Y-%m-%d"),
        all_dates[-1].strftime("%Y-%m-%d"),
    )

    # Simulation Portfolio State
    cash = float(initial_capital)
    # open_positions: ticker -> dict(side, shares, entry_price, entry_date, entry_atr, tp_price, sl_price, bars_held)
    open_positions: Dict[str, Dict] = {}
    pending_orders: List[Dict] = []  # orders generated at close, filled at next open
    last_exit_date: Dict[str, pd.Timestamp] = {}  # for 5-bar cooldown
    closed_trades: List[Dict] = []
    daily_equity_curve: List[Dict] = []

    slippage_rate = slippage_bps / 10000.0

    # Chronological day-by-day loop
    for current_date in all_dates:
        # 1. MORNING & INTRADAY EXECUTION: Process pending barrier exits and pending entry orders
        # 1a. Check active positions for barrier hits (overnight gap at Open or intraday High/Low)
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
                # Check overnight gap beyond barriers at Open first
                if open_p >= pos["tp_price"]:
                    exit_reason = "TAKE_PROFIT"
                    exit_fill_base = open_p
                elif open_p <= pos["sl_price"]:
                    exit_reason = "STOP_LOSS"
                    exit_fill_base = open_p
                else:
                    # Intraday barrier inspection:
                    # Conservative tie-breaking: if both hit in the same bar, Stop-Loss triggers first
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
                # Check overnight gap beyond barriers at Open first
                if open_p <= pos["tp_price"]:
                    exit_reason = "TAKE_PROFIT"
                    exit_fill_base = open_p
                elif open_p >= pos["sl_price"]:
                    exit_reason = "STOP_LOSS"
                    exit_fill_base = open_p
                else:
                    # Intraday barrier inspection:
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

        # Execute position exits with adverse exit slippage & commission
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

        # 1b. Execute pending entry orders generated at previous close at today's OPEN
        executed_orders = []
        for order in pending_orders:
            sym = order["ticker"]
            if current_date not in ticker_dfs[sym].index:
                continue
            if sym in open_positions:
                continue  # already occupied

            bar_today = ticker_dfs[sym].loc[current_date]
            open_p = float(bar_today["Open"])
            side = order["signal"]

            # Calculate fill with adverse entry slippage
            entry_fill = open_p * (1.0 + slippage_rate) if side == "BUY" else open_p * (1.0 - slippage_rate)

            # Position sizing: 10% of current total estimated equity
            current_portfolio_est = cash + sum(
                p["shares"] * float(ticker_dfs[s].loc[current_date]["Close"]) for s, p in open_positions.items()
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
                tp = entry_fill + (1.5 * atr_val)
                sl = entry_fill - (2.0 * atr_val)
            else:
                tp = entry_fill - (1.5 * atr_val)
                sl = entry_fill + (2.0 * atr_val)

            open_positions[sym] = {
                "side": "LONG" if side == "BUY" else "SHORT",
                "shares": shares,
                "entry_fill": entry_fill,
                "entry_commission": entry_commission,
                "entry_date": current_date,
                "entry_atr": atr_val,
                "tp_price": tp,
                "sl_price": sl,
                "bars_held": 0,
            }
            executed_orders.append(order)

        # Clear executed pending orders
        pending_orders = [o for o in pending_orders if o not in executed_orders]

        # 2. EVENING SIGNAL GENERATION: At CLOSE of current_date, evaluate models for next session entry
        new_pending_orders = []
        for sym, df_sym in ticker_dfs.items():
            if current_date not in df_sym.index:
                continue
            if sym in open_positions:
                continue

            # Check 5-bar cooldown since last exit
            if sym in last_exit_date:
                bars_since = len(df_sym.loc[last_exit_date[sym]:current_date]) - 1
                if bars_since < 5:
                    continue

            # Locate integer index in df_sym
            idx_loc = df_sym.index.get_loc(current_date)
            if idx_loc < 50:
                continue

            current_row = df_sym.iloc[idx_loc]
            row_features = current_row[kept_features].values.reshape(1, -1)
            scaled_features = scaler.transform(row_features)

            # Predict XGBoost and LightGBM probabilities
            xgb_raw = xgb_model.predict_proba(scaled_features)[0]
            lgbm_raw = lgbm_model.predict_proba(scaled_features)[0]

            xgb_cal = calibrator.calibrate("XGB", xgb_raw)
            lgbm_cal = calibrator.calibrate("LGBM", lgbm_raw)

            # Single Authoritative Production Strategy Logic:
            # Primary Alpha Driver: XGBoost >= 0.60
            # Secondary Asymmetric Veto: LightGBM opposing >= 0.65
            signal = "HOLD"
            p_buy_xgb = xgb_cal[2]
            p_sell_xgb = xgb_cal[0]
            p_sell_lgbm = lgbm_cal[0]
            p_buy_lgbm = lgbm_cal[2]

            if p_buy_xgb >= 0.60:
                if p_sell_lgbm >= 0.65:
                    signal = "VETOED"
                else:
                    signal = "BUY"
            elif p_sell_xgb >= 0.60:
                if p_buy_lgbm >= 0.65:
                    signal = "VETOED"
                else:
                    signal = "SELL"

            # Macro Regime Gate (SPY 200 SMA)
            if signal == "BUY" and current_date in spy_df.index:
                spy_slice = spy_df.loc[:current_date]
                if len(spy_slice) >= 200:
                    spy_sma200 = spy_slice["Close"].rolling(200).mean().iloc[-1]
                    if float(spy_slice["Close"].iloc[-1]) < float(spy_sma200):
                        signal = "VETOED"

            if signal in ["BUY", "SELL"]:
                new_pending_orders.append({
                    "ticker": sym,
                    "signal": signal,
                    "generated_date": current_date,
                    "p_buy": float(p_buy_xgb),
                    "p_sell": float(p_sell_xgb),
                })

        pending_orders = new_pending_orders

        # 3. END-OF-DAY MARK-TO-MARKET PORTFOLIO VALUATION
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

    # Close any remaining open positions at the end of the simulation
    if open_positions:
        last_date = all_dates[-1]
        for sym, pos in list(open_positions.items()):
            if last_date in ticker_dfs[sym].index:
                close_p = float(ticker_dfs[sym].loc[last_date]["Close"])
                shares = pos["shares"]
                exit_fill = close_p * (1.0 - slippage_rate) if pos["side"] == "LONG" else close_p * (1.0 + slippage_rate)
                gross_pnl = (exit_fill - pos["entry_fill"]) * shares if pos["side"] == "LONG" else (pos["entry_fill"] - exit_fill) * shares
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

    report = f"""
    ===============================================================
       HYDRA CHRONOLOGICAL MULTI-ASSET PORTFOLIO BACKTEST
    ===============================================================
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
    Exit Mechanics:         Dynamic Triple Barrier (1.5x ATR TP,
                            2.0x ATR SL, 15-day Max Horizon)
    ===============================================================
    """
    print(report)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "period": period,
        "tickers": tickers,
        "calendar_sessions": n_days,
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_net_return_pct": round(total_return_pct, 2),
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
    parser.add_argument("--ticker", type=str, help="Specific ticker to backtest")
    parser.add_argument("--period", type=str, default="2y", help="Historical period (e.g. 1y, 2y, 5y)")
    parser.add_argument("--output", type=str, default="backtest_results", help="Output directory")
    parser.add_argument("--use-snapshots", action="store_true", help="Use local immutable market snapshots")
    parser.add_argument("--snapshot-dir", type=str, help="Path to immutable snapshot directory")
    args = parser.parse_args()

    target_tickers = [args.ticker.upper()] if args.ticker else ["AAPL", "MSFT", "NVDA", "AMZN"]
    run_chronological_backtest(
        tickers=target_tickers,
        period=args.period,
        output_dir=args.output,
        use_snapshots=args.use_snapshots,
        snapshot_dir=args.snapshot_dir,
    )
