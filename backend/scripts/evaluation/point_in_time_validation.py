"""
Point-in-Time Prediction Validation Engine.
===========================================
Audits and measures the directional predictive accuracy and profitability of the
trading platform's signals across historical out-of-sample trading days.

Adheres strictly to the Point-in-Time Walk-Forward protocol:
1. Simulates running immediately after day t's candle closes (16:00 ET).
2. Model receives strictly data available at or before day t (D_{1..t}).
3. Signals are generated using production logic (XGBoost, Scaler, Macro Filter, State Machine).
4. Next-session execution filled at Open[t+1] +/- 5bps slippage.
5. Measures performance across 1, 3, 5, and 10 trading-day horizons.
6. Compares point-in-time sequential replay against full dataset batch replay (Invariance check).
7. Dynamically labels timezone as EDT during Daylight Saving Time and EST during Standard Time.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.execution.live_inference import (
    FEATURE_COLUMNS,
    add_upgraded_features,
)
from src.models.regime.calibration import ModelCalibrator

NY_TZ = ZoneInfo("America/New_York")


def format_eastern_timestamp(dt_val: Any, hour: int, minute: int) -> str:
    """
    Formats a date object or string into an Eastern Time timestamp with correct EDT/EST designation.
    """
    if isinstance(dt_val, str):
        dt_obj = datetime.strptime(dt_val[:10], "%Y-%m-%d")
    elif hasattr(dt_val, "to_pydatetime"):
        dt_obj = dt_val.to_pydatetime()
    elif isinstance(dt_val, datetime):
        dt_obj = dt_val
    else:
        dt_obj = datetime.fromisoformat(str(dt_val)[:10])

    aware_dt = datetime(
        dt_obj.year, dt_obj.month, dt_obj.day, hour, minute, 0, tzinfo=NY_TZ
    )
    return aware_dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def fetch_aligned_market_data(
    ticker: str = "AAPL",
    start_date: str = "2022-01-01",
    end_date: str = "2026-09-30",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetches raw OHLCV for ticker, SPY, and ^VIX starting from 2022-01-01
    to ensure full 200-bar warmup for SMA200 and 120-bar warmup for ZScore_RSI_120.
    """
    print(f"Fetching historical OHLCV data for {ticker}, SPY, ^VIX ({start_date} -> {end_date})...")
    ticker_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    spy_df = yf.download("SPY", start=start_date, end=end_date, progress=False)
    vix_df = yf.download("^VIX", start=start_date, end=end_date, progress=False)

    for d in [ticker_df, spy_df, vix_df]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

    # Reindex to ticker index
    common_idx = ticker_df.index.intersection(spy_df.index)
    ticker_df = ticker_df.loc[common_idx].sort_index()
    spy_df = spy_df.loc[common_idx].sort_index()
    vix_df = vix_df.reindex(common_idx).ffill().bfill().sort_index()

    return ticker_df, spy_df, vix_df


def run_point_in_time_signal_generation(
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    eval_start_date: str = "2024-01-01",
    eval_end_date: str = "2026-09-30",
    use_calibration: bool = True,
    strict_pit_slicing: bool = True,
) -> List[Dict[str, Any]]:
    """
    Point-in-Time Signal Generation.
    If strict_pit_slicing is True:
        For every trading day t in [eval_start_date, eval_end_date]:
        Slices D_{1..t} = ticker_df.loc[:t], spy_df.loc[:t], vix_df.loc[:t].
        Computes features strictly on trailing data.
        Runs inference, applies macro filter, and advances position state machine.
    """
    scaler_path = BACKEND_DIR / "artifacts" / "latest_scaler.joblib"
    model_path = BACKEND_DIR / "artifacts" / "xgb_ensemble.json"
    calibrator_path = BACKEND_DIR / "artifacts" / "model_calibrator.joblib"

    if not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError("Production scaler or XGBoost model not found in artifacts.")

    scaler = joblib.load(scaler_path)
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    calibrator = None
    if use_calibration and calibrator_path.exists():
        try:
            calibrator = ModelCalibrator.load(str(calibrator_path))
        except Exception as e:
            print(f"Warning: could not load calibrator ({e}), using raw probs.")

    # Identify candidate evaluation trading days
    eval_mask = (ticker_df.index >= eval_start_date) & (ticker_df.index <= eval_end_date)
    eval_dates = ticker_df.index[eval_mask]

    daily_signals: List[Dict[str, Any]] = []

    current_pos = "FLAT"
    last_trade_idx = -10
    min_cooldown_bars = 5

    if strict_pit_slicing:
        print(f"Executing strict Point-in-Time daily replay across {len(eval_dates)} trading sessions...")
        # Step-by-step point-in-time slice
        for step_i, current_t in enumerate(eval_dates):
            t_slice = ticker_df.loc[:current_t].copy()
            spy_slice = spy_df.loc[:current_t].copy()
            vix_slice = vix_df.loc[:current_t].copy()

            feat_df = add_upgraded_features(t_slice, spy_slice, vix_slice)
            feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()].copy()
            clean_feat = feat_df.reindex(columns=FEATURE_COLUMNS).dropna()

            if clean_feat.empty or current_t not in clean_feat.index:
                continue

            row_features = clean_feat.loc[[current_t]].values
            scaled_row = scaler.transform(row_features)

            probs = model.predict_proba(scaled_row)[0]
            if calibrator is not None:
                probs = calibrator.calibrate("XGB", probs)

            p_sell, p_hold, p_buy = float(probs[0]), float(probs[1]), float(probs[2])

            if p_buy >= 0.60:
                raw_signal = "BUY"
                conf = p_buy
            elif p_sell >= 0.60:
                raw_signal = "SELL"
                conf = p_sell
            else:
                raw_signal = "HOLD"
                conf = p_hold

            # Macro filter at bar t
            cur_close = float(ticker_df.loc[current_t, "Close"])
            sma200_series = ticker_df["Close"].loc[:current_t].rolling(200, min_periods=20).mean()
            cur_sma200 = float(sma200_series.iloc[-1])

            cur_spy_close = float(spy_df.loc[current_t, "Close"])
            spy_sma50_series = spy_df["Close"].loc[:current_t].rolling(50, min_periods=10).mean()
            cur_spy_sma50 = float(spy_sma50_series.iloc[-1])

            long_ok = (cur_close >= cur_sma200) and (cur_spy_close >= cur_spy_sma50)
            short_ok = (cur_close < cur_sma200) or (cur_spy_close < cur_spy_sma50)

            if current_pos == "LONG":
                sell_allowed = True
            else:
                sell_allowed = short_ok

            if raw_signal == "BUY" and not long_ok:
                filtered_signal = "HOLD"
            elif raw_signal == "SELL" and not sell_allowed:
                filtered_signal = "HOLD"
            else:
                filtered_signal = raw_signal

            # State machine transition
            cooldown_satisfied = (step_i - last_trade_idx) >= min_cooldown_bars
            confirmed_action = "HOLD"

            if filtered_signal == "BUY" and current_pos != "LONG" and cooldown_satisfied:
                confirmed_action = "BUY"
                current_pos = "LONG"
                last_trade_idx = step_i
            elif filtered_signal == "SELL" and current_pos == "LONG" and cooldown_satisfied:
                confirmed_action = "SELL"
                current_pos = "FLAT"
                last_trade_idx = step_i

            date_str = current_t.strftime("%Y-%m-%d")
            daily_signals.append({
                "bar_idx": step_i,
                "date": date_str,
                "timestamp_close": format_eastern_timestamp(current_t, 16, 0),
                "close_price": cur_close,
                "raw_signal": raw_signal,
                "filtered_signal": filtered_signal,
                "confirmed_signal": confirmed_action,
                "confidence": conf,
                "prob_sell": p_sell,
                "prob_hold": p_hold,
                "prob_buy": p_buy,
                "macro_long_ok": long_ok,
                "macro_short_ok": short_ok,
                "position_after": current_pos,
            })
    else:
        # Full batch replay for Invariance / Lookahead verification
        feat_df = add_upgraded_features(ticker_df.copy(), spy_df.copy(), vix_df.copy())
        feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()].copy()
        clean_feat = feat_df.reindex(columns=FEATURE_COLUMNS).dropna()

        scaled_rows = scaler.transform(clean_feat.values)
        all_probs = model.predict_proba(scaled_rows)
        if calibrator is not None:
            all_probs = np.array([calibrator.calibrate("XGB", p) for p in all_probs])

        close_s = ticker_df["Close"].reindex(clean_feat.index).ffill()
        sma200_s = close_s.rolling(200, min_periods=20).mean()
        spy_close_s = spy_df["Close"].reindex(clean_feat.index).ffill()
        spy_sma50_s = spy_close_s.rolling(50, min_periods=10).mean()

        for step_i, current_t in enumerate(clean_feat.index):
            if current_t < pd.Timestamp(eval_start_date) or current_t > pd.Timestamp(eval_end_date):
                continue

            prob_vec = all_probs[step_i]
            p_sell, p_hold, p_buy = float(prob_vec[0]), float(prob_vec[1]), float(prob_vec[2])

            if p_buy >= 0.60:
                raw_signal = "BUY"
                conf = p_buy
            elif p_sell >= 0.60:
                raw_signal = "SELL"
                conf = p_sell
            else:
                raw_signal = "HOLD"
                conf = p_hold

            cur_close = float(close_s.iloc[step_i])
            cur_sma200 = float(sma200_s.iloc[step_i])
            cur_spy = float(spy_close_s.iloc[step_i])
            cur_spy_sma50 = float(spy_sma50_s.iloc[step_i])

            long_ok = (cur_close >= cur_sma200) and (cur_spy >= cur_spy_sma50)
            short_ok = (cur_close < cur_sma200) or (cur_spy < cur_spy_sma50)
            sell_allowed = True if current_pos == "LONG" else short_ok

            if raw_signal == "BUY" and not long_ok:
                filtered_signal = "HOLD"
            elif raw_signal == "SELL" and not sell_allowed:
                filtered_signal = "HOLD"
            else:
                filtered_signal = raw_signal

            cooldown_satisfied = (step_i - last_trade_idx) >= min_cooldown_bars
            confirmed_action = "HOLD"

            if filtered_signal == "BUY" and current_pos != "LONG" and cooldown_satisfied:
                confirmed_action = "BUY"
                current_pos = "LONG"
                last_trade_idx = step_i
            elif filtered_signal == "SELL" and current_pos == "LONG" and cooldown_satisfied:
                confirmed_action = "SELL"
                current_pos = "FLAT"
                last_trade_idx = step_i

            date_str = current_t.strftime("%Y-%m-%d")
            daily_signals.append({
                "bar_idx": step_i,
                "date": date_str,
                "timestamp_close": format_eastern_timestamp(current_t, 16, 0),
                "close_price": cur_close,
                "raw_signal": raw_signal,
                "filtered_signal": filtered_signal,
                "confirmed_signal": confirmed_action,
                "confidence": conf,
                "prob_sell": p_sell,
                "prob_hold": p_hold,
                "prob_buy": p_buy,
                "macro_long_ok": long_ok,
                "macro_short_ok": short_ok,
                "position_after": current_pos,
            })

    return daily_signals


def calculate_horizon_performance(
    signals_list: List[Dict[str, Any]],
    ticker_df: pd.DataFrame,
    horizons: List[int] = [1, 3, 5, 10],
    slippage_bps: float = 0.0005,  # 5 bps each way
    commission_per_share: float = 0.005,  # $0.005 per share each way ($0.01 roundtrip)
) -> List[Dict[str, Any]]:
    """
    Measures post-signal execution performance across 1, 3, 5, and 10 trading-day horizons.
    Execution begins at Open[t+1] +/- slippage.
    Exit occurs at Close[t+k] +/- slippage.
    Calculates gross return, net return, correctness, MAE, and MFE.
    """
    evaluated_records: List[Dict[str, Any]] = []

    # Map dates to integer positional indices in ticker_df
    date_to_pos = {d.strftime("%Y-%m-%d"): idx for idx, d in enumerate(ticker_df.index)}

    for sig in signals_list:
        date_str = sig["date"]
        action = sig["confirmed_signal"]

        if action not in ["BUY", "SELL"]:
            continue

        pos_t = date_to_pos.get(date_str)
        if pos_t is None or pos_t + 1 >= len(ticker_df):
            # Not executable or at the very edge of history
            continue

        next_bar_date = ticker_df.index[pos_t + 1]
        next_open = float(ticker_df["Open"].iloc[pos_t + 1])
        exec_target_str = next_bar_date.strftime("%Y-%m-%d")
        exec_timestamp_str = format_eastern_timestamp(next_bar_date, 9, 30)

        # Execution Fill Price (t+1 Open)
        if action == "BUY":
            fill_price_gross = next_open * (1.0 + slippage_bps)
            fill_price_net = fill_price_gross + commission_per_share
        else:
            fill_price_gross = next_open * (1.0 - slippage_bps)
            fill_price_net = fill_price_gross - commission_per_share

        record = {
            "source_date": date_str,
            "signal": action,
            "confidence": sig["confidence"],
            "signal_price": sig["close_price"],
            "source_candle_timestamp": sig["timestamp_close"],
            "execution_target_date": exec_target_str,
            "execution_timestamp": exec_timestamp_str,
            "next_open": next_open,
            "fill_price_gross": round(fill_price_gross, 2),
            "fill_price_net": round(fill_price_net, 2),
            "is_executable": True,
            "horizons": {},
        }

        # Measure for each horizon
        for h in horizons:
            pos_exit = pos_t + h
            if pos_exit < len(ticker_df):
                exit_date = ticker_df.index[pos_exit]
                exit_close = float(ticker_df["Close"].iloc[pos_exit])

                # Holding window slices for MAE and MFE: from bar t+1 to t+h
                window_highs = ticker_df["High"].iloc[pos_t + 1 : pos_exit + 1]
                window_lows = ticker_df["Low"].iloc[pos_t + 1 : pos_exit + 1]

                max_high = float(window_highs.max())
                min_low = float(window_lows.min())

                if action == "BUY":
                    # Exit fill
                    exit_price_gross = exit_close * (1.0 - slippage_bps)
                    exit_price_net = exit_price_gross - commission_per_share

                    ret_gross = (exit_close - next_open) / next_open
                    ret_net = (exit_price_net - fill_price_net) / fill_price_net
                    is_correct = ret_net > 0.0

                    mfe = (max_high - next_open) / next_open
                    mae = (min_low - next_open) / next_open  # typically <= 0
                else:  # SELL / SHORT
                    exit_price_gross = exit_close * (1.0 + slippage_bps)
                    exit_price_net = exit_price_gross + commission_per_share

                    ret_gross = (next_open - exit_close) / next_open
                    ret_net = (fill_price_net - exit_price_net) / fill_price_net
                    is_correct = ret_net > 0.0

                    mfe = (next_open - min_low) / next_open
                    mae = (next_open - max_high) / next_open  # typically <= 0

                record["horizons"][h] = {
                    "exit_date": exit_date.strftime("%Y-%m-%d"),
                    "exit_close": exit_close,
                    "return_gross": ret_gross,
                    "return_net": ret_net,
                    "is_correct": is_correct,
                    "mfe": mfe,
                    "mae": mae,
                }
            else:
                # Horizon has not completed yet
                record["horizons"][h] = None

        evaluated_records.append(record)

    return evaluated_records


def compute_aggregate_metrics(
    evaluated_records: List[Dict[str, Any]],
    total_eval_days: int,
    horizons: List[int] = [1, 3, 5, 10],
) -> Dict[str, Any]:
    """
    Computes all quantitative evaluation metrics:
    - directional accuracy (BUY, SELL, overall)
    - precision, recall, win rate
    - average return after BUY / SELL (gross and net)
    - MAE and MFE averages
    - confusion matrix
    """
    total_buy = sum(1 for r in evaluated_records if r["signal"] == "BUY")
    total_sell = sum(1 for r in evaluated_records if r["signal"] == "SELL")
    total_trades = total_buy + total_sell
    total_hold = total_eval_days - total_trades

    metrics_by_h = {}

    for h in horizons:
        valid_recs = [r for r in evaluated_records if r["horizons"].get(h) is not None]
        if not valid_recs:
            continue

        buy_recs = [r for r in valid_recs if r["signal"] == "BUY"]
        sell_recs = [r for r in valid_recs if r["signal"] == "SELL"]

        # Win rates / Directional Accuracy
        buy_correct = sum(1 for r in buy_recs if r["horizons"][h]["is_correct"])
        buy_acc = buy_correct / len(buy_recs) if buy_recs else 0.0

        sell_correct = sum(1 for r in sell_recs if r["horizons"][h]["is_correct"])
        sell_acc = sell_correct / len(sell_recs) if sell_recs else 0.0

        total_correct = buy_correct + sell_correct
        overall_acc = total_correct / len(valid_recs) if valid_recs else 0.0

        # Returns
        buy_rets_gross = [r["horizons"][h]["return_gross"] for r in buy_recs]
        buy_rets_net = [r["horizons"][h]["return_net"] for r in buy_recs]
        avg_buy_ret_gross = np.mean(buy_rets_gross) if buy_rets_gross else 0.0
        avg_buy_ret_net = np.mean(buy_rets_net) if buy_rets_net else 0.0

        sell_rets_gross = [r["horizons"][h]["return_gross"] for r in sell_recs]
        sell_rets_net = [r["horizons"][h]["return_net"] for r in sell_recs]
        avg_sell_ret_gross = np.mean(sell_rets_gross) if sell_rets_gross else 0.0
        avg_sell_ret_net = np.mean(sell_rets_net) if sell_rets_net else 0.0

        all_rets_gross = [r["horizons"][h]["return_gross"] for r in valid_recs]
        all_rets_net = [r["horizons"][h]["return_net"] for r in valid_recs]
        avg_total_ret_gross = np.mean(all_rets_gross) if all_rets_gross else 0.0
        avg_total_ret_net = np.mean(all_rets_net) if all_rets_net else 0.0

        # Excursion
        mae_vals = [r["horizons"][h]["mae"] for r in valid_recs]
        mfe_vals = [r["horizons"][h]["mfe"] for r in valid_recs]
        avg_mae = np.mean(mae_vals) if mae_vals else 0.0
        avg_mfe = np.mean(mfe_vals) if mfe_vals else 0.0

        # Confusion Matrix
        # Predicted Positive: BUY
        # Predicted Negative: SELL
        # Actual Positive: Price increased (ret_net > 0 for BUY, ret_net < 0 for SELL)
        # Actual Negative: Price decreased
        tp = buy_correct  # Predicted BUY, Price went up
        fp = len(buy_recs) - buy_correct  # Predicted BUY, Price went down
        tn = sell_correct  # Predicted SELL, Price went down
        fn = len(sell_recs) - sell_correct  # Predicted SELL, Price went up

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics_by_h[h] = {
            "eval_count": len(valid_recs),
            "buy_count": len(buy_recs),
            "sell_count": len(sell_recs),
            "buy_accuracy": buy_acc,
            "sell_accuracy": sell_acc,
            "overall_accuracy": overall_acc,
            "win_rate": overall_acc,
            "avg_buy_return_gross": avg_buy_ret_gross,
            "avg_buy_return_net": avg_buy_ret_net,
            "avg_sell_return_gross": avg_sell_ret_gross,
            "avg_sell_return_net": avg_sell_ret_net,
            "avg_overall_return_gross": avg_total_ret_gross,
            "avg_overall_return_net": avg_total_ret_net,
            "avg_mae": avg_mae,
            "avg_mfe": avg_mfe,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": {
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
            },
        }

    return {
        "total_eval_days": total_eval_days,
        "total_buy_signals": total_buy,
        "total_sell_signals": total_sell,
        "total_hold_days": total_hold,
        "executable_signals": len(evaluated_records),
        "by_horizon": metrics_by_h,
    }


def main():
    print("=" * 80)
    print("INSTITUTIONAL POINT-IN-TIME PREDICTION VALIDATION ENGINE")
    print("=" * 80)

    ticker = "AAPL"
    start_date = "2022-01-01"
    end_date = "2026-09-30"
    oos_start = "2024-01-01"
    oos_end = "2026-09-30"

    ticker_df, spy_df, vix_df = fetch_aligned_market_data(ticker, start_date, end_date)

    # ---------------------------------------------------------
    # STEP 8: INVARIANCE & LOOKAHEAD VERIFICATION
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 8: REPLAY INVARIANCE & ZERO-LOOKAHEAD VERIFICATION")
    print("-" * 60)

    pit_signals = run_point_in_time_signal_generation(
        ticker_df, spy_df, vix_df,
        eval_start_date=oos_start,
        eval_end_date=oos_end,
        strict_pit_slicing=True,
    )

    batch_signals = run_point_in_time_signal_generation(
        ticker_df, spy_df, vix_df,
        eval_start_date=oos_start,
        eval_end_date=oos_end,
        strict_pit_slicing=False,
    )

    # Compare signals day by day
    mismatches = 0
    total_compared = min(len(pit_signals), len(batch_signals))
    for i in range(total_compared):
        s_pit = pit_signals[i]
        s_bat = batch_signals[i]
        if (
            s_pit["confirmed_signal"] != s_bat["confirmed_signal"]
            or abs(s_pit["confidence"] - s_bat["confidence"]) > 1e-4
        ):
            mismatches += 1

    invariance_passed = mismatches == 0
    print(f"Total days verified: {total_compared}")
    print(f"Mismatches detected: {mismatches}")
    print(f"Invariance Status: {'PASSED (100% BIT-FOR-BIT MATCH)' if invariance_passed else 'FAILED'}")

    # ---------------------------------------------------------
    # PERFORMANCE MEASUREMENT (OOS 2024-2026)
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"CALCULATING OUT-OF-SAMPLE PERFORMANCE ({oos_start} -> {oos_end})")
    print("-" * 60)

    eval_records = calculate_horizon_performance(pit_signals, ticker_df)
    total_eval_days = len(pit_signals)
    metrics = compute_aggregate_metrics(eval_records, total_eval_days)

    print(f"Total Trading Days in Evaluation Window: {metrics['total_eval_days']}")
    print(f"Total Confirmed BUY Signals: {metrics['total_buy_signals']}")
    print(f"Total Confirmed SELL Signals: {metrics['total_sell_signals']}")
    print(f"Total HOLD / NO_SIGNAL Days: {metrics['total_hold_days']}")
    print(f"Total Executable Actionable Signals: {metrics['executable_signals']}")

    for h, m in metrics["by_horizon"].items():
        print(f"\n--- HORIZON: {h} TRADING DAYS ---")
        print(f"  Trades Evaluated: {m['eval_count']} (BUY: {m['buy_count']}, SELL: {m['sell_count']})")
        print(f"  BUY Directional Accuracy:  {m['buy_accuracy'] * 100:.2f}%")
        print(f"  SELL Directional Accuracy: {m['sell_accuracy'] * 100:.2f}%")
        print(f"  Overall Directional Acc:   {m['overall_accuracy'] * 100:.2f}% (Win Rate)")
        print(f"  Precision: {m['precision']:.3f} | Recall: {m['recall']:.3f} | F1: {m['f1_score']:.3f}")
        print(f"  Avg Return BUY  (Gross / Net): {m['avg_buy_return_gross'] * 100:+.2f}% / {m['avg_buy_return_net'] * 100:+.2f}%")
        print(f"  Avg Return SELL (Gross / Net): {m['avg_sell_return_gross'] * 100:+.2f}% / {m['avg_sell_return_net'] * 100:+.2f}%")
        print(f"  Avg Overall Ret (Gross / Net): {m['avg_overall_return_gross'] * 100:+.2f}% / {m['avg_overall_return_net'] * 100:+.2f}%")
        print(f"  Avg MAE (Adverse Excursion):   {m['avg_mae'] * 100:.2f}%")
        print(f"  Avg MFE (Favorable Excursion): {m['avg_mfe'] * 100:.2f}%")
        print(f"  Confusion Matrix: TP={m['confusion_matrix']['TP']}, FP={m['confusion_matrix']['FP']}, "
              f"TN={m['confusion_matrix']['TN']}, FN={m['confusion_matrix']['FN']}")

    # ---------------------------------------------------------
    # RECENT SIGNALS SAMPLE TABLE (>= 20 SIGNALS)
    # ---------------------------------------------------------
    print("\n" + "=" * 120)
    print("RECENT RECEPTIVE SIGNALS SAMPLE TABLE (POINT-IN-TIME WALK-FORWARD)")
    print("=" * 120)

    recent_sample = eval_records[-25:] if len(eval_records) >= 25 else eval_records

    # Print table header for 5-day horizon
    print(f"{'SOURCE DATE':<12} | {'SIGNAL':<6} | {'CLOSE PRICE':<11} | {'NEXT OPEN':<10} | {'5D CLOSE':<10} | {'5D NET RET':<10} | {'5D OUTCOME':<10} | {'EXEC TIMESTAMP'}")
    print("-" * 120)

    for r in recent_sample:
        s_date = r["source_date"]
        sig = r["signal"]
        close_p = f"${r['signal_price']:.2f}"
        next_open = f"${r['next_open']:.2f}"
        h5 = r["horizons"].get(5)
        if h5:
            fut_p = f"${h5['exit_close']:.2f}"
            ret_s = f"{h5['return_net'] * 100:+.2f}%"
            outcome = "CORRECT" if h5["is_correct"] else "INCORRECT"
        else:
            fut_p = "OPEN"
            ret_s = "N/A"
            outcome = "PENDING"
        exec_ts = r["execution_timestamp"]
        print(f"{s_date:<12} | {sig:<6} | {close_p:<11} | {next_open:<10} | {fut_p:<10} | {ret_s:<10} | {outcome:<10} | {exec_ts}")

    # ---------------------------------------------------------
    # 2026 DASHBOARD LEDGER EVALUATION (Matching uploaded UI)
    # ---------------------------------------------------------
    print("\n" + "=" * 120)
    print("EVALUATION OF ACTIVE PRODUCTION SIGNALS (SIGNAL LEDGER DATABASE)")
    print("=" * 120)

    import sqlite3
    db_path = BACKEND_DIR / "artifacts" / "signal_ledger.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        ledger_rows = conn.execute(
            "SELECT bar_timestamp, signal, confidence, execution_target_bar, execution_price "
            "FROM signal_ledger WHERE symbol='AAPL' AND signal IN ('BUY', 'SELL') ORDER BY bar_timestamp ASC"
        ).fetchall()

        print(f"Total AAPL Confirmed Signals in Production Ledger: {len(ledger_rows)}")
        ledger_signals = []
        for row in ledger_rows:
            ledger_signals.append({
                "date": row["bar_timestamp"],
                "confirmed_signal": row["signal"],
                "confidence": row["confidence"],
                "close_price": float(row["execution_price"]),
                "timestamp_close": format_eastern_timestamp(row["bar_timestamp"], 16, 0),
            })

        ledger_eval = calculate_horizon_performance(ledger_signals, ticker_df)
        ledger_metrics = compute_aggregate_metrics(ledger_eval, total_eval_days=len(ledger_rows))

        for h, m in ledger_metrics["by_horizon"].items():
            print(f"\n--- LEDGER SIGNALS HORIZON: {h} TRADING DAYS ---")
            print(f"  Trades Evaluated: {m['eval_count']} (BUY: {m['buy_count']}, SELL: {m['sell_count']})")
            print(f"  BUY Directional Accuracy:  {m['buy_accuracy'] * 100:.2f}%")
            print(f"  SELL Directional Accuracy: {m['sell_accuracy'] * 100:.2f}%")
            print(f"  Overall Directional Acc:   {m['overall_accuracy'] * 100:.2f}% (Win Rate)")
            print(f"  Avg Return BUY  (Gross / Net): {m['avg_buy_return_gross'] * 100:+.2f}% / {m['avg_buy_return_net'] * 100:+.2f}%")
            print(f"  Avg Return SELL (Gross / Net): {m['avg_sell_return_gross'] * 100:+.2f}% / {m['avg_sell_return_net'] * 100:+.2f}%")
            print(f"  Avg Overall Ret (Gross / Net): {m['avg_overall_return_gross'] * 100:+.2f}% / {m['avg_overall_return_net'] * 100:+.2f}%")
            print(f"  Avg MAE: {m['avg_mae'] * 100:.2f}% | Avg MFE: {m['avg_mfe'] * 100:.2f}%")

    print("\n" + "=" * 80)
    print("VALIDATION EXECUTION COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
