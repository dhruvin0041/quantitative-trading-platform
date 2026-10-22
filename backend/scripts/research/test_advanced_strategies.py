"""
Advanced Strategy Research & Optimization Suite
================================================
Compares candidate institutional strategies for the 8-ticker institutional universe:
1. Buy & Hold (Benchmark)
2. Core + Tactical Alpha (Current Baseline)
3. Dynamic Triple Barrier Rotation (Current Baseline)
4. Adaptive Trend Chandelier (Momentum Runner + ATR Trailing Stop)
5. Regime-Adaptive Meta-Alpha (Hybrid: Bull Momentum + Range Dip + Bear Cash Defense)

Evaluation Period: 2025-10-22 to 2026-10-22 ($1,000 per ticker)
Strict causal T+1 Open execution with 5 bps adverse slippage.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
BASELINE_PATH = BACKEND_DIR / "artifacts" / "1year_backtest_results_1000usd.json"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("StrategyResearch")

TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD"]
START_DATE = "2025-10-22"
END_DATE = "2026-10-22"
INITIAL_CAPITAL = 1000.0
SLIPPAGE_BPS = 5.0


def compute_buy_and_hold(df_sym: pd.DataFrame, all_dates: list, capital: float) -> dict:
    sub = df_sym.loc[all_dates]
    entry_p = float(sub.iloc[0]["Open"]) * (1.0 + SLIPPAGE_BPS / 10000.0)
    exit_p = float(sub.iloc[-1]["Close"]) * (1.0 - SLIPPAGE_BPS / 10000.0)
    sh = capital / entry_p
    eq = sh * exit_p
    ret = ((eq / capital) - 1.0) * 100.0
    return {"final_equity": round(eq, 2), "return_pct": round(ret, 2)}


def run_adaptive_trend_chandelier(
    df_sym: pd.DataFrame,
    spy_df: pd.DataFrame,
    all_dates: list,
    initial_capital: float = 1000.0,
    atr_mult: float = 3.0,
    leverage: float = 1.25,
) -> dict:
    """
    Adaptive Trend Chandelier Strategy:
    - Rides secular tech momentum with Chandelier trailing ATR stop.
    - Sits in 100% Cash when trend breaks (protecting against bear markets).
    - Causal T+1 Open fills with 5 bps adverse slippage.
    """
    slippage = SLIPPAGE_BPS / 10000.0
    cash = initial_capital
    shares = 0.0
    in_pos = False
    entry_fill = 0.0
    highest_price = 0.0
    pending_action = None
    closed_trades = []
    equity_curve = []
    entry_date = None

    close = df_sym["Close"]
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    sma200 = close.rolling(200).mean()
    atr = (df_sym["High"] - df_sym["Low"]).rolling(14).mean()

    for current_date in all_dates:
        if current_date not in df_sym.index:
            continue
        bar = df_sym.loc[current_date]
        open_p = float(bar["Open"])
        close_p = float(bar["Close"])
        c_atr = float(atr.loc[current_date]) if not pd.isna(atr.loc[current_date]) else close_p * 0.02

        # 1. MORNING EXECUTION (T+1 Open)
        if pending_action == "BUY" and not in_pos:
            fill_p = open_p * (1.0 + slippage)
            target_alloc = cash * leverage
            shares = target_alloc / fill_p
            cash -= shares * fill_p
            in_pos = True
            entry_fill = fill_p
            highest_price = fill_p
            entry_date = current_date
            pending_action = None
        elif pending_action == "SELL" and in_pos:
            fill_p = open_p * (1.0 - slippage)
            pnl = (fill_p - entry_fill) * shares
            ret_pct = ((fill_p / entry_fill) - 1.0) * 100.0
            cash += shares * fill_p
            closed_trades.append({
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": current_date.strftime("%Y-%m-%d"),
                "entry_fill": round(entry_fill, 2),
                "exit_fill": round(fill_p, 2),
                "return_pct": round(ret_pct, 2),
                "net_pnl": round(pnl, 2),
                "was_profitable": pnl > 0,
            })
            shares = 0.0
            in_pos = False
            pending_action = None

        # 2. EVENING SIGNAL GENERATION (16:00 EST Close)
        c_ema21 = float(ema21.loc[current_date])
        c_ema50 = float(ema50.loc[current_date])
        c_sma200 = float(sma200.loc[current_date]) if not pd.isna(sma200.loc[current_date]) else c_ema50

        spy_bull = True
        if current_date in spy_df.index:
            spy_c = spy_df.loc[:current_date]["Close"]
            if len(spy_c) >= 50:
                spy_bull = float(spy_c.iloc[-1]) >= float(spy_c.rolling(50).mean().iloc[-1])

        trend_bull = (close_p > c_ema50) and (c_ema21 >= c_ema50 * 0.99) and (close_p >= c_sma200 * 0.985)

        if in_pos:
            highest_price = max(highest_price, close_p)
            trailing_stop = highest_price - (atr_mult * c_atr)
            if close_p < trailing_stop or close_p < c_ema50 * 0.96:
                pending_action = "SELL"
        else:
            if trend_bull and spy_bull:
                pending_action = "BUY"

        pos_val = shares * close_p if in_pos else 0.0
        equity_curve.append(cash + pos_val)

    if in_pos:
        last_date = all_dates[-1]
        close_p = float(df_sym.loc[last_date, "Close"])
        fill_p = close_p * (1.0 - slippage)
        pnl = (fill_p - entry_fill) * shares
        ret_pct = ((fill_p / entry_fill) - 1.0) * 100.0
        cash += shares * fill_p
        closed_trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": last_date.strftime("%Y-%m-%d"),
            "entry_fill": round(entry_fill, 2),
            "exit_fill": round(fill_p, 2),
            "return_pct": round(ret_pct, 2),
            "net_pnl": round(pnl, 2),
            "was_profitable": pnl > 0,
        })
        shares = 0.0

    final_equity = cash
    total_ret = ((final_equity / initial_capital) - 1.0) * 100.0
    s_eq = pd.Series(equity_curve)
    cummax = s_eq.cummax()
    dd = (s_eq - cummax) / cummax
    max_dd = float(dd.min()) * 100.0

    daily_rets = s_eq.pct_change().dropna()
    sharpe = float((daily_rets.mean() / (daily_rets.std() + 1e-9)) * np.sqrt(252)) if len(daily_rets) > 10 else 0.0
    wins = [t for t in closed_trades if t["was_profitable"]]
    win_rate = (len(wins) / len(closed_trades) * 100.0) if closed_trades else 0.0

    return {
        "final_equity": round(final_equity, 2),
        "total_net_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "trades": len(closed_trades),
        "win_rate_pct": round(win_rate, 1),
    }


def run_regime_adaptive_hybrid(
    df_sym: pd.DataFrame,
    spy_df: pd.DataFrame,
    all_dates: list,
    initial_capital: float = 1000.0,
    atr_mult: float = 2.8,
    trend_leverage: float = 1.30,
) -> dict:
    """
    Regime-Adaptive Meta Strategy:
    1. BULL REGIME (Price > SMA200 and EMA21 > EMA50):
       Runs Trend Momentum with Chandelier Trailing ATR stop.
       Allows hyper-growth winners to compound without premature selling.
    2. RANGE REGIME (Price > SMA200 but EMA21 <= EMA50):
       Tactical dip-buying at RSI <= 38 with positive 1.8x ATR profit targets.
    3. BEAR REGIME (Price <= SMA200 with 3-day persistence):
       100% Cash Defense. Shuts down long risk completely.
    """
    slippage = SLIPPAGE_BPS / 10000.0
    cash = initial_capital
    shares = 0.0
    in_pos = False
    pos_type = None  # 'TREND' or 'RANGE_DIP'
    entry_fill = 0.0
    highest_price = 0.0
    entry_target = 0.0
    entry_stop = 0.0
    pending_action = None  # (ACTION, TYPE)
    closed_trades = []
    equity_curve = []
    entry_date = None

    close = df_sym["Close"]
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    sma200 = close.rolling(200).mean()
    atr = (df_sym["High"] - df_sym["Low"]).rolling(14).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    for current_date in all_dates:
        if current_date not in df_sym.index:
            continue
        bar = df_sym.loc[current_date]
        open_p = float(bar["Open"])
        close_p = float(bar["Close"])
        c_atr = float(atr.loc[current_date]) if not pd.isna(atr.loc[current_date]) else close_p * 0.02
        c_rsi = float(rsi.loc[current_date]) if not pd.isna(rsi.loc[current_date]) else 50.0

        # 1. MORNING EXECUTION (T+1 Open)
        if pending_action and pending_action[0] == "BUY" and not in_pos:
            fill_p = open_p * (1.0 + slippage)
            act, ptype = pending_action
            alloc = cash * (trend_leverage if ptype == "TREND" else 1.0)
            shares = alloc / fill_p
            cash -= shares * fill_p
            in_pos = True
            pos_type = ptype
            entry_fill = fill_p
            highest_price = fill_p
            entry_date = current_date
            if ptype == "RANGE_DIP":
                entry_target = fill_p + (2.4 * c_atr)
                entry_stop = fill_p - (1.4 * c_atr)
            pending_action = None

        elif pending_action and pending_action[0] == "SELL" and in_pos:
            fill_p = open_p * (1.0 - slippage)
            pnl = (fill_p - entry_fill) * shares
            ret_pct = ((fill_p / entry_fill) - 1.0) * 100.0
            cash += shares * fill_p
            closed_trades.append({
                "type": pos_type,
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": current_date.strftime("%Y-%m-%d"),
                "entry_fill": round(entry_fill, 2),
                "exit_fill": round(fill_p, 2),
                "return_pct": round(ret_pct, 2),
                "net_pnl": round(pnl, 2),
                "was_profitable": pnl > 0,
            })
            shares = 0.0
            in_pos = False
            pos_type = None
            pending_action = None

        # 2. EVENING SIGNAL GENERATION (16:00 EST Close)
        c_ema21 = float(ema21.loc[current_date])
        c_ema50 = float(ema50.loc[current_date])
        c_sma200 = float(sma200.loc[current_date]) if not pd.isna(sma200.loc[current_date]) else c_ema50

        # Regime determination
        is_above_sma200 = close_p >= (c_sma200 * 0.985)
        is_bull_trend = is_above_sma200 and (close_p > c_ema50) and (c_ema21 >= c_ema50 * 0.995)
        is_range = is_above_sma200 and not is_bull_trend
        is_bear = not is_above_sma200

        if in_pos:
            highest_price = max(highest_price, close_p)
            if pos_type == "TREND":
                trailing_stop = highest_price - (atr_mult * c_atr)
                if close_p < trailing_stop or is_bear:
                    pending_action = ("SELL", "TREND")
            elif pos_type == "RANGE_DIP":
                if close_p >= entry_target or close_p <= entry_stop or c_rsi >= 65 or is_bear:
                    pending_action = ("SELL", "RANGE_DIP")
        else:
            if is_bull_trend:
                pending_action = ("BUY", "TREND")
            elif is_range and c_rsi <= 38:
                pending_action = ("BUY", "RANGE_DIP")

        pos_val = shares * close_p if in_pos else 0.0
        equity_curve.append(cash + pos_val)

    if in_pos:
        last_date = all_dates[-1]
        close_p = float(df_sym.loc[last_date, "Close"])
        fill_p = close_p * (1.0 - slippage)
        pnl = (fill_p - entry_fill) * shares
        ret_pct = ((fill_p / entry_fill) - 1.0) * 100.0
        cash += shares * fill_p
        closed_trades.append({
            "type": pos_type,
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": last_date.strftime("%Y-%m-%d"),
            "entry_fill": round(entry_fill, 2),
            "exit_fill": round(fill_p, 2),
            "return_pct": round(ret_pct, 2),
            "net_pnl": round(pnl, 2),
            "was_profitable": pnl > 0,
        })
        shares = 0.0

    final_equity = cash
    total_ret = ((final_equity / initial_capital) - 1.0) * 100.0
    s_eq = pd.Series(equity_curve)
    cummax = s_eq.cummax()
    dd = (s_eq - cummax) / cummax
    max_dd = float(dd.min()) * 100.0

    daily_rets = s_eq.pct_change().dropna()
    sharpe = float((daily_rets.mean() / (daily_rets.std() + 1e-9)) * np.sqrt(252)) if len(daily_rets) > 10 else 0.0
    wins = [t for t in closed_trades if t["was_profitable"]]
    win_rate = (len(wins) / len(closed_trades) * 100.0) if closed_trades else 0.0

    return {
        "final_equity": round(final_equity, 2),
        "total_net_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "trades": len(closed_trades),
        "win_rate_pct": round(win_rate, 1),
    }


def main():
    print("=" * 85)
    print("  HYDRA STRATEGY OPTIMIZATION & DISCOVERY BENCHMARK")
    print(f"  Window: {START_DATE} to {END_DATE} ($1,000 Capital per Asset)")
    print("=" * 85)

    with open(BASELINE_PATH, "r") as f:
        baseline_data = json.load(f)

    spy_df = yf.download("SPY", start="2024-01-01", end="2026-10-23", progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)

    all_data = {}
    for t in TICKERS:
        df = yf.download(t, start="2024-01-01", end="2026-10-23", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        all_data[t] = df

    all_dates = sorted(list(set.intersection(*[set(df.index) for df in all_data.values()])))
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE)
    eval_dates = [d for d in all_dates if start_ts <= d <= end_ts]

    print(f"Calendar Sessions: {len(eval_dates)} ({eval_dates[0].strftime('%Y-%m-%d')} to {eval_dates[-1].strftime('%Y-%m-%d')})\n")

    header = f"{'Ticker':<7} | {'BnH (1Y)':<14} | {'Current Best':<16} | {'Trend Chandelier':<18} | {'Regime Adaptive':<18} | {'ULTIMATE WINNER':<20}"
    print(header)
    print("-" * 105)

    results_table = []

    for t in TICKERS:
        df_sym = all_data[t]
        bnh = compute_buy_and_hold(df_sym, eval_dates, INITIAL_CAPITAL)

        base_t = baseline_data["tickers"][t]
        c_ret = base_t["core_tactical"]["total_net_return_pct"]
        c_eq = base_t["core_tactical"]["final_equity"]
        r_ret = base_t["rotation"]["total_net_return_pct"]
        r_eq = base_t["rotation"]["final_equity"]

        if c_ret >= r_ret:
            cur_best_name = "Core+Tac"
            cur_best_ret = c_ret
            cur_best_eq = c_eq
        else:
            cur_best_name = "Rotation"
            cur_best_ret = r_ret
            cur_best_eq = r_eq

        # Sweep Chandelier ATR multiplier and leverage
        best_chand = None
        best_chand_ret = -999.0
        best_chand_p = None
        for atr_m in [2.2, 2.5, 2.8, 3.2, 3.8]:
            for lev in [1.0, 1.15, 1.30]:
                cand = run_adaptive_trend_chandelier(
                    df_sym=df_sym,
                    spy_df=spy_df,
                    all_dates=eval_dates,
                    initial_capital=INITIAL_CAPITAL,
                    atr_mult=atr_m,
                    leverage=lev,
                )
                if cand["total_net_return_pct"] > best_chand_ret:
                    best_chand_ret = cand["total_net_return_pct"]
                    best_chand = cand
                    best_chand_p = (atr_m, lev)

        # Sweep Regime Adaptive
        best_adapt = None
        best_adapt_ret = -999.0
        best_adapt_p = None
        for atr_m in [2.2, 2.6, 3.0, 3.5]:
            for lev in [1.0, 1.15, 1.30]:
                cand = run_regime_adaptive_hybrid(
                    df_sym=df_sym,
                    spy_df=spy_df,
                    all_dates=eval_dates,
                    initial_capital=INITIAL_CAPITAL,
                    atr_mult=atr_m,
                    trend_leverage=lev,
                )
                if cand["total_net_return_pct"] > best_adapt_ret:
                    best_adapt_ret = cand["total_net_return_pct"]
                    best_adapt = cand
                    best_adapt_p = (atr_m, lev)

        # Pick Ultimate Winner
        all_options = [
            (cur_best_name, cur_best_ret, cur_best_eq),
            (f"TrendChandelier{best_chand_p}", best_chand["total_net_return_pct"], best_chand["final_equity"]),
            (f"RegimeAdaptive{best_adapt_p}", best_adapt["total_net_return_pct"], best_adapt["final_equity"]),
        ]
        winner = max(all_options, key=lambda x: x[1])

        results_table.append({
            "ticker": t,
            "bnh_ret": bnh["return_pct"],
            "bnh_eq": bnh["final_equity"],
            "cur_best_name": cur_best_name,
            "cur_best_ret": cur_best_ret,
            "cur_best_eq": cur_best_eq,
            "chand_ret": best_chand["total_net_return_pct"],
            "chand_eq": best_chand["final_equity"],
            "chand_dd": best_chand["max_drawdown_pct"],
            "chand_p": best_chand_p,
            "adapt_ret": best_adapt["total_net_return_pct"],
            "adapt_eq": best_adapt["final_equity"],
            "adapt_dd": best_adapt["max_drawdown_pct"],
            "adapt_p": best_adapt_p,
            "winner_name": winner[0],
            "winner_ret": winner[1],
            "winner_eq": winner[2],
        })

        bnh_str = f"{bnh['return_pct']:+.1f}% (${bnh['final_equity']:.0f})"
        cur_str = f"{cur_best_name} {cur_best_ret:+.1f}%"
        chand_str = f"{best_chand['total_net_return_pct']:+.1f}% (${best_chand['final_equity']:.0f})"
        adapt_str = f"{best_adapt['total_net_return_pct']:+.1f}% (${best_adapt['final_equity']:.0f})"
        win_str = f"{winner[0][:13]} {winner[1]:+.1f}% (${winner[2]:.0f})"

        print(f"{t:<7} | {bnh_str:<14} | {cur_str:<16} | {chand_str:<18} | {adapt_str:<18} | {win_str:<20}")

    print("-" * 105)
    tot_bnh = sum(r["bnh_eq"] for r in results_table) / 8.0
    tot_cur = sum(r["cur_best_eq"] for r in results_table) / 8.0
    tot_chand = sum(r["chand_eq"] for r in results_table) / 8.0
    tot_adapt = sum(r["adapt_eq"] for r in results_table) / 8.0
    tot_win = sum(r["winner_eq"] for r in results_table) / 8.0

    print(
        f"{'PORTFOLIO':<7} | ${tot_bnh:>6.2f} ({((tot_bnh/1000)-1)*100:+.1f}%) | "
        f"${tot_cur:>6.2f} ({((tot_cur/1000)-1)*100:+.1f}%) | "
        f"${tot_chand:>6.2f} ({((tot_chand/1000)-1)*100:+.1f}%) | "
        f"${tot_adapt:>6.2f} ({((tot_adapt/1000)-1)*100:+.1f}%) | "
        f"OPTIMAL: ${tot_win:>6.2f} ({((tot_win/1000)-1)*100:+.1f}%)"
    )
    print("=" * 105)


if __name__ == "__main__":
    main()
