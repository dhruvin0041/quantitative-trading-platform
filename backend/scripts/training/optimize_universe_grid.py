import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import json

import pandas as pd

TICKERS = [
    # US Tech
    "AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    # US Defensives & Financials
    "COST", "WMT", "JPM", "LLY",
    # Indian NSE Giants
    "RELIANCE.NS", "TCS.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "LT.NS", "ITC.NS"
]

def backtest_ticker(df, strat_type, params, is_indian=False, spy_df=None):
    init_entry = params.get("init_entry", False)
    base_atr = params.get("base_atr", 3.8)
    ratchet_15 = params.get("ratchet_15", 3.0)
    ratchet_25 = params.get("ratchet_25", 2.2)
    climax_rsi = params.get("climax_rsi", 82.0)
    climax_gain = params.get("climax_gain", 0.18)
    cooldown_max = params.get("cooldown", 2)
    ema_fast_p = int(params.get("ema_fast", 8))
    ema_slow_p = int(params.get("ema_slow", 21))
    dip_rsi = params.get("dip_rsi", 44.0)
    macro_bear_exit = params.get("macro_bear_exit", True)

    close_s = df["Close"]
    open_s = df["Open"]
    high_s = df["High"]
    low_s = df["Low"]
    rsi_s = df["RSI"] if "RSI" in df.columns else pd.Series(50.0, index=df.index)

    # 14 ATR
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    atr_14_s = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14, min_periods=5).mean().bfill()

    # Indicators
    ema_fast_s = close_s.ewm(span=ema_fast_p).mean()
    ema_slow_s = close_s.ewm(span=ema_slow_p).mean()
    sma200_s = close_s.rolling(200, min_periods=20).mean().bfill()
    ema50_s = close_s.ewm(span=50).mean()

    # MACD
    ema12 = close_s.ewm(span=12).mean()
    ema26 = close_s.ewm(span=26).mean()
    macd_hist_s = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()

    # Macro regime
    if spy_df is not None and not is_indian:
        spy_close = spy_df["Close"].reindex(df.index).ffill().bfill()
        spy_sma200 = spy_close.rolling(200, min_periods=20).mean().bfill()
        macro_spy_bull_s = spy_close >= spy_sma200 * 0.985
    else:
        macro_spy_bull_s = close_s >= sma200_s * 0.980

    slippage_rate = 0.0005
    initial_capital = 1000.0
    cash = initial_capital
    shares = 0.0
    trades = []
    last_buy = None

    current_pos = "FLAT"
    entry_price = 0.0
    peak_price = 0.0
    trailing_stop = 0.0
    cooldown = 0
    consec_below = 0
    pending_action = None

    chart_start_idx = max(0, len(df) - 252)
    chart_indices = df.index[chart_start_idx:]

    # Initial entry
    if init_entry and len(chart_indices) > 0:
        c0 = float(close_s.loc[chart_indices[0]])
        s0 = float(sma200_s.loc[chart_indices[0]])
        e50_0 = float(ema50_s.loc[chart_indices[0]])
        is_init_bull = (c0 >= s0 * 0.980 or c0 >= e50_0)
        if is_init_bull and len(chart_indices) > 1:
            p_exec = round(float(open_s.loc[chart_indices[1]]) * (1.0 + slippage_rate), 2)
            shares = cash / p_exec
            cash = 0.0
            last_buy = {"buy_time": str(chart_indices[0])[:10], "buy_price": p_exec}
            current_pos = "LONG"
            entry_price = p_exec
            peak_price = p_exec
            c_atr0 = float(atr_14_s.loc[chart_indices[0]])
            trailing_stop = p_exec - (base_atr * c_atr0)

    for i in range(len(chart_indices)):
        dt = chart_indices[i]
        cur_c = float(close_s.loc[dt])
        cur_o = float(open_s.loc[dt])
        cur_h = float(high_s.loc[dt])
        cur_l = float(low_s.loc[dt])
        cur_rsi = float(rsi_s.loc[dt])
        cur_atr = float(atr_14_s.loc[dt])
        cur_sma200 = float(sma200_s.loc[dt])
        cur_ema_fast = float(ema_fast_s.loc[dt])
        cur_ema_slow = float(ema_slow_s.loc[dt])
        cur_hist = float(macd_hist_s.loc[dt])
        macro_bull = bool(macro_spy_bull_s.loc[dt])
        asset_bull = cur_c >= cur_sma200 * 0.980
        regime_bull = macro_bull and asset_bull

        # Morning execution
        if pending_action == "BUY" and current_pos == "FLAT":
            p_exec = round(cur_o * (1.0 + slippage_rate), 2)
            shares = cash / p_exec
            cash = 0.0
            last_buy = {"buy_time": str(dt)[:10], "buy_price": p_exec}
            current_pos = "LONG"
            entry_price = p_exec
            peak_price = cur_h
            init_stop_dist = min(base_atr * cur_atr, p_exec * 0.08)
            trailing_stop = entry_price - init_stop_dist
            pending_action = None
            consec_below = 0
        elif pending_action == "SELL" and current_pos == "LONG":
            p_exec = round(cur_o * (1.0 - slippage_rate), 2)
            sell_proceeds = shares * p_exec
            pnl = sell_proceeds - (shares * last_buy["buy_price"])
            ret_pct = ((p_exec / last_buy["buy_price"]) - 1.0) * 100.0
            trades.append({
                "buy_time": last_buy["buy_time"],
                "buy_price": last_buy["buy_price"],
                "sell_time": str(dt)[:10],
                "sell_price": p_exec,
                "pnl": pnl,
                "return_pct": ret_pct
            })
            cash = sell_proceeds
            shares = 0.0
            last_buy = None
            current_pos = "FLAT"
            entry_price = 0.0
            pending_action = None
            cooldown = cooldown_max

        if current_pos == "FLAT" and cooldown > 0:
            cooldown -= 1

        bullish_candle = (cur_c > cur_o) or (cur_c > cur_l + 0.35 * (cur_h - cur_l))

        if cur_c < cur_sma200 * 0.970:
            consec_below += 1
        else:
            consec_below = 0

        # Evening evaluation
        if current_pos == "LONG" and not pending_action:
            peak_price = max(peak_price, cur_h)
            gain_pct = (cur_c - entry_price) / max(entry_price, 1e-6)

            # Trailing ratchets
            if gain_pct >= 0.25:
                new_stop = max(entry_price * 1.06, peak_price - (ratchet_25 * cur_atr))
            elif gain_pct >= 0.15:
                new_stop = max(entry_price * 1.02, peak_price - (ratchet_15 * cur_atr))
            else:
                new_stop = peak_price - (base_atr * cur_atr)
            trailing_stop = max(trailing_stop, new_stop)

            hit_macro_bear = ((not macro_bull and cur_c < cur_sma200 * 0.965) or (consec_below >= 5)) if macro_bear_exit else False

            # Climax tightening (tighten stop instead of instant dump)
            hit_climax = False
            if climax_rsi and cur_rsi >= climax_rsi and gain_pct >= climax_gain:
                tight_stop = peak_price - (1.6 * cur_atr)
                trailing_stop = max(trailing_stop, tight_stop)
                if cur_c < trailing_stop:
                    hit_climax = True

            hit_trailing = (cur_c < trailing_stop)

            if hit_macro_bear or hit_climax or hit_trailing:
                pending_action = "SELL"

        elif current_pos == "FLAT" and not pending_action and cooldown == 0:
            trigger_buy = False
            rsi_max = 84.0 if (cur_c >= cur_sma200 or cur_ema_fast >= cur_ema_slow) else 74.0

            if strat_type == "VOLATILITY_BREAKOUT_SWING":
                cond_macd = (cur_hist > 0)
                cond_break = (cur_c >= cur_ema_slow * 0.995) and (45.0 <= cur_rsi <= rsi_max) and bullish_candle
                if regime_bull and cond_macd and cond_break:
                    trigger_buy = True
            elif strat_type in ["CORE_TACTICAL_ACCUMULATION", "DYNAMIC_REGIME_HYBRID"]:
                if regime_bull:
                    cond_pb = (cur_rsi <= dip_rsi) and bullish_candle
                    cond_break = (cur_c >= cur_ema_slow * 0.995) and (cur_hist > 0) and (45.0 <= cur_rsi <= rsi_max) and bullish_candle
                    if cond_pb or cond_break:
                        trigger_buy = True
                else:
                    if cur_rsi <= 28.0 and bullish_candle:
                        trigger_buy = True
            elif strat_type == "ASYMMETRIC_MEAN_REVERSION":
                if regime_bull:
                    cond_pb = (cur_rsi <= dip_rsi) and bullish_candle
                    cond_break = (cur_c >= cur_ema_slow * 0.995) and (cur_hist > 0) and (45.0 <= cur_rsi <= rsi_max) and bullish_candle
                    if cond_pb or cond_break:
                        trigger_buy = True
                else:
                    if cur_rsi <= 30.0 and bullish_candle:
                        trigger_buy = True

            if trigger_buy:
                pending_action = "BUY"

    cur_p = float(close_s.iloc[-1])
    if shares > 0 and cur_p > 0:
        final_val = shares * cur_p
    else:
        final_val = cash

    p0 = float(close_s.loc[chart_indices[0]])
    pt = float(close_s.loc[chart_indices[-1]])
    bh_ret = ((pt - p0) / p0) * 100.0
    strat_ret = ((final_val - initial_capital) / initial_capital) * 100.0

    wins = [t for t in trades if t["pnl"] > 0]
    wr = (len(wins) / len(trades) * 100.0) if trades else (100.0 if shares > 0 and final_val > initial_capital else 0.0)

    return {
        "final_val": final_val,
        "strat_ret": strat_ret,
        "bh_ret": bh_ret,
        "alpha": strat_ret - bh_ret,
        "trades": len(trades),
        "win_rate": wr,
        "has_open": shares > 0
    }

spy_df = pd.read_parquet("backend/data/cache/SPY_2y_features.parquet") if Path("backend/data/cache/SPY_2y_features.parquet").exists() else None

# Parameter search space per asset class
best_configs = {}

print("Optimizing strategy parameters across institutional universe...")

for sym in TICKERS:
    pq_path = Path(f"backend/data/cache/{sym}_2y_features.parquet")
    if not pq_path.exists():
        continue
    df = pd.read_parquet(pq_path)
    is_indian = sym.endswith(".NS")

    # Candidate strategy types
    strat_types = ["CORE_TACTICAL_ACCUMULATION", "VOLATILITY_BREAKOUT_SWING", "ASYMMETRIC_MEAN_REVERSION"]
    # Candidate parameter grids
    base_atrs = [3.2, 3.8, 4.2]
    ratchet_15s = [2.8, 3.2]
    ratchet_25s = [2.0, 2.5]
    dip_rsis = [40.0, 44.0, 46.0]
    climax_rsis = [80.0, 84.0]
    init_entries = [True, False]
    cooldowns = [1, 2, 3]
    ema_fasts = [5, 7, 9]
    ema_slows = [13, 21, 30]

    best_res = None
    best_params = None
    best_st = None

    # Sample a diverse high-performance grid
    for st in strat_types:
        for batr in [3.2, 3.8, 4.2]:
            for drsi in [42.0, 45.0]:
                for ie in [True, False]:
                    for cd in [1, 2]:
                        for ef, es in [(6, 15), (8, 21), (9, 26)]:
                            p = {
                                "init_entry": ie,
                                "base_atr": batr,
                                "ratchet_15": batr * 0.8,
                                "ratchet_25": batr * 0.6,
                                "climax_rsi": 82.0,
                                "climax_gain": 0.18,
                                "dip_rsi": drsi,
                                "cooldown": cd,
                                "ema_fast": ef,
                                "ema_slow": es,
                                "macro_bear_exit": True,
                                "bb_window": 20,
                                "bb_mult": 2.1,
                                "bb_adaptive": True,
                                "ema_adaptive": True
                            }
                            res = backtest_ticker(df, st, p, is_indian=is_indian, spy_df=spy_df)
                            if best_res is None or res["final_val"] > best_res["final_val"]:
                                best_res = res
                                best_params = p
                                best_st = st

    best_configs[sym] = {
        "strategy_type": best_st,
        "strategy_name": best_st.replace("_", " ").title(),
        "params": best_params,
        "performance": best_res
    }

print("\n" + "=" * 115)
print(f"{'Ticker':<14} | {'Best Strategy':<28} | {'Final Val':<11} | {'Strat Ret%':<11} | {'B&H Ret%':<10} | {'Alpha %':<9} | {'Win Rate':<8} | {'Trades'}")
print("=" * 115)

tot_bh = 0.0
tot_best = 0.0

for sym, cfg in best_configs.items():
    p = cfg["performance"]
    tot_bh += 1000.0 * (1.0 + p["bh_ret"] / 100.0)
    tot_best += p["final_val"]
    print(f"{sym:<14} | {cfg['strategy_type']:<28} | ${p['final_val']:<10.2f} | {p['strat_ret']:>+9.2f}% | {p['bh_ret']:>+8.2f}% | {p['alpha']:>+7.2f}% | {p['win_rate']:>6.1f}% | {p['trades']}")

tot_bh_ret = ((tot_bh - 18000.0) / 18000.0) * 100.0
tot_best_ret = ((tot_best - 18000.0) / 18000.0) * 100.0
print("-" * 115)
print(f"{'PORTFOLIO':<14} | {'INSTITUTIONAL ENSEMBLE':<28} | ${tot_best:<10.2f} | {tot_best_ret:>+9.2f}% | {tot_bh_ret:>+8.2f}% | {tot_best_ret - tot_bh_ret:>+7.2f}% | {'-':<8} | -")

# Export champion configs
with open("backend/scratch/optimized_champion_configs.json", "w") as f:
    json.dump(best_configs, f, indent=2)

print("\nOptimized configurations saved to backend/scratch/optimized_champion_configs.json")
