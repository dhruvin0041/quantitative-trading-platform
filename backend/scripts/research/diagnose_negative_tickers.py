import pandas as pd
import yfinance as yf

tickers = ["MSFT", "META", "TSLA"]
print("=" * 80)
print("  DIAGNOSING AND FIXING NEGATIVE TICKERS: MSFT, META, TSLA")
print("=" * 80)

for sym in tickers:
    df = yf.download(sym, start="2024-01-01", end="2026-10-23", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    sub = df.loc["2025-10-22":"2026-10-22"]
    c = sub["Close"]
    sma200 = df["Close"].rolling(200).mean().loc["2025-10-22":"2026-10-22"]
    ema50 = df["Close"].ewm(span=50).mean().loc["2025-10-22":"2026-10-22"]
    ema20 = df["Close"].ewm(span=20).mean().loc["2025-10-22":"2026-10-22"]
    atr = (df["High"] - df["Low"]).rolling(14).mean().loc["2025-10-22":"2026-10-22"]
    delta = c.diff()
    rsi = 100.0 - (100.0 / (1.0 + (delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean())))

    start_o = float(sub.iloc[0]["Open"])
    end_c = float(sub.iloc[-1]["Close"])
    bnh_ret = ((end_c / start_o) - 1.0) * 100.0

    print(f"\n=================== {sym} ===================")
    print(f"Underlying Buy & Hold: {bnh_ret:+.2f}% (${start_o:.2f} -> ${end_c:.2f})")

    # TEST STRATEGY 1: Pure Trend Following (Only Long when Close > EMA50 and EMA20 > EMA50, Exit to Cash immediately when trend breaks)
    cash = 1000.0
    shares = 0.0
    in_pos = False
    entry_p = 0.0
    trades_trend = []
    for dt in sub.index:
        cp = float(c.loc[dt])
        e50 = float(ema50.loc[dt])
        e20 = float(ema20.loc[dt])
        trend_up = (cp > e50) and (e20 > e50)

        if not in_pos and trend_up:
            entry_p = cp * 1.0005
            shares = cash / entry_p
            cash = 0.0
            in_pos = True
        elif in_pos and not trend_up:
            exit_p = cp * 0.9995
            pnl = (exit_p - entry_p) * shares
            ret_pct = ((exit_p / entry_p) - 1.0) * 100.0
            cash = shares * exit_p
            trades_trend.append(ret_pct)
            shares = 0.0
            in_pos = False

    if in_pos:
        cash = shares * float(c.iloc[-1]) * 0.9995
    trend_ret = ((cash / 1000.0) - 1.0) * 100.0
    print(f"Strategy 1 (Pure Trend Following + Cash Defense): Return = {trend_ret:+.2f}% (${cash:.2f}), Trades = {len(trades_trend)}")

    # TEST STRATEGY 2: Trend Long + Bear Short
    # When Close > EMA50 & EMA20 > EMA50: LONG
    # When Close < EMA50 & EMA20 < EMA50: SHORT (Ride the downtrend down!)
    cash = 1000.0
    pos_side = None
    shares = 0.0
    entry_p = 0.0
    trades_ls = []
    for dt in sub.index:
        cp = float(c.loc[dt])
        e50 = float(ema50.loc[dt])
        e20 = float(ema20.loc[dt])
        trend_up = (cp > e50) and (e20 > e50)
        trend_down = (cp < e50) and (e20 < e50)

        if pos_side == "LONG" and not trend_up:
            exit_p = cp * 0.9995
            ret_pct = ((exit_p / entry_p) - 1.0) * 100.0
            cash = shares * exit_p
            trades_ls.append(("LONG", ret_pct))
            pos_side = None
            shares = 0.0
        elif pos_side == "SHORT" and not trend_down:
            exit_p = cp * 1.0005
            ret_pct = ((entry_p / exit_p) - 1.0) * 100.0
            cash = cash + (shares * (entry_p - exit_p))
            trades_ls.append(("SHORT", ret_pct))
            pos_side = None
            shares = 0.0

        if pos_side is None:
            if trend_up:
                entry_p = cp * 1.0005
                shares = cash / entry_p
                cash = 0.0
                pos_side = "LONG"
            elif trend_down:
                entry_p = cp * 0.9995
                shares = cash / entry_p
                pos_side = "SHORT"

    if pos_side == "LONG":
        cash = shares * float(c.iloc[-1]) * 0.9995
    elif pos_side == "SHORT":
        cash = cash + (shares * (entry_p - float(c.iloc[-1]) * 1.0005))
    ls_ret = ((cash / 1000.0) - 1.0) * 100.0
    print(f"Strategy 2 (Long Bull Trend / Short Bear Trend): Return = {ls_ret:+.2f}% (${cash:.2f}), Trades = {len(trades_ls)}")

    # TEST STRATEGY 3: Mean-Reversion Oversold Rebounds (Bollinger %B / RSI <= 30 dip buying with quick profit taking at 20 EMA)
    # Buy when RSI <= 32. Sell when Close >= EMA20 or RSI >= 55. Stop loss at -1.5 ATR.
    cash = 1000.0
    shares = 0.0
    in_pos = False
    entry_p = 0.0
    trades_mr = []
    for dt in sub.index:
        cp = float(c.loc[dt])
        c_rsi = float(rsi.loc[dt]) if not pd.isna(rsi.loc[dt]) else 50.0
        c_atr = float(atr.loc[dt]) if not pd.isna(atr.loc[dt]) else cp * 0.02
        e20 = float(ema20.loc[dt])

        if not in_pos and c_rsi <= 32:
            entry_p = cp * 1.0005
            shares = cash / entry_p
            cash = 0.0
            in_pos = True
        elif in_pos:
            # Profit target: reached 20 EMA or RSI >= 55
            # Stop loss: dropped > 1.8 ATR
            if cp >= e20 or c_rsi >= 55 or cp <= entry_p - (1.8 * c_atr):
                exit_p = cp * 0.9995
                ret_pct = ((exit_p / entry_p) - 1.0) * 100.0
                cash = shares * exit_p
                trades_mr.append(ret_pct)
                shares = 0.0
                in_pos = False

    if in_pos:
        cash = shares * float(c.iloc[-1]) * 0.9995
    mr_ret = ((cash / 1000.0) - 1.0) * 100.0
    print(f"Strategy 3 (Oversold Mean-Reversion Dip Rebounds): Return = {mr_ret:+.2f}% (${cash:.2f}), Trades = {len(trades_mr)}")
