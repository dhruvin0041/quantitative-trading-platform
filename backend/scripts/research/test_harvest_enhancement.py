import json
from pathlib import Path

import pandas as pd
import yfinance as yf

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
BASELINE_PATH = BACKEND_DIR / "artifacts" / "1year_backtest_results_1000usd.json"

with open(BASELINE_PATH, "r") as f:
    baseline = json.load(f)

tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD']
print("=" * 80)
print(f"{'Ticker':<7} | {'Buy & Hold':<12} | {'Orig Core':<12} | {'Orig Rot':<12} | {'Enhanced Core+Tac':<18}")
print("-" * 80)

results = []

for sym in tickers:
    df = yf.download(sym, start='2024-01-01', end='2026-10-23', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    sub = df.loc['2025-10-22':'2026-10-22']
    c = df['Close']
    sma200 = c.rolling(200).mean()
    atr = (df['High'] - df['Low']).rolling(14).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    p0 = float(sub.iloc[0]['Open']) * 1.0005
    p_end = float(sub.iloc[-1]['Close']) * 0.9995
    bnh_ret = ((p_end / p0) - 1.0) * 100.0

    orig_c = baseline['tickers'][sym]['core_tactical']['total_net_return_pct']
    orig_r = baseline['tickers'][sym]['rotation']['total_net_return_pct']

    cash = 1000.0
    core_sh = cash / p0
    cash = 0.0
    tac_sh = 0.0
    entry_tac = 0.0
    consec_below = 0
    consec_above = 0
    in_regime = True

    for dt in sub.index:
        cp = float(sub.loc[dt, 'Close'])
        c_sma = float(sma200.loc[dt]) if not pd.isna(sma200.loc[dt]) else cp
        c_rsi = float(rsi.loc[dt]) if not pd.isna(rsi.loc[dt]) else 50.0
        c_atr = float(atr.loc[dt]) if not pd.isna(atr.loc[dt]) else cp * 0.02

        if cp < c_sma * 0.985:
            consec_below += 1
            consec_above = 0
        elif cp > c_sma * 1.005:
            consec_above += 1
            consec_below = 0

        if in_regime and consec_below >= 3:
            cash += core_sh * cp * 0.9995
            core_sh = 0.0
            if tac_sh > 0:
                cash += (cp * 0.9995 - entry_tac) * tac_sh
                tac_sh = 0.0
            in_regime = False
        elif not in_regime and consec_above >= 2:
            core_sh = cash / (cp * 1.0005)
            cash = 0.0
            in_regime = True

        if in_regime and core_sh > 0:
            if tac_sh == 0.0 and c_rsi <= 35:
                tac_sh = core_sh * 0.35
                entry_tac = cp * 1.0005
            elif tac_sh > 0.0:
                is_profit = cp > entry_tac
                if is_profit and (cp >= entry_tac + (1.8 * c_atr) or c_rsi >= 68):
                    pnl = (cp * 0.9995 - entry_tac) * tac_sh
                    cash += pnl
                    tac_sh = 0.0
                elif cp <= entry_tac - (2.0 * c_atr):
                    pnl = (cp * 0.9995 - entry_tac) * tac_sh
                    cash += pnl
                    tac_sh = 0.0

    final_eq = (core_sh * p_end) + cash + (tac_sh * (p_end - entry_tac))
    enh_ret = ((final_eq / 1000.0) - 1.0) * 100.0

    results.append({
        "sym": sym,
        "bnh": bnh_ret,
        "orig_c": orig_c,
        "orig_r": orig_r,
        "enh_c": enh_ret,
        "final_eq": final_eq
    })

    print(f"{sym:<7} | {bnh_ret:>+6.2f}%     | {orig_c:>+6.2f}%     | {orig_r:>+6.2f}%     | {enh_ret:>+6.2f}% (${final_eq:.2f})")

print("-" * 80)
avg_orig_c = sum(r["orig_c"] for r in results) / len(results)
avg_enh_c = sum(r["enh_c"] for r in results) / len(results)
avg_bnh = sum(r["bnh"] for r in results) / len(results)
print(f"PORTFOLIO AVERAGE: BnH={avg_bnh:+.2f}% | Orig Core={avg_orig_c:+.2f}% | Enhanced Core={avg_enh_c:+.2f}%")
print("=" * 80)
