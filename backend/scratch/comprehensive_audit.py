import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf

# pyrefly: ignore [missing-import]
from scripts.evaluation.point_in_time_validation import (
    fetch_aligned_market_data,
    run_point_in_time_signal_generation,
    calculate_horizon_performance,
    compute_aggregate_metrics,
)

print("=" * 80)
print("COMPREHENSIVE STATISTICAL AUDIT & ROBUSTNESS ENGINE")
print("=" * 80)

ticker_df, spy_df, vix_df = fetch_aligned_market_data("AAPL", "2022-01-01", "2026-09-30")

# 1. Walk-Forward Signals
wf_signals = run_point_in_time_signal_generation(
    ticker_df, spy_df, vix_df, "2024-01-01", "2026-09-30", strict_pit_slicing=False
)
wf_eval = calculate_horizon_performance(wf_signals, ticker_df)
wf_metrics = compute_aggregate_metrics(wf_eval, len(wf_signals))

# 2. Production Ledger Signals
conn = sqlite3.connect("artifacts/signal_ledger.db")
conn.row_factory = sqlite3.Row
ledger_rows = conn.execute(
    "SELECT bar_timestamp, signal, confidence, execution_price "
    "FROM signal_ledger WHERE symbol='AAPL' AND signal IN ('BUY', 'SELL') ORDER BY bar_timestamp ASC"
).fetchall()

ledger_sigs = []
for r in ledger_rows:
    b_date = r["bar_timestamp"]
    if b_date in ticker_df.index:
        close_p = float(ticker_df.loc[b_date, "Close"])
    else:
        close_p = float(r["execution_price"])
    ledger_sigs.append({
        "date": b_date,
        "confirmed_signal": r["signal"],
        "confidence": r["confidence"],
        "close_price": close_p,
        "timestamp_close": f"{b_date} 16:00:00",
    })

ledger_eval = calculate_horizon_performance(ledger_sigs, ticker_df)
ledger_metrics = compute_aggregate_metrics(ledger_eval, len(ledger_sigs))

# =========================================================
# STATISTICAL ROBUSTNESS (5-Day Horizon)
# =========================================================
def analyze_expectancy_and_stats(eval_records, horizon=5):
    valid = [r for r in eval_records if r["horizons"].get(horizon) is not None]
    
    buy_recs = [r for r in valid if r["signal"] == "BUY"]
    sell_recs = [r for r in valid if r["signal"] == "SELL"]
    
    def get_stats_for_subset(subset):
        if not subset:
            return {}
        rets = [r["horizons"][horizon]["return_net"] for r in subset]
        n_trades = len(rets)
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        
        n_win = len(wins)
        n_loss = len(losses)
        win_rate = n_win / n_trades if n_trades > 0 else 0
        loss_rate = n_loss / n_trades if n_trades > 0 else 0
        
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        median_ret = float(np.median(rets))
        std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        
        # Cumulative return & Max Drawdown
        cum_equity = np.cumprod(1 + np.array(rets))
        peak = np.maximum.accumulate(cum_equity)
        dd = (cum_equity - peak) / peak
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
        cum_ret = float(cum_equity[-1] - 1.0) if len(cum_equity) > 0 else 0.0
        
        # Profit factor
        sum_gains = sum(wins) if wins else 0.0
        sum_losses = abs(sum(losses)) if losses else 0.0
        profit_factor = sum_gains / sum_losses if sum_losses > 0 else float("inf")
        
        # Sharpe (per trade annualized: ~252 / 5 = ~50 trades/year)
        sharpe = (np.mean(rets) / std_ret) * np.sqrt(50) if std_ret > 0 else 0.0
        
        # 95% Confidence Interval for win rate
        ci_low, ci_high = stats.binomtest(n_win, n_trades).proportion_ci(confidence_level=0.95, method="wilson")
        
        return {
            "n_trades": n_trades,
            "n_win": n_win,
            "n_loss": n_loss,
            "win_rate": win_rate,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "expectancy": expectancy,
            "median_ret": median_ret,
            "std_ret": std_ret,
            "max_dd": max_dd,
            "cum_ret": cum_ret,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "rets": rets,
        }
    
    return {
        "ALL": get_stats_for_subset(valid),
        "BUY": get_stats_for_subset(buy_recs),
        "SELL": get_stats_for_subset(sell_recs),
    }

wf_stats_5d = analyze_expectancy_and_stats(wf_eval, 5)
ledger_stats_5d = analyze_expectancy_and_stats(ledger_eval, 5)

print("\n" + "=" * 60)
print("WALK-FORWARD (17 TRADES) EXPECTANCY & ROBUSTNESS (5-DAY)")
print("=" * 60)
for k in ["ALL", "BUY", "SELL"]:
    s = wf_stats_5d[k]
    print(f"\n--- SUBSET: {k} ---")
    print(f"  Trades: {s['n_trades']} (Wins: {s['n_win']}, Losses: {s['n_loss']})")
    print(f"  Win Rate: {s['win_rate']*100:.2f}% (95% CI: [{s['ci_95_low']*100:.2f}%, {s['ci_95_high']*100:.2f}%])")
    print(f"  Avg Win: {s['avg_win']*100:+.2f}% | Avg Loss: {s['avg_loss']*100:+.2f}%")
    print(f"  Win/Loss Ratio: {s['win_loss_ratio']:.2f}")
    print(f"  Expectancy per Trade: {s['expectancy']*100:+.2f}%")
    print(f"  Median Return: {s['median_ret']*100:+.2f}%")
    print(f"  Std Deviation: {s['std_ret']*100:.2f}%")
    print(f"  Profit Factor: {s['profit_factor']:.2f}")
    print(f"  Max Drawdown: {s['max_dd']*100:.2f}%")
    print(f"  Cumulative Compounded Return: {s['cum_ret']*100:+.2f}%")
    print(f"  Annualized Sharpe (50 trades/yr): {s['sharpe']:.2f}")

# BASELINE COMPARISON OVER 2024-01-01 to 2026-09-30
eval_mask = (ticker_df.index >= "2024-01-01") & (ticker_df.index <= "2026-09-30")
eval_df = ticker_df.loc[eval_mask]

# Buy and hold return
bnh_start_open = float(eval_df["Open"].iloc[0])
bnh_end_close = float(eval_df["Close"].iloc[-1])
bnh_return = (bnh_end_close - bnh_start_open) / bnh_start_open
bnh_equity = eval_df["Close"] / bnh_start_open
bnh_peak = bnh_equity.cummax()
bnh_dd = (bnh_equity - bnh_peak) / bnh_peak
bnh_max_dd = float(bnh_dd.min())

# Baseline 5-day drift
drift_5d = [(eval_df["Close"].iloc[i+5] - eval_df["Open"].iloc[i+1]) / eval_df["Open"].iloc[i+1] for i in range(len(eval_df) - 6)]
pct_positive_5d = np.mean([r > 0 for r in drift_5d])
pct_negative_5d = np.mean([r < 0 for r in drift_5d])
avg_drift_5d = float(np.mean(drift_5d))

print("\n" + "=" * 60)
print("BENCHMARK & MARKET DRIFT BASELINE (2024-01-01 -> 2026-09-30)")
print("=" * 60)
print(f"AAPL Buy & Hold Start: ${bnh_start_open:.2f} (2024-01-02 Open)")
print(f"AAPL Buy & Hold End:   ${bnh_end_close:.2f} (2026-09-29 Close)")
print(f"AAPL Buy & Hold Return: {bnh_return*100:+.2f}%")
print(f"AAPL Buy & Hold Max Drawdown: {bnh_max_dd*100:.2f}%")
print(f"All 5-Day Forward Windows Count: {len(drift_5d)}")
print(f"Always-LONG 5D Win Rate (Positive Drift): {pct_positive_5d*100:.2f}%")
print(f"Always-SHORT 5D Win Rate (Negative Drift): {pct_negative_5d*100:.2f}%")
print(f"Always-LONG 5D Average Return: {avg_drift_5d*100:+.2f}%")

# Statistical significance test
# Null Hypothesis 1: p = 0.50 (random guessing)
# Null Hypothesis 2: p = pct_positive_5d (beating long drift)
n_trades = wf_stats_5d["ALL"]["n_trades"]
n_win = wf_stats_5d["ALL"]["n_win"]

p_val_random = stats.binomtest(n_win, n_trades, p=0.5, alternative="greater").pvalue
p_val_drift = stats.binomtest(n_win, n_trades, p=pct_positive_5d, alternative="greater").pvalue

print("\n" + "=" * 60)
print("HYPOTHESIS TESTING (BINOMIAL TESTS)")
print("=" * 60)
print(f"H0: p = 0.50 (Random Guessing) -> p-value: {p_val_random:.4f} ({'Statistically Significant (p < 0.05)' if p_val_random < 0.05 else 'Not Significant'})")
print(f"H0: p = {pct_positive_5d:.4f} (Prevailing Market Drift) -> p-value: {p_val_drift:.4f} ({'Statistically Significant (p < 0.05)' if p_val_drift < 0.05 else 'Not Significant'})")

# Ledger stats summary
print("\n" + "=" * 60)
print("PRODUCTION LEDGER (45 TRADES) EXPECTANCY (5-DAY)")
print("=" * 60)
for k in ["ALL", "BUY", "SELL"]:
    s = ledger_stats_5d[k]
    print(f"\n--- LEDGER SUBSET: {k} ---")
    print(f"  Trades: {s['n_trades']} (Wins: {s['n_win']}, Losses: {s['n_loss']})")
    print(f"  Win Rate: {s['win_rate']*100:.2f}% (95% CI: [{s['ci_95_low']*100:.2f}%, {s['ci_95_high']*100:.2f}%])")
    print(f"  Avg Win: {s['avg_win']*100:+.2f}% | Avg Loss: {s['avg_loss']*100:+.2f}%")
    print(f"  Win/Loss Ratio: {s['win_loss_ratio']:.2f}")
    print(f"  Expectancy per Trade: {s['expectancy']*100:+.2f}%")
    print(f"  Profit Factor: {s['profit_factor']:.2f}")
    print(f"  Max Drawdown: {s['max_dd']*100:.2f}%")
    print(f"  Cumulative Compounded Return: {s['cum_ret']*100:+.2f}%")
    print(f"  Sharpe: {s['sharpe']:.2f}")

p_val_ledger_random = stats.binomtest(ledger_stats_5d["ALL"]["n_win"], ledger_stats_5d["ALL"]["n_trades"], p=0.5, alternative="greater").pvalue
print(f"\nLedger H0: p = 0.50 -> p-value: {p_val_ledger_random:.4f} ({'p < 0.05' if p_val_ledger_random < 0.05 else 'Not Significant at alpha=0.05'})")
