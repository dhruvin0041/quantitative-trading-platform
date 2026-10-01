import sqlite3
import pandas as pd
# pyrefly: ignore [missing-import]
from scripts.evaluation.point_in_time_validation import (
    fetch_aligned_market_data,
    calculate_horizon_performance,
    format_eastern_timestamp,
)

ticker_df, spy_df, vix_df = fetch_aligned_market_data('AAPL', '2024-01-01', '2026-09-30')

conn = sqlite3.connect('artifacts/signal_ledger.db')
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT bar_timestamp, signal, confidence, execution_price "
    "FROM signal_ledger WHERE symbol='AAPL' AND signal IN ('BUY', 'SELL') ORDER BY bar_timestamp ASC"
).fetchall()

sigs = []
for r in rows:
    b_date = r['bar_timestamp']
    if b_date in ticker_df.index:
        close_p = float(ticker_df.loc[b_date, 'Close'])
    else:
        close_p = float(r['execution_price'])
    sigs.append({
        'date': b_date,
        'confirmed_signal': r['signal'],
        'confidence': r['confidence'],
        'close_price': close_p,
        'timestamp_close': format_eastern_timestamp(b_date, 16, 0),
    })

eval_recs = calculate_horizon_performance(sigs, ticker_df)
print(f"Total evaluated ledger signals: {len(eval_recs)}")

print("\n" + "=" * 135)
print(f"{'DATE':<12} | {'SIGNAL':<6} | {'SIGNAL PRICE':<12} | {'NEXT OPEN':<10} | {'5D CLOSE':<10} | {'1D NET RET':<10} | {'3D NET RET':<10} | {'5D NET RET':<10} | {'10D NET RET':<11} | {'5D OUTCOME':<10}")
print("=" * 135)

# Print recent 24 signals (all 23 signals from 2026 + late 2025)
for r in eval_recs[-24:]:
    d = r['source_date']
    sig = r['signal']
    sig_p = f"${r['signal_price']:.2f}"
    nxt_o = f"${r['next_open']:.2f}"
    
    h1 = r['horizons'].get(1)
    h3 = r['horizons'].get(3)
    h5 = r['horizons'].get(5)
    h10 = r['horizons'].get(10)
    
    r1_str = f"{h1['return_net']*100:+.2f}%" if h1 else "N/A"
    r3_str = f"{h3['return_net']*100:+.2f}%" if h3 else "N/A"
    r5_str = f"{h5['return_net']*100:+.2f}%" if h5 else "N/A"
    r10_str = f"{h10['return_net']*100:+.2f}%" if h10 else "N/A"
    fut_p = f"${h5['exit_close']:.2f}" if h5 else "N/A"
    outcome = "CORRECT" if (h5 and h5['is_correct']) else ("INCORRECT" if h5 else "PENDING")
    
    print(f"{d:<12} | {sig:<6} | {sig_p:<12} | {nxt_o:<10} | {fut_p:<10} | {r1_str:<10} | {r3_str:<10} | {r5_str:<10} | {r10_str:<11} | {outcome:<10}")
