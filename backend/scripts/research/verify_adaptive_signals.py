from pathlib import Path

import pandas as pd

csv_path = Path("backend/backtest_results/backtest_trades.csv")
if not csv_path.exists():
    print(f"File {csv_path} not found")
    exit(1)

df = pd.read_csv(csv_path)
print("=" * 90)
print("  AUDIT OF BACKTEST TRADES IN INSTITUTIONAL ADAPTIVE MESH")
print("=" * 90)
print(f"Total trades: {len(df)}")
print(f"Unique tickers: {df['ticker'].unique().tolist()}")

for sym in ["META", "TSLA", "MSFT", "NVDA", "AMZN", "AAPL", "GOOGL", "AMD"]:
    sub = df[df["ticker"] == sym]
    wins = len(sub[sub["was_profitable"]])
    total = len(sub)
    win_rate = (wins / total * 100) if total > 0 else 0
    tot_pnl = sub["net_pnl"].sum()
    print(f"\n[{sym}] Total Trades: {total} | Wins: {wins} ({win_rate:.1f}%) | Total Net PnL: ${tot_pnl:,.2f}")
    for _, r in sub.head(4).iterrows():
        print(f"   {r['side']:<14} | {r['entry_date']} -> {r['exit_date']} | Entry: ${r['entry_fill']:.2f} -> Exit: ${r['exit_fill']:.2f} | Net: ${r['net_pnl']:+6.2f} ({r['return_pct']:+5.2f}%) | Reason: {r['exit_reason']}")
    if len(sub) > 4:
        print(f"   ... and {len(sub) - 4} more trades")
