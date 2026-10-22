import json

with open("backend/artifacts/1year_backtest_results_1000usd.json") as f:
    res = json.load(f)

print("=" * 110)
print(f"{'Ticker':<7} | {'Buy & Hold':<12} | {'Core+Tactical':<15} | {'Dynamic Rotation':<18} | {'Oversold Rebound':<18} | {'Adaptive Mesh':<18}")
print("=" * 110)
for t in ["AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD"]:
    td = res["tickers"][t]
    b = td["buy_and_hold"]["net_return_pct"]
    c = td["core_tactical"]["total_net_return_pct"]
    r = td["rotation"]["total_net_return_pct"]
    o = td["oversold_rebound"]["total_net_return_pct"]
    a = td["adaptive"]["total_net_return_pct"]
    print(f"{t:<7} | {b:>+7.2f}%    | {c:>+7.2f}%       | {r:>+7.2f}%          | {o:>+7.2f}%          | {a:>+7.2f}%")

print("-" * 110)
p = res["portfolio"]
b_spy = res["benchmarks"]["SPY"]["net_return_pct"]
p_c = p["core_tactical"]["total_net_return_pct"]
p_r = p["rotation"]["total_net_return_pct"]
p_o = p["oversold_rebound"]["total_net_return_pct"]
p_a = p["adaptive"]["total_net_return_pct"]
print(f"{'PORT':<7} | {b_spy:>+7.2f}%    | {p_c:>+7.2f}%       | {p_r:>+7.2f}%          | {p_o:>+7.2f}%          | {p_a:>+7.2f}%")
print("=" * 110)
