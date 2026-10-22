"""
Verification script for live signal generation and strategy states across all 8 institutional tickers.
"""
import json
import urllib.request

TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD"]

print("=" * 85)
print(f"{'Ticker':<7} | {'Signal':<6} | {'Confidence':<10} | {'Regime':<12} | {'Markers':<8} | {'Strategy Mode':<15}")
print("=" * 85)

for ticker in TICKERS:
    url = f"http://localhost:8000/predict?ticker={ticker}"
    req = urllib.request.Request(url, headers={"X-API-Key": "hydra-secure-key-2026"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            sig = data.get("signal", "N/A")
            conf = f"{float(data.get('confidence_score', 0)):.2%}"
            regime = data.get("market_regime", "N/A")
            markers = len(data.get("markers", []))
            mode = data.get("strategy_mode", "N/A")
            tactical = data.get("tactical_action", "N/A")
            print(f"{ticker:<7} | {sig:<6} | {conf:<10} | {regime:<12} | {markers:<8} | {mode:<15}")
    except Exception as e:
        print(f"{ticker:<7} | ERROR: {e}")

print("=" * 85)
