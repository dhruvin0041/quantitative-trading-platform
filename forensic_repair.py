import json
import os
import asyncio
from src.execution.fx_engine import FXEngine
from datetime import datetime

async def forensic_repair():
    db_path = "data/paper_trading.json"
    if not os.path.exists(db_path):
        print("No DB found.")
        return

    with open(db_path, "r") as f:
        db = json.load(f)

    fx = FXEngine()
    await fx.update_rates()
    
    base_curr = db.get("base_currency", "USD")
    print(f"Base Currency identified as: {base_curr}")
    
    # 1. FIX THE BASELINE (SSOT)
    # The baseline capital must be normalized to the same currency as the current equity.
    # If initial_capital is 1M and base is INR, it MUST be ~95M.
    if base_curr != "USD":
        expected_initial = fx.convert_to_base(1000000.0, "USD", base_curr)
        print(f"Repairing Initial Capital: {db.get('initial_capital')} -> {expected_initial}")
        db["initial_capital"] = expected_initial
        
        # Ensure capital (cash) is also normalized
        if db["capital"] < expected_initial * 0.1: # Heuristic: if cash is suspiciously low
            old_cash = db["capital"]
            db["capital"] = fx.convert_to_base(old_cash, "USD", base_curr)
            print(f"Repairing Cash: {old_cash} -> {db['capital']}")
    else:
        db["initial_capital"] = 1000000.0

    # 2. SANITIZE EQUITY CURVE
    # We must remove snapshots that are in the wrong currency or are outliers.
    valid_snaps = []
    threshold = db["initial_capital"] * 0.5 # Sanity check: must be at least 50% of initial
    
    for s in db.get("portfolio_snapshots", []):
        # Only keep snapshots that match current base currency and are above threshold
        if s.get("base_currency") == base_curr and s["equity"] > threshold:
            valid_snaps.append(s)
    
    if not valid_snaps:
        print("Equity curve completely contaminated. Re-baselining from current state.")
        valid_snaps = [{
            "time": datetime.now().isoformat(),
            "equity": db["capital"],
            "cash": db["capital"],
            "base_currency": base_curr
        }]
    
    db["portfolio_snapshots"] = valid_snaps
    print(f"Sanitized equity curve: {len(db['portfolio_snapshots'])} clean points remaining.")

    # 3. FIX TRADE OUTCOMES (SSOT)
    # Ensure every closed trade has an outcome label
    for trade in db.get("history", []):
        if trade["action"] == "SELL" and "pnl" in trade:
            if trade["pnl"] > 0:
                trade["outcome"] = "WIN"
            elif trade["pnl"] < 0:
                trade["outcome"] = "LOSS"
            else:
                trade["outcome"] = "FLAT"

    with open(db_path, "w") as f:
        json.dump(db, f, indent=4)
    
    print("FORENSIC REPAIR COMPLETE. SSOT ESTABLISHED.")

if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    asyncio.run(forensic_repair())
