import json
import os
import asyncio
from src.execution.fx_engine import FXEngine

async def repair():
    db_path = "data/paper_trading.json"
    if not os.path.exists(db_path):
        print("No DB found.")
        return

    with open(db_path, "r") as f:
        db = json.load(f)

    fx = FXEngine()
    await fx.update_rates()
    
    base_curr = db.get("base_currency", "USD")
    print(f"Current Base: {base_curr}")
    
    # 1. Fix Initial Capital
    # If it's exactly 1,000,000 and base is NOT USD, it was likely never converted.
    if db.get("initial_capital") == 1000000.0 and base_curr != "USD":
        old_val = db["initial_capital"]
        new_val = fx.convert_to_base(1000000.0, "USD", base_curr)
        db["initial_capital"] = new_val
        print(f"Repaired Initial Capital: {old_val} -> {new_val}")

    # 2. Fix Cash
    if db.get("capital") < 2000000.0 and base_curr != "USD":
         old_val = db["capital"]
         new_val = fx.convert_to_base(old_val, "USD", base_curr)
         db["capital"] = new_val
         print(f"Repaired Cash: {old_val} -> {new_val}")

    # 3. Fix Snapshots (Remove jumps and outliers)
    # We will re-baseline snapshots if they are too low (near 1M when base is INR)
    threshold = 2000000.0 if base_curr != "USD" else 0.0
    valid_snaps = []
    for s in db.get("portfolio_snapshots", []):
        if s["equity"] > threshold:
            valid_snaps.append(s)
    
    # If no snapshots left, create one from current state
    if not valid_snaps:
        from datetime import datetime
        valid_snaps = [{
            "time": datetime.now().isoformat(),
            "equity": db["capital"],
            "cash": db["capital"],
            "base_currency": base_curr
        }]
    
    db["portfolio_snapshots"] = valid_snaps
    print(f"Cleaned snapshots: {len(db.get('portfolio_snapshots', []))} remaining.")

    with open(db_path, "w") as f:
        json.dump(db, f, indent=4)
    print("DB Repaired Successfully.")

if __name__ == "__main__":
    import sys
    # Add project root to path
    sys.path.append(os.getcwd())
    asyncio.run(repair())
