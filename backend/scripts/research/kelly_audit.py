import numpy as np
import pandas as pd
import json
import joblib
import os

def run_kelly_safety_audit():
    print("=== HYDRA KELLY SAFETY AUDIT ===")
    
    # 1. Load calibration and performance data
    try:
        with open('backtest_results/calibration_audit.json', 'r') as f:
            calib = json.load(f)
        
        # We'll use the Consensus model win rates for this calculation
        expected_win_rate = 54.6 # Target %
        actual_win_rate = 32.7 # Observed in previous validation sprint %
        
        # 2. Basic Kelly Formula: K% = (p*b - q) / b
        # p = probability of win
        # q = probability of loss (1-p)
        # b = odds (win amount / loss amount) - using Profit Factor as proxy for b
        
        profit_factor = 1.19 # from previous backtest
        p = actual_win_rate / 100
        q = 1 - p
        b = profit_factor
        
        raw_kelly = (p * b - q) / b
        
        print(f"Empirical Win Rate: {p*100:.1f}%")
        print(f"Profit Factor: {b:.2f}")
        print(f"Raw Kelly Fraction: {raw_kelly*100:.2f}%")
        
        # 3. Safety Buffers
        print("\nSafety Recommendations:")
        half_kelly = raw_kelly / 2
        quarter_kelly = raw_kelly / 4
        
        # If raw_kelly is negative, it means no edge.
        if raw_kelly <= 0:
            print(">>> WARNING: Empirical edge is insufficient for Kelly sizing.")
            print("Recommendation: Use fixed 1% exposure until calibration improves.")
            safe_cap = 0.01
        else:
            print(f" - Full Kelly: {raw_kelly*100:.1f}% (High risk of ruin)")
            print(f" - Half Kelly: {half_kelly*100:.1f}% (Recommended for aggressive)")
            print(f" - Quarter Kelly: {quarter_kelly*100:.1f}% (Recommended for institutional safety)")
            safe_cap = min(0.10, quarter_kelly)
            
        print(f"\nFinal Recommended Safety Cap: {safe_cap*100:.1f}%")
        
        # Save recommendation
        with open('configs/risk_params.json', 'w') as f:
            json.dump({"kelly_cap": safe_cap, "last_audit": str(pd.Timestamp.now())}, f, indent=2)
            
    except Exception as e:
        print(f"Audit failed: {e}")

if __name__ == "__main__":
    run_kelly_safety_audit()
