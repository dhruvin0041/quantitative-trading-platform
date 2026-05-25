import pandas as pd
import numpy as np
import yfinance as yf
import os
import json
from src.execution.live_inference import add_upgraded_features, FEATURE_COLUMNS
from src.data_ingestion.market_data import fetch_historical_data

def analyze_correlations():
    print("Fetching data for correlation analysis...")
    # Using SPY as a broad proxy for feature correlations
    spy_df = yf.download('SPY', period='2y', interval='1d', progress=False)
    vix_df = yf.download('^VIX', period='2y', interval='1d', progress=False)
    
    # Simple workaround for MultiIndex
    if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.droplevel(1)

    df = add_upgraded_features(spy_df.copy(), spy_df, vix_df)
    
    # Only check FEATURE_COLUMNS
    df = df[FEATURE_COLUMNS]
    
    corr_matrix = df.corr()
    high_corr = (corr_matrix.abs() > 0.85) & (corr_matrix != 1.0)
    pairs = [(i,j) for i in corr_matrix.columns 
                    for j in corr_matrix.columns 
                    if high_corr.loc[i,j] and i < j]
                    
    print("\n=== CORRELATION REPORT ===")
    print(f"Highly correlated pairs (>0.85 abs):")
    for p1, p2 in pairs:
        print(f" - {p1} vs {p2}: {corr_matrix.loc[p1, p2]:.4f}")
        
    # Check importance for these pairs
    importances = pd.read_csv('backtest_results/feature_importances.csv', index_col=0)
    
    print("\nComparison for Resolution:")
    to_drop = []
    for p1, p2 in pairs:
        imp1 = importances.loc[p1].values[0]
        imp2 = importances.loc[p2].values[0]
        better = p1 if imp1 > imp2 else p2
        worse = p2 if imp1 > imp2 else p1
        print(f" - {p1} ({imp1:.4f}) vs {p2} ({imp2:.4f}) -> Keeping {better}")
        if worse not in to_drop:
            to_drop.append(worse)
            
    print(f"\nFinal drop list based on correlations: {to_drop}")

if __name__ == "__main__":
    analyze_correlations()
