import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(backend_dir))

from api import inference_service as service
from src.data_ingestion.market_data import fetch_historical_data

def run_forensic_test():
    print("Initializing Forensic Hypothesis Testing Framework...")
    ticker = "AAPL"
    
    ticker_df = fetch_historical_data(ticker, start_date="2020-01-01", end_date="2026-10-08")
    spy_df = fetch_historical_data("SPY", start_date="2020-01-01", end_date="2026-10-08")
    
    ticker_df = ticker_df.loc["2020-01-01":]
    spy_df = spy_df.loc["2020-01-01":]
    
    print("Replaying Causal ML Signals (Frozen Strategy Stage 3)...")
    signals_list = service._replay_causal_ml_signals(ticker, ticker_df, spy_df)
    
    if not signals_list:
        print("No signals generated.")
        return
        
    df_sig = pd.DataFrame(signals_list)
    df_sig["bar_timestamp"] = pd.to_datetime(df_sig["bar_timestamp"])
    df_sig = df_sig.sort_values("bar_timestamp").reset_index(drop=True)
    
    # Calculate baseline unconditional returns
    ticker_df["Fwd_5d_Ret"] = ticker_df["Close"].shift(-5) / ticker_df["Close"] - 1
    ticker_df["5d_Momentum"] = ticker_df["Close"] / ticker_df["Close"].shift(5) - 1
    spy_df["SPY_SMA200"] = spy_df["Close"].rolling(200).mean()
    ticker_df["SPY_Regime"] = np.where(spy_df["Close"] >= spy_df["SPY_SMA200"], "BULL", "BEAR")
    
    baseline_mean = ticker_df["Fwd_5d_Ret"].mean()
    baseline_std = ticker_df["Fwd_5d_Ret"].std()
    
    # Merge forward returns into signals
    df_sig["Fwd_5d_Ret"] = np.nan
    df_sig["5d_Momentum"] = np.nan
    df_sig["SPY_Regime"] = "UNKNOWN"
    
    for i, row in df_sig.iterrows():
        ts = row["bar_timestamp"].strftime("%Y-%m-%d")
        if ts in ticker_df.index:
            df_sig.at[i, "Fwd_5d_Ret"] = ticker_df.loc[ts, "Fwd_5d_Ret"]
            df_sig.at[i, "5d_Momentum"] = ticker_df.loc[ts, "5d_Momentum"]
            df_sig.at[i, "SPY_Regime"] = ticker_df.loc[ts, "SPY_Regime"]
            
    df_sig = df_sig.dropna(subset=["Fwd_5d_Ret"])
    
    # Flips logic
    df_sig["Prev_Signal"] = df_sig["signal"].shift(1)
    df_sig["Days_Since_Last"] = (df_sig["bar_timestamp"] - df_sig["bar_timestamp"].shift(1)).dt.days
    df_sig["Is_Flip"] = (df_sig["signal"] != df_sig["Prev_Signal"]) & (df_sig["Prev_Signal"].notna())
    df_sig["Is_Rapid_Flip"] = df_sig["Is_Flip"] & (df_sig["Days_Since_Last"] <= 5)
    
    total_signals = len(df_sig)
    total_flips = df_sig["Is_Flip"].sum()
    rapid_flips = df_sig["Is_Rapid_Flip"].sum()
    
    print("\n==================================================")
    print("FORENSIC HYPOTHESIS TESTING REPORT")
    print("==================================================")
    print(f"Total Signals: {total_signals}")
    print(f"Total Flips: {total_flips}")
    print(f"Rapid Flips (<= 5 days): {rapid_flips} ({(rapid_flips/total_flips*100) if total_flips > 0 else 0:.1f}% of flips)")
    print(f"Unconditional AAPL 5d Return Mean: {baseline_mean*100:.2f}%")
    
    def print_stats(name, subset_series):
        if len(subset_series) == 0:
            print(f"\n--- {name} ---")
            print("N=0")
            return
        n = len(subset_series)
        mean = subset_series.mean()
        median = subset_series.median()
        std = subset_series.std()
        win_rate = (subset_series > 0).mean()
        ci_95 = 1.96 * std / np.sqrt(n)
        
        print(f"\n--- {name} ---")
        print(f"N = {n}")
        print(f"Mean = {mean*100:.2f}% (Median = {median*100:.2f}%)")
        print(f"Std Dev = {std*100:.2f}%")
        print(f"95% CI = [{ (mean - ci_95)*100:.2f}%, { (mean + ci_95)*100:.2f}% ]")
        print(f"Win Rate = {win_rate*100:.1f}%")

    # 1. Rapid Flips Analysis
    rapid_flip_rets = df_sig[df_sig["Is_Rapid_Flip"]]["Fwd_5d_Ret"]
    print_stats("Signals after Rapid Flips (<= 5 days)", rapid_flip_rets)
    
    normal_flip_rets = df_sig[~df_sig["Is_Rapid_Flip"] & df_sig["Is_Flip"]]["Fwd_5d_Ret"]
    print_stats("Signals after Normal Flips (> 5 days)", normal_flip_rets)
    
    # 2. Standing in Front of a Train (SELL during high momentum)
    sells_high_mom = df_sig[(df_sig["signal"] == "SELL") & (df_sig["5d_Momentum"] > 0.05)]["Fwd_5d_Ret"]
    sells_low_mom = df_sig[(df_sig["signal"] == "SELL") & (df_sig["5d_Momentum"] <= 0.05)]["Fwd_5d_Ret"]
    print_stats("SELL Signals when 5d Momentum > +5%", sells_high_mom)
    print_stats("SELL Signals when 5d Momentum <= +5%", sells_low_mom)
    
    # 3. Falling Knife (BUY during Bear Regime)
    buys_bear = df_sig[(df_sig["signal"] == "BUY") & (df_sig["SPY_Regime"] == "BEAR")]["Fwd_5d_Ret"]
    buys_bull = df_sig[(df_sig["signal"] == "BUY") & (df_sig["SPY_Regime"] == "BULL")]["Fwd_5d_Ret"]
    print_stats("BUY Signals during SPY Bear Regime (SPY < 200 SMA)", buys_bear)
    print_stats("BUY Signals during SPY Bull Regime (SPY >= 200 SMA)", buys_bull)
    
if __name__ == "__main__":
    run_forensic_test()
