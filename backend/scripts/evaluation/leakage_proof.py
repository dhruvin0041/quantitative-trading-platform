import pandas as pd
import yfinance as yf

from src.execution.live_inference import add_upgraded_features


def run_invariant_test():
    print("=== FEATURE INVARIANT TEST (LOOKAHEAD AUDIT) ===")

    ticker = "AAPL"
    # Fetch data up to a specific date
    full_data = yf.download(
        ticker, start="2023-01-01", end="2024-01-01", progress=False
    )
    spy_data = yf.download("SPY", start="2023-01-01", end="2024-01-01", progress=False)
    vix_data = yf.download("^VIX", start="2023-01-01", end="2024-01-01", progress=False)

    if isinstance(full_data.columns, pd.MultiIndex):
        full_data.columns = full_data.columns.droplevel(1)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.droplevel(1)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.droplevel(1)

    # 1. Compute features on the "Short" dataset (up to June 1st)
    short_limit = "2023-06-01"
    df_short = full_data[full_data.index <= short_limit].copy()
    spy_short = spy_data[spy_data.index <= short_limit].copy()
    vix_short = vix_data[vix_data.index <= short_limit].copy()

    feat_short = add_upgraded_features(df_short, spy_short, vix_short)
    val_short = feat_short.loc[short_limit]

    # 2. Compute features on the "Full" dataset
    feat_full = add_upgraded_features(full_data.copy(), spy_data, vix_data)
    val_full = feat_full.loc[short_limit]

    # 3. Compare
    diff = (val_short - val_full).abs()
    leaking_cols = diff[diff > 1e-6]

    if not leaking_cols.empty:
        print("\n[CRITICAL] LEAKAGE DETECTED!")
        print(
            "The following features change their historical values when future data is added:"
        )
        for col in leaking_cols.index:
            print(f" - {col}: diff={diff[col]:.8f}")
    else:
        print("\n[SUCCESS] No feature lookahead detected in current window.")


if __name__ == "__main__":
    run_invariant_test()
