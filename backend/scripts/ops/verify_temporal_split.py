import sys
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd
import yfinance as yf

from src.data_ingestion.market_data import apply_dynamic_triple_barrier, fetch_historical_data
from src.execution.data_firewall import TemporalFirewall
from src.execution.live_inference import FEATURE_COLUMNS, add_upgraded_features


def audit_temporal_split(ticker: str = "AAPL", horizon: int = 15):
    print("=" * 80)
    print(f"HYDRA TEMPORAL DATA-SPLIT & OOS AUDIT: {ticker} (horizon={horizon})")
    print("=" * 80)

    # ----------------------------------------------------
    # STAGE 1: 2016-2024 MODEL DEVELOPMENT UNIVERSE
    # ----------------------------------------------------
    print("\n[STAGE 1] Auditing 2016-2024 Model Development Data...")
    df_raw_dev = fetch_historical_data(ticker, start_date="2016-01-01", end_date="2024-12-31")
    spy_dev = yf.download("SPY", start="2016-01-01", end="2024-12-31", progress=False)
    vix_dev = yf.download("^VIX", start="2016-01-01", end="2024-12-31", progress=False)
    if isinstance(spy_dev.columns, pd.MultiIndex):
        spy_dev.columns = spy_dev.columns.droplevel(1)
    if isinstance(vix_dev.columns, pd.MultiIndex):
        vix_dev.columns = vix_dev.columns.droplevel(1)

    raw_dev_count = len(df_raw_dev)
    dev_start_date = df_raw_dev.index[0].strftime("%Y-%m-%d")
    dev_end_date = df_raw_dev.index[-1].strftime("%Y-%m-%d")

    # Hard Firewall Check
    TemporalFirewall.validate_development_data(df_raw_dev, "raw_development_data")
    TemporalFirewall.validate_no_2026_leakage(df_raw_dev, "raw_development_data")

    # Feature generation
    df_feat_dev = add_upgraded_features(df_raw_dev.copy(), spy_dev, vix_dev)

    # Warmup removal (Section 6: No hidden pre-2016 data)
    # The longest indicator window is 120 bars (rolling z-score).
    warmup_removed_count = 119
    df_clean_dev = df_feat_dev.iloc[warmup_removed_count:].copy()
    first_valid_date = df_clean_dev.index[0].strftime("%Y-%m-%d")

    # Triple barrier labeling confined strictly to 2016-2024
    df_labeled_dev = apply_dynamic_triple_barrier(
        df_clean_dev.copy(), tp_atr_multiplier=3.0, sl_atr_multiplier=0.5, horizon=horizon
    )

    label_horizon_dropped_dev = len(df_clean_dev) - len(df_labeled_dev)
    final_train_samples = len(df_labeled_dev)
    final_train_last_date = df_labeled_dev.index[-1].strftime("%Y-%m-%d")

    # Verify no post-2024 dates
    TemporalFirewall.validate_development_data(df_labeled_dev, "final_training_samples")

    # ----------------------------------------------------
    # STAGE 2: 2025 DEDICATED VALIDATION / CALIBRATION
    # ----------------------------------------------------
    print("\n[STAGE 2] Auditing 2025 Dedicated Validation / Calibration Data...")
    # Fetch 2024-01-01 to 2025-12-31 to provide sufficient warmup for 2025 features
    df_raw_val_full = fetch_historical_data(ticker, start_date="2024-01-01", end_date="2025-12-31")
    spy_val_full = yf.download("SPY", start="2024-01-01", end="2025-12-31", progress=False)
    vix_val_full = yf.download("^VIX", start="2024-01-01", end="2025-12-31", progress=False)
    if isinstance(spy_val_full.columns, pd.MultiIndex):
        spy_val_full.columns = spy_val_full.columns.droplevel(1)
    if isinstance(vix_val_full.columns, pd.MultiIndex):
        vix_val_full.columns = vix_val_full.columns.droplevel(1)

    # Compute features across the continuous stream
    df_feat_val_full = add_upgraded_features(df_raw_val_full.copy(), spy_val_full, vix_val_full)

    # Slice STRICTLY 2025 (2025-01-01 to 2025-12-31)
    val_2025_mask = (df_feat_val_full.index >= "2025-01-01") & (df_feat_val_full.index <= "2025-12-31")
    df_val_2025_clean = df_feat_val_full[val_2025_mask].dropna(subset=FEATURE_COLUMNS).copy()
    val_2025_raw_count = len(df_val_2025_clean)
    val_start_date = df_val_2025_clean.index[0].strftime("%Y-%m-%d")
    val_end_date = df_val_2025_clean.index[-1].strftime("%Y-%m-%d")

    # Hard Firewall Check
    TemporalFirewall.validate_validation_data(df_val_2025_clean, "2025_validation_data")
    TemporalFirewall.validate_no_2026_leakage(df_val_2025_clean, "2025_validation_data")

    # Validation labels (confined strictly to 2025, zero 2026 prices)
    df_labeled_val = apply_dynamic_triple_barrier(
        df_val_2025_clean.copy(), tp_atr_multiplier=3.0, sl_atr_multiplier=0.5, horizon=horizon
    )
    val_label_horizon_dropped = len(df_val_2025_clean) - len(df_labeled_val)
    final_val_samples = len(df_labeled_val)
    final_val_last_date = df_labeled_val.index[-1].strftime("%Y-%m-%d")

    # Verify no 2026 dates
    TemporalFirewall.validate_validation_data(df_labeled_val, "labeled_validation_samples")

    # ----------------------------------------------------
    # STAGE 3: 2026+ TRUE UNTOUCHED OUT-OF-SAMPLE (OOS)
    # ----------------------------------------------------
    print("\n[STAGE 3] Auditing 2026+ Out-Of-Sample Boundary...")
    oos_start_date = "2026-01-01"

    # Print Full Audit Summary
    print("\n" + "=" * 80)
    print("HYDRA TEMPORAL SPLIT PROVENANCE SUMMARY")
    print("=" * 80)
    print("DEVELOPMENT PERIOD (2016-2024):")
    print(f"  First trading date:             {dev_start_date}")
    print(f"  Last trading date:              {dev_end_date}")
    print(f"  Raw bars fetched:               {raw_dev_count}")
    print(f"  Warmup bars removed:            {warmup_removed_count} (first valid date: {first_valid_date})")
    print(f"  Incomplete-label bars removed:  {label_horizon_dropped_dev} (horizon={horizon})")
    print(f"  Final clean training samples:   {final_train_samples} (last sample: {final_train_last_date})")
    print()
    print("VALIDATION / CALIBRATION PERIOD (2025):")
    print(f"  First trading date:             {val_start_date}")
    print(f"  Last trading date:              {val_end_date}")
    print(f"  Raw bars in 2025:               {val_2025_raw_count}")
    print("  Warmup bars removed:            0 (warmed up via continuous 2024 stream)")
    print(f"  Incomplete-label bars removed:  {val_label_horizon_dropped} (horizon={horizon})")
    print(f"  Final clean validation samples: {final_val_samples} (last sample: {final_val_last_date})")
    print()
    print("OUT-OF-SAMPLE (OOS) BOUNDARY:")
    print(f"  Untouched OOS Start Date:       {oos_start_date}")
    print("  2026 Data Influence on Dev:     ZERO (Firewall PASS)")
    print("  2026 Data Influence on Val:     ZERO (Firewall PASS)")
    print("=" * 80)

    results = {
        "ticker": ticker,
        "horizon": horizon,
        "dev_start_date": dev_start_date,
        "dev_end_date": dev_end_date,
        "raw_dev_count": raw_dev_count,
        "warmup_removed_count": warmup_removed_count,
        "first_valid_date": first_valid_date,
        "label_horizon_dropped_dev": label_horizon_dropped_dev,
        "final_train_samples": final_train_samples,
        "final_train_last_date": final_train_last_date,
        "val_start_date": val_start_date,
        "val_end_date": val_end_date,
        "val_2025_raw_count": val_2025_raw_count,
        "val_label_horizon_dropped": val_label_horizon_dropped,
        "final_val_samples": final_val_samples,
        "final_val_last_date": final_val_last_date,
        "oos_start_date": oos_start_date,
    }
    return results


if __name__ == "__main__":
    audit_temporal_split("AAPL", horizon=15)
