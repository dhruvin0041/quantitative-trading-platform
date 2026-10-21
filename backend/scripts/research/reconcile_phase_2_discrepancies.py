import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add backend to path
BACKEND_DIR = Path("backend")
if str(BACKEND_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR.resolve()))

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from scripts.research.master_research_validation_sprint import (
    InstitutionalPortfolioSimulator,
    compute_all_feature_candidates,
)
from scripts.research.v2_4_3_hierarchical_regime_strategy import (
    HierarchicalSignalStateMachine,
)
from scripts.research.v2_4_4_optimization_lineage_and_generalization import (
    EVAL_END,
    EVAL_START,
    LOCKED_DELTA,
    LOCKED_FEATURE_SET,
    LOCKED_LGB_PARAMS,
    LOCKED_THRESHOLD,
    LOCKED_VOL,
    WARMUP_START,
    fetch_clean_market_data,
)

REPORTS_DIR = Path("backend/reports")
ARTIFACTS_DIR = Path("backend/artifacts")

print("================================================================================")
print("HYDRA V2.5 PHASE 2 — FORENSIC DISCREPANCY RECONCILIATION ENGINE")
print("================================================================================")

# 1. Fetch Market Data for AAPL
print("\n[Step 1] Ingesting AAPL Market Data...")
df_raw = fetch_clean_market_data("AAPL", start=WARMUP_START, end=EVAL_END)
df_full, feature_sets = compute_all_feature_candidates(df_raw)
selected_features = feature_sets[LOCKED_FEATURE_SET]
eval_dates = df_full.loc[EVAL_START:"2026-10-13"].index
eval_prices = df_raw.loc[eval_dates].copy()
eval_features_df = df_full.loc[eval_dates].copy()

embargo_dt = pd.Timestamp(EVAL_START) - pd.Timedelta(days=20)
dev_df = df_full.loc[:embargo_dt].dropna(subset=["SMA_200"]).copy()

# Target labeling
tp_mult, sl_mult, h_len = 2.0, 1.5, 15
fh_list = [dev_df["High"].shift(-h) for h in range(1, h_len + 1)]
fl_list = [dev_df["Low"].shift(-h) for h in range(1, h_len + 1)]
fh_df = pd.concat(fh_list, axis=1)
fl_df = pd.concat(fl_list, axis=1)

labels = []
valid_idx = []
for dt in dev_df.index:
    c = float(dev_df.loc[dt, "Close"])
    atr = float(dev_df.loc[dt, "ATR"])
    ub = c + (atr * tp_mult)
    lb = c - (atr * sl_mult)

    fh_row = fh_df.loc[dt].values
    fl_row = fl_df.loc[dt].values
    if np.any(np.isnan(fh_row)) or np.any(np.isnan(fl_row)):
        continue
    u_hit = np.where(fh_row >= ub)[0]
    l_hit = np.where(fl_row <= lb)[0]
    u_s = u_hit[0] + 1 if len(u_hit) > 0 else h_len + 1
    l_s = l_hit[0] + 1 if len(l_hit) > 0 else h_len + 1

    if u_s > h_len and l_s > h_len:
        labels.append(1)
    elif u_s < l_s:
        labels.append(2)
    else:
        labels.append(0)
    valid_idx.append(dt)

y_dev = np.array(labels, dtype=int)
dev_clean = dev_df.loc[valid_idx].copy()
dev_clean[selected_features] = dev_clean[selected_features].ffill().fillna(0.0)

# 2. RUN 1: ARCHIVED PROBABILITIES ($1,188.34)
print("\n[Step 2] Executing Run 1: Archived Probabilities ($1,188.34)...")
archived_log_path = REPORTS_DIR / "v2_4_3_regime_state_log.csv"
assert archived_log_path.exists(), "Archived log not found!"
df_archived_log = pd.read_csv(archived_log_path)
probs_run1 = df_archived_log[["prob_sell", "prob_hold", "prob_buy"]].values

sm_run1 = HierarchicalSignalStateMachine(delta=LOCKED_DELTA, vol_threshold=LOCKED_VOL, theta=LOCKED_THRESHOLD, overlay_mode=False)
sigs_run1, logs_run1 = sm_run1.process_sequence(probs_run1, eval_features_df)
sim1 = InstitutionalPortfolioSimulator(initial_capital=1000.0, slippage_bps=5.0)
res_run1 = sim1.simulate(eval_prices, sigs_run1, "Run_1_Archived")

print(f"  Run 1 Final Equity: ${res_run1['summary']['final_equity']:.2f} | Trades: {res_run1['summary']['total_trades']} | Return: {res_run1['summary']['net_return_pct']:.2f}%")

# 3. RUN 2: MULTI-ASSET RETRAINED PIPELINE ($1,162.55)
print("\n[Step 3] Executing Run 2: Multi-Asset Pipeline ($1,162.55)...")
scaler_run2 = StandardScaler()
X_dev_s2 = scaler_run2.fit_transform(dev_clean[selected_features].values)
model_run2 = lgb.LGBMClassifier(**LOCKED_LGB_PARAMS)
model_run2.fit(X_dev_s2, y_dev)

eval_features_clean = eval_features_df[selected_features].ffill().fillna(0.0)
X_eval_s2 = scaler_run2.transform(eval_features_clean.values)
probs_run2 = model_run2.predict_proba(X_eval_s2)

sm_run2 = HierarchicalSignalStateMachine(delta=LOCKED_DELTA, vol_threshold=LOCKED_VOL, theta=LOCKED_THRESHOLD, overlay_mode=False)
sigs_run2, logs_run2 = sm_run2.process_sequence(probs_run2, eval_features_df)
sim2 = InstitutionalPortfolioSimulator(initial_capital=1000.0, slippage_bps=5.0)
res_run2 = sim2.simulate(eval_prices, sigs_run2, "Run_2_MultiAssetPipeline")

print(f"  Run 2 Final Equity: ${res_run2['summary']['final_equity']:.2f} | Trades: {res_run2['summary']['total_trades']} | Return: {res_run2['summary']['net_return_pct']:.2f}%")

# 4. RUN 3: RE-EXECUTED HIERARCHICAL RUNNER ($1,155.66)
print("\n[Step 4] Executing Run 3: Fresh Script Execution ($1,155.66)...")
# Check master_sprint_phase_4_signal_state_log.csv
p4_path = REPORTS_DIR / "master_sprint_phase_4_signal_state_log.csv"
df_p4 = pd.read_csv(p4_path)
probs_run3 = df_p4[["prob_sell", "prob_hold", "prob_buy"]].values

sm_run3 = HierarchicalSignalStateMachine(delta=LOCKED_DELTA, vol_threshold=LOCKED_VOL, theta=LOCKED_THRESHOLD, overlay_mode=False)
sigs_run3, logs_run3 = sm_run3.process_sequence(probs_run3, eval_features_df)
sim3 = InstitutionalPortfolioSimulator(initial_capital=1000.0, slippage_bps=5.0)
res_run3 = sim3.simulate(eval_prices, sigs_run3, "Run_3_FreshRunner")

print(f"  Run 3 Final Equity: ${res_run3['summary']['final_equity']:.2f} | Trades: {res_run3['summary']['total_trades']} | Return: {res_run3['summary']['net_return_pct']:.2f}%")

# 5. DIVERGENCE ANALYSIS
print("\n[Step 5] Detailed Divergence Analysis Across Runs...")
divergences_1_2 = []
divergences_1_3 = []
divergences_2_3 = []

for i in range(len(eval_dates)):
    dt = eval_dates[i]
    p1 = probs_run1[i]
    p2 = probs_run2[i]
    p3 = probs_run3[i]

    s1 = sigs_run1.iloc[i]
    s2 = sigs_run2.iloc[i]
    s3 = sigs_run3.iloc[i]

    l1 = logs_run1[i]
    l2 = logs_run2[i]
    l3 = logs_run3[i]

    if s1 != s2 or abs(p1[2] - p2[2]) > 0.001 or abs(p1[0] - p2[0]) > 0.001:
        divergences_1_2.append({
            "bar": i, "date": str(dt),
            "p1_buy": p1[2], "p2_buy": p2[2],
            "p1_sell": p1[0], "p2_sell": p2[0],
            "sig1": s1, "sig2": s2,
            "regime1": l1["regime"], "regime2": l2["regime"],
            "state1": l1["position_state_after_bar"], "state2": l2["position_state_after_bar"]
        })

    if s1 != s3 or abs(p1[2] - p3[2]) > 0.001 or abs(p1[0] - p3[0]) > 0.001:
        divergences_1_3.append({
            "bar": i, "date": str(dt),
            "p1_buy": p1[2], "p3_buy": p3[2],
            "p1_sell": p1[0], "p3_sell": p3[0],
            "sig1": s1, "sig3": s3,
            "regime1": l1["regime"], "regime3": l3["regime"],
            "state1": l1["position_state_after_bar"], "state3": l3["position_state_after_bar"]
        })

print(f"Run 1 vs Run 2: Total bars with diff > 0.001 or signal diff: {len(divergences_1_2)}")
if divergences_1_2:
    print("First signal divergence between Run 1 and Run 2:")
    sig_diffs_1_2 = [d for d in divergences_1_2 if d['sig1'] != d['sig2']]
    if sig_diffs_1_2:
        print(f"  First SIGNAL divergence at bar {sig_diffs_1_2[0]['bar']} ({sig_diffs_1_2[0]['date']}):")
        print(f"    Run 1: {sig_diffs_1_2[0]['sig1']} (P(Buy)={sig_diffs_1_2[0]['p1_buy']:.4f}, P(Sell)={sig_diffs_1_2[0]['p1_sell']:.4f}, State={sig_diffs_1_2[0]['state1']})")
        print(f"    Run 2: {sig_diffs_1_2[0]['sig2']} (P(Buy)={sig_diffs_1_2[0]['p2_buy']:.4f}, P(Sell)={sig_diffs_1_2[0]['p2_sell']:.4f}, State={sig_diffs_1_2[0]['state2']})")
    else:
        print("  Zero signal divergences! Signals are identical.")

print(f"\nRun 1 vs Run 3: Total bars with diff > 0.001 or signal diff: {len(divergences_1_3)}")
if divergences_1_3:
    sig_diffs_1_3 = [d for d in divergences_1_3 if d['sig1'] != d['sig3']]
    if sig_diffs_1_3:
        print(f"  First SIGNAL divergence at bar {sig_diffs_1_3[0]['bar']} ({sig_diffs_1_3[0]['date']}):")
        print(f"    Run 1: {sig_diffs_1_3[0]['sig1']} (P(Buy)={sig_diffs_1_3[0]['p1_buy']:.4f}, P(Sell)={sig_diffs_1_3[0]['p1_sell']:.4f}, State={sig_diffs_1_3[0]['state1']})")
        print(f"    Run 3: {sig_diffs_1_3[0]['sig3']} (P(Buy)={sig_diffs_1_3[0]['p3_buy']:.4f}, P(Sell)={sig_diffs_1_3[0]['p3_sell']:.4f}, State={sig_diffs_1_3[0]['state3']})")

# Trade comparisons
print("\n[Step 6] Comparing Trade Ledgers Across Runs...")
t1 = res_run1["trades"]
t2 = res_run2["trades"]
t3 = res_run3["trades"]
print(f"Trades Count: Run 1={len(t1)}, Run 2={len(t2)}, Run 3={len(t3)}")

for idx in range(min(len(t1), len(t2))):
    tr1 = t1[idx]
    tr2 = t2[idx]
    if tr1["entry_timestamp"] != tr2["entry_timestamp"] or tr1["exit_timestamp"] != tr2["exit_timestamp"] or abs(tr1["net_pnl"] - tr2["net_pnl"]) > 0.01:
        print(f"  First Divergent Trade between Run 1 and Run 2: Trade #{idx+1}")
        print(f"    Run 1 Trade #{idx+1}: Entry={tr1['entry_timestamp']} @ {tr1['entry_fill_price']:.2f}, Exit={tr1['exit_timestamp']} @ {tr1['exit_fill_price']:.2f}, Shares={tr1['entry_shares']}, Net PnL=${tr1['net_pnl']:.2f}")
        print(f"    Run 2 Trade #{idx+1}: Entry={tr2['entry_timestamp']} @ {tr2['entry_fill_price']:.2f}, Exit={tr2['exit_timestamp']} @ {tr2['exit_fill_price']:.2f}, Shares={tr2['entry_shares']}, Net PnL=${tr2['net_pnl']:.2f}")
        break

for idx in range(min(len(t1), len(t3))):
    tr1 = t1[idx]
    tr3 = t3[idx]
    if tr1["entry_timestamp"] != tr3["entry_timestamp"] or tr1["exit_timestamp"] != tr3["exit_timestamp"] or abs(tr1["net_pnl"] - tr3["net_pnl"]) > 0.01:
        print(f"  First Divergent Trade between Run 1 and Run 3: Trade #{idx+1}")
        print(f"    Run 1 Trade #{idx+1}: Entry={tr1['entry_timestamp']} @ {tr1['entry_fill_price']:.2f}, Exit={tr1['exit_timestamp']} @ {tr1['exit_fill_price']:.2f}, Shares={tr1['entry_shares']}, Net PnL=${tr1['net_pnl']:.2f}")
        print(f"    Run 3 Trade #{idx+1}: Entry={tr3['entry_timestamp']} @ {tr3['entry_fill_price']:.2f}, Exit={tr3['exit_timestamp']} @ {tr3['exit_fill_price']:.2f}, Shares={tr3['entry_shares']}, Net PnL=${tr3['net_pnl']:.2f}")
        break
