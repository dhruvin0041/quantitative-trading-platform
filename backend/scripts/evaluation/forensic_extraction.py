# =============================================================================
# HYDRA V2.2 — Complete Forensic Extraction Script (READ-ONLY)
# =============================================================================
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

BACKEND = Path(r"D:\DataScience\Projects\Data_Science_Projects\Stock_Indicator\backend")
ARTIFACTS = BACKEND / "artifacts"
CONFIGS = BACKEND / "configs"
sys.path.insert(0, str(BACKEND))

from src.data_ingestion.market_data import (
    apply_dynamic_triple_barrier,
    fetch_historical_data,
)
from src.execution.live_inference import (
    FEATURE_COLUMNS,
    add_upgraded_features,
    load_config,
)
from src.models.neural.fusion_network import build_fusion_model
from src.models.regime.calibration import ModelCalibrator
from src.models.rl.dqn_agent import DQNAgent

config = load_config(ticker="AAPL")
TIME_STEPS = config["data"]["time_steps"]

# Load frozen artifacts
scaler = joblib.load(ARTIFACTS / "latest_scaler.joblib")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(ARTIFACTS / "xgb_ensemble.json"))
lgbm_model = joblib.load(ARTIFACTS / "lgbm_agent.joblib")
dl_model = build_fusion_model(config)
dl_model.load_weights(str(ARTIFACTS / "latest_fusion_weights.weights.h5"))
dqn_agent = DQNAgent(state_size=33)
dqn_agent.load(str(ARTIFACTS / "dqn_model.pth"))
calibrator = ModelCalibrator.load(str(ARTIFACTS / "model_calibrator.joblib"))
meta_model = joblib.load(ARTIFACTS / "meta_ensemble.joblib")

# Data fetch
start_dt = "2024-01-01"
end_dt = "2025-12-31"
df_raw = fetch_historical_data("AAPL", start_dt, end_dt)
spy_df = yf.download("SPY", start=start_dt, end=end_dt, progress=False)
vix_df = yf.download("^VIX", start=start_dt, end=end_dt, progress=False)
if isinstance(spy_df.columns, pd.MultiIndex):
    spy_df.columns = spy_df.columns.droplevel(1)
if isinstance(vix_df.columns, pd.MultiIndex):
    vix_df.columns = vix_df.columns.droplevel(1)

df_feat = add_upgraded_features(df_raw.copy(), spy_df, vix_df, lag_vix=True)
df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()].copy()
tb = apply_dynamic_triple_barrier(df_feat.copy(), tp_atr_multiplier=1.5, sl_atr_multiplier=2.0, horizon=15)

def calc_entropy(probs, eps=1e-12):
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)

# ---------------------------------------------------------------------------
# Investigation 1 & 2: H2 Observations
# ---------------------------------------------------------------------------
df_h2 = df_feat.loc["2025-07-01":"2025-12-31"].dropna(subset=FEATURE_COLUMNS)
common_h2 = df_h2.index.intersection(tb.dropna().index)
eval_h2 = df_h2.loc[common_h2]
y_true_h2 = tb.loc[common_h2, "target_signal"].values.astype(int)

scaled_h2 = scaler.transform(eval_h2[FEATURE_COLUMNS].values)
full_scaled = scaler.transform(df_feat[FEATURE_COLUMNS].dropna().values)
full_idx = df_feat[FEATURE_COLUMNS].dropna().index

xgb_probs_raw = xgb_model.predict_proba(scaled_h2)
lgbm_probs_raw = lgbm_model.predict_proba(scaled_h2)

# Pass-through calibration for XGB/LGBM
xgb_probs = np.array([calibrator.calibrate("XGB", p) for p in xgb_probs_raw])
lgbm_probs = np.array([calibrator.calibrate("LGBM", p) for p in lgbm_probs_raw])

dl_raw_list = []
dl_cal_list = []
dqn_dash_list = []
dqn_auth_list = []
meta_probs_list = []

for i, dt in enumerate(common_h2):
    loc_in_full = full_idx.get_loc(dt)
    if isinstance(loc_in_full, slice):
        loc_in_full = loc_in_full.stop - 1
    ts_window = full_scaled[loc_in_full - TIME_STEPS + 1 : loc_in_full + 1]
    ts_seq = ts_window.reshape(1, TIME_STEPS, -1)

    # DL prediction
    raw_dl = dl_model.predict([ts_seq]*6, verbose=0)[2][0]
    dl_raw_list.append(raw_dl)
    cal_dl = calibrator.calibrate("DL_FUSION", raw_dl)
    dl_cal_list.append(cal_dl)

    # DQN Dashboard (100,000 dummy bug)
    row = scaled_h2[i]
    dash_state = np.concatenate([row, [0, 0, 0, 100000, 100000, 0]])
    p_dash = dqn_agent.predict_proba(dash_state, temperature=1.5)
    dqn_dash_list.append(p_dash)

    # DQN Authentic state [tabular, dl_preds, xgb_preds]
    auth_state = np.hstack([row, cal_dl, xgb_probs[i]])
    p_auth = dqn_agent.predict_proba(auth_state, temperature=1.5)
    dqn_auth_list.append(p_auth)

    # Meta-Ensemble
    base_preds = {
        "LSTM": cal_dl,
        "XGBoost": xgb_probs[i],
        "LightGBM": lgbm_probs[i],
        "DQN": p_dash,
    }
    m_p = meta_model.predict_proba(base_preds, 0)
    meta_probs_list.append(m_p)

dl_raw_arr = np.array(dl_raw_list)
dl_cal_arr = np.array(dl_cal_list)
dqn_dash_arr = np.array(dqn_dash_list)
dqn_auth_arr = np.array(dqn_auth_list)
meta_arr = np.array(meta_probs_list)

# ---------------------------------------------------------------------------
# Investigation 2: Production Decision Trace for all 112 H2 observations
# ---------------------------------------------------------------------------
close_s = df_feat["Close"].reindex(eval_h2.index).ffill()
sma200 = close_s.rolling(200, min_periods=20).mean()
spy_c = spy_df["Close"].reindex(eval_h2.index).ffill()
spy_sma50 = spy_c.rolling(50, min_periods=10).mean()

h2_traces = []
pos = "FLAT"
last_trade_idx = -10

for i, dt in enumerate(common_h2):
    dt_str = dt.strftime("%Y-%m-%d")
    c = float(close_s.iloc[i])
    sma = float(sma200.iloc[i]) if pd.notna(sma200.iloc[i]) else c
    sc = float(spy_c.iloc[i])
    ssma = float(spy_sma50.iloc[i]) if pd.notna(spy_sma50.iloc[i]) else sc

    long_ok = bool((c >= sma) and (sc >= ssma))
    short_ok = bool((c < sma) or (sc < ssma))

    xgb_p = xgb_probs[i]
    lgbm_p = lgbm_probs[i]
    dl_p = dl_cal_arr[i]
    dqn_p = dqn_dash_arr[i]
    meta_p = meta_arr[i]

    xgb_act = int(np.argmax(xgb_p))
    lgbm_act = int(np.argmax(lgbm_p))
    dl_act = int(np.argmax(dl_p))
    dqn_act = int(np.argmax(dqn_p))

    # Primary driver: XGBoost
    primary_sig = ["SELL", "HOLD", "BUY"][xgb_act]
    primary_conf = float(np.max(xgb_p))

    # Veto check (production default veto_threshold=1.01 -> veto disabled)
    veto_triggered = False

    # Threshold check: primary_conf >= 0.60
    conf_pass = bool(primary_conf >= 0.60)

    # Pre-signal
    if conf_pass:
        pre_sig = primary_sig
    else:
        pre_sig = "HOLD"

    # Macro filter & Position state logic
    sig_note = "Normal"
    if pre_sig == "BUY" and not long_ok:
        final_sig = "HOLD"
        sig_note = "Suppressed by Macro Filter (Long Not Allowed)"
    elif pre_sig == "SELL" and not short_ok and pos != "LONG":
        final_sig = "HOLD"
        sig_note = "Suppressed by Macro Filter (Naked Short Forbidden in Bull)"
    else:
        final_sig = pre_sig

    # Order generation & Position update
    order_gen = False
    order_type = "NONE"
    if final_sig == "BUY" and pos != "LONG" and (i - last_trade_idx >= 5):
        pos = "LONG"
        last_trade_idx = i
        order_gen = True
        order_type = "BUY (ENTER LONG)"
    elif final_sig == "SELL" and pos == "LONG" and (i - last_trade_idx >= 5):
        pos = "FLAT"
        last_trade_idx = i
        order_gen = True
        order_type = "SELL (EXIT LONG)"
    elif final_sig == "SELL" and pos == "FLAT":
        order_gen = False
        order_type = "HOLD (NO SHORT)"

    h2_traces.append({
        "index": i + 1,
        "date": dt_str,
        "close": round(c, 2),
        "true_label": ["SELL", "HOLD", "BUY"][y_true_h2[i]],
        "raw_xgb_p": [round(float(x), 4) for x in xgb_probs_raw[i]],
        "cal_xgb_p": [round(float(x), 4) for x in xgb_p],
        "cal_lgbm_p": [round(float(x), 4) for x in lgbm_p],
        "raw_dl_p": [round(float(x), 4) for x in dl_raw_arr[i]],
        "cal_dl_p": [round(float(x), 4) for x in dl_p],
        "dqn_q_dash": [round(float(x), 1) for x in dqn_agent.predict_q_values(dash_state)],
        "dqn_dash_p": [round(float(x), 4) for x in dqn_p],
        "dqn_auth_p": [round(float(x), 4) for x in dqn_auth_arr[i]],
        "meta_p": [round(float(x), 4) for x in meta_p],
        "primary_model": "XGB_AGENT",
        "primary_signal": primary_sig,
        "primary_conf": round(primary_conf, 4),
        "veto_logic": "Disabled (veto_threshold=1.01)",
        "conf_threshold_pass": conf_pass,
        "macro_long_allowed": long_ok,
        "macro_short_allowed": short_ok,
        "position_before": pos,
        "final_decision": final_sig,
        "order_generated": order_gen,
        "order_type": order_type,
        "signal_note": sig_note,
    })

# ---------------------------------------------------------------------------
# Investigation 6: The 5 Flagship Trades
# ---------------------------------------------------------------------------
def run_trade_sim(use_macro_filter=True):
    pos = "FLAT"
    last_t = -10
    sigs = []
    for i in range(len(eval_h2)):
        pred = np.argmax(xgb_probs[i])
        c = float(close_s.iloc[i])
        sma = float(sma200.iloc[i]) if pd.notna(sma200.iloc[i]) else c
        sc = float(spy_c.iloc[i])
        ssma = float(spy_sma50.iloc[i]) if pd.notna(spy_sma50.iloc[i]) else sc
        long_ok = (c >= sma) and (sc >= ssma) if use_macro_filter else True

        if pred == 2 and long_ok and pos != "LONG" and (i - last_t >= 5):
            pos = "LONG"
            last_t = i
            sigs.append("BUY")
        elif pred == 0 and pos == "LONG" and (i - last_t >= 5):
            pos = "FLAT"
            last_t = i
            sigs.append("SELL")
        else:
            sigs.append("HOLD")

    sig_series = pd.Series(sigs, index=eval_h2.index)
    trades = []
    entry_p = None
    entry_d = None
    open_vals = df_feat["Open"].squeeze()

    for i, sig_d in enumerate(sig_series.index):
        sig = sig_series.iloc[i]
        orig_idx = df_feat.index.get_indexer([sig_d])[0]
        if orig_idx + 1 >= len(df_feat):
            continue
        next_open = float(open_vals.iloc[orig_idx + 1])
        if sig == "BUY" and entry_p is None:
            entry_p = next_open * 1.0005
            entry_d = sig_d
        elif sig == "SELL" and entry_p is not None:
            exit_p = next_open * 0.9995
            pnl_pct = (exit_p - entry_p) / entry_p * 100
            comm_pct = (0.005 * 100 * 2) / (entry_p * 100) * 100
            net_pct = pnl_pct - comm_pct
            h_days = len(pd.bdate_range(entry_d, sig_d)) - 1
            trades.append({
                "signal_date": str(sig_d.date()),
                "direction": "LONG",
                "entry_date": str(entry_d.date()),
                "exit_date": str(sig_d.date()),
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_p, 2),
                "gross_return_pct": round(pnl_pct, 4),
                "slippage_bps": 10.0,
                "commission_pct": round(comm_pct, 4),
                "net_return_pct": round(net_pct, 4),
                "holding_days": h_days,
                "is_winner": net_pct > 0,
            })
            entry_p = None
            entry_d = None

    tdf = pd.DataFrame(trades)
    net_rets = tdf["net_return_pct"].values
    cum_factor = np.cumprod(1 + net_rets / 100)
    total_ret = (cum_factor[-1] - 1) * 100
    peak = np.maximum.accumulate(cum_factor)
    drawdown = (cum_factor - peak) / peak * 100
    max_dd = float(np.min(drawdown))
    gp = float(tdf[tdf["is_winner"]]["gross_return_pct"].sum())
    gl = float(abs(tdf[~tdf["is_winner"]]["gross_return_pct"].sum()))
    pf = round(gp / gl, 4) if gl > 0 else float("inf")
    return trades, total_ret, pf, max_dd

trades_flagship, total_ret, pf, max_dd = run_trade_sim(use_macro_filter=True)
trades_no_macro, ret_no_macro, pf_no_macro, dd_no_macro = run_trade_sim(use_macro_filter=False)

# ---------------------------------------------------------------------------
# Investigation 4: Baseline Calculations
# ---------------------------------------------------------------------------
# True label counts in H2
n_h2 = len(y_true_h2)
counts_h2 = {
    "SELL": int(np.sum(y_true_h2 == 0)),
    "HOLD": int(np.sum(y_true_h2 == 1)),
    "BUY": int(np.sum(y_true_h2 == 2)),
}

# 1. Majority-Class Baseline (always predict BUY / Class 2)
y_maj = np.full(n_h2, 2)
acc_maj = accuracy_score(y_true_h2, y_maj)
bal_maj = balanced_accuracy_score(y_true_h2, y_maj)
prec_maj = precision_score(y_true_h2, y_maj, labels=[0, 1, 2], average=None, zero_division=0)
rec_maj = recall_score(y_true_h2, y_maj, labels=[0, 1, 2], average=None, zero_division=0)
f1_maj = f1_score(y_true_h2, y_maj, labels=[0, 1, 2], average=None, zero_division=0)
macro_f1_maj = f1_score(y_true_h2, y_maj, average="macro", zero_division=0)
weighted_f1_maj = f1_score(y_true_h2, y_maj, average="weighted", zero_division=0)

# 2. Uniform Random Baseline (expected theoretical values)
# Expected accuracy = 1/3 * (36/112 + 16/112 + 60/112) = 1/3 = 33.33%
# Expected balanced accuracy = 1/3 = 33.33%
# Expected precision for class c = support_c / 112
# Expected recall for class c = 1/3
# Expected F1 = 2 * P * R / (P + R)
p_sell_rnd = counts_h2["SELL"] / n_h2
p_hold_rnd = counts_h2["HOLD"] / n_h2
p_buy_rnd = counts_h2["BUY"] / n_h2
r_rnd = 1.0 / 3.0
f1_sell_rnd = 2 * p_sell_rnd * r_rnd / (p_sell_rnd + r_rnd)
f1_hold_rnd = 2 * p_hold_rnd * r_rnd / (p_hold_rnd + r_rnd)
f1_buy_rnd = 2 * p_buy_rnd * r_rnd / (p_buy_rnd + r_rnd)
macro_f1_rnd = (f1_sell_rnd + f1_hold_rnd + f1_buy_rnd) / 3.0
weighted_f1_rnd = (p_sell_rnd * f1_sell_rnd + p_hold_rnd * f1_hold_rnd + p_buy_rnd * f1_buy_rnd)

# ---------------------------------------------------------------------------
# Investigation 3: H1 Observation Date Extraction
# ---------------------------------------------------------------------------
df_h1 = df_feat.loc["2025-01-02":"2025-06-30"].dropna(subset=FEATURE_COLUMNS)
common_h1 = df_h1.index.intersection(tb.dropna().index)
all_106_dates = [d.strftime("%Y-%m-%d") for d in common_h1]

# Eligible 48 calibration dates from manifest: 2025-03-31 to 2025-06-06
eligible_48 = [d for d in all_106_dates if "2025-03-31" <= d <= "2025-06-06"]
# Purged 15 crossing dates from manifest: 2025-06-09 to 2025-06-30
purged_15 = [d for d in all_106_dates if "2025-06-09" <= d <= "2025-06-30"]
# Early unaligned dates: 2025-01-02 to 2025-03-28 (43 dates dropped by train.py 60-bar sequence window)
early_43 = [d for d in all_106_dates if d < "2025-03-31"]

# Build master summary dictionary
results_summary = {
    "inv1_dl_fusion": {
        "raw_counts": {
            "SELL": int(np.sum(np.argmax(dl_raw_arr, axis=1) == 0)),
            "HOLD": int(np.sum(np.argmax(dl_raw_arr, axis=1) == 1)),
            "BUY": int(np.sum(np.argmax(dl_raw_arr, axis=1) == 2)),
        },
        "raw_distribution_pct": {
            "SELL": round(float(np.mean(np.argmax(dl_raw_arr, axis=1) == 0) * 100), 2),
            "HOLD": round(float(np.mean(np.argmax(dl_raw_arr, axis=1) == 1) * 100), 2),
            "BUY": round(float(np.mean(np.argmax(dl_raw_arr, axis=1) == 2) * 100), 2),
        },
        "calibrated_counts": {
            "SELL": int(np.sum(np.argmax(dl_cal_arr, axis=1) == 0)),
            "HOLD": int(np.sum(np.argmax(dl_cal_arr, axis=1) == 1)),
            "BUY": int(np.sum(np.argmax(dl_cal_arr, axis=1) == 2)),
        },
        "calibrated_distribution_pct": {
            "SELL": round(float(np.mean(np.argmax(dl_cal_arr, axis=1) == 0) * 100), 2),
            "HOLD": round(float(np.mean(np.argmax(dl_cal_arr, axis=1) == 1) * 100), 2),
            "BUY": round(float(np.mean(np.argmax(dl_cal_arr, axis=1) == 2) * 100), 2),
        },
        "calibrated_prob_stats": {
            "mean": [round(float(x), 4) for x in np.mean(dl_cal_arr, axis=0)],
            "median": [round(float(x), 4) for x in np.median(dl_cal_arr, axis=0)],
            "min": [round(float(x), 4) for x in np.min(dl_cal_arr, axis=0)],
            "max": [round(float(x), 4) for x in np.max(dl_cal_arr, axis=0)],
            "std": [round(float(x), 4) for x in np.std(dl_cal_arr, axis=0)],
        },
        "entropy": {
            "mean": round(float(np.mean(calc_entropy(dl_cal_arr))), 4),
            "std": round(float(np.std(calc_entropy(dl_cal_arr))), 4),
        }
    },
    "inv1_dqn": {
        "dashboard_counts": {
            "SELL": int(np.sum(np.argmax(dqn_dash_arr, axis=1) == 0)),
            "HOLD": int(np.sum(np.argmax(dqn_dash_arr, axis=1) == 1)),
            "BUY": int(np.sum(np.argmax(dqn_dash_arr, axis=1) == 2)),
        },
        "dashboard_distribution_pct": {
            "SELL": round(float(np.mean(np.argmax(dqn_dash_arr, axis=1) == 0) * 100), 2),
            "HOLD": round(float(np.mean(np.argmax(dqn_dash_arr, axis=1) == 1) * 100), 2),
            "BUY": round(float(np.mean(np.argmax(dqn_dash_arr, axis=1) == 2) * 100), 2),
        },
        "dashboard_prob_stats": {
            "mean": [round(float(x), 4) for x in np.mean(dqn_dash_arr, axis=0)],
            "median": [round(float(x), 4) for x in np.median(dqn_dash_arr, axis=0)],
            "min": [round(float(x), 4) for x in np.min(dqn_dash_arr, axis=0)],
            "max": [round(float(x), 4) for x in np.max(dqn_dash_arr, axis=0)],
            "std": [round(float(x), 4) for x in np.std(dqn_dash_arr, axis=0)],
        },
        "dashboard_entropy": {
            "mean": round(float(np.mean(calc_entropy(dqn_dash_arr))), 4),
            "std": round(float(np.std(calc_entropy(dqn_dash_arr))), 4),
        },
        "authentic_counts": {
            "SELL": int(np.sum(np.argmax(dqn_auth_arr, axis=1) == 0)),
            "HOLD": int(np.sum(np.argmax(dqn_auth_arr, axis=1) == 1)),
            "BUY": int(np.sum(np.argmax(dqn_auth_arr, axis=1) == 2)),
        },
        "authentic_distribution_pct": {
            "SELL": round(float(np.mean(np.argmax(dqn_auth_arr, axis=1) == 0) * 100), 2),
            "HOLD": round(float(np.mean(np.argmax(dqn_auth_arr, axis=1) == 1) * 100), 2),
            "BUY": round(float(np.mean(np.argmax(dqn_auth_arr, axis=1) == 2) * 100), 2),
        },
        "authentic_prob_stats": {
            "mean": [round(float(x), 4) for x in np.mean(dqn_auth_arr, axis=0)],
            "median": [round(float(x), 4) for x in np.median(dqn_auth_arr, axis=0)],
            "min": [round(float(x), 4) for x in np.min(dqn_auth_arr, axis=0)],
            "max": [round(float(x), 4) for x in np.max(dqn_auth_arr, axis=0)],
            "std": [round(float(x), 4) for x in np.std(dqn_auth_arr, axis=0)],
        },
        "authentic_entropy": {
            "mean": round(float(np.mean(calc_entropy(dqn_auth_arr))), 4),
            "std": round(float(np.std(calc_entropy(dqn_auth_arr))), 4),
        }
    },
    "inv2_traces": h2_traces,
    "inv3_h1_dates": {
        "all_106_dates": all_106_dates,
        "eligible_48_dates": eligible_48,
        "purged_15_dates": purged_15,
        "early_43_dates": early_43,
    },
    "inv4_baselines": {
        "majority_class": {
            "accuracy": round(float(acc_maj), 4),
            "balanced_accuracy": round(float(bal_maj), 4),
            "precision": {
                "SELL": round(float(prec_maj[0]), 4),
                "HOLD": round(float(prec_maj[1]), 4),
                "BUY": round(float(prec_maj[2]), 4),
            },
            "recall": {
                "SELL": round(float(rec_maj[0]), 4),
                "HOLD": round(float(rec_maj[1]), 4),
                "BUY": round(float(rec_maj[2]), 4),
            },
            "f1": {
                "SELL": round(float(f1_maj[0]), 4),
                "HOLD": round(float(f1_maj[1]), 4),
                "BUY": round(float(f1_maj[2]), 4),
            },
            "macro_f1": round(float(macro_f1_maj), 4),
            "weighted_f1": round(float(weighted_f1_maj), 4),
        },
        "uniform_random": {
            "accuracy": round(float(1.0 / 3.0), 4),
            "balanced_accuracy": round(float(1.0 / 3.0), 4),
            "precision": {
                "SELL": round(float(p_sell_rnd), 4),
                "HOLD": round(float(p_hold_rnd), 4),
                "BUY": round(float(p_buy_rnd), 4),
            },
            "recall": {
                "SELL": round(float(r_rnd), 4),
                "HOLD": round(float(r_rnd), 4),
                "BUY": round(float(r_rnd), 4),
            },
            "f1": {
                "SELL": round(float(f1_sell_rnd), 4),
                "HOLD": round(float(f1_hold_rnd), 4),
                "BUY": round(float(f1_buy_rnd), 4),
            },
            "macro_f1": round(float(macro_f1_rnd), 4),
            "weighted_f1": round(float(weighted_f1_rnd), 4),
        }
    },
    "inv6_trades": {
        "flagship_5_trades": trades_flagship,
        "compounded_return_pct": round(float(total_ret), 4),
        "profit_factor": round(float(pf), 4),
        "max_drawdown_pct": round(float(max_dd), 4),
        "controlled_comparison_no_macro": {
            "trade_count": len(trades_no_macro),
            "compounded_return_pct": round(float(ret_no_macro), 4),
            "profit_factor": round(float(pf_no_macro), 4),
            "max_drawdown_pct": round(float(dd_no_macro), 4),
        }
    }
}

out_path = BACKEND / "reports" / "v2_2_dashboard" / "forensic_diagnostic_data.json"
with open(out_path, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"[SUCCESS] Forensic extraction complete! Saved to {out_path}")
print(f"H2 Trades Count: {len(trades_flagship)}, Compounded Return: {total_ret:.2f}%, PF: {pf:.2f}, MaxDD: {max_dd:.2f}%")
print(f"Counterfactual without Macro: Trades={len(trades_no_macro)}, Return={ret_no_macro:.2f}%, PF={pf_no_macro:.2f}, MaxDD={dd_no_macro:.2f}%")
