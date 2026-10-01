# =============================================================================
# HYDRA V2.2 — Frozen Model Signal Accuracy & Performance Dashboard
# =============================================================================
# EVALUATION-ONLY: Does NOT retrain, refit, or modify any model, scaler,
# calibrator, hyperparameter, threshold, or manifest artifact.
# =============================================================================
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND = Path(r"D:\DataScience\Projects\Data_Science_Projects\Stock_Indicator\backend")
ARTIFACTS = BACKEND / "artifacts"
CONFIGS = BACKEND / "configs"
REPORTS_OUT = BACKEND / "reports" / "v2_2_dashboard"
REPORTS_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND))

# ---------------------------------------------------------------------------
# 0. Load Frozen Manifest (read-only verification)
# ---------------------------------------------------------------------------
manifest = json.load(open(ARTIFACTS / "frozen_strategy_manifest_v2.2.json"))
assert manifest["strategy_version"] == "HYDRA_PROSPECTIVE_V2.2"
print(f"[AUDIT] Strategy: {manifest['strategy_version']}  Status: {manifest['status']}")

# ---------------------------------------------------------------------------
# 1. Feature Engineering (reuse exact production pipeline)
# ---------------------------------------------------------------------------
from src.data_ingestion.market_data import (
    apply_dynamic_triple_barrier,
    fetch_historical_data,
)
from src.execution.live_inference import (
    FEATURE_COLUMNS,
    add_upgraded_features,
    load_config,
)

config = load_config(ticker="AAPL")
TICKER = "AAPL"
TIME_STEPS = config["data"]["time_steps"]  # 60


def fetch_and_engineer(start, end):
    """Fetch OHLCV and compute features using the exact frozen pipeline."""
    import yfinance as yf
    warmup_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    df = fetch_historical_data(TICKER, warmup_start, end)
    spy_df = yf.download("SPY", start=warmup_start, end=end, progress=False)
    vix_df = yf.download("^VIX", start=warmup_start, end=end, progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.droplevel(1)

    df_feat = add_upgraded_features(df.copy(), spy_df, vix_df, lag_vix=True)
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()].copy()

    df_full = df_feat.copy()
    df_feat = df_feat.loc[start:end]
    return df_full, df_feat, spy_df, vix_df


def generate_labels(df_full, eval_start, eval_end):
    """Generate triple-barrier labels for the evaluation period."""
    tb = apply_dynamic_triple_barrier(
        df_full.copy(),
        tp_atr_multiplier=1.5,
        sl_atr_multiplier=2.0,
        horizon=15,
    )
    tb = tb.loc[eval_start:eval_end]
    if "target_signal" not in tb.columns:
        return None
    return tb["target_signal"]


# ---------------------------------------------------------------------------
# 2. Load All Frozen Models (read-only)
# ---------------------------------------------------------------------------
print("\n[LOADING] Frozen V2.2 model artifacts...")

scaler = joblib.load(str(ARTIFACTS / "latest_scaler.joblib"))

import xgboost as xgb

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(ARTIFACTS / "xgb_ensemble.json"))

lgbm_model = joblib.load(str(ARTIFACTS / "lgbm_agent.joblib"))

from src.models.neural.fusion_network import build_fusion_model

dl_model = build_fusion_model(config)
dl_model.load_weights(str(ARTIFACTS / "latest_fusion_weights.weights.h5"))

from src.models.rl.dqn_agent import DQNAgent

dqn_agent = DQNAgent(state_size=len(FEATURE_COLUMNS) + 6)
dqn_agent.load(str(ARTIFACTS / "dqn_model.pth"))
dqn_agent.epsilon = 0.0

from src.models.ensemble.meta_ensemble import MetaEnsemble

meta_ensemble = MetaEnsemble.load(str(ARTIFACTS / "meta_ensemble.joblib"))

from src.models.regime.calibration import ModelCalibrator

calibrator = ModelCalibrator.load(str(ARTIFACTS / "model_calibrator.joblib"))

print("[LOADED] All 6 model components loaded from frozen V2.2 artifacts.\n")

# ---------------------------------------------------------------------------
# 3. Evaluation Periods
# ---------------------------------------------------------------------------
PERIODS = {
    "Development (2020-2024)": ("2020-01-02", "2024-12-31", "DEVELOPMENT"),
    "Calibration H1 (2025 Q1-Q2)": ("2025-01-02", "2025-06-30", "CALIBRATION_H1"),
    "Evaluation H2 (2025 Q3-Q4)": ("2025-07-01", "2025-12-31", "EVALUATION_H2"),
}

# ---------------------------------------------------------------------------
# 4. Per-Bar Inference Engine (strictly point-in-time)
# ---------------------------------------------------------------------------
def run_model_inference(df_feat, df_full, spy_df):
    """Run all frozen models on each bar, producing per-bar probability vectors."""
    df_filtered = df_feat.reindex(columns=FEATURE_COLUMNS).dropna()
    if df_filtered.empty:
        return pd.DataFrame()

    scaled_all = scaler.transform(df_filtered.values)

    xgb_probs = xgb_model.predict_proba(scaled_all)
    lgbm_probs = lgbm_model.predict_proba(scaled_all)

    # DL Fusion
    dl_probs = np.full((len(df_filtered), 3), 1.0 / 3.0)
    full_filtered = df_full.reindex(columns=FEATURE_COLUMNS).dropna()
    full_scaled = scaler.transform(full_filtered.values)
    full_idx = full_filtered.index

    from src.data_ingestion.market_data import get_sector_peer
    peer_ticker = get_sector_peer(TICKER)
    import yfinance as yf
    peer_start = (pd.Timestamp(df_full.index[0]) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    peer_end = df_full.index[-1].strftime("%Y-%m-%d") if hasattr(df_full.index[-1], 'strftime') else str(df_full.index[-1])[:10]
    try:
        peer_df_raw = fetch_historical_data(peer_ticker, peer_start, peer_end)
        spy_for_peer = yf.download("SPY", start=peer_start, end=peer_end, progress=False)
        vix_for_peer = yf.download("^VIX", start=peer_start, end=peer_end, progress=False)
        if isinstance(spy_for_peer.columns, pd.MultiIndex):
            spy_for_peer.columns = spy_for_peer.columns.droplevel(1)
        if isinstance(vix_for_peer.columns, pd.MultiIndex):
            vix_for_peer.columns = vix_for_peer.columns.droplevel(1)
        peer_feat = add_upgraded_features(peer_df_raw.copy(), spy_for_peer, vix_for_peer, lag_vix=True)
        peer_feat = peer_feat.loc[:, ~peer_feat.columns.duplicated()].copy()
        peer_filtered = peer_feat.reindex(columns=FEATURE_COLUMNS).dropna()
        peer_scaled = scaler.transform(peer_filtered.values)
        peer_full_idx = peer_filtered.index
    except Exception:
        peer_scaled = full_scaled
        peer_full_idx = full_idx

    for i, eval_date in enumerate(df_filtered.index):
        loc_in_full = full_idx.get_loc(eval_date) if eval_date in full_idx else -1
        if isinstance(loc_in_full, slice):
            loc_in_full = loc_in_full.stop - 1
        if loc_in_full < TIME_STEPS:
            continue
        ts_window = full_scaled[loc_in_full - TIME_STEPS + 1: loc_in_full + 1]
        ts_seq = ts_window.reshape(1, TIME_STEPS, -1)

        if eval_date in peer_full_idx:
            peer_loc = peer_full_idx.get_loc(eval_date)
            if isinstance(peer_loc, slice):
                peer_loc = peer_loc.stop - 1
            if peer_loc >= TIME_STEPS:
                peer_window = peer_scaled[peer_loc - TIME_STEPS + 1: peer_loc + 1]
                peer_seq = peer_window.reshape(1, TIME_STEPS, -1)
            else:
                peer_seq = ts_seq
        else:
            peer_seq = ts_seq

        try:
            raw_dl = dl_model.predict(
                [ts_seq, ts_seq, ts_seq, ts_seq, ts_seq, peer_seq], verbose=0
            )
            dl_p = raw_dl[2][0] if isinstance(raw_dl, list) else raw_dl[0]
            if len(dl_p) == 3:
                dl_probs[i] = dl_p
        except Exception:
            pass

    # DQN
    dqn_probs = np.full((len(df_filtered), 3), 1.0 / 3.0)
    for i in range(len(df_filtered)):
        row = scaled_all[i]
        augmented_state = np.concatenate([row, [0, 0, 0, 100000, 100000, 0]])
        dqn_p = dqn_agent.predict_proba(augmented_state, temperature=1.5)
        dqn_probs[i] = dqn_p

    # Calibrate
    xgb_cal = np.array([calibrator.calibrate("XGB", p) for p in xgb_probs])
    lgbm_cal = np.array([calibrator.calibrate("LGBM", p) for p in lgbm_probs])
    dl_cal = np.array([calibrator.calibrate("DL_FUSION", p) for p in dl_probs])

    # Meta-Ensemble
    meta_probs = np.full((len(df_filtered), 3), 1.0 / 3.0)
    for i in range(len(df_filtered)):
        base_preds = {
            "LSTM": dl_cal[i],
            "XGBoost": xgb_cal[i],
            "LightGBM": lgbm_cal[i],
            "DQN": dqn_probs[i],
        }
        try:
            meta_p = meta_ensemble.predict_proba(base_preds, 0)
            if hasattr(meta_p, '__len__') and len(meta_p) == 3:
                meta_probs[i] = meta_p
        except Exception:
            meta_probs[i] = np.mean([dl_cal[i], xgb_cal[i], lgbm_cal[i], dqn_probs[i]], axis=0)

    # HYDRA Final Signal
    close_series = df_full["Close"].reindex(df_filtered.index).ffill()
    sma200 = close_series.rolling(200, min_periods=20).mean()
    if spy_df is not None and "Close" in spy_df.columns:
        spy_close = spy_df["Close"].reindex(df_filtered.index).ffill()
        spy_sma50 = spy_close.rolling(50, min_periods=10).mean()
    else:
        spy_close = close_series
        spy_sma50 = close_series

    hydra_signals = []
    position = "FLAT"
    last_trade_idx = -10
    threshold = 0.60
    cooldown = 5

    for i in range(len(df_filtered)):
        p = meta_probs[i]
        p_sell, _, p_buy = float(p[0]), float(p[1]), float(p[2])

        if p_buy >= threshold:
            raw_sig = "BUY"
        elif p_sell >= threshold:
            raw_sig = "SELL"
        else:
            raw_sig = "HOLD"

        cur_c = float(close_series.iloc[i]) if i < len(close_series) else 0
        cur_sma = float(sma200.iloc[i]) if i < len(sma200) and pd.notna(sma200.iloc[i]) else cur_c
        cur_spy = float(spy_close.iloc[i]) if i < len(spy_close) else cur_c
        cur_spy_sma = float(spy_sma50.iloc[i]) if i < len(spy_sma50) and pd.notna(spy_sma50.iloc[i]) else cur_spy

        long_ok = (cur_c >= cur_sma) and (cur_spy >= cur_spy_sma)
        sell_allowed = True if position == "LONG" else ((cur_c < cur_sma) or (cur_spy < cur_spy_sma))

        if raw_sig == "BUY" and not long_ok:
            filtered = "HOLD"
        elif raw_sig == "SELL" and not sell_allowed:
            filtered = "HOLD"
        else:
            filtered = raw_sig

        if filtered == "BUY" and position != "LONG" and (i - last_trade_idx >= cooldown):
            position = "LONG"
            last_trade_idx = i
            hydra_signals.append("BUY")
        elif filtered == "SELL" and position == "LONG" and (i - last_trade_idx >= cooldown):
            position = "FLAT"
            last_trade_idx = i
            hydra_signals.append("SELL")
        else:
            hydra_signals.append("HOLD")

    results = pd.DataFrame(index=df_filtered.index)
    results["XGB_SELL"] = xgb_cal[:, 0]
    results["XGB_HOLD"] = xgb_cal[:, 1]
    results["XGB_BUY"] = xgb_cal[:, 2]
    results["XGB_pred"] = np.argmax(xgb_cal, axis=1)
    results["LGBM_SELL"] = lgbm_cal[:, 0]
    results["LGBM_HOLD"] = lgbm_cal[:, 1]
    results["LGBM_BUY"] = lgbm_cal[:, 2]
    results["LGBM_pred"] = np.argmax(lgbm_cal, axis=1)
    results["DL_SELL"] = dl_cal[:, 0]
    results["DL_HOLD"] = dl_cal[:, 1]
    results["DL_BUY"] = dl_cal[:, 2]
    results["DL_pred"] = np.argmax(dl_cal, axis=1)
    results["DQN_SELL"] = dqn_probs[:, 0]
    results["DQN_HOLD"] = dqn_probs[:, 1]
    results["DQN_BUY"] = dqn_probs[:, 2]
    results["DQN_pred"] = np.argmax(dqn_probs, axis=1)
    results["META_SELL"] = meta_probs[:, 0]
    results["META_HOLD"] = meta_probs[:, 1]
    results["META_BUY"] = meta_probs[:, 2]
    results["META_pred"] = np.argmax(meta_probs, axis=1)
    results["HYDRA_signal"] = hydra_signals
    results["HYDRA_SELL"] = meta_probs[:, 0]
    results["HYDRA_HOLD"] = meta_probs[:, 1]
    results["HYDRA_BUY"] = meta_probs[:, 2]

    return results


# ---------------------------------------------------------------------------
# 5. Metrics Functions
# ---------------------------------------------------------------------------
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def multiclass_brier(y_true, y_prob, n_classes=3):
    y_onehot = np.zeros((len(y_true), n_classes))
    for i, y in enumerate(y_true):
        if 0 <= int(y) < n_classes:
            y_onehot[i, int(y)] = 1.0
    return np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies_arr = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_true) * abs(accuracies_arr[mask].mean() - confidences[mask].mean())
    return ece


def compute_all_metrics(y_true, y_prob, model_name, y_pred=None):
    if y_pred is None:
        y_pred = np.argmax(y_prob, axis=1)
    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    labels = [0, 1, 2]
    label_names = ["SELL", "HOLD", "BUY"]
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = np.array([(y_true == c).sum() for c in labels])
    brier = multiclass_brier(y_true, y_prob)
    try:
        ll = log_loss(y_true, y_prob, labels=labels)
    except Exception:
        ll = float("nan")
    ece = expected_calibration_error(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    z = 1.96
    p_hat = acc
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    halfwidth = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return {
        "model": model_name, "n_samples": n,
        "accuracy": round(acc, 4),
        "accuracy_ci_95": f"[{max(0, center - halfwidth):.4f}, {min(1, center + halfwidth):.4f}]",
        "balanced_accuracy": round(bal_acc, 4),
        "precision_per_class": {label_names[i]: round(float(prec[i]), 4) for i in range(3)},
        "recall_per_class": {label_names[i]: round(float(rec[i]), 4) for i in range(3)},
        "f1_per_class": {label_names[i]: round(float(f1[i]), 4) for i in range(3)},
        "support_per_class": {label_names[i]: int(support[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "brier_score": round(brier, 4), "log_loss": round(ll, 4) if not np.isnan(ll) else "N/A",
        "ece": round(ece, 4),
    }


def compute_trading_metrics(df_full, hydra_signals_series):
    trades = []
    entry_price = None
    entry_date = None
    close_s = df_full["Close"].squeeze()
    open_s = df_full["Open"].squeeze() if "Open" in df_full.columns else close_s

    for i, sig_date in enumerate(hydra_signals_series.index):
        sig = hydra_signals_series.iloc[i]
        loc_in_full = df_full.index.get_indexer([sig_date])
        orig_idx = int(loc_in_full[0]) if len(loc_in_full) > 0 and loc_in_full[0] >= 0 else -1
        if orig_idx < 0 or orig_idx + 1 >= len(df_full):
            continue
        next_open = float(open_s.iloc[orig_idx + 1])
        if sig == "BUY" and entry_price is None:
            entry_price = next_open * 1.0005
            entry_date = sig_date
        elif sig == "SELL" and entry_price is not None:
            exit_price = next_open * 0.9995
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            commission_pct = (0.005 * 100 * 2) / (entry_price * 100) * 100
            holding_days = len(pd.bdate_range(entry_date, sig_date)) - 1
            trades.append({
                "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date)[:10],
                "exit_date": str(sig_date.date()) if hasattr(sig_date, 'date') else str(sig_date)[:10],
                "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2),
                "gross_pnl_pct": round(pnl_pct, 4), "commission_pct": round(commission_pct, 4),
                "net_pnl_pct": round(pnl_pct - commission_pct, 4),
                "holding_days": holding_days, "is_winner": pnl_pct > 0,
            })
            entry_price = None
            entry_date = None

    if not trades:
        return {"total_trades": 0, "trades": [], "cumulative_factors": [], "drawdown_series": []}

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    winners = trades_df[trades_df["is_winner"]]
    losers = trades_df[~trades_df["is_winner"]]
    gross_profit = float(winners["gross_pnl_pct"].sum()) if len(winners) > 0 else 0
    gross_loss = float(abs(losers["gross_pnl_pct"].sum())) if len(losers) > 0 else 0
    net_returns = trades_df["net_pnl_pct"].values
    cum_factor = np.cumprod(1 + net_returns / 100)
    total_return = (cum_factor[-1] - 1) * 100
    peak = np.maximum.accumulate(cum_factor)
    drawdown = (cum_factor - peak) / peak * 100
    max_dd = float(np.min(drawdown))
    avg_hold = float(trades_df["holding_days"].mean())
    trades_per_year = 252 / max(avg_hold, 1)
    mean_ret = float(np.mean(net_returns))
    std_ret = float(np.std(net_returns)) if np.std(net_returns) > 0 else 1e-9
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)
    downside = net_returns[net_returns < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year)
    return {
        "total_trades": n, "winning_trades": len(winners), "losing_trades": len(losers),
        "win_rate": round(len(winners) / n * 100, 2),
        "gross_profit_pct": round(gross_profit, 4), "gross_loss_pct": round(gross_loss, 4),
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf"),
        "avg_profit_per_trade_pct": round(float(winners["gross_pnl_pct"].mean()), 4) if len(winners) > 0 else 0,
        "avg_loss_per_trade_pct": round(float(losers["gross_pnl_pct"].mean()), 4) if len(losers) > 0 else 0,
        "net_return_pct": round(total_return, 4), "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4), "sortino_ratio": round(sortino, 4),
        "avg_holding_days": round(avg_hold, 1),
        "note": "Modeled costs: 5 bps slippage + $0.005/share commission. NOT actual brokerage fills.",
        "trades": trades, "cumulative_factors": cum_factor.tolist(), "drawdown_series": drawdown.tolist(),
    }


# ---------------------------------------------------------------------------
# 6. MAIN EVALUATION LOOP
# ---------------------------------------------------------------------------
all_period_results = {}
all_period_metrics = {}
all_trading_metrics = {}

for period_name, (start, end, role) in PERIODS.items():
    print(f"\n{'='*70}")
    print(f"EVALUATING: {period_name}  [{start} -> {end}]  Role: {role}")
    print(f"{'='*70}")

    try:
        df_full, df_feat, spy_df, vix_df = fetch_and_engineer(start, end)
    except Exception as e:
        print(f"  [SKIP] Data fetch failed: {e}")
        continue

    if df_feat.empty or len(df_feat) < 30:
        print(f"  [SKIP] Insufficient data: {len(df_feat)} bars")
        continue

    labels = generate_labels(df_full, start, end)
    results = run_model_inference(df_feat, df_full, spy_df)
    if results.empty:
        print("  [SKIP] Inference returned empty results")
        continue

    if labels is not None:
        common_idx = results.index.intersection(labels.index)
        results = results.loc[common_idx]
        y_true = labels.loc[common_idx].values.astype(int)
        print(f"  Aligned {len(common_idx)} bars with triple-barrier labels")
    else:
        y_true = None

    all_period_results[period_name] = results

    if y_true is not None and len(y_true) > 0:
        model_metrics = {}
        models_map = {
            "XGBoost": ("XGB_pred", np.column_stack([results["XGB_SELL"], results["XGB_HOLD"], results["XGB_BUY"]])),
            "LightGBM": ("LGBM_pred", np.column_stack([results["LGBM_SELL"], results["LGBM_HOLD"], results["LGBM_BUY"]])),
            "DL Fusion": ("DL_pred", np.column_stack([results["DL_SELL"], results["DL_HOLD"], results["DL_BUY"]])),
            "DQN": ("DQN_pred", np.column_stack([results["DQN_SELL"], results["DQN_HOLD"], results["DQN_BUY"]])),
            "Meta-Ensemble": ("META_pred", np.column_stack([results["META_SELL"], results["META_HOLD"], results["META_BUY"]])),
        }
        for mname, (pred_col, probs) in models_map.items():
            y_pred = results[pred_col].values.astype(int)
            m = compute_all_metrics(y_true, probs, mname, y_pred)
            model_metrics[mname] = m
            print(f"  {mname}: Acc={m['accuracy']:.4f}  Brier={m['brier_score']:.4f}  ECE={m['ece']:.4f}")

        hydra_pred = np.array([{"BUY": 2, "SELL": 0, "HOLD": 1}[s] for s in results["HYDRA_signal"]])
        hydra_probs = np.column_stack([results["HYDRA_SELL"], results["HYDRA_HOLD"], results["HYDRA_BUY"]])
        model_metrics["HYDRA Final"] = compute_all_metrics(y_true, hydra_probs, "HYDRA Final", hydra_pred)
        all_period_metrics[period_name] = model_metrics
    else:
        all_period_metrics[period_name] = {}

    hydra_sigs = results["HYDRA_signal"]
    trading = compute_trading_metrics(df_full, hydra_sigs)
    all_trading_metrics[period_name] = trading
    if trading["total_trades"] > 0:
        print(f"  Trading: {trading['total_trades']} trades, WR={trading['win_rate']}%, PF={trading['profit_factor']}, Sharpe={trading['sharpe_ratio']}")


# ---------------------------------------------------------------------------
# 7. CHARTS
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "text.color": "#c9d1d9", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d", "grid.color": "#21262d",
    "font.family": "sans-serif", "font.size": 11,
})

COLORS = {"XGBoost": "#58a6ff", "LightGBM": "#3fb950", "DL Fusion": "#d2a8ff",
          "DQN": "#f0883e", "Meta-Ensemble": "#f778ba", "HYDRA Final": "#ffd700"}
CLASS_COLORS = {"SELL": "#f85149", "HOLD": "#8b949e", "BUY": "#3fb950"}


def save_chart(fig, name):
    path = REPORTS_OUT / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [CHART] Saved: {path.name}")


eval_period = None
for pname in ["Evaluation H2 (2025 Q3-Q4)", "Calibration H1 (2025 Q1-Q2)", "Development (2020-2024)"]:
    if pname in all_period_metrics and all_period_metrics[pname]:
        eval_period = pname
        break

if eval_period:
    metrics = all_period_metrics[eval_period]
    results = all_period_results[eval_period]
    models = list(metrics.keys())

    # Chart 1: Accuracy by Model
    fig, ax = plt.subplots(figsize=(10, 6))
    accs = [metrics[m]["accuracy"] for m in models]
    bars = ax.barh(models, accs, color=[COLORS.get(m, "#58a6ff") for m in models], height=0.6, edgecolor="#30363d")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{acc:.1%}", va="center", fontsize=10, color="#c9d1d9")
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Model Accuracy - {eval_period}", fontsize=14, fontweight="bold", color="#ffd700")
    ax.set_xlim(0, max(accs) * 1.2)
    ax.axvline(1/3, color="#f85149", linestyle="--", alpha=0.5, label="Random (33.3%)")
    ax.legend(loc="lower right")
    save_chart(fig, "01_accuracy_by_model")

    # Chart 2: Precision & Recall by Class
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(models))
    width = 0.25
    for ax_idx, (mn, mk) in enumerate([("Precision", "precision_per_class"), ("Recall", "recall_per_class")]):
        ax = axes[ax_idx]
        for ci, cls in enumerate(["SELL", "HOLD", "BUY"]):
            vals = [metrics[m][mk][cls] for m in models]
            ax.bar(x + ci * width, vals, width, label=cls, color=CLASS_COLORS[cls], alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(mn)
        ax.set_title(mn, fontsize=13, fontweight="bold")
        ax.legend()
        ax.set_ylim(0, 1)
    fig.suptitle(f"Precision & Recall - {eval_period}", fontsize=14, fontweight="bold", color="#ffd700")
    fig.tight_layout()
    save_chart(fig, "02_precision_recall")

    # Chart 3: Confusion Matrix
    for mname in ["Meta-Ensemble", "XGBoost"]:
        if mname in metrics:
            cm = np.array(metrics[mname]["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(cm, cmap="Blues", aspect="auto")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                            color="white" if cm[i, j] > cm.max()/2 else "#c9d1d9")
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(["SELL", "HOLD", "BUY"])
            ax.set_yticklabels(["SELL", "HOLD", "BUY"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix: {mname}", fontsize=13, fontweight="bold", color="#ffd700")
            save_chart(fig, "03_confusion_matrix")
            break

    # Chart 4: Brier Score & Log Loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, (mn, mk) in enumerate([("Brier Score", "brier_score"), ("Log Loss", "log_loss")]):
        ax = axes[ax_idx]
        vals = [(m, metrics[m][mk]) for m in models if metrics[m][mk] != "N/A"]
        names = [v[0] for v in vals]
        values = [v[1] for v in vals]
        ax.barh(names, values, color=[COLORS.get(n, "#58a6ff") for n in names], height=0.6, edgecolor="#30363d")
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=10, color="#c9d1d9")
        ax.set_xlabel(mn)
        ax.set_title(mn, fontsize=13, fontweight="bold")
    fig.suptitle(f"Probability Metrics - {eval_period}", fontsize=14, fontweight="bold", color="#ffd700")
    fig.tight_layout()
    save_chart(fig, "04_brier_logloss")

    # Chart 5: Reliability Diagram (BUY class for XGBoost)
    fig, ax = plt.subplots(figsize=(7, 7))
    labels_for_diag = generate_labels(
        fetch_and_engineer(*PERIODS[eval_period][:2])[0],
        *PERIODS[eval_period][:2]
    )
    if labels_for_diag is not None:
        common = results.index.intersection(labels_for_diag.index)
        y_t = (labels_for_diag.loc[common].values == 2).astype(float)
        p_b = results.loc[common, "XGB_BUY"].values
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        centers, accs_bin = [], []
        for b in range(n_bins):
            mask = (p_b > bin_edges[b]) & (p_b <= bin_edges[b + 1])
            if mask.sum() > 0:
                centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                accs_bin.append(y_t[mask].mean())
        ax.plot([0, 1], [0, 1], "--", color="#f85149", alpha=0.5, label="Perfect calibration")
        ax.bar(centers, accs_bin, width=0.08, alpha=0.7, color="#58a6ff", label="XGBoost")
        ax.set_xlabel("Predicted P(BUY)")
        ax.set_ylabel("Observed frequency")
        ax.set_title("Reliability Diagram (BUY)", fontsize=13, fontweight="bold", color="#ffd700")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    save_chart(fig, "05_reliability_diagram")


# Trading charts
for period_name, trading in all_trading_metrics.items():
    if trading["total_trades"] < 2:
        continue
    trades_df = pd.DataFrame(trading["trades"])
    trade_dates = pd.to_datetime(trades_df["exit_date"])
    tag = period_name.split("(")[0].strip().lower().replace(" ", "_")

    # Chart 6: Cumulative Returns
    fig, ax = plt.subplots(figsize=(12, 6))
    cum_f = np.array(trading["cumulative_factors"])
    ax.plot(trade_dates, (cum_f - 1) * 100, color="#ffd700", linewidth=2, label="HYDRA V2.2")
    ax.fill_between(trade_dates, 0, (cum_f - 1) * 100, where=(cum_f - 1) * 100 >= 0, alpha=0.15, color="#3fb950")
    ax.fill_between(trade_dates, 0, (cum_f - 1) * 100, where=(cum_f - 1) * 100 < 0, alpha=0.15, color="#f85149")
    ax.axhline(0, color="#30363d")
    ax.set_xlabel("Exit Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(f"Cumulative Returns - {period_name}", fontsize=14, fontweight="bold", color="#ffd700")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_chart(fig, f"06_cumulative_{tag}")

    # Chart 7: Drawdown
    fig, ax = plt.subplots(figsize=(12, 5))
    dd = np.array(trading["drawdown_series"])
    ax.fill_between(trade_dates, dd, 0, alpha=0.4, color="#f85149")
    ax.plot(trade_dates, dd, color="#f85149", linewidth=1.5)
    ax.axhline(0, color="#30363d")
    ax.set_xlabel("Exit Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"Drawdown - {period_name}", fontsize=14, fontweight="bold", color="#ffd700")
    ax.grid(True, alpha=0.3)
    save_chart(fig, f"07_drawdown_{tag}")

    # Chart 8: Individual Trades
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_t = ["#3fb950" if t["is_winner"] else "#f85149" for t in trading["trades"]]
    ax.bar(range(len(trades_df)), trades_df["net_pnl_pct"], color=colors_t, edgecolor="#30363d", alpha=0.85)
    ax.axhline(0, color="#ffd700", linewidth=1)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Net P&L (%)")
    ax.set_title(f"Trade Outcomes - {period_name}", fontsize=14, fontweight="bold", color="#ffd700")
    ax.grid(True, alpha=0.3)
    save_chart(fig, f"08_trade_outcomes_{tag}")

    # Chart 9: Monthly Performance
    trades_df["exit_month"] = pd.to_datetime(trades_df["exit_date"]).dt.to_period("M")
    monthly = trades_df.groupby("exit_month")["net_pnl_pct"].sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    month_colors = ["#3fb950" if v >= 0 else "#f85149" for v in monthly.values]
    ax.bar([str(m) for m in monthly.index], monthly.values, color=month_colors, edgecolor="#30363d")
    ax.axhline(0, color="#ffd700", linewidth=1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Net Return (%)")
    ax.set_title(f"Monthly Performance - {period_name}", fontsize=14, fontweight="bold", color="#ffd700")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    save_chart(fig, f"09_monthly_{tag}")

# Chart 10: Predicted class distribution
if eval_period and eval_period in all_period_metrics:
    metrics = all_period_metrics[eval_period]
    models = list(metrics.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    for ci, cls in enumerate(["SELL", "HOLD", "BUY"]):
        pred_counts = []
        for m in models:
            cm = np.array(metrics[m]["confusion_matrix"])
            pred_counts.append(cm[:, ci].sum())
        bottom = np.zeros(len(models))
        for prev_ci in range(ci):
            prev = [np.array(metrics[m]["confusion_matrix"])[:, prev_ci].sum() for m in models]
            bottom += np.array(prev)
        ax.bar(x, pred_counts, 0.6, bottom=bottom, label=f"Pred {cls}", color=CLASS_COLORS[cls], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(f"Predicted Class Distribution - {eval_period}", fontsize=14, fontweight="bold", color="#ffd700")
    ax.legend()
    save_chart(fig, "10_predicted_distribution")

print(f"\n[CHARTS] All charts saved to: {REPORTS_OUT}")


# ---------------------------------------------------------------------------
# 8. JSON Report
# ---------------------------------------------------------------------------
final_report = {
    "strategy_version": "HYDRA_PROSPECTIVE_V2.2",
    "evaluation_timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "evaluation_type": "READ_ONLY_FROZEN_MODEL_EVALUATION",
    "periods": {},
}

for period_name in PERIODS:
    entry = {"role": PERIODS[period_name][2]}
    if period_name in all_period_metrics:
        entry["classification_metrics"] = all_period_metrics[period_name]
    if period_name in all_trading_metrics:
        t = all_trading_metrics[period_name]
        entry["trading_performance"] = {k: v for k, v in t.items() if k not in ["trades", "cumulative_factors", "drawdown_series"]}
    final_report["periods"][period_name] = entry

wl = []
for pn, mets in all_period_metrics.items():
    for mn, m in mets.items():
        if m["n_samples"] < 30:
            wl.append(f"{pn}/{mn}: {m['n_samples']} samples < 30 threshold")
        if m["brier_score"] > 0.8:
            wl.append(f"{pn}/{mn}: Brier {m['brier_score']:.4f} > 0.8 indicates poor calibration")
for pn, t in all_trading_metrics.items():
    if 0 < t["total_trades"] < 10:
        wl.append(f"{pn}: {t['total_trades']} trades insufficient for reliable statistics")

final_report["data_quality_warnings"] = wl
final_report["prospective_note"] = "No V2.2 prospective signals exist yet. First eligible: 2026-10-01 16:00:02 EDT."

with open(REPORTS_OUT / "v2_2_evaluation_report.json", "w") as f:
    json.dump(final_report, f, indent=2, default=str)
print(f"\n[REPORT] Saved: {REPORTS_OUT / 'v2_2_evaluation_report.json'}")

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("HYDRA V2.2 FROZEN MODEL EVALUATION SUMMARY")
print("=" * 72)

for period_name, mets in all_period_metrics.items():
    if not mets:
        continue
    print(f"\n--- {period_name} ---")
    print(f"{'Model':<20} {'Acc':>7} {'Bal.Acc':>8} {'Brier':>7} {'ECE':>7} {'N':>6}")
    print("-" * 55)
    for mn, m in mets.items():
        print(f"{mn:<20} {m['accuracy']:>7.4f} {m['balanced_accuracy']:>8.4f} {m['brier_score']:>7.4f} {m['ece']:>7.4f} {m['n_samples']:>6d}")

for period_name, t in all_trading_metrics.items():
    if t["total_trades"] == 0:
        continue
    print(f"\n--- Trading: {period_name} ---")
    print(f"  Trades: {t['total_trades']} | WR: {t['win_rate']}% | PF: {t['profit_factor']}")
    print(f"  Net Return: {t['net_return_pct']:.2f}% | Max DD: {t['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe: {t['sharpe_ratio']:.4f} | Sortino: {t['sortino_ratio']:.4f} | Avg Hold: {t['avg_holding_days']}d")
    print(f"  [{t['note']}]")

if wl:
    print(f"\n--- Warnings ({len(wl)}) ---")
    for w in wl:
        print(f"  [!] {w}")

print("\n[PROSPECTIVE] 0 signals. First eligible: 2026-10-01 16:00:02 EDT")
print(f"[COMPLETE] All outputs: {REPORTS_OUT}")
print("=" * 72)
