import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

BACKEND = Path(r"D:\DataScience\Projects\Data_Science_Projects\Stock_Indicator\backend")
ARTIFACTS_DIR = BACKEND / "artifacts"
REPORTS_OUT = BACKEND / "reports" / "v2_2_dashboard"
REPORTS_OUT.mkdir(parents=True, exist_ok=True)
BRAIN_ARTIFACTS = Path(r"C:\Users\dhruv\.gemini\antigravity-ide\brain\0907f3e7-eca0-454d-8074-78e79fc8c669")

sys.path.insert(0, str(BACKEND))

from src.data_ingestion.market_data import (
    apply_dynamic_triple_barrier,
    fetch_historical_data,
)
from src.execution.live_inference import (
    FEATURE_COLUMNS,
    add_upgraded_features,
)

# ---------------------------------------------------------------------------
# Load frozen models
# ---------------------------------------------------------------------------
print("[LOADING] Loading frozen models...")
scaler = joblib.load(str(ARTIFACTS_DIR / "latest_scaler.joblib"))
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(ARTIFACTS_DIR / "xgb_ensemble.json"))
lgbm_model = joblib.load(str(ARTIFACTS_DIR / "lgbm_agent.joblib"))
calibrator = joblib.load(str(ARTIFACTS_DIR / "model_calibrator.joblib"))

# Load existing report JSON for DL & DQN metrics
with open(REPORTS_OUT / "v2_2_evaluation_report.json") as f:
    existing_rep = json.load(f)

# Matplotlib dark styling
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "text.color": "#c9d1d9",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 10,
})

def save_fig(fig, filename):
    p1 = REPORTS_OUT / filename
    p2 = BRAIN_ARTIFACTS / filename
    fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(p2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVED] {filename}")

# ---------------------------------------------------------------------------
# 1. Fetch & Compute Data for Development, H1, and H2
# ---------------------------------------------------------------------------
PERIODS = {
    "Development (2020-2024)": ("2020-01-02", "2024-12-31"),
    "Calibration H1 (2025 Q1-Q2)": ("2025-01-02", "2025-06-30"),
    "Evaluation H2 (2025 Q3-Q4)": ("2025-07-01", "2025-12-31"),
}

period_data = {}

for pname, (start, end) in PERIODS.items():
    print(f"Fetching {pname}...")
    warmup = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    df = fetch_historical_data("AAPL", warmup, end)
    spy_df = yf.download("SPY", start=warmup, end=end, progress=False)
    vix_df = yf.download("^VIX", start=warmup, end=end, progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.droplevel(1)

    df_feat = add_upgraded_features(df.copy(), spy_df, vix_df, lag_vix=True)
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()].copy()

    tb = apply_dynamic_triple_barrier(df_feat.copy(), tp_atr_multiplier=1.5, sl_atr_multiplier=2.0, horizon=15)
    tb_sub = tb.loc[start:end]

    df_sub = df_feat.loc[start:end]
    df_filt = df_sub.reindex(columns=FEATURE_COLUMNS).dropna()
    scaled = scaler.transform(df_filt.values)

    xgb_raw = xgb_model.predict_proba(scaled)
    lgbm_raw = lgbm_model.predict_proba(scaled)

    common_idx = df_filt.index.intersection(tb_sub.index)
    y_true = tb_sub.loc[common_idx, "target_signal"].values.astype(int)

    period_data[pname] = {
        "df_full": df_feat,
        "df_filt": df_filt,
        "common_idx": common_idx,
        "y_true": y_true,
        "scaled": scaled,
        "xgb_raw": xgb_raw,
        "lgbm_raw": lgbm_raw,
        "spy_df": spy_df,
    }

# ---------------------------------------------------------------------------
# Trading Execution Function
# ---------------------------------------------------------------------------
def simulate_trades(df_full, signals_series):
    trades = []
    entry_price = None
    entry_date = None
    open_s = df_full["Open"].squeeze()

    for i, sig_date in enumerate(signals_series.index):
        sig = signals_series.iloc[i]
        loc = df_full.index.get_indexer([sig_date])
        orig_idx = int(loc[0]) if len(loc) > 0 and loc[0] >= 0 else -1
        if orig_idx < 0 or orig_idx + 1 >= len(df_full):
            continue
        next_open = float(open_s.iloc[orig_idx + 1])

        if sig == "BUY" and entry_price is None:
            entry_price = next_open * 1.0005  # +5 bps slippage
            entry_date = sig_date
        elif sig == "SELL" and entry_price is not None:
            exit_price = next_open * 0.9995  # -5 bps slippage
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            commission_pct = (0.005 * 2) / entry_price * 100
            net_pct = pnl_pct - commission_pct
            holding_days = len(pd.bdate_range(entry_date, sig_date)) - 1
            trades.append({
                "entry_date": str(entry_date)[:10],
                "exit_date": str(sig_date)[:10],
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "gross_pnl_pct": round(pnl_pct, 4),
                "commission_pct": round(commission_pct, 4),
                "net_pnl_pct": round(net_pct, 4),
                "holding_days": holding_days,
                "is_winner": net_pct > 0,
            })
            entry_price = None
            entry_date = None

    if not trades:
        return {"total_trades": 0, "trades": [], "cumulative_factors": [], "drawdown_series": []}

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    winners = trades_df[trades_df["is_winner"]]
    losers = trades_df[~trades_df["is_winner"]]
    gp = float(winners["gross_pnl_pct"].sum()) if len(winners) > 0 else 0
    gl = float(abs(losers["gross_pnl_pct"].sum())) if len(losers) > 0 else 0
    net_rets = trades_df["net_pnl_pct"].values
    cum_factor = np.cumprod(1 + net_rets / 100)
    total_ret = (cum_factor[-1] - 1) * 100
    peak = np.maximum.accumulate(cum_factor)
    drawdown = (cum_factor - peak) / peak * 100
    max_dd = float(np.min(drawdown))
    avg_hold = float(trades_df["holding_days"].mean())
    tpy = 252 / max(avg_hold, 1)
    mean_r = float(np.mean(net_rets))
    std_r = float(np.std(net_rets)) if np.std(net_rets) > 0 else 1e-9
    sharpe = (mean_r / std_r) * np.sqrt(tpy)
    downside = net_rets[net_rets < 0]
    down_std = float(np.std(downside)) if len(downside) > 1 and np.std(downside) > 0 else 1e-9
    sortino = (mean_r / down_std) * np.sqrt(tpy)

    return {
        "total_trades": n,
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": round(len(winners) / n * 100, 2),
        "gross_profit_pct": round(gp, 4),
        "gross_loss_pct": round(gl, 4),
        "profit_factor": round(gp / gl, 4) if gl > 0 else float("inf"),
        "avg_profit_per_trade_pct": round(float(winners["gross_pnl_pct"].mean()), 4) if len(winners) > 0 else 0,
        "avg_loss_per_trade_pct": round(float(losers["gross_pnl_pct"].mean()), 4) if len(losers) > 0 else 0,
        "net_return_pct": round(total_ret, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "avg_holding_days": round(avg_hold, 1),
        "trades": trades,
        "cumulative_factors": cum_factor.tolist(),
        "drawdown_series": drawdown.tolist(),
    }

# Compute trades for XGBoost on Development and H2
xgb_dev_trades = None
for pname, pdata in period_data.items():
    df_full = pdata["df_full"]
    df_filt = pdata["df_filt"]
    scaled = pdata["scaled"]
    probs = pdata["xgb_raw"]
    close_s = df_full["Close"].reindex(df_filt.index).ffill()
    sma200 = close_s.rolling(200, min_periods=20).mean()
    spy_c = pdata["spy_df"]["Close"].reindex(df_filt.index).ffill()
    spy_sma50 = spy_c.rolling(50, min_periods=10).mean()

    pos = "FLAT"
    last_t = -10
    sigs = []
    for i in range(len(df_filt)):
        pred = np.argmax(probs[i])
        c = float(close_s.iloc[i])
        sma = float(sma200.iloc[i]) if pd.notna(sma200.iloc[i]) else c
        sc = float(spy_c.iloc[i])
        ssma = float(spy_sma50.iloc[i]) if pd.notna(spy_sma50.iloc[i]) else sc
        long_ok = (c >= sma) and (sc >= ssma)

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

    sig_series = pd.Series(sigs, index=df_filt.index)
    res = simulate_trades(df_full, sig_series)
    pdata["xgb_trades"] = res
    if "Development" in pname:
        xgb_dev_trades = res

# ---------------------------------------------------------------------------
# CHART 1: Actual vs Predicted BUY/SELL/HOLD
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
class_labels = ["SELL", "HOLD", "BUY"]
class_colors = ["#f85149", "#8b949e", "#3fb950"]

for idx, (pname, pdata) in enumerate(period_data.items()):
    ax = axes[idx]
    y_t = pdata["y_true"]
    pred_xgb = np.argmax(pdata["xgb_raw"][:len(y_t)], axis=1)
    pred_lgb = np.argmax(pdata["lgbm_raw"][:len(y_t)], axis=1)

    act_counts = [np.sum(y_t == c) for c in [0, 1, 2]]
    xgb_counts = [np.sum(pred_xgb == c) for c in [0, 1, 2]]
    lgb_counts = [np.sum(pred_lgb == c) for c in [0, 1, 2]]

    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, act_counts, w, label="Actual Ground Truth", color="#ffd700", alpha=0.9, edgecolor="#30363d")
    ax.bar(x, xgb_counts, w, label="XGBoost Pred", color="#58a6ff", alpha=0.85, edgecolor="#30363d")
    ax.bar(x + w, lgb_counts, w, label="LightGBM Pred", color="#3fb950", alpha=0.85, edgecolor="#30363d")

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontweight="bold")
    ax.set_ylabel("Number of Bars")
    period_short = pname.split("(")[0].strip()
    ax.set_title(f"{period_short} (N={len(y_t)})", fontsize=12, fontweight="bold")
    if idx == 0:
        ax.legend(framealpha=0.3)
    ax.grid(axis="y", alpha=0.2)

fig.suptitle("Chart 1: Actual vs Predicted BUY / SELL / HOLD Class Frequencies", fontsize=14, fontweight="bold", color="#ffd700", y=1.02)
fig.tight_layout()
save_fig(fig, "01_actual_vs_predicted.png")

# ---------------------------------------------------------------------------
# CHART 2: Confusion Matrices (2x3 Grid)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
eval_pdata = period_data["Evaluation H2 (2025 Q3-Q4)"]
y_h2 = eval_pdata["y_true"]
models_h2 = existing_rep["periods"]["Evaluation H2 (2025 Q3-Q4)"]["classification_metrics"]

m_names = ["XGBoost", "LightGBM", "DL Fusion", "DQN", "Meta-Ensemble", "HYDRA Final"]
for idx, mname in enumerate(m_names):
    ax = axes[idx // 3, idx % 3]
    cm = np.array(models_h2[mname]["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "#c9d1d9"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=13, fontweight="bold", color=color)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["SELL", "HOLD", "BUY"], fontsize=9)
    ax.set_yticklabels(["SELL", "HOLD", "BUY"], fontsize=9)
    ax.set_xlabel("Predicted Class", fontsize=9)
    ax.set_ylabel("Actual Class", fontsize=9)
    acc = models_h2[mname]["accuracy"]
    ax.set_title(f"{mname} (Acc: {acc:.1%})", fontsize=11, fontweight="bold", color="#ffd700")

fig.suptitle("Chart 2: Confusion Matrices — Evaluation H2 2025 (Out-of-Sample, N=112)", fontsize=14, fontweight="bold", color="#ffd700", y=1.01)
fig.tight_layout()
save_fig(fig, "02_confusion_matrices.png")

# ---------------------------------------------------------------------------
# CHART 3: Accuracy by Model Across All Periods
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
p_keys = ["Development (2020-2024)", "Calibration H1 (2025 Q1-Q2)", "Evaluation H2 (2025 Q3-Q4)"]
p_labels = ["Development (2020-24)", "H1 Calibration (2025)", "H2 Evaluation (2025)"]
m_keys = ["XGBoost", "LightGBM", "DL Fusion", "DQN", "Meta-Ensemble", "HYDRA Final"]
colors_bars = ["#58a6ff", "#3fb950", "#d2a8ff"]

x = np.arange(len(m_keys))
w = 0.25

for p_i, p_key in enumerate(p_keys):
    accs = [existing_rep["periods"][p_key]["classification_metrics"][m]["accuracy"] for m in m_keys]
    bars = ax.bar(x + (p_i - 1) * w, accs, w, label=p_labels[p_i], color=colors_bars[p_i], alpha=0.85, edgecolor="#30363d")
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.1%}", ha="center", va="bottom", fontsize=8, color="#c9d1d9")

ax.axhline(1 / 3, color="#f85149", linestyle="--", linewidth=1.5, alpha=0.7, label="Random Baseline (33.3%)")
ax.set_xticks(x)
ax.set_xticklabels(m_keys, fontweight="bold", fontsize=10)
ax.set_ylabel("Classification Accuracy")
ax.set_ylim(0, 0.85)
ax.set_title("Chart 3: Model Accuracy Across Development, Calibration H1, and Out-of-Sample H2", fontsize=14, fontweight="bold", color="#ffd700")
ax.legend(framealpha=0.3, loc="upper right")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
save_fig(fig, "03_accuracy_by_model.png")

# ---------------------------------------------------------------------------
# CHART 4: Precision and Recall by Class (H2 Evaluation)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
h2_metrics = existing_rep["periods"]["Evaluation H2 (2025 Q3-Q4)"]["classification_metrics"]
models = ["XGBoost", "LightGBM", "DL Fusion", "DQN", "Meta-Ensemble"]

x = np.arange(len(models))
w = 0.25

for ax_i, (metric_title, metric_key) in enumerate([("Precision by Class", "precision_per_class"), ("Recall by Class", "recall_per_class")]):
    ax = axes[ax_i]
    for c_i, c_name in enumerate(["SELL", "HOLD", "BUY"]):
        vals = [h2_metrics[m][metric_key][c_name] for m in models]
        ax.bar(x + (c_i - 1) * w, vals, w, label=c_name, color=class_colors[c_i], alpha=0.85, edgecolor="#30363d")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(metric_title)
    ax.set_ylim(0, 1.05)
    ax.set_title(metric_title, fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.3)
    ax.grid(axis="y", alpha=0.2)

fig.suptitle("Chart 4: Per-Class Precision & Recall — Evaluation H2 2025 (N=112)", fontsize=14, fontweight="bold", color="#ffd700", y=1.02)
fig.tight_layout()
save_fig(fig, "04_precision_recall_by_class.png")

# ---------------------------------------------------------------------------
# CHART 5: Brier Score and Log Loss by Model Across Periods
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
models_prob = ["XGBoost", "LightGBM", "DL Fusion", "DQN", "Meta-Ensemble"]
colors_p = ["#58a6ff", "#3fb950", "#d2a8ff"]

# Brier
ax = axes[0]
for p_i, p_key in enumerate(p_keys):
    briers = [existing_rep["periods"][p_key]["classification_metrics"][m]["brier_score"] for m in models_prob]
    ax.bar(x + (p_i - 1) * w, briers, w, label=p_labels[p_i], color=colors_p[p_i], alpha=0.85, edgecolor="#30363d")
ax.axhline(0.6667, color="#f85149", linestyle="--", label="Random Uninformative (0.667)")
ax.set_xticks(x)
ax.set_xticklabels(models_prob, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Multiclass Brier Score (Lower is Better)")
ax.set_title("Multiclass Brier Score", fontsize=12, fontweight="bold")
ax.legend(framealpha=0.3)
ax.grid(axis="y", alpha=0.2)

# Log Loss
ax2 = axes[1]
for p_i, p_key in enumerate(p_keys):
    lls = []
    for m in models_prob:
        v = existing_rep["periods"][p_key]["classification_metrics"][m]["log_loss"]
        lls.append(min(v, 3.0) if v != "N/A" and not np.isnan(v) else 0)
    ax2.bar(x + (p_i - 1) * w, lls, w, label=p_labels[p_i], color=colors_p[p_i], alpha=0.85, edgecolor="#30363d")
ax2.axhline(1.0986, color="#f85149", linestyle="--", label="Random Log Loss ln(3)=1.099")
ax2.set_xticks(x)
ax2.set_xticklabels(models_prob, rotation=25, ha="right", fontsize=9)
ax2.set_ylabel("Multiclass Log Loss (Capped at 3.0 for display)")
ax2.set_title("Multiclass Log Loss", fontsize=12, fontweight="bold")
ax2.legend(framealpha=0.3)
ax2.grid(axis="y", alpha=0.2)

fig.suptitle("Chart 5: Probability Calibration Metrics Across Evaluation Eras", fontsize=14, fontweight="bold", color="#ffd700", y=1.02)
fig.tight_layout()
save_fig(fig, "05_brier_and_log_loss.png")

# ---------------------------------------------------------------------------
# CHART 6: Calibration Reliability Diagrams
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_i, (target_cls, cls_name) in enumerate([(2, "BUY"), (0, "SELL")]):
    ax = axes[ax_i]
    dev_pdata = period_data["Development (2020-2024)"]
    y_t = (dev_pdata["y_true"] == target_cls).astype(int)
    p_xgb = dev_pdata["xgb_raw"][:len(y_t), target_cls]
    p_lgb = dev_pdata["lgbm_raw"][:len(y_t), target_cls]

    bins = np.linspace(0, 1, 9)
    centers = (bins[:-1] + bins[1:]) / 2

    acc_xgb = []
    acc_lgb = []
    for b in range(len(bins) - 1):
        m_x = (p_xgb >= bins[b]) & (p_xgb < bins[b + 1])
        acc_xgb.append(y_t[m_x].mean() if m_x.sum() > 5 else np.nan)
        m_l = (p_lgb >= bins[b]) & (p_lgb < bins[b + 1])
        acc_lgb.append(y_t[m_l].mean() if m_l.sum() > 5 else np.nan)

    ax.plot([0, 1], [0, 1], "--", color="#8b949e", alpha=0.6, label="Perfect Calibration")
    ax.plot(centers, acc_xgb, "o-", color="#58a6ff", linewidth=2, label="XGBoost")
    ax.plot(centers, acc_lgb, "s-", color="#3fb950", linewidth=2, label="LightGBM")

    ax.set_xlabel(f"Predicted Probability P({cls_name})")
    ax.set_ylabel(f"Observed Frequency P({cls_name})")
    ax.set_title(f"Reliability Diagram: {cls_name} Class", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2)

fig.suptitle("Chart 6: Calibration Reliability Diagrams (Predicted Confidence vs Observed Frequency)", fontsize=14, fontweight="bold", color="#ffd700", y=1.02)
fig.tight_layout()
save_fig(fig, "06_calibration_reliability.png")

# ---------------------------------------------------------------------------
# CHART 7: Cumulative Portfolio Returns
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

if xgb_dev_trades and xgb_dev_trades["total_trades"] > 0:
    tdf = pd.DataFrame(xgb_dev_trades["trades"])
    tdf["exit_date"] = pd.to_datetime(tdf["exit_date"])
    cum_pct = (np.array(xgb_dev_trades["cumulative_factors"]) - 1) * 100

    ax.plot(tdf["exit_date"], cum_pct, color="#ffd700", linewidth=2.5, label=f"HYDRA V2.2 XGBoost Alpha (+{cum_pct[-1]:.1f}%)")
    ax.fill_between(tdf["exit_date"], 0, cum_pct, where=(cum_pct >= 0), color="#3fb950", alpha=0.15)
    ax.fill_between(tdf["exit_date"], 0, cum_pct, where=(cum_pct < 0), color="#f85149", alpha=0.15)

    # Add AAPL buy & hold benchmark
    df_dev = period_data["Development (2020-2024)"]["df_full"].loc["2020-01-02":"2024-12-31"]
    aapl_close = df_dev["Close"].squeeze()
    aapl_bench = (aapl_close / aapl_close.iloc[0] - 1) * 100
    ax.plot(df_dev.index, aapl_bench, color="#8b949e", linestyle=":", linewidth=1.5, label=f"AAPL Buy & Hold (+{aapl_bench.iloc[-1]:.1f}%)")

    # Add SPY benchmark
    spy_dev = period_data["Development (2020-2024)"]["spy_df"].loc["2020-01-02":"2024-12-31"]
    if not spy_dev.empty:
        spy_c = spy_dev["Close"].squeeze()
        spy_bench = (spy_c / spy_c.iloc[0] - 1) * 100
        ax.plot(spy_dev.index, spy_bench, color="#58a6ff", linestyle="--", linewidth=1.5, label=f"SPY Benchmark (+{spy_bench.iloc[-1]:.1f}%)")

    ax.axhline(0, color="#30363d", linewidth=1)
    ax.set_ylabel("Net Cumulative Return (%)")
    ax.set_title("Chart 7: Cumulative Portfolio Returns — Development Period (2020–2024)", fontsize=14, fontweight="bold", color="#ffd700")
    ax.legend(loc="upper left", framealpha=0.3)
    ax.grid(True, alpha=0.2)

fig.tight_layout()
save_fig(fig, "07_cumulative_returns.png")

# ---------------------------------------------------------------------------
# CHART 8: Drawdown Over Time
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

if xgb_dev_trades and xgb_dev_trades["total_trades"] > 0:
    tdf = pd.DataFrame(xgb_dev_trades["trades"])
    tdf["exit_date"] = pd.to_datetime(tdf["exit_date"])
    dd = np.array(xgb_dev_trades["drawdown_series"])

    ax.plot(tdf["exit_date"], dd, color="#f85149", linewidth=2)
    ax.fill_between(tdf["exit_date"], dd, 0, color="#f85149", alpha=0.35)
    ax.axhline(0, color="#30363d")
    ax.set_ylabel("Drawdown (%)")
    max_dd_val = np.min(dd)
    ax.set_title(f"Chart 8: Underwater Drawdown Profile — Development Period (Max Drawdown: {max_dd_val:.2f}%)", fontsize=14, fontweight="bold", color="#ffd700")
    ax.grid(True, alpha=0.2)

fig.tight_layout()
save_fig(fig, "08_drawdown_over_time.png")

# ---------------------------------------------------------------------------
# CHART 9: Monthly Trading Performance
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))

if xgb_dev_trades and xgb_dev_trades["total_trades"] > 0:
    tdf = pd.DataFrame(xgb_dev_trades["trades"])
    tdf["exit_month"] = pd.to_datetime(tdf["exit_date"]).dt.to_period("M")
    monthly = tdf.groupby("exit_month")["net_pnl_pct"].sum()

    m_colors = ["#3fb950" if v >= 0 else "#f85149" for v in monthly.values]
    ax.bar([str(m) for m in monthly.index], monthly.values, color=m_colors, edgecolor="#30363d", width=0.6)
    ax.axhline(0, color="#ffd700", linewidth=1)
    ax.set_ylabel("Net Monthly Return (%)")
    ax.set_title("Chart 9: Monthly Trading Performance — Net P&L % by Realized Month", fontsize=14, fontweight="bold", color="#ffd700")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
save_fig(fig, "09_monthly_performance.png")

# ---------------------------------------------------------------------------
# CHART 10: Individual Trade Outcomes
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))

if xgb_dev_trades and xgb_dev_trades["total_trades"] > 0:
    tdf = pd.DataFrame(xgb_dev_trades["trades"])
    colors_bars = ["#3fb950" if t["is_winner"] else "#f85149" for _, t in tdf.iterrows()]
    bars = ax.bar(range(len(tdf)), tdf["net_pnl_pct"], color=colors_bars, edgecolor="#30363d", alpha=0.85)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        hold = tdf["holding_days"].iloc[i]
        offset = 0.8 if h >= 0 else -1.8
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{hold}d", ha="center", fontsize=7, color="#8b949e")

    ax.axhline(0, color="#ffd700", linewidth=1)
    ax.set_xlabel("Sequential Trade Index (#)")
    ax.set_ylabel("Net Return Per Trade (%)")
    wr = xgb_dev_trades["win_rate"]
    pf = xgb_dev_trades["profit_factor"]
    ax.set_title(f"Chart 10: Individual Trade Outcomes (32 Trades, Win Rate: {wr}%, Profit Factor: {pf}, Holding Days Above Bars)", fontsize=13, fontweight="bold", color="#ffd700")
    ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
save_fig(fig, "10_individual_trade_outcomes.png")

# ---------------------------------------------------------------------------
# Save Enhanced Report JSON
# ---------------------------------------------------------------------------
enhanced_report = existing_rep.copy()
for pname in PERIODS:
    if pname in period_data and "xgb_trades" in period_data[pname]:
        t = period_data[pname]["xgb_trades"]
        clean_t = {k: v for k, v in t.items() if k not in ["trades", "cumulative_factors", "drawdown_series"]}
        clean_t["modeled_costs_description"] = "5 bps slippage per side (10 bps round trip) + $0.005/share institutional commission"
        enhanced_report["periods"][pname]["trading_performance_xgb_argmax"] = clean_t

with open(REPORTS_OUT / "v2_2_evaluation_report.json", "w") as f:
    json.dump(enhanced_report, f, indent=2, default=str)

print("[COMPLETE] All 10 charts and enhanced evaluation report generated successfully!")
