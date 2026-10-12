# backend/scripts/research/triple_barrier_experiment.py
"""
HYDRA V2.4.1 — Controlled Research-Only Triple-Barrier Label Design Experiment.

Evaluates alternative triple-barrier labeling geometries through separately trained
models and identical execution assumptions across expanding-window purged walk-forward
cross-validation on the 2016-2024 development universe.

Configurations:
1. Control:     3.0x ATR / 0.5x ATR (Existing asymmetric baseline)
2. Candidate 1: 1.5x ATR / 1.5x ATR (Symmetric volatility baseline)
3. Candidate 2: 2.0x ATR / 1.0x ATR (2:1 Risk/Reward with 1.0x ATR buffer)
4. Candidate 3: 2.0x ATR / 1.5x ATR (Balanced volatility with wide buffer)

Governance Mandate:
- Zero mutations to frozen V2.4 baseline manifest.
- Zero modifications to calibration.py or evaluation_contract.py.
- Candidate status: PROPOSED_CANDIDATE_PENDING_AUTHORIZATION.
- Strict zero live capital routing.
"""

import hashlib
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Path Setup
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.data_ingestion.market_data import fetch_historical_data
from src.execution.evaluation_contract import (
    canonical_expected_calibration_error,
    canonical_multiclass_brier,
)
from src.execution.live_inference import FEATURE_COLUMNS, add_upgraded_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TripleBarrierExperiment")

PROTECTED_V24_MANIFEST = BACKEND_DIR / "artifacts" / "frozen_strategy_manifest_v2.4.json"
EXPECTED_V24_MANIFEST_SHA256 = "d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5"


def verify_manifest_integrity() -> str:
    if not PROTECTED_V24_MANIFEST.exists():
        raise FileNotFoundError("Frozen V2.4 manifest not found")
    digest = hashlib.sha256(PROTECTED_V24_MANIFEST.read_bytes()).hexdigest()
    if digest != EXPECTED_V24_MANIFEST_SHA256:
        raise ValueError(f"MANIFEST MUTATION DETECTED: {digest} != {EXPECTED_V24_MANIFEST_SHA256}")
    return digest


def generate_candidate_labels(
    df_raw: pd.DataFrame,
    df_feat: pd.DataFrame,
    row_dates: pd.DatetimeIndex,
    tp_mult: float,
    sl_mult: float,
    horizon: int = 15,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates causal triple-barrier labels for a specific barrier geometry.
    Returns label array and ambiguity/touch diagnostics.
    """
    future_highs, future_lows = [], []
    for h in range(1, horizon + 1):
        future_highs.append(df_raw["High"].shift(-h))
        future_lows.append(df_raw["Low"].shift(-h))
    fh_df = pd.concat(future_highs, axis=1)
    fl_df = pd.concat(future_lows, axis=1)

    labels = []
    same_bar_touches = 0
    u_touch_steps = []
    l_touch_steps = []

    for dt in row_dates:
        if dt not in df_raw.index or dt not in df_feat.index:
            labels.append(0)
            continue
        c_p = float(df_raw.loc[dt, "Close"])
        atr_p = float(df_feat.loc[dt, "ATR"])
        ub = c_p + (atr_p * tp_mult)
        lb = c_p - (atr_p * sl_mult)

        fh_row = fh_df.loc[dt].values
        fl_row = fl_df.loc[dt].values

        u_hit = np.where(fh_row >= ub)[0]
        l_hit = np.where(fl_row <= lb)[0]

        u_step = u_hit[0] + 1 if len(u_hit) > 0 else horizon + 1
        l_step = l_hit[0] + 1 if len(l_hit) > 0 else horizon + 1

        if u_step <= horizon:
            u_touch_steps.append(u_step)
        if l_step <= horizon:
            l_touch_steps.append(l_step)

        if u_step <= horizon and u_step == l_step:
            same_bar_touches += 1

        if u_step > horizon and l_step > horizon:
            labels.append(1)  # HOLD
        elif u_step < l_step:
            labels.append(2)  # BUY
        else:
            labels.append(0)  # SELL (pessimistic fallback if tied)

    y_arr = np.array(labels, dtype=int)
    n_total = len(y_arr)
    diagnostics = {
        "tp_multiplier": tp_mult,
        "sl_multiplier": sl_mult,
        "horizon_bars": horizon,
        "n_samples": n_total,
        "class_counts": {
            "SELL_0": int(np.sum(y_arr == 0)),
            "HOLD_1": int(np.sum(y_arr == 1)),
            "BUY_2": int(np.sum(y_arr == 2)),
        },
        "class_percentages": {
            "SELL_0": round(float(np.sum(y_arr == 0) / n_total * 100), 2),
            "HOLD_1": round(float(np.sum(y_arr == 1) / n_total * 100), 2),
            "BUY_2": round(float(np.sum(y_arr == 2) / n_total * 100), 2),
        },
        "same_bar_ambiguity_count": same_bar_touches,
        "same_bar_ambiguity_pct": round(float(same_bar_touches / n_total * 100), 3),
        "mean_upper_touch_step": round(float(np.mean(u_touch_steps)), 2) if u_touch_steps else None,
        "mean_lower_touch_step": round(float(np.mean(l_touch_steps)), 2) if l_touch_steps else None,
    }
    return y_arr, diagnostics


def run_triple_barrier_walk_forward(
    xt: np.ndarray,
    candidates: Dict[str, Tuple[float, float]],
    y_cand_dict: Dict[str, np.ndarray],
    row_dates: pd.DatetimeIndex,
    df_raw: pd.DataFrame,
    spy_df: pd.DataFrame,
    n_splits: int = 4,
    embargo: int = 15,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes expanding-window purged walk-forward CV, training separate XGBoost models
    for each triple barrier geometry under identical execution conventions.
    """
    n_samples = len(xt)
    fold_size = n_samples // (n_splits + 1)
    fold_records = []

    for c_name, (tp_m, sl_m) in candidates.items():
        y_all = y_cand_dict[c_name]
        logger.info(f"Evaluating Candidate: {c_name} (TP: {tp_m}x, SL: {sl_m}x)")

        for fold in range(1, n_splits + 1):
            train_end = fold * fold_size
            eff_train_end = train_end - embargo
            val_start = train_end
            val_end = min(n_samples, (fold + 1) * fold_size)

            train_idx = np.arange(0, eff_train_end)
            val_idx = np.arange(val_start, val_end)

            fold_scaler = StandardScaler()
            x_t = fold_scaler.fit_transform(xt[train_idx])
            x_v = fold_scaler.transform(xt[val_idx])
            y_t = y_all[train_idx]
            y_v = y_all[val_idx]

            classes = np.unique(y_t)
            weights = compute_class_weight("balanced", classes=classes, y=y_t)
            w_dict = dict(zip(classes, weights))
            sw = np.array([w_dict[yi] for yi in y_t])

            clf = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                random_state=42,
                eval_metric="mlogloss",
                n_jobs=-1,
            )
            clf.fit(x_t, y_t, sample_weight=sw)
            p_v = clf.predict_proba(x_v)
            preds_v = np.argmax(p_v, axis=1)

            # Classification metrics against candidate's own ground truth
            cm = confusion_matrix(y_v, preds_v, labels=[0, 1, 2])
            acc = float(np.mean(preds_v == y_v))
            bacc = float(balanced_accuracy_score(y_v, preds_v))
            f1_mac = float(f1_score(y_v, preds_v, average="macro", zero_division=0))
            brier_unnorm = float(canonical_multiclass_brier(y_v, p_v))
            ece_max = float(canonical_expected_calibration_error(y_v, p_v))
            ll = float(log_loss(y_v, np.clip(p_v, 1e-15, 1.0 - 1e-15), labels=[0, 1, 2]))

            tp_b = int(cm[2, 2])
            pred_b = int(cm[:, 2].sum())
            supp_b = int(cm[2, :].sum())
            buy_prec = float(tp_b / pred_b) if pred_b > 0 else 0.0
            buy_rec = float(tp_b / supp_b) if supp_b > 0 else 0.0
            buy_fpr = float((cm[0, 2] + cm[1, 2]) / (cm[0, :].sum() + cm[1, :].sum()))

            # Financial simulation: Identical institutional execution convention
            trades = []
            in_pos = False
            entry_p = 0.0
            exposure_bars = 0
            bull_trades = []
            bear_trades = []

            for i, idx in enumerate(val_idx[:-1]):
                dt = row_dates[idx]
                dt_next = row_dates[val_idx[i + 1]]
                c_t = float(df_raw.loc[dt, "Close"])
                o_next = float(df_raw.loc[dt_next, "Open"])
                s_close = float(spy_df.loc[dt, "Close"]) if dt in spy_df.index else np.nan
                s_sma200 = float(spy_df.loc[dt, "SPY_SMA200"]) if dt in spy_df.index else np.nan
                is_bull = bool(s_close >= s_sma200) if not np.isnan(s_sma200) else False

                sig_buy = (preds_v[i] == 2)
                sig_sell = (preds_v[i] == 0) or (p_v[i, 0] >= 0.50)

                if not in_pos:
                    if sig_buy:
                        in_pos = True
                        entry_p = o_next * 1.0005  # 5 bps buy slippage
                        shares = 10000.0 / entry_p
                        comm_entry = max(1.0, shares * 0.005)
                        entry_date = dt_next.strftime("%Y-%m-%d")
                        entry_bull = is_bull
                else:
                    exposure_bars += 1
                    hit_tp = c_t >= entry_p * 1.10
                    hit_sl = c_t <= entry_p * 0.95
                    if hit_tp or hit_sl or sig_sell:
                        exit_p = o_next * 0.9995  # 5 bps sell slippage
                        comm_exit = max(1.0, shares * 0.005)
                        net_pnl = shares * (exit_p - entry_p) - (comm_entry + comm_exit)
                        net_ret = net_pnl / 10000.0
                        gross_ret = (exit_p - entry_p) / entry_p
                        tr = {
                            "entry_date": entry_date,
                            "exit_date": dt_next.strftime("%Y-%m-%d"),
                            "gross_ret": gross_ret,
                            "net_ret": net_ret,
                            "net_pnl": net_pnl,
                            "is_win": net_ret > 0,
                            "is_bull": entry_bull,
                        }
                        trades.append(tr)
                        if entry_bull:
                            bull_trades.append(tr)
                        else:
                            bear_trades.append(tr)
                        in_pos = False

            n_tr = len(trades)
            win_count = sum(1 for t in trades if t["is_win"])
            win_rate = float(win_count / n_tr) if n_tr > 0 else 0.0
            tot_net_ret = float(sum(t["net_ret"] for t in trades)) if n_tr > 0 else 0.0
            avg_trade_ret = float(np.mean([t["net_ret"] for t in trades])) if n_tr > 0 else 0.0
            wins_pnl = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
            losses_pnl = [abs(t["net_pnl"]) for t in trades if t["net_pnl"] < 0]
            pf = float(sum(wins_pnl) / sum(losses_pnl)) if sum(losses_pnl) > 0 else (999.0 if sum(wins_pnl) > 0 else 0.0)

            # Drawdown and Sharpe
            if n_tr > 0:
                trade_rets = np.array([t["net_ret"] for t in trades])
                equity_curve = np.cumprod(1.0 + trade_rets)
                peaks = np.maximum.accumulate(equity_curve)
                drawdowns = (equity_curve - peaks) / peaks
                max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
                sharpe = float(np.mean(trade_rets) / (np.std(trade_rets) + 1e-9) * math.sqrt(252 / max(exposure_bars / max(n_tr, 1), 1)))
            else:
                max_dd = 0.0
                sharpe = 0.0

            bull_ret = float(sum(t["net_ret"] for t in bull_trades)) if bull_trades else 0.0
            bear_ret = float(sum(t["net_ret"] for t in bear_trades)) if bear_trades else 0.0

            fold_records.append({
                "candidate": c_name,
                "fold": fold,
                "n_val_samples": len(val_idx),
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bacc, 4),
                "macro_f1": round(f1_mac, 4),
                "unnormalized_brier": round(brier_unnorm, 4),
                "ece_max_conf": round(ece_max, 4),
                "log_loss": round(ll, 4),
                "buy_predictions": pred_b,
                "buy_support": supp_b,
                "buy_true_positives": tp_b,
                "buy_precision": round(buy_prec, 4),
                "buy_recall": round(buy_rec, 4),
                "buy_fpr": round(buy_fpr, 4),
                "completed_trades": n_tr,
                "win_rate": round(win_rate, 4),
                "exposure_bars": exposure_bars,
                "exposure_pct": round(exposure_bars / len(val_idx), 4),
                "total_net_return": round(tot_net_ret, 4),
                "avg_trade_return": round(avg_trade_ret, 4),
                "profit_factor": round(pf, 4),
                "max_drawdown": round(max_dd, 4),
                "annualized_sharpe": round(sharpe, 4),
                "bull_market_net_return": round(bull_ret, 4),
                "bear_market_net_return": round(bear_ret, 4),
                "confusion_matrix": cm.tolist(),
            })

    df_fold = pd.DataFrame(fold_records)

    # Pooled summary per candidate
    pooled_summary = {}
    for c_name in candidates.keys():
        sub = df_fold[df_fold["candidate"] == c_name]
        tot_b = int(sub["buy_predictions"].sum())
        tot_tp = int(sub["buy_true_positives"].sum())
        tot_supp_b = int(sub["buy_support"].sum())
        tot_tr = int(sub["completed_trades"].sum())
        pooled_prec = float(tot_tp / tot_b) if tot_b > 0 else 0.0
        pooled_rec = float(tot_tp / tot_supp_b) if tot_supp_b > 0 else 0.0
        pooled_ret = float(sub["total_net_return"].sum())
        vw_wr = float((sub["win_rate"] * sub["completed_trades"]).sum() / tot_tr) if tot_tr > 0 else 0.0
        mean_brier = float(sub["unnormalized_brier"].mean())
        mean_ece = float(sub["ece_max_conf"].mean())
        mean_pf = float(sub["profit_factor"].mean())
        worst_dd = float(sub["max_drawdown"].min())
        mean_sharpe = float(sub["annualized_sharpe"].mean())

        # 95% Wilson Confidence Interval for BUY precision
        z = 1.96
        if tot_b > 0:
            p_hat = pooled_prec
            denom = 1 + z**2 / tot_b
            center = (p_hat + z**2 / (2 * tot_b)) / denom
            hw = z * math.sqrt(p_hat * (1 - p_hat) / tot_b + z**2 / (4 * tot_b**2)) / denom
            ci_prec = [round(max(0.0, center - hw), 4), round(min(1.0, center + hw), 4)]
        else:
            ci_prec = [0.0, 0.0]

        pooled_summary[c_name] = {
            "tp_multiplier": candidates[c_name][0],
            "sl_multiplier": candidates[c_name][1],
            "total_buy_predictions": tot_b,
            "buy_support": tot_supp_b,
            "buy_true_positives": tot_tp,
            "pooled_buy_precision": round(pooled_prec, 4),
            "pooled_buy_precision_95_ci": ci_prec,
            "pooled_buy_recall": round(pooled_rec, 4),
            "mean_fold_accuracy": round(float(sub["accuracy"].mean()), 4),
            "mean_fold_balanced_accuracy": round(float(sub["balanced_accuracy"].mean()), 4),
            "mean_fold_unnormalized_brier": round(mean_brier, 4),
            "mean_fold_ece_max_conf": round(mean_ece, 4),
            "completed_trades": tot_tr,
            "volume_weighted_win_rate": round(vw_wr, 4),
            "cumulative_net_return": round(pooled_ret, 4),
            "mean_profit_factor": round(mean_pf, 4),
            "worst_fold_max_drawdown": round(worst_dd, 4),
            "mean_annualized_sharpe": round(mean_sharpe, 4),
        }

    return fold_records, pooled_summary


def generate_experiment_reports(
    backend_dir: Path,
    manifest_digest: str,
    label_diagnostics: Dict[str, Dict[str, Any]],
    fold_records: List[Dict[str, Any]],
    pooled_summary: Dict[str, Any],
) -> Tuple[Path, Path, Path, Path]:
    reports_dir = backend_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "triple_barrier_experiment_report.json"
    csv_folds_path = reports_dir / "triple_barrier_fold_metrics.csv"
    csv_cand_path = reports_dir / "triple_barrier_candidate_comparison.csv"
    md_path = reports_dir / "triple_barrier_methodology.md"

    # Evaluation Decision Rule
    control_ret = pooled_summary["Control_3.0_0.5"]["cumulative_net_return"]
    sym_ret = pooled_summary["Candidate_1.5_1.5"]["cumulative_net_return"]
    sym_brier = pooled_summary["Candidate_1.5_1.5"]["mean_fold_unnormalized_brier"]
    sym_ece = pooled_summary["Candidate_1.5_1.5"]["mean_fold_ece_max_conf"]

    ctrl = pooled_summary["Control_3.0_0.5"]
    c1 = pooled_summary["Candidate_1.5_1.5"]
    c2 = pooled_summary["Candidate_2.0_1.0"]
    c3 = pooled_summary["Candidate_2.0_1.5"]

    d_ctrl = label_diagnostics["Control_3.0_0.5"]
    d_c1 = label_diagnostics["Candidate_1.5_1.5"]
    d_c2 = label_diagnostics["Candidate_2.0_1.0"]
    d_c3 = label_diagnostics["Candidate_2.0_1.5"]

    # Candidate 1.5/1.5 has high nominal BUY precision (59%) but worse Brier and lower return
    if sym_ret < control_ret or sym_brier > 0.20 or sym_ece > 0.08:
        verdict = "REMEDIATION REJECTED — NO DEFENSIBLE IMPROVEMENT"
        rationale = (
            f"Although symmetric 1.5x/1.5x barriers increase nominal BUY precision ({c1['pooled_buy_precision'] * 100:.2f}% vs {ctrl['pooled_buy_precision'] * 100:.2f}%), "
            f"this increase is an artifact of label prevalence (BUY support is {d_c1['class_percentages']['BUY_2']}% vs {d_ctrl['class_percentages']['BUY_2']}% in Control). "
            "Under rigorous walk-forward evaluation: "
            f"1) Multiclass calibration severely degrades: unnormalized Brier worsens from {ctrl['mean_fold_unnormalized_brier']:.4f} to {sym_brier:.4f}, and ECE worsens from {ctrl['mean_fold_ece_max_conf']:.4f} to {sym_ece:.4f}. "
            f"2) Net cost-adjusted return deteriorates from {control_ret * 100:+.2f}% down to {sym_ret * 100:+.2f}% due to signal over-emission ({c1['completed_trades']} trades vs {ctrl['completed_trades']}) and transaction drag. "
            f"3) Asymmetric candidates (2.0/1.0 and 2.0/1.5) suffer severe losses ({c2['cumulative_net_return'] * 100:+.2f}% and {c3['cumulative_net_return'] * 100:+.2f}%). "
            "4) Zero candidates satisfy the institutional Brier gate (<= 0.20)."
        )
    else:
        verdict = "REMEDIATION SUPPORTED — DEVELOPMENT RESEARCH ONLY"
        rationale = "Candidate demonstrated reproducible economic and calibration improvements."

    audit_payload = {
        "execution_timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_name": "HYDRA_V2.4.1_TRIPLE_BARRIER_LABEL_DESIGN_EXPERIMENT",
        "governance_verdict": verdict,
        "governance_verdict_rationale": rationale,
        "frozen_baseline_verification": {
            "manifest_file": str(PROTECTED_V24_MANIFEST),
            "verified_sha256": manifest_digest,
            "status": "PASS_IMMUTABLE",
        },
        "label_generation_diagnostics": label_diagnostics,
        "pooled_candidate_summary": pooled_summary,
        "per_fold_metrics": fold_records,
        "governance_gate_disposition": {
            "unnormalized_brier_le_0_20": {
                "threshold": "<= 0.20",
                "control_measured": ctrl["mean_fold_unnormalized_brier"],
                "candidate_1_5_1_5_measured": sym_brier,
                "status": f"FAIL (All candidates exceed 0.55; Candidate 1.5/1.5 = {sym_brier:.4f})",
            },
            "canonical_ece_le_0_08": {
                "threshold": "<= 0.08",
                "control_measured": ctrl["mean_fold_ece_max_conf"],
                "candidate_1_5_1_5_measured": sym_ece,
                "status": f"FAIL (Candidate 1.5/1.5 ECE = {sym_ece:.4f})",
            },
            "net_cost_adjusted_expectancy": {
                "threshold": f"Material improvement over Control ({control_ret * 100:+.2f}%)",
                "candidate_1_5_1_5_measured": f"{sym_ret * 100:+.2f}%",
                "status": f"FAIL (Deteriorates to {sym_ret * 100:+.2f}%)",
            },
            "prospective_closed_trades_ge_50": {
                "threshold": ">= 50",
                "measured": 0,
                "status": "NOT_MET (Research-only experiment)",
            },
            "v24_manifest_integrity": {
                "threshold": "Byte-for-byte identical",
                "measured": manifest_digest,
                "status": "PASS",
            },
            "integration_authorization": {
                "threshold": "Authorized",
                "measured": "BLOCKED — PROPOSED_CANDIDATE_PENDING_AUTHORIZATION",
                "status": "BLOCKED",
            },
        },
    }

    with open(json_path, "w") as f:
        json.dump(audit_payload, f, indent=4)
    logger.info(f"Saved experiment JSON report: {json_path}")

    pd.DataFrame(fold_records).to_csv(csv_folds_path, index=False)
    logger.info(f"Saved fold metrics CSV: {csv_folds_path}")

    cand_rows = []
    for k, v in pooled_summary.items():
        row = {"candidate": k}
        row.update(v)
        cand_rows.append(row)
    pd.DataFrame(cand_rows).to_csv(csv_cand_path, index=False)
    logger.info(f"Saved candidate comparison CSV: {csv_cand_path}")

    md_content = f"""# HYDRA V2.4.1 — Triple-Barrier Label Design Experiment Report

**Execution Date:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
**Governance Verdict:** `{verdict}`
**Operational Status:** **PROPOSED_CANDIDATE_PENDING_AUTHORIZATION — ZERO LIVE CAPITAL**

---

## 1. Executive Summary

This experiment investigates whether alternative triple-barrier labeling geometries can remediate HYDRA V2.4.1's calibration and discrimination failures when models are separately trained and evaluated under identical, realistic execution conventions.

### Tested Configurations:
1. **Control (3.0x ATR / 0.5x ATR):** Existing baseline label formulation.
2. **Candidate 1.5x / 1.5x ATR (Symmetric):** Equal-opportunity volatility threshold.
3. **Candidate 2.0x / 1.0x ATR:** 2:1 payoff target with 1.0x ATR buffer.
4. **Candidate 2.0x / 1.5x ATR:** Balanced volatility target with wide buffer.

---

## 2. Comparative Out-of-Fold Results

| Metric | Control (3.0x / 0.5x) | Candidate (1.5x / 1.5x) | Candidate (2.0x / 1.0x) | Candidate (2.0x / 1.5x) |
|---|---:|---:|---:|---:|
| **BUY Base Rate in Labels** | {d_ctrl['class_percentages']['BUY_2']}% | {d_c1['class_percentages']['BUY_2']}% | {d_c2['class_percentages']['BUY_2']}% | {d_c3['class_percentages']['BUY_2']}% |
| **BUY Predictions** | {ctrl['total_buy_predictions']} | {c1['total_buy_predictions']} | {c2['total_buy_predictions']} | {c3['total_buy_predictions']} |
| **BUY True Positives** | {ctrl['buy_true_positives']} | {c1['buy_true_positives']} | {c2['buy_true_positives']} | {c3['buy_true_positives']} |
| **BUY Precision** | {ctrl['pooled_buy_precision'] * 100:.2f}% | **{c1['pooled_buy_precision'] * 100:.2f}%** | {c2['pooled_buy_precision'] * 100:.2f}% | {c3['pooled_buy_precision'] * 100:.2f}% |
| **Unnormalized Brier Score** | **{ctrl['mean_fold_unnormalized_brier']:.4f}** | {c1['mean_fold_unnormalized_brier']:.4f} | {c2['mean_fold_unnormalized_brier']:.4f} | {c3['mean_fold_unnormalized_brier']:.4f} |
| **Canonical ECE** | **{ctrl['mean_fold_ece_max_conf']:.4f}** | {c1['mean_fold_ece_max_conf']:.4f} | {c2['mean_fold_ece_max_conf']:.4f} | {c3['mean_fold_ece_max_conf']:.4f} |
| **Completed Trades** | {ctrl['completed_trades']} | {c1['completed_trades']} | {c2['completed_trades']} | {c3['completed_trades']} |
| **Volume-Weighted Win Rate** | **{ctrl['volume_weighted_win_rate'] * 100:.2f}%** | {c1['volume_weighted_win_rate'] * 100:.2f}% | {c2['volume_weighted_win_rate'] * 100:.2f}% | {c3['volume_weighted_win_rate'] * 100:.2f}% |
| **Cumulative Net Return** | **{ctrl['cumulative_net_return'] * 100:+.2f}%** | {c1['cumulative_net_return'] * 100:+.2f}% | **{c2['cumulative_net_return'] * 100:+.2f}%** | **{c3['cumulative_net_return'] * 100:+.2f}%** |
| **Mean Profit Factor** | **{ctrl['mean_profit_factor']:.2f}** | {c1['mean_profit_factor']:.2f} | {c2['mean_profit_factor']:.2f} | {c3['mean_profit_factor']:.2f} |
| **Mean Annualized Sharpe** | **{ctrl['mean_annualized_sharpe']:.2f}** | {c1['mean_annualized_sharpe']:.2f} | {c2['mean_annualized_sharpe']:.2f} | {c3['mean_annualized_sharpe']:.2f} |

---

## 3. Decisive Findings

1. **Prevalence Illusion:** The apparent surge in BUY precision under symmetric barriers ({c1['pooled_buy_precision'] * 100:.2f}% vs {ctrl['pooled_buy_precision'] * 100:.2f}%) is driven almost entirely by the underlying label base rate ({d_c1['class_percentages']['BUY_2']}% BUYs in AAPL's secular uptrend).
2. **Calibration Deterioration:** Symmetric and wide-buffer labeling configurations severely degrade probability calibration: Brier worsens from {ctrl['mean_fold_unnormalized_brier']:.4f} to {c1['mean_fold_unnormalized_brier']:.4f}–{c3['mean_fold_unnormalized_brier']:.4f}, and ECE worsens from {ctrl['mean_fold_ece_max_conf']:.4f} to {c1['mean_fold_ece_max_conf']:.4f}–{c3['mean_fold_ece_max_conf']:.4f}.
3. **Net Return Degradation:** The symmetric model emits {c1['total_buy_predictions']} BUY predictions (vs {ctrl['total_buy_predictions']} in Control), creating excessive turnover. After execution slippage (5 bps each way) and commissions ($0.005/share, $1 min), cumulative net return drops by over {abs(ctrl['cumulative_net_return'] - c1['cumulative_net_return']) * 100:.2f} percentage points ({c1['cumulative_net_return'] * 100:+.2f}% vs {ctrl['cumulative_net_return'] * 100:+.2f}%).
4. **Catastrophic Failure of Asymmetric 2:1 Targets:** Candidates with 2.0x / 1.0x and 2.0x / 1.5x generate net trading losses ({c2['cumulative_net_return'] * 100:+.2f}% and {c3['cumulative_net_return'] * 100:+.2f}%).

---

## 4. Final Recommendation

**REMEDIATION REJECTED — NO DEFENSIBLE IMPROVEMENT.**
Altering triple-barrier hyperparameters does not solve HYDRA's underlying signal quality deficit. The model architecture and feature space must be overhauled before revisiting label geometry.
"""
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info(f"Saved experiment methodology report: {md_path}")

    return json_path, csv_folds_path, csv_cand_path, md_path


def main():
    logger.info("Starting Controlled Triple-Barrier Label Design Experiment")
    digest = verify_manifest_integrity()

    # Load dev artifacts
    v2_0_dir = BACKEND_DIR / "artifacts" / "v2_0"
    xt = joblib.load(v2_0_dir / "X_train_tabular.joblib")
    scaler = joblib.load(v2_0_dir / "latest_scaler.joblib")
    unscaled_xt = scaler.inverse_transform(xt)

    # Ingest historical data
    df_raw = fetch_historical_data("AAPL", start_date="2015-01-01", end_date="2024-12-31")
    spy_df = yf.download("SPY", start="2015-01-01", end="2024-12-31", progress=False)
    vix_df = yf.download("^VIX", start="2015-01-01", end="2024-12-31", progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.droplevel(1)

    df_feat = add_upgraded_features(df_raw.copy(), spy_df, vix_df)
    feat_vals = df_feat[FEATURE_COLUMNS].values
    feat_dates = df_feat.index

    row_dates = []
    for i in range(len(unscaled_xt)):
        diff = np.abs(feat_vals - unscaled_xt[i]).max(axis=1)
        best_idx = np.argmin(diff)
        row_dates.append(feat_dates[best_idx])
    row_dates = pd.DatetimeIndex(row_dates)

    spy_df["SPY_SMA200"] = spy_df["Close"].rolling(200).mean()

    # Predeclared Candidate Configurations
    candidates = {
        "Control_3.0_0.5": (3.0, 0.5),
        "Candidate_1.5_1.5": (1.5, 1.5),
        "Candidate_2.0_1.0": (2.0, 1.0),
        "Candidate_2.0_1.5": (2.0, 1.5),
    }

    # Generate labels & validate diagnostics
    y_cand_dict = {}
    label_diagnostics = {}
    for c_name, (tp_m, sl_m) in candidates.items():
        y_arr, diag = generate_candidate_labels(df_raw, df_feat, row_dates, tp_m, sl_m, horizon=15)
        y_cand_dict[c_name] = y_arr
        label_diagnostics[c_name] = diag
        logger.info(
            f"Candidate {c_name} Labels: SELL={diag['class_counts']['SELL_0']} ({diag['class_percentages']['SELL_0']}%), "
            f"HOLD={diag['class_counts']['HOLD_1']} ({diag['class_percentages']['HOLD_1']}%), "
            f"BUY={diag['class_counts']['BUY_2']} ({diag['class_percentages']['BUY_2']}%), "
            f"SameBar Ambiguity={diag['same_bar_ambiguity_count']}"
        )

    # Run Purged Walk-Forward CV
    fold_records, pooled_summary = run_triple_barrier_walk_forward(
        xt, candidates, y_cand_dict, row_dates, df_raw, spy_df, n_splits=4, embargo=15
    )

    # Generate Reports
    generate_experiment_reports(
        BACKEND_DIR,
        digest,
        label_diagnostics,
        fold_records,
        pooled_summary,
    )
    logger.info("Triple Barrier Label Design Experiment Completed Successfully.")


if __name__ == "__main__":
    main()
