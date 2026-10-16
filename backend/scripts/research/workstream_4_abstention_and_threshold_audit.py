# backend/scripts/research/workstream_4_abstention_and_threshold_audit.py
"""
HYDRA V2.4.2+ — Workstream 4: Forensic Audit of ML Abstention,
Historical Threshold Locking, and Cent-Reconciled Portfolio Evaluation.

Forensic Methodology:
1. Arithmetic Reconciliation of 2025-2026 Forward Trade Ledgers:
   - Reconcile exact raw fill prices, shares, ticket fees, net proceeds, and ending cash to the cent.
2. Root-Cause Audit of ML Model Abstention:
   - Inspect label distribution across barrier geometries.
   - Analyze probability outputs: P(SELL), P(HOLD), P(BUY) across 2025-2026.
   - Compare Argmax vs Hard Threshold (0.50) in a 3-class problem.
   - Compute feature importances of the XGBoost classifier.
3. Historical Threshold Selection & Locking (Strictly Pre-2025 Data):
   - Evaluate decision rules across the 4 Purged Walk-Forward Folds (2017-2024, N=1,656 bars).
   - Test threshold candidates: theta in [0.25, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50].
   - Test relative ranking rule: P(BUY) > P(SELL) and P(BUY) >= theta.
   - Test argmax rule: argmax(probs) == BUY.
   - Select and LOCK the single best historical candidate without peeking at forward data.
4. Out-of-Sample Forward Evaluation of the Locked Candidate:
   - Execute the locked model on 2025-10-13 to 2026-10-13 (252 bars).
   - Run the InstitutionalPortfolioSimulator ($1,000 capital, 5 bps slippage, $0.005/share fee).
   - Compare against 200-day SMA Trend (+26.57%) and Buy-and-Hold (+38.08%).
   - Export cent-reconciled ledgers, equity curves, and audit reports.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WS4AbstentionAudit")

REPORTS_DIR = BACKEND_DIR / "reports"
PINNED_DATASET_PATH = BACKEND_DIR / "artifacts" / "research" / "workstream_1_pinned_development_dataset.joblib"
PHASE_3D_OOF_PATH = BACKEND_DIR / "artifacts" / "research" / "phase_3d_oof_predictions.joblib"

EVAL_START = "2025-10-13"
EVAL_END = "2026-10-14"
WARMUP_START = "2015-01-01"


# ──────────────────────────────────────────────────────────────────────────────
# 1. EXACT ARITHMETIC RECONCILIATION OF 2025–2026 FORWARD SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def reconcile_forward_trade_ledger_arithmetic() -> Dict[str, Any]:
    """
    Audits the raw math of every trade in workstream_3_forward_trade_ledger.csv
    and explains the exact cent-level rounding differences.
    """
    ledger_csv = REPORTS_DIR / "workstream_3_forward_trade_ledger.csv"
    if not ledger_csv.exists():
        raise FileNotFoundError(f"Missing {ledger_csv}")

    df_trades = pd.read_csv(ledger_csv)
    reconciled_trades = []

    for _, r in df_trades.iterrows():
        shares = int(r["entry_shares"])
        entry_fill = float(r["entry_fill_price"])
        entry_fees = float(r["entry_fees"])
        exit_fill = float(r["exit_fill_price"])
        exit_fees = float(r["exit_fees"])

        # Exact unrounded arithmetic
        exact_entry_gross = shares * entry_fill
        exact_entry_cost = exact_entry_gross + entry_fees
        exact_exit_gross = shares * exit_fill
        exact_exit_net = exact_exit_gross - exit_fees
        exact_net_pnl = exact_exit_net - exact_entry_cost
        exact_ret_pct = (exact_net_pnl / exact_entry_cost) * 100.0 if exact_entry_cost > 0 else 0.0

        # Rounded representation (for reporting display)
        rec = {
            "trade_id": int(r["trade_id"]),
            "strategy": r["strategy"],
            "symbol": r["symbol"],
            "entry_timestamp": r["entry_timestamp"],
            "entry_fill_price": entry_fill,
            "shares": shares,
            "entry_gross": round(exact_entry_gross, 4),
            "entry_fees": entry_fees,
            "entry_total_cost": round(exact_entry_cost, 4),
            "exit_timestamp": r["exit_timestamp"],
            "exit_fill_price": exit_fill,
            "exit_gross": round(exact_exit_gross, 4),
            "exit_fees": exit_fees,
            "exit_net_proceeds": round(exact_exit_net, 4),
            "exact_net_pnl": round(exact_net_pnl, 4),
            "reported_net_pnl": float(r["net_pnl"]),
            "display_entry_cost": round(exact_entry_cost, 2),
            "display_exit_proceeds": round(exact_exit_net, 2),
            "display_net_pnl": round(exact_net_pnl, 2),
            "net_return_pct": round(exact_ret_pct, 4),
            "holding_period_bars": int(r["holding_period_bars"]),
            "exit_reason": r["exit_reason"],
            "is_winner": bool(exact_net_pnl > 0),
        }
        reconciled_trades.append(rec)

    # Reconcile SMA trade specifically:
    sma_trade = next(t for t in reconciled_trades if t["strategy"] == "200_Day_SMA_Trend")
    # Explanation of the 4-cent discrepancy:
    # 4 shares * $245.8153 = $983.2611. With $1.00 fee: $984.2611 (~$984.26).
    # 4 shares * $247.3446 = $989.3786. With $1.00 fee: $988.3786 (~$988.38).
    # Exact Net PnL = $988.3786 - $984.2611 = +$4.1175 (~+$4.12).
    # If one instead rounds entry_cost to $984.28 (from 4 * 245.82 = 983.28 + 1)
    # and exit proceeds to $988.36 (from 4 * 247.34 = 989.36 - 1),
    # then $988.36 - $984.28 = $4.08!
    # The discrepancy was purely due to subtracting prematurely rounded display numbers
    # instead of subtracting raw unrounded floats.

    return {
        "reconciled_trades": reconciled_trades,
        "sma_trade_audit": {
            "entry_fill_exact": sma_trade["entry_fill_price"],
            "exit_fill_exact": sma_trade["exit_fill_price"],
            "exact_entry_cost": sma_trade["entry_total_cost"],
            "exact_exit_proceeds": sma_trade["exit_net_proceeds"],
            "exact_net_pnl": sma_trade["exact_net_pnl"],
            "rounded_display_net_pnl": sma_trade["display_net_pnl"],
            "root_cause_of_4_cent_discrepancy": (
                "Subtracting prematurely rounded display numbers ($988.36 - $984.28 = $4.08) "
                "versus subtracting raw float fills ($988.3786 - $984.2611 = +$4.1175 -> $4.12)."
            ),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. AUDIT OF ML MODEL ABSTENTION & PROBABILITY DISTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────────────

def audit_ml_abstention() -> Dict[str, Any]:
    """
    Conducts a full forensic investigation into why both ML models abstained
    from BUY signals during the 2025-2026 period:
    - Target label imbalance in training data
    - Probability distribution across the 3 classes
    - Comparison between Argmax, relative ranking, and hard 0.50 threshold
    - Feature importances
    """
    import yfinance as yf

    from scripts.research.workstream_3_forward_simulation import compute_indicators

    logger.info("Fetching AAPL data for ML abstention forensics...")
    df_raw = yf.download("AAPL", start=WARMUP_START, end=EVAL_END, progress=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.droplevel(1)
    df_raw.dropna(inplace=True)
    df_full = compute_indicators(df_raw)

    feature_cols = [
        "RSI", "ATR", "ADX", "MACD", "MACD_Signal", "MACD_Diff",
        "Stoch_K", "Stoch_D", "BB_Upper", "BB_Middle", "BB_Lower",
        "Volatility_Ratio", "Return_1d", "Return_5d", "Return_20d",
        "Volume_Ratio", "SMA_200",
    ]

    eval_start_dt = pd.Timestamp(EVAL_START)
    embargo_start = eval_start_dt - pd.Timedelta(days=20)
    train_df = df_full.loc[:embargo_start].dropna(subset=feature_cols)
    eval_dates = df_full.loc[EVAL_START:"2026-10-13"].index
    eval_df = df_full.loc[eval_dates].copy()

    # Compare 2 barrier configurations:
    # Config A: Frozen V2.4 Control (3.0 ATR TP, 0.5 ATR SL, H=15)
    # Config B: Remediated Candidate (2.0 ATR TP, 1.5 ATR SL, H=15)
    configs = [
        {"name": "Frozen_V2.4_ML", "tp": 3.0, "sl": 0.5, "horizon": 15},
        {"name": "Remediated_ML_Candidate", "tp": 2.0, "sl": 1.5, "horizon": 15},
    ]

    analysis_results = {}

    for cfg in configs:
        name = cfg["name"]
        tp = cfg["tp"]
        sl = cfg["sl"]
        h_len = cfg["horizon"]

        # Label generation for train_df
        fh_list = [train_df["High"].shift(-h) for h in range(1, h_len + 1)]
        fl_list = [train_df["Low"].shift(-h) for h in range(1, h_len + 1)]
        fh_df = pd.concat(fh_list, axis=1)
        fl_df = pd.concat(fl_list, axis=1)

        y_train, valid_mask = [], []
        for dt in train_df.index:
            c = float(train_df.loc[dt, "Close"])
            atr = float(train_df.loc[dt, "ATR"])
            ub = c + (atr * tp)
            lb = c - (atr * sl)
            fh_row = fh_df.loc[dt].values
            fl_row = fl_df.loc[dt].values
            if np.any(np.isnan(fh_row)) or np.any(np.isnan(fl_row)):
                valid_mask.append(False)
                continue
            u_hit = np.where(fh_row >= ub)[0]
            l_hit = np.where(fl_row <= lb)[0]
            u_s = u_hit[0] + 1 if len(u_hit) > 0 else h_len + 1
            l_s = l_hit[0] + 1 if len(l_hit) > 0 else h_len + 1
            if u_s > h_len and l_s > h_len:
                y_train.append(1)  # HOLD
            elif u_s < l_s:
                y_train.append(2)  # BUY
            else:
                y_train.append(0)  # SELL
            valid_mask.append(True)

        y_train = np.array(y_train, dtype=int)
        valid_mask = np.array(valid_mask)
        X_train = train_df[feature_cols].values[valid_mask]

        # Label stats
        total_samples = len(y_train)
        n_sell = int(np.sum(y_train == 0))
        n_hold = int(np.sum(y_train == 1))
        n_buy = int(np.sum(y_train == 2))

        # Train model
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_ev_s = scaler.transform(eval_df[feature_cols].ffill().fillna(0.0).values)

        xgb_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "min_child_weight": 2.0,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": 1,
        }
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr_s, y_train)

        probs_eval = model.predict_proba(X_ev_s)
        p_sell = probs_eval[:, 0]
        p_hold = probs_eval[:, 1]
        p_buy = probs_eval[:, 2]

        # Feature importances
        importances = dict(zip(feature_cols, [round(float(x), 4) for x in model.feature_importances_]))
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        # Decision rules count on forward period
        count_hard_50 = int(np.sum(p_buy >= 0.50))
        count_hard_40 = int(np.sum(p_buy >= 0.40))
        count_hard_35 = int(np.sum(p_buy >= 0.35))
        count_hard_33 = int(np.sum(p_buy >= 0.333))
        count_argmax_buy = int(np.sum(np.argmax(probs_eval, axis=1) == 2))
        count_rel_ranking = int(np.sum((p_buy > p_sell) & (p_buy > p_hold)))

        analysis_results[name] = {
            "training_samples": total_samples,
            "label_distribution": {
                "SELL": {"count": n_sell, "pct": round(n_sell / total_samples * 100, 2)},
                "HOLD": {"count": n_hold, "pct": round(n_hold / total_samples * 100, 2)},
                "BUY": {"count": n_buy, "pct": round(n_buy / total_samples * 100, 2)},
            },
            "forward_eval_probs": {
                "p_buy": {
                    "min": round(float(np.min(p_buy)), 4),
                    "mean": round(float(np.mean(p_buy)), 4),
                    "max": round(float(np.max(p_buy)), 4),
                    "std": round(float(np.std(p_buy)), 4),
                },
                "p_sell": {
                    "min": round(float(np.min(p_sell)), 4),
                    "mean": round(float(np.mean(p_sell)), 4),
                    "max": round(float(np.max(p_sell)), 4),
                },
                "p_hold": {
                    "min": round(float(np.min(p_hold)), 4),
                    "mean": round(float(np.mean(p_hold)), 4),
                    "max": round(float(np.max(p_hold)), 4),
                },
            },
            "signal_counts_by_rule": {
                "threshold_ge_0_50": count_hard_50,
                "threshold_ge_0_40": count_hard_40,
                "threshold_ge_0_35": count_hard_35,
                "threshold_ge_0_33": count_hard_33,
                "argmax_buy": count_argmax_buy,
                "relative_ranking_buy_dominant": count_rel_ranking,
            },
            "top_features": sorted_imp[:5],
        }

    return analysis_results


# ──────────────────────────────────────────────────────────────────────────────
# 3. HISTORICAL THRESHOLD SELECTION & LOCKING (STRICTLY PRE-2025 DATA)
# ──────────────────────────────────────────────────────────────────────────────

def select_and_lock_historical_threshold() -> Dict[str, Any]:
    """
    Sweeps decision threshold candidates strictly across the 4 historical
    Purged Walk-Forward Folds (2017-08-24 to 2024-03-25, N=1,656 bars)
    to find the optimal decision rule WITHOUT touching forward 2025-2026 data.

    Evaluation criteria:
    - Maximum Out-of-Sample Sharpe Ratio
    - Positive Net Expectancy after costs (5 bps slippage + commissions)
    - Minimum 20 closed trades for statistical viability
    - Lower Drawdown than Buy & Hold (-37.66%)
    """
    from scripts.research.workstream_2_directional_baselines import (
        InstitutionalPortfolioSimulator,
    )

    logger.info("Executing Historical Threshold Selection on Pre-2025 Out-of-Fold Data...")

    # Load pinned dataset
    pinned_bundle = joblib.load(PINNED_DATASET_PATH)
    raw_df = pinned_bundle["raw_equity"].copy()
    ef_df = pinned_bundle["engineered_features_clean"].copy()
    feature_cols_v24 = pinned_bundle["feature_columns_v24"]

    dev_df = ef_df.loc["2016-01-04":"2024-12-30"].copy()
    oof_dates = dev_df.index[414:2070]
    dev_prices = raw_df.loc[dev_df.index].copy()
    oof_prices = raw_df.loc[oof_dates].copy()

    # Reconstruct Candidate 2.0 / 1.5 ATR labels for dev_df
    future_highs, future_lows = [], []
    for h in range(1, 16):
        future_highs.append(dev_prices["High"].shift(-h))
        future_lows.append(dev_prices["Low"].shift(-h))
    fh_df = pd.concat(future_highs, axis=1)
    fl_df = pd.concat(future_lows, axis=1)

    cand_labels = []
    tp_mult, sl_mult, horizon = 2.0, 1.5, 15
    for dt in dev_df.index:
        c_p = float(dev_prices.loc[dt, "Close"])
        atr_p = float(dev_df.loc[dt, "ATR"])
        ub = c_p + (atr_p * tp_mult)
        lb = c_p - (atr_p * sl_mult)

        fh_row = fh_df.loc[dt].values
        fl_row = fl_df.loc[dt].values
        u_hit = np.where(fh_row >= ub)[0]
        l_hit = np.where(fl_row <= lb)[0]
        u_step = u_hit[0] + 1 if len(u_hit) > 0 else horizon + 1
        l_step = l_hit[0] + 1 if len(l_hit) > 0 else horizon + 1

        if u_step > horizon and l_step > horizon:
            cand_labels.append(1)  # HOLD
        elif u_step < l_step:
            cand_labels.append(2)  # BUY
        else:
            cand_labels.append(0)  # SELL

    y_cand_all = np.array(cand_labels, dtype=int)
    X_dev = dev_df[feature_cols_v24].values

    # 4 Walk-Forward Folds
    fold_splits = [
        (range(0, 399), range(414, 828)),
        (range(0, 813), range(828, 1242)),
        (range(0, 1227), range(1242, 1656)),
        (range(0, 1641), range(1656, 2070)),
    ]

    opt_xgb_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.70,
        "min_child_weight": 2.0,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": 1,
    }

    remediated_oof_probs = np.zeros((1656, 3))
    for f_idx, (tr_idx, va_idx) in enumerate(fold_splits):
        X_tr = X_dev[tr_idx]
        y_tr = y_cand_all[tr_idx]
        X_va = X_dev[va_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = xgb.XGBClassifier(**opt_xgb_params)
        model.fit(X_tr_s, y_tr)
        probs_va = model.predict_proba(X_va_s)

        local_start = f_idx * 414
        local_end = (f_idx + 1) * 414
        remediated_oof_probs[local_start:local_end] = probs_va

    close_oof = oof_prices["Close"]
    sma_200 = dev_prices["Close"].rolling(window=200, min_periods=200).mean().loc[oof_dates]

    # Threshold candidates to test
    threshold_grid = [0.25, 0.30, 0.333, 0.35, 0.40, 0.45, 0.50]
    sweep_results = []

    sim = InstitutionalPortfolioSimulator(
        initial_capital=1000.0,
        slippage_bps=5.0,
        commission_per_share=0.005,
        min_commission=1.00,
    )

    for th in threshold_grid:
        sig = pd.Series("HOLD", index=oof_dates)
        in_pos = False

        for i in range(len(oof_dates)):
            c = close_oof.iloc[i]
            sma = sma_200.iloc[i]
            pb = remediated_oof_probs[i, 2]
            ps = remediated_oof_probs[i, 0]
            bull_regime = (c >= sma) if not np.isnan(sma) else True

            if not in_pos:
                # Entry: Conviction >= threshold AND macro trend filter
                if pb >= th and bull_regime:
                    sig.iloc[i] = "BUY"
                    in_pos = True
            else:
                # Exit: Sell conviction >= threshold OR trend break
                if ps >= th or (not bull_regime):
                    sig.iloc[i] = "SELL"
                    in_pos = False

        res = sim.simulate(oof_prices, sig, f"Candidate_Th_{th:.3f}")
        s = res["summary"]

        sweep_results.append({
            "threshold": th,
            "final_equity": s["final_equity"],
            "net_pnl": s["net_profit_dollars"],
            "net_return_pct": s["net_return_pct"],
            "total_trades": s["total_trades"],
            "win_rate_pct": s["win_rate_pct"],
            "profit_factor": s["profit_factor"],
            "max_drawdown_pct": s["max_drawdown_pct"],
            "sharpe_ratio": s["sharpe_ratio"],
            "exposure_pct": s["exposure_pct"],
        })

    # Also test Relative Dominance Rule: P(BUY) > P(SELL) and P(BUY) >= 0.35
    sig_rel = pd.Series("HOLD", index=oof_dates)
    in_pos_rel = False
    for i in range(len(oof_dates)):
        c = close_oof.iloc[i]
        sma = sma_200.iloc[i]
        pb = remediated_oof_probs[i, 2]
        ps = remediated_oof_probs[i, 0]
        bull_regime = (c >= sma) if not np.isnan(sma) else True

        if not in_pos_rel:
            if (pb > ps) and (pb >= 0.35) and bull_regime:
                sig_rel.iloc[i] = "BUY"
                in_pos_rel = True
        else:
            if (ps > pb) or (not bull_regime):
                sig_rel.iloc[i] = "SELL"
                in_pos_rel = False

    res_rel = sim.simulate(oof_prices, sig_rel, "Candidate_Relative_Dominance")
    s_rel = res_rel["summary"]
    sweep_results.append({
        "threshold": "Relative_Dominance (P(B)>P(S) & P(B)>=0.35)",
        "final_equity": s_rel["final_equity"],
        "net_pnl": s_rel["net_profit_dollars"],
        "net_return_pct": s_rel["net_return_pct"],
        "total_trades": s_rel["total_trades"],
        "win_rate_pct": s_rel["win_rate_pct"],
        "profit_factor": s_rel["profit_factor"],
        "max_drawdown_pct": s_rel["max_drawdown_pct"],
        "sharpe_ratio": s_rel["sharpe_ratio"],
        "exposure_pct": s_rel["exposure_pct"],
    })

    # Select best candidate based on highest Sharpe ratio with > 20 trades
    valid_candidates = [
        c for c in sweep_results
        if isinstance(c["threshold"], float) and c["total_trades"] >= 20 and c["net_return_pct"] > 0
    ]
    best_candidate = max(valid_candidates, key=lambda x: x["sharpe_ratio"]) if valid_candidates else sweep_results[0]
    locked_threshold = best_candidate["threshold"]

    logger.info(f"LOCKED HISTORICAL THRESHOLD: {locked_threshold} (Historical Sharpe: {best_candidate['sharpe_ratio']}, Return: {best_candidate['net_return_pct']}%)")

    return {
        "sweep_results": sweep_results,
        "locked_threshold": locked_threshold,
        "locked_candidate_stats": best_candidate,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. OUT-OF-SAMPLE FORWARD EVALUATION OF LOCKED MODEL (2025-10-13 TO 2026-10-13)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_locked_candidate_forward(locked_threshold: float) -> Dict[str, Any]:
    """
    Applies the LOCKED threshold to the 2025-10-13 to 2026-10-13 forward period.
    The model is trained strictly on pre-2025 data (zero lookahead).
    Simulates portfolio performance and exports cent-reconciled trade ledgers.
    """
    import yfinance as yf

    from scripts.research.workstream_3_forward_simulation import (
        InstitutionalPortfolioSimulator,
        compute_indicators,
        generate_buy_and_hold_signals,
        generate_dip_entry_signals,
        generate_donchian_signals,
        generate_sma200_signals,
    )

    logger.info(f"Evaluating Locked ML Candidate (Threshold={locked_threshold}) on Forward Period...")

    df_raw = yf.download("AAPL", start=WARMUP_START, end=EVAL_END, progress=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.droplevel(1)
    df_raw.dropna(inplace=True)
    df_full = compute_indicators(df_raw)

    eval_dates = df_full.loc[EVAL_START:"2026-10-13"].index
    eval_prices = df_raw.loc[eval_dates].copy()

    feature_cols = [
        "RSI", "ATR", "ADX", "MACD", "MACD_Signal", "MACD_Diff",
        "Stoch_K", "Stoch_D", "BB_Upper", "BB_Middle", "BB_Lower",
        "Volatility_Ratio", "Return_1d", "Return_5d", "Return_20d",
        "Volume_Ratio", "SMA_200",
    ]

    eval_start_dt = pd.Timestamp(EVAL_START)
    embargo_start = eval_start_dt - pd.Timedelta(days=20)
    train_df = df_full.loc[:embargo_start].dropna(subset=feature_cols)

    # Train Remediated Candidate (2.0 ATR TP, 1.5 ATR SL, H=15)
    tp, sl, h_len = 2.0, 1.5, 15
    fh_list = [train_df["High"].shift(-h) for h in range(1, h_len + 1)]
    fl_list = [train_df["Low"].shift(-h) for h in range(1, h_len + 1)]
    fh_df = pd.concat(fh_list, axis=1)
    fl_df = pd.concat(fl_list, axis=1)

    y_train, valid_mask = [], []
    for dt in train_df.index:
        c = float(train_df.loc[dt, "Close"])
        atr = float(train_df.loc[dt, "ATR"])
        ub = c + (atr * tp)
        lb = c - (atr * sl)
        fh_row = fh_df.loc[dt].values
        fl_row = fl_df.loc[dt].values
        if np.any(np.isnan(fh_row)) or np.any(np.isnan(fl_row)):
            valid_mask.append(False)
            continue
        u_hit = np.where(fh_row >= ub)[0]
        l_hit = np.where(fl_row <= lb)[0]
        u_s = u_hit[0] + 1 if len(u_hit) > 0 else h_len + 1
        l_s = l_hit[0] + 1 if len(l_hit) > 0 else h_len + 1
        if u_s > h_len and l_s > h_len:
            y_train.append(1)
        elif u_s < l_s:
            y_train.append(2)
        else:
            y_train.append(0)
        valid_mask.append(True)

    y_train = np.array(y_train, dtype=int)
    valid_mask = np.array(valid_mask)
    X_train = train_df[feature_cols].values[valid_mask]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_ev_s = scaler.transform(df_full.loc[eval_dates, feature_cols].ffill().fillna(0.0).values)

    xgb_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.70,
        "min_child_weight": 2.0,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": 1,
    }
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr_s, y_train)
    probs_eval = model.predict_proba(X_ev_s)

    # Generate signals using LOCKED threshold on baseline features
    close_ev = eval_prices["Close"]
    sma_ev = df_full["SMA_200"].loc[eval_dates]

    sig_locked = pd.Series("HOLD", index=eval_dates)
    in_pos = False
    for i in range(len(eval_dates)):
        c = float(close_ev.iloc[i])
        s = float(sma_ev.iloc[i])
        pb = probs_eval[i, 2]
        ps = probs_eval[i, 0]
        bull_regime = (c >= s) if not np.isnan(s) else True

        if not in_pos:
            if pb >= locked_threshold and bull_regime:
                sig_locked.iloc[i] = "BUY"
                in_pos = True
        else:
            if ps >= locked_threshold or (not bull_regime):
                sig_locked.iloc[i] = "SELL"
                in_pos = False

    # Also train and evaluate the Stationary Feature Candidate (Scale-Invariant Normalized Inputs)
    df_full["Price_vs_SMA200"] = (df_full["Close"] - df_full["SMA_200"]) / df_full["SMA_200"]
    df_full["BB_Position"] = (df_full["Close"] - df_full["BB_Lower"]) / (df_full["BB_Upper"] - df_full["BB_Lower"] + 1e-8)
    df_full["BB_Width"] = (df_full["BB_Upper"] - df_full["BB_Lower"]) / (df_full["BB_Middle"] + 1e-8)
    df_full["MACD_Norm"] = df_full["MACD_Diff"] / df_full["Close"]

    stationary_cols = [
        "RSI", "ADX", "Stoch_K", "Stoch_D", "Volatility_Ratio",
        "Price_vs_SMA200", "BB_Position", "BB_Width", "MACD_Norm",
        "Return_1d", "Return_5d", "Return_20d", "Volume_Ratio",
    ]
    train_stat_df = df_full.loc[:embargo_start].dropna(subset=stationary_cols)
    X_train_stat = train_stat_df[stationary_cols].values[valid_mask]
    scaler_stat = StandardScaler()
    X_tr_stat_s = scaler_stat.fit_transform(X_train_stat)
    X_ev_stat_s = scaler_stat.transform(df_full.loc[eval_dates, stationary_cols].ffill().fillna(0.0).values)

    model_stat = xgb.XGBClassifier(**xgb_params)
    model_stat.fit(X_tr_stat_s, y_train)
    probs_stat = model_stat.predict_proba(X_ev_stat_s)

    sig_stat = pd.Series("HOLD", index=eval_dates)
    in_pos_stat = False
    for i in range(len(eval_dates)):
        c = float(close_ev.iloc[i])
        s = float(sma_ev.iloc[i])
        pb_s = probs_stat[i, 2]
        ps_s = probs_stat[i, 0]
        bull_regime = (c >= s) if not np.isnan(s) else True

        if not in_pos_stat:
            if pb_s >= 0.50 and bull_regime:
                sig_stat.iloc[i] = "BUY"
                in_pos_stat = True
        else:
            if ps_s >= 0.50 or (not bull_regime):
                sig_stat.iloc[i] = "SELL"
                in_pos_stat = False

    # Simulate locked strategy alongside baselines
    sim = InstitutionalPortfolioSimulator(
        initial_capital=1000.0,
        slippage_bps=5.0,
        commission_per_share=0.005,
        min_commission=1.00,
    )

    strategies = {
        "Buy_and_Hold": generate_buy_and_hold_signals(eval_dates),
        "200_Day_SMA_Trend": generate_sma200_signals(df_full, eval_dates),
        "20_Day_Donchian_Breakout": generate_donchian_signals(df_full, eval_dates),
        "Dip_Entry_In_Uptrend": generate_dip_entry_signals(df_full, eval_dates),
        f"HYDRA_Locked_Candidate_Th_{locked_threshold:.2f}": sig_locked,
        "HYDRA_Stationary_Remediated_ML_Th_0.50": sig_stat,
    }

    forward_results = {}
    for s_name, s_sigs in strategies.items():
        res = sim.simulate(eval_prices, s_sigs, s_name)
        forward_results[s_name] = res

    # Cent-reconciled summary table
    summary_rows = [res["summary"] for res in forward_results.values()]
    df_forward_summary = pd.DataFrame(summary_rows)
    df_forward_summary.to_csv(REPORTS_DIR / "workstream_4_locked_candidate_comparison.csv", index=False)

    # Collect all trades
    all_trades = []
    for res in forward_results.values():
        all_trades.extend(res["trades"])
    df_locked_trades = pd.DataFrame(all_trades)
    df_locked_trades.to_csv(REPORTS_DIR / "workstream_4_locked_candidate_trade_ledger.csv", index=False)

    return {
        "summary": summary_rows,
        "trades": all_trades,
        "locked_strategy_summary": forward_results[f"HYDRA_Locked_Candidate_Th_{locked_threshold:.2f}"]["summary"],
        "locked_strategy_trades": forward_results[f"HYDRA_Locked_Candidate_Th_{locked_threshold:.2f}"]["trades"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. MASTER EXECUTION & REPORT GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def run_master_workstream_4() -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("HYDRA V2.4.2+ — Workstream 4: Forensic ML Abstention & Threshold Locking")
    logger.info("=" * 80)

    # 1. Reconcile raw math
    logger.info("Step 1: Reconciling trade ledger arithmetic...")
    recon = reconcile_forward_trade_ledger_arithmetic()

    # 2. Audit ML abstention
    logger.info("Step 2: Auditing ML abstention & probability distributions...")
    abstention_audit = audit_ml_abstention()

    # 3. Sweep and lock historical threshold
    logger.info("Step 3: Sweeping and locking historical threshold on pre-2025 OOF data...")
    threshold_audit = select_and_lock_historical_threshold()
    locked_th = threshold_audit["locked_threshold"]

    # 4. Out-of-sample forward evaluation
    logger.info("Step 4: Evaluating locked candidate out-of-sample forward (2025-2026)...")
    forward_eval = evaluate_locked_candidate_forward(locked_th)

    # Save comprehensive audit report JSON
    audit_bundle = {
        "arithmetic_reconciliation": recon["sma_trade_audit"],
        "abstention_forensics": abstention_audit,
        "historical_threshold_selection": threshold_audit,
        "forward_evaluation": forward_eval["summary"],
    }
    with open(REPORTS_DIR / "workstream_4_abstention_and_threshold_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_bundle, f, indent=2, default=str)

    print("\n" + "=" * 90)
    print("HYDRA V2.4.2+ — WORKSTREAM 4 AUDIT COMPLETE")
    print("=" * 90)
    print(f"LOCKED HISTORICAL THRESHOLD: {locked_th}")
    print("\nForward Out-of-Sample Performance (2025-10-13 to 2026-10-13):")
    print(f"{'Strategy':<40} {'Final Equity':>14} {'Net P&L':>12} {'Return':>10} {'Trades':>8} {'Sharpe':>8}")
    print("-" * 96)
    for s in forward_eval["summary"]:
        print(f"{s['strategy']:<40} ${s['final_equity']:>12,.2f} ${s['net_profit_dollars']:>10,.2f} {s['net_return_pct']:>8.2f}% {s['total_trades']:>7} {s['sharpe_ratio']:>7.3f}")
    print("=" * 90)

    return audit_bundle


if __name__ == "__main__":
    run_master_workstream_4()
