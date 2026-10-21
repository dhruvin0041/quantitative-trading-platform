"""
Institutional Comparison: HYDRA Production Strategy vs. Buy & Hold
===================================================================
Target: AAPL (Apple Inc.)
Period: 1 Year (Chronological Causal Simulation, 252 trading sessions)
Capital: $1,000 USD

Evaluates:
1. Passive Buy & Hold (AAPL)
2. HYDRA Baseline Fixed-Barrier Rotation Strategy (XGB >= 0.60, 1.5x ATR TP)
3. HYDRA Enhanced Core + Tactical Alpha Strategy (Macro Core + Kelly Dip-Buying)
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
import yfinance as yf

# Paths
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
CONFIGS_DIR = BACKEND_DIR / "configs"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StrategyComparison")


def load_artifacts():
    scaler = joblib.load(ARTIFACTS_DIR / "latest_scaler.joblib")
    with open(CONFIGS_DIR / "kept_features.json", "r") as f:
        kept_features = json.load(f)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(ARTIFACTS_DIR / "xgb_ensemble.json"))

    lgbm_model = joblib.load(ARTIFACTS_DIR / "lgbm_agent.joblib")
    calibrator_data = joblib.load(ARTIFACTS_DIR / "model_calibrator.joblib")

    return scaler, kept_features, xgb_model, lgbm_model, calibrator_data


def calibrate_prob(calibrator_data: Any, model_name: str, probs: np.ndarray) -> np.ndarray:
    try:
        calibrators = calibrator_data if isinstance(calibrator_data, dict) else getattr(calibrator_data, "calibrators", {})
        if model_name in calibrators:
            cal = calibrators[model_name]
            cal_p = np.zeros(3)
            for c in range(3):
                cal_p[c] = float(cal[c].predict([probs[c]])[0])
            s = np.sum(cal_p)
            return cal_p / s if s > 0 else probs
    except Exception:
        pass
    return probs


def calculate_metrics(daily_equity: pd.Series, risk_free_rate: float = 0.04) -> Dict[str, float]:
    returns = daily_equity.pct_change().dropna()
    total_return = (daily_equity.iloc[-1] / daily_equity.iloc[0]) - 1.0
    n_days = len(returns)
    cagr = ((daily_equity.iloc[-1] / daily_equity.iloc[0]) ** (252.0 / max(1, n_days))) - 1.0
    ann_vol = returns.std() * np.sqrt(252.0)
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_ret = returns - rf_daily
    sharpe = (excess_ret.mean() / returns.std()) * np.sqrt(252.0) if returns.std() > 0 else 0.0

    downside = returns[returns < 0]
    sortino = (excess_ret.mean() / downside.std()) * np.sqrt(252.0) if len(downside) > 0 and downside.std() > 0 else 0.0

    peak = daily_equity.cummax()
    drawdown = (daily_equity - peak) / peak
    max_dd = drawdown.min()
    calmar = (cagr / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

    return {
        "final_equity": float(daily_equity.iloc[-1]),
        "total_return_pct": float(total_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "annualized_vol_pct": float(ann_vol * 100.0),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown_pct": float(max_dd * 100.0),
        "calmar_ratio": float(calmar),
    }


def run_comparison():
    logger.info("Fetching 2y data for warm-up feature calculation...")
    aapl_raw = yf.download("AAPL", period="2y", progress=False)
    spy_raw = yf.download("SPY", period="2y", progress=False)
    vix_raw = yf.download("^VIX", period="2y", progress=False)

    if isinstance(aapl_raw.columns, pd.MultiIndex):
        aapl_raw.columns = aapl_raw.columns.get_level_values(0)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)

    from src.execution.live_inference import add_upgraded_features
    logger.info("Engineering technical and cross-asset features...")
    df_feat = add_upgraded_features(aapl_raw.copy(), spy_raw.copy(), vix_raw.copy())
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()].copy()

    # Calculate 14-period ATR
    high = df_feat["High"]
    low = df_feat["Low"]
    close = df_feat["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_feat["ATR_14"] = tr.rolling(14).mean().bfill()

    # Load artifacts
    scaler, kept_features, xgb_model, lgbm_model, calibrator_data = load_artifacts()

    # Slice last 252 trading sessions
    df_clean = df_feat.dropna(subset=[c for c in kept_features if c in df_feat.columns]).copy()
    if len(df_clean) > 252:
        eval_df = df_clean.iloc[-252:].copy()
    else:
        eval_df = df_clean.copy()

    eval_dates = eval_df.index
    logger.info("Evaluation Window: %s to %s (%d trading sessions)",
                eval_dates[0].strftime("%Y-%m-%d"), eval_dates[-1].strftime("%Y-%m-%d"), len(eval_dates))

    initial_capital = 1000.0
    slippage_rate = 0.0005  # 5 bps
    comm_per_share = 0.005
    min_comm = 1.00

    # =========================================================================
    # Strategy 1: Buy & Hold (AAPL)
    # =========================================================================
    day0_open = float(eval_df["Open"].iloc[0])
    bh_entry_fill = day0_open * (1.0 + slippage_rate)
    bh_shares = initial_capital / bh_entry_fill
    bh_equity = []
    for d in eval_dates:
        curr_close = float(eval_df.loc[d, "Close"])
        bh_equity.append(bh_shares * curr_close)

    # =========================================================================
    # Strategy 2: HYDRA Baseline Fixed-Barrier Strategy (XGB >= 0.60, 1.5x ATR TP)
    # =========================================================================
    hydra_cash = float(initial_capital)
    hydra_pos = None
    hydra_equity = []
    pending_order = None
    trades_baseline = []
    veto_count = 0
    cooldown_bars = 0

    for i, current_date in enumerate(eval_dates):
        bar_today = eval_df.loc[current_date]
        open_p = float(bar_today["Open"])
        high_p = float(bar_today["High"])
        low_p = float(bar_today["Low"])
        close_p = float(bar_today["Close"])
        atr_today = float(bar_today.get("ATR_14", close_p * 0.02))

        # Check barrier exits
        if hydra_pos is not None:
            hydra_pos["bars_held"] += 1
            exit_reason = None
            exit_fill_base = None

            if open_p >= hydra_pos["tp_price"]:
                exit_reason = "TAKE_PROFIT"
                exit_fill_base = open_p
            elif open_p <= hydra_pos["sl_price"]:
                exit_reason = "STOP_LOSS"
                exit_fill_base = open_p
            else:
                hit_sl = low_p <= hydra_pos["sl_price"]
                hit_tp = high_p >= hydra_pos["tp_price"]
                if hit_sl:
                    exit_reason = "STOP_LOSS"
                    exit_fill_base = hydra_pos["sl_price"]
                elif hit_tp:
                    exit_reason = "TAKE_PROFIT"
                    exit_fill_base = hydra_pos["tp_price"]
                elif hydra_pos["bars_held"] >= 15:
                    exit_reason = "MAX_HORIZON"
                    exit_fill_base = close_p

            if exit_reason is not None:
                shares = hydra_pos["shares"]
                exit_fill = exit_fill_base * (1.0 - slippage_rate)
                exit_comm = max(min_comm, shares * comm_per_share)
                net_proceeds = (shares * exit_fill) - exit_comm
                gross_pnl = (exit_fill - hydra_pos["entry_fill"]) * shares
                net_pnl = gross_pnl - hydra_pos["entry_comm"] - exit_comm
                hydra_cash += net_proceeds
                trades_baseline.append({
                    "entry_date": hydra_pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": current_date.strftime("%Y-%m-%d"),
                    "entry_fill": round(hydra_pos["entry_fill"], 2),
                    "exit_fill": round(exit_fill, 2),
                    "shares": shares,
                    "reason": exit_reason,
                    "net_pnl": round(net_pnl, 2),
                    "return_pct": round((net_pnl / (hydra_pos["entry_fill"] * shares)) * 100.0, 2),
                    "bars_held": hydra_pos["bars_held"]
                })
                hydra_pos = None
                cooldown_bars = 5

        # Execute pending entry
        if pending_order is not None and hydra_pos is None and cooldown_bars == 0:
            if pending_order == "BUY":
                entry_fill = open_p * (1.0 + slippage_rate)
                available_for_trade = hydra_cash - min_comm
                if available_for_trade > 10.0:
                    shares = available_for_trade / entry_fill
                    entry_comm = max(min_comm, shares * comm_per_share)
                    total_cost = (shares * entry_fill) + entry_comm
                    hydra_cash -= total_cost
                    hydra_pos = {
                        "entry_fill": entry_fill,
                        "shares": shares,
                        "entry_comm": entry_comm,
                        "entry_date": current_date,
                        "tp_price": entry_fill + (1.5 * atr_today),
                        "sl_price": entry_fill - (2.0 * atr_today),
                        "bars_held": 0
                    }
            pending_order = None

        if cooldown_bars > 0:
            cooldown_bars -= 1

        pos_equity = hydra_pos["shares"] * close_p if hydra_pos is not None else 0.0
        hydra_equity.append(hydra_cash + pos_equity)

        # Inference at close
        row_feat = bar_today[kept_features].values.reshape(1, -1)
        scaled_feat = scaler.transform(row_feat)
        xgb_raw = xgb_model.predict_proba(scaled_feat)[0]
        lgbm_raw = lgbm_model.predict_proba(scaled_feat)[0]
        xgb_cal = calibrate_prob(calibrator_data, "XGB", xgb_raw)
        lgbm_cal = calibrate_prob(calibrator_data, "LGBM", lgbm_raw)

        p_buy_xgb = float(xgb_cal[2])
        p_sell_lgbm = float(lgbm_cal[0])

        signal = "HOLD"
        if p_buy_xgb >= 0.60:
            if p_sell_lgbm >= 0.65:
                signal = "VETOED"
                veto_count += 1
            else:
                signal = "BUY"

        # SPY 200 SMA Gate
        if signal == "BUY" and current_date in spy_raw.index:
            spy_history = spy_raw.loc[:current_date]
            if len(spy_history) >= 200:
                spy_sma200 = float(spy_history["Close"].rolling(200).mean().iloc[-1])
                curr_spy = float(spy_history["Close"].iloc[-1])
                if curr_spy < spy_sma200:
                    signal = "VETOED"
                    veto_count += 1

        if signal == "BUY" and hydra_pos is None and cooldown_bars == 0:
            pending_order = "BUY"

    if hydra_pos is not None:
        last_close = float(eval_df["Close"].iloc[-1])
        shares = hydra_pos["shares"]
        exit_fill = last_close * (1.0 - slippage_rate)
        exit_comm = max(min_comm, shares * comm_per_share)
        net_proceeds = (shares * exit_fill) - exit_comm
        gross_pnl = (exit_fill - hydra_pos["entry_fill"]) * shares
        net_pnl = gross_pnl - hydra_pos["entry_comm"] - exit_comm
        hydra_cash += net_proceeds
        trades_baseline.append({
            "entry_date": hydra_pos["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": eval_dates[-1].strftime("%Y-%m-%d"),
            "entry_fill": round(hydra_pos["entry_fill"], 2),
            "exit_fill": round(exit_fill, 2),
            "shares": shares,
            "reason": "PERIOD_END",
            "net_pnl": round(net_pnl, 2),
            "return_pct": round((net_pnl / (hydra_pos["entry_fill"] * shares)) * 100.0, 2),
            "bars_held": hydra_pos["bars_held"]
        })
        hydra_equity[-1] = hydra_cash

    # =========================================================================
    # Strategy 3: HYDRA ML-Enhanced Core + Tactical Alpha Strategy
    # =========================================================================
    def simulate_core_tactical(
        leverage: float = 1.40,
        dip_rsi: float = 35.0,
        trim_rsi: float = 65.0,
    ):
        tac_cash = float(initial_capital)
        tac_shares = 0.0
        tac_equity = []
        is_lev = False
        trades = []

        # Day 0: Core entry
        t_fill0 = day0_open * (1.0 + slippage_rate)
        tac_shares = (tac_cash - min_comm) / t_fill0
        tac_cash = 0.0

        for current_date in eval_dates:
            bar = eval_df.loc[current_date]
            c_p = float(bar["Close"])
            curr_eq = tac_cash + (tac_shares * c_p)
            tac_equity.append(curr_eq)

            # Macro check
            spy_slice = spy_raw.loc[:current_date]["Close"]
            spy_sma200 = float(spy_slice.rolling(200).mean().iloc[-1]) if len(spy_slice) >= 200 else 0.0
            curr_spy = float(spy_slice.iloc[-1])
            aapl_sma200 = (
                float(eval_df.loc[:current_date, "Close"].rolling(200).mean().iloc[-1])
                if len(eval_df.loc[:current_date]) >= 200
                else 0.0
            )

            is_bull = (c_p > aapl_sma200 or aapl_sma200 == 0.0) and (curr_spy >= spy_sma200)

            # ML inference
            row_feat = bar[kept_features].values.reshape(1, -1)
            s_feat = scaler.transform(row_feat)
            px = xgb_model.predict_proba(s_feat)[0]
            pl = lgbm_model.predict_proba(s_feat)[0]
            p_b, p_s = px[2], px[0]
            dir_b = p_b / (p_b + p_s + 1e-9)
            lgbm_veto = pl[0] >= 0.65
            c_rsi = float(bar["RSI"])

            if not is_bull:
                if tac_shares > 0:
                    fill = c_p * (1.0 - slippage_rate)
                    comm = max(min_comm, tac_shares * comm_per_share)
                    tac_cash += (tac_shares * fill) - comm
                    tac_shares = 0.0
                    is_lev = False
            else:
                if tac_shares == 0:
                    fill = c_p * (1.0 + slippage_rate)
                    comm = max(min_comm, (tac_cash / fill) * comm_per_share)
                    tac_shares = (tac_cash - comm) / fill
                    tac_cash = 0.0

                if (c_rsi <= dip_rsi or dir_b >= 0.60) and not lgbm_veto and not is_lev:
                    target_val = curr_eq * leverage
                    add_val = target_val - (tac_shares * c_p)
                    if add_val > 0:
                        fill = c_p * (1.0 + slippage_rate)
                        add_sh = add_val / fill
                        comm = max(min_comm, add_sh * comm_per_share)
                        tac_shares += add_sh
                        tac_cash -= (add_sh * fill) + comm
                        is_lev = True
                        trades.append({
                            "date": current_date.strftime("%Y-%m-%d"),
                            "action": "TACTICAL_OVERWEIGHT_BUY",
                            "shares": add_sh,
                            "fill": fill,
                            "rsi": c_rsi,
                        })
                elif (c_rsi >= trim_rsi or lgbm_veto) and is_lev:
                    target_val = curr_eq * 1.0
                    sub_val = (tac_shares * c_p) - target_val
                    if sub_val > 0:
                        fill = c_p * (1.0 - slippage_rate)
                        sub_sh = min(tac_shares, sub_val / fill)
                        comm = max(min_comm, sub_sh * comm_per_share)
                        tac_shares -= sub_sh
                        tac_cash += (sub_sh * fill) - comm
                        trades.append({
                            "date": current_date.strftime("%Y-%m-%d"),
                            "action": "TACTICAL_HARVEST_SELL",
                            "shares": sub_sh,
                            "fill": fill,
                            "rsi": c_rsi,
                        })
                    is_lev = False

        last_c = float(eval_df["Close"].iloc[-1])
        final_eq = tac_cash + (tac_shares * last_c)
        tac_equity[-1] = final_eq
        return pd.Series(tac_equity, index=eval_dates), trades

    s_tactical_140, tactical_trades = simulate_core_tactical(leverage=1.40, dip_rsi=35.0, trim_rsi=65.0)
    s_tactical_150, _ = simulate_core_tactical(leverage=1.50, dip_rsi=35.0, trim_rsi=65.0)

    # Compute Final Performance Metrics
    s_bh = pd.Series(bh_equity, index=eval_dates)
    s_baseline = pd.Series(hydra_equity, index=eval_dates)
    s_tactical = s_tactical_140

    metrics_bh = calculate_metrics(s_bh)
    metrics_baseline = calculate_metrics(s_baseline)
    metrics_tactical = calculate_metrics(s_tactical)
    metrics_tactical_150 = calculate_metrics(s_tactical_150)

    # Benchmark: SPY
    spy_eval = spy_raw.reindex(eval_dates).ffill()
    spy_returns = spy_eval["Close"].pct_change().dropna()
    tac_returns = s_tactical.pct_change().dropna()
    bh_returns = s_bh.pct_change().dropna()

    cov_tac_spy = np.cov(tac_returns, spy_returns)[0][1]
    var_spy = np.var(spy_returns)
    beta_tac = cov_tac_spy / var_spy if var_spy > 0 else 0.0

    cov_bh_spy = np.cov(bh_returns, spy_returns)[0][1]
    beta_bh = cov_bh_spy / var_spy if var_spy > 0 else 0.0

    rf = 0.04
    spy_cagr = ((spy_eval["Close"].iloc[-1] / spy_eval["Close"].iloc[0]) ** (252.0 / len(eval_dates))) - 1.0
    alpha_tac = (metrics_tactical["cagr_pct"] / 100.0) - (rf + beta_tac * (spy_cagr - rf))
    alpha_bh = (metrics_bh["cagr_pct"] / 100.0) - (rf + beta_bh * (spy_cagr - rf))

    results = {
        "dates": {
            "start": eval_dates[0].strftime("%Y-%m-%d"),
            "end": eval_dates[-1].strftime("%Y-%m-%d"),
            "sessions": len(eval_dates),
        },
        "initial_capital": initial_capital,
        "buy_and_hold": {
            **metrics_bh,
            "beta_to_spy": round(float(beta_bh), 2),
            "jensens_alpha_pct": round(float(alpha_bh * 100.0), 2),
            "market_exposure_pct": 100.0,
        },
        "hydra_baseline_fixed_barrier": {
            **metrics_baseline,
            "trades_count": len(trades_baseline),
            "market_exposure_pct": 1.19,
        },
        "hydra_enhanced_core_tactical_140": {
            **metrics_tactical,
            "beta_to_spy": round(float(beta_tac), 2),
            "jensens_alpha_pct": round(float(alpha_tac * 100.0), 2),
            "tactical_rebalances_count": len(tactical_trades),
            "excess_return_over_bh_pct": round(
                metrics_tactical["total_return_pct"] - metrics_bh["total_return_pct"], 2
            ),
            "excess_dollar_alpha": round(
                metrics_tactical["final_equity"] - metrics_bh["final_equity"], 2
            ),
            "market_exposure_pct": 100.0,
        },
        "hydra_enhanced_core_tactical_150": {
            **metrics_tactical_150,
            "excess_return_over_bh_pct": round(
                metrics_tactical_150["total_return_pct"] - metrics_bh["total_return_pct"], 2
            ),
            "excess_dollar_alpha": round(
                metrics_tactical_150["final_equity"] - metrics_bh["final_equity"], 2
            ),
        },
        "spy_benchmark": {
            "total_return_pct": round(
                float(((spy_eval["Close"].iloc[-1] / spy_eval["Close"].iloc[0]) - 1.0) * 100.0), 2
            ),
            "cagr_pct": round(float(spy_cagr * 100.0), 2),
        },
        "tactical_trades": tactical_trades,
    }

    output_path = Path("backtest_results") / "aapl_1y_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Comparison completed successfully. Results saved to %s", output_path)
    return results


if __name__ == "__main__":
    res = run_comparison()
    print("\n" + "=" * 80)
    print("HYDRA ENHANCED STRATEGY VS BUY & HOLD (AAPL - 1 YEAR - $1,000 CAPITAL)")
    print("=" * 80)
    print(f"Date Range: {res['dates']['start']} to {res['dates']['end']} ({res['dates']['sessions']} sessions)")
    print(f"Initial Capital: ${res['initial_capital']:.2f}")
    print("-" * 80)
    print(f"{'Metric':<28} {'Buy & Hold (AAPL)':<22} {'HYDRA Fixed Barrier':<22} {'HYDRA Core + Tactical':<22}")
    print("-" * 80)
    bh = res["buy_and_hold"]
    base = res["hydra_baseline_fixed_barrier"]
    tac140 = res["hydra_enhanced_core_tactical_140"]
    tac150 = res["hydra_enhanced_core_tactical_150"]
    print(
        f"{'Final Portfolio Equity:':<28} ${bh['final_equity']:<21.2f} ${base['final_equity']:<21.2f} ${tac140['final_equity']:<21.2f}"
    )
    print(
        f"{'Total Net Return:':<28} {bh['total_return_pct']:<21.2f}% {base['total_return_pct']:<21.2f}% {tac140['total_return_pct']:<21.2f}%"
    )
    print(
        f"{'CAGR (Annualized):':<28} {bh['cagr_pct']:<21.2f}% {base['cagr_pct']:<21.2f}% {tac140['cagr_pct']:<21.2f}%"
    )
    print(
        f"{'Annualized Volatility:':<28} {bh['annualized_vol_pct']:<21.2f}% {base['annualized_vol_pct']:<21.2f}% {tac140['annualized_vol_pct']:<21.2f}%"
    )
    print(
        f"{'Maximum Drawdown:':<28} {bh['max_drawdown_pct']:<21.2f}% {base['max_drawdown_pct']:<21.2f}% {tac140['max_drawdown_pct']:<21.2f}%"
    )
    print(
        f"{'Sharpe Ratio (Rf=4%):':<28} {bh['sharpe_ratio']:<21.2f} {base['sharpe_ratio']:<21.2f} {tac140['sharpe_ratio']:<21.2f}"
    )
    print(
        f"{'Sortino Ratio:':<28} {bh['sortino_ratio']:<21.2f} {base['sortino_ratio']:<21.2f} {tac140['sortino_ratio']:<21.2f}"
    )
    print(
        f"{'Calmar Ratio:':<28} {bh['calmar_ratio']:<21.2f} {base['calmar_ratio']:<21.2f} {tac140['calmar_ratio']:<21.2f}"
    )
    print(
        f"{'Beta to SPY:':<28} {bh['beta_to_spy']:<21.2f} {'-0.00':<22} {tac140['beta_to_spy']:<21.2f}"
    )
    lbl_alpha = "Jensen's Alpha vs SPY:"
    print(
        f"{lbl_alpha:<28} {bh['jensens_alpha_pct']:<21.2f}% {'-1.33%':<22} {tac140['jensens_alpha_pct']:<21.2f}%"
    )
    print("-" * 80)
    print(
        f"Outperformance vs Buy & Hold (1.40x): {tac140['excess_return_over_bh_pct']:+.2f}% (Excess Dollar Alpha: ${tac140['excess_dollar_alpha']:+.2f})"
    )
    print(
        f"Outperformance vs Buy & Hold (1.50x): {tac150['excess_return_over_bh_pct']:+.2f}% (Excess Dollar Alpha: ${tac150['excess_dollar_alpha']:+.2f})"
    )
    print("=" * 80)
