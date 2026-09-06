import argparse
import json
import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Suppress verbose warnings during audit
warnings.filterwarnings("ignore")
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from src.execution.consensus_engine import WeightedConsensusEngine
from src.execution.live_inference import add_upgraded_features, load_config
from src.models.neural.fusion_network import build_fusion_model
from src.models.rl.dqn_agent import DQNAgent
from src.optimization.objective_functions import (
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
)

# Load canonical kept features (institutional standard: 27 stationary features)
KEPT_FEATURES_PATH = BACKEND_DIR / "configs" / "kept_features.json"
if KEPT_FEATURES_PATH.exists():
    with open(KEPT_FEATURES_PATH, "r") as f:
        STATIONARY_FEATURES = json.load(f)
else:
    STATIONARY_FEATURES = [
        "MA20_vs_MA50",
        "EMA9_vs_EMA21",
        "Price_vs_EMA9",
        "Price_vs_EMA21",
        "VIX_Level",
        "BB_Width",
        "BB_Position",
        "RSI",
        "ADX",
        "MACD_Hist",
        "Relative_Strength",
        "OBV_Change",
        "Return",
        "Volume_Ratio",
    ]


def fetch_data_clean(ticker: str, start: str, end: str, spy: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Fetch ticker OHLCV and compute institutional stationary features and Triple Barrier targets."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = add_upgraded_features(df, spy, vix)
    df["forward_5d_ret"] = df["Close"].shift(-5) / df["Close"] - 1
    # Target signal: BUY if ret > 2%, SELL if ret < -1.5%
    df["target_signal"] = 1  # HOLD
    df.loc[df["forward_5d_ret"] > 0.02, "target_signal"] = 2
    df.loc[df["forward_5d_ret"] < -0.015, "target_signal"] = 0
    return df.dropna()


def build_models(num_features: int):
    """
    Instantiate retraining trees (XGBoost, Regularized LightGBM)
    and pre-trained neural checkpoints (DL Fusion, DQN Agent).
    """
    # 1. XGBoost Classifier (Conservative Tree Profile)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
    )

    # 2. Regularized LightGBM Classifier (Constrained Leaf & Bagging Profile)
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        learning_rate=0.05,
        objective="multiclass",
        num_class=3,
        random_state=42,
        verbose=-1,
    )

    # 3. Pre-trained DL Fusion Checkpoint (Inference-Only)
    dl_model = None
    try:
        config = load_config("AAPL")
        config["data"]["num_features"] = num_features
        dl_model = build_fusion_model(config)
        weights_path = BACKEND_DIR / "artifacts" / "latest_fusion_weights.weights.h5"
        if weights_path.exists():
            dl_model.load_weights(str(weights_path), skip_mismatch=False)
    except Exception as e:
        print(f"[WARN] Could not load DL Fusion weights: {e}")

    # 4. Pre-trained DQN Agent Checkpoint (Inference-Only)
    dqn_agent = None
    try:
        dqn_path = BACKEND_DIR / "artifacts" / "dqn_model.pth"
        if dqn_path.exists():
            dqn_agent = DQNAgent(state_size=num_features + 6)
            dqn_agent.load(str(dqn_path))
    except Exception as e:
        print(f"[WARN] Could not load DQN Agent: {e}")

    return xgb_model, lgbm_model, dl_model, dqn_agent


def run_walk_forward_simulation(
    tickers=None,
    full_start="2021-01-01",
    full_end="2026-05-23",
    sim_start="2024-01-01",
    sim_end="2026-05-01",
):
    """
    Executes walk-forward evaluation across AAPL, MSFT, and NVDA.
    Retrains tree models monthly on a 2-year rolling window with a 10-day embargo,
    while running inference across pre-trained DL Fusion and DQN checkpoints.
    Returns complete out-of-sample daily prediction stream.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA"]

    print(f"Downloading benchmark macro indexes (SPY, ^VIX) from {full_start} to {full_end}...")
    spy_full = yf.download("SPY", start=full_start, end=full_end, progress=False)
    vix_full = yf.download("^VIX", start=full_start, end=full_end, progress=False)
    if isinstance(spy_full.columns, pd.MultiIndex):
        spy_full.columns = spy_full.columns.droplevel(1)
    if isinstance(vix_full.columns, pd.MultiIndex):
        vix_full.columns = vix_full.columns.droplevel(1)

    print("Fetching and sanitizing ticker datasets...")
    all_data = {}
    for t in tickers:
        df = fetch_data_clean(t, full_start, full_end, spy_full, vix_full)
        all_data[t] = df

    features = STATIONARY_FEATURES
    num_features = len(features)
    print(f"Stationary Feature Vector: {num_features} dimensions.")

    _, _, dl_model, dqn_agent = build_models(num_features)

    sim_start_ts = pd.Timestamp(sim_start)
    sim_end_ts = pd.Timestamp(sim_end)
    current_date = sim_start_ts
    history = []

    print("Beginning walk-forward monthly out-of-sample slices...")
    fold_idx = 0
    while current_date < sim_end_ts:
        next_date = current_date + timedelta(weeks=4)
        train_start = current_date - timedelta(days=365 * 2)

        # 1. Retrain trees strictly on data before current_date with 10-day embargo
        X_train_list = []
        y_train_list = []
        for t in tickers:
            df = all_data[t]
            mask = (df.index >= train_start) & (df.index < current_date - timedelta(days=10))
            train_chunk = df.loc[mask]
            if len(train_chunk) > 100:
                X_train_list.append(train_chunk[features].values)
                y_train_list.append(train_chunk["target_signal"].values)

        if not X_train_list:
            current_date = next_date
            continue

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Retrain XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
        )
        xgb_model.fit(X_train_scaled, y_train)

        # Retrain Regularized LightGBM
        lgbm_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            num_leaves=15,
            min_child_samples=50,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            learning_rate=0.05,
            objective="multiclass",
            num_class=3,
            random_state=42,
            verbose=-1,
        )
        lgbm_model.fit(X_train_scaled, y_train)

        # 2. Evaluate out-of-sample test slice
        for t in tickers:
            df = all_data[t]
            test_mask = (df.index >= current_date) & (df.index < next_date)
            test_chunk = df.loc[test_mask]
            if test_chunk.empty:
                continue

            X_test_scaled = scaler.transform(test_chunk[features].values)
            fwd_ret = test_chunk["forward_5d_ret"].values
            dates = test_chunk.index

            # Vectorized Tree Predictions
            probs_xgb = xgb_model.predict_proba(X_test_scaled)
            probs_lgbm = lgbm_model.predict_proba(X_test_scaled)

            # Vectorized DL Fusion Predictions
            if dl_model is not None:
                seq_list = []
                for d in dates:
                    loc_idx = df.index.get_loc(d)
                    seq = df.iloc[max(0, loc_idx - 59):loc_idx + 1][features].values
                    if len(seq) < 60:
                        pad = np.tile(seq[0], (60 - len(seq), 1))
                        seq = np.vstack([pad, seq])
                    seq_list.append(scaler.transform(seq))
                X_seq_batch = np.array(seq_list, dtype=np.float32)
                probs_dl = dl_model.predict([X_seq_batch] * 6, verbose=0)[2]
            else:
                probs_dl = np.full((len(test_chunk), 3), 1.0 / 3.0)

            # Vectorized DQN Predictions
            if dqn_agent is not None:
                dqn_states = np.hstack([X_test_scaled, probs_dl, probs_xgb]).astype(np.float32)
                with torch.no_grad():
                    t_states = torch.FloatTensor(dqn_states).to(dqn_agent.device)
                    q_vals = dqn_agent.policy_net(t_states).cpu().numpy()
                    tau = 1.5
                    scaled_q = q_vals / tau
                    exp_vals = np.exp(scaled_q - np.max(scaled_q, axis=1, keepdims=True))
                    probs_dqn = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
            else:
                probs_dqn = np.full((len(test_chunk), 3), 1.0 / 3.0)

            for i in range(len(test_chunk)):
                history.append(
                    {
                        "date": dates[i],
                        "ticker": t,
                        "ret": fwd_ret[i],
                        "p_xgb": probs_xgb[i],
                        "p_lgbm": probs_lgbm[i],
                        "p_dl": probs_dl[i],
                        "p_dqn": probs_dqn[i],
                    }
                )

        fold_idx += 1
        current_date = next_date

    print(f"Completed {fold_idx} walk-forward folds. Total test instances: {len(history)}")
    return pd.DataFrame(history)


def compute_strategy_metrics(results_list: list) -> dict:
    """Compute institutional quantitative metrics including Sharpe, MaxDD, and MC bootstrap intervals."""
    if not results_list:
        return {
            "signals": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "prob_pf_12": 0.0,
        }

    res_df = pd.DataFrame(results_list)
    returns = res_df["ret"].values
    wr = float(res_df["correct"].mean() * 100)
    pf = float(calculate_profit_factor(returns))
    sharpe = float(calculate_sharpe_ratio(returns))
    max_dd = float(calculate_max_drawdown(returns) * 100)

    # 1,000 Trial Monte Carlo Bootstrap
    pfs = []
    for _ in range(1000):
        sample = res_df.sample(frac=1.0, replace=True)
        gains = sample[sample["ret"] > 0]["ret"].sum()
        losses = abs(sample[sample["ret"] < 0]["ret"].sum())
        pfs.append(gains / losses if losses > 0 else 0.0)

    ci_lower = float(np.percentile(pfs, 2.5))
    ci_upper = float(np.percentile(pfs, 97.5))
    prob_pf_12 = float(np.mean(np.array(pfs) > 1.20) * 100.0)

    return {
        "signals": len(res_df),
        "win_rate": wr,
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "prob_pf_12": prob_pf_12,
    }


def evaluate_ablation(hist_df: pd.DataFrame, mode: str, threshold: float = 60.0):
    """
    Evaluates a specific model configuration or consensus weighting across the walk-forward history.
    """
    consensus_engine = WeightedConsensusEngine()
    results = []

    if mode in ["pure_xgb", "pure_lgbm", "pure_dl", "pure_dqn"]:
        col_map = {
            "pure_xgb": "p_xgb",
            "pure_lgbm": "p_lgbm",
            "pure_dl": "p_dl",
            "pure_dqn": "p_dqn",
        }
        col = col_map[mode]
        for _, row in hist_df.iterrows():
            p = row[col]
            ret = row["ret"]
            if np.max(p) >= (threshold / 100.0):
                action = int(np.argmax(p))
                if action == 2:
                    results.append({"ret": ret, "correct": ret > 0})
                elif action == 0:
                    results.append({"ret": -ret, "correct": ret < 0})

    elif mode == "static_consensus":
        # Static 50/50 consensus between XGBoost and LightGBM
        weights = {"XGB_AGENT": 0.50, "LGBM_AGENT": 0.50, "DL_FUSION": 0.0, "DQN_AGENT": 0.0}
        for _, row in hist_df.iterrows():
            base_probs = {
                "XGB_AGENT": row["p_xgb"],
                "LGBM_AGENT": row["p_lgbm"],
                "DL_FUSION": row["p_dl"],
                "DQN_AGENT": row["p_dqn"],
            }
            cons = consensus_engine.compute_agreement(base_probs, weights)
            score = cons["agreement_score"]
            direction = cons["dominant_direction"]
            ret = row["ret"]
            if score >= threshold:
                if direction == "BUY":
                    results.append({"ret": ret, "correct": ret > 0})
                elif direction == "SELL":
                    results.append({"ret": -ret, "correct": ret < 0})

    elif mode == "dynamic_consensus":
        # Dynamic Rolling Performance Weighting:
        # Trailing 90-day performance checks out-of-sample accuracy and edge.
        # Underperforming models (WR <= 50% or PF < 1.15) decay to zero.
        # When competing models exhibit negative or degrading alpha, pure XGBoost is preserved.
        dates = sorted(hist_df["date"].unique())
        step_size = 20
        model_trades = {"XGB_AGENT": [], "LGBM_AGENT": [], "DL_FUSION": [], "DQN_AGENT": []}

        for i in range(0, len(dates), step_size):
            chunk_dates = dates[i:i + step_size]
            chunk_df = hist_df[hist_df["date"].isin(chunk_dates)]

            # Calculate trailing out-of-sample edge
            wrs, pfs = {}, {}
            for m_key in ["XGB_AGENT", "LGBM_AGENT", "DL_FUSION", "DQN_AGENT"]:
                recent = [t for t in model_trades[m_key] if (chunk_dates[0] - t["date"]).days <= 90]
                if len(recent) >= 10:
                    wr = np.mean([t["correct"] for t in recent])
                    profits = sum([t["ret"] for t in recent if t["ret"] > 0])
                    losses = abs(sum([t["ret"] for t in recent if t["ret"] < 0]))
                    pf = profits / losses if losses > 0 else 1.0
                    wrs[m_key] = wr
                    pfs[m_key] = pf
                else:
                    wrs[m_key] = 0.55 if m_key == "XGB_AGENT" else 0.50
                    pfs[m_key] = 1.30 if m_key == "XGB_AGENT" else 1.00

            xgb_edge = max(0.01, wrs["XGB_AGENT"] - 0.50)

            # Strict Performance Decay: XGBoost is the institutional baseline.
            # Competing models earn allocation ONLY if they clear the positive edge hurdle.
            weights = {"XGB_AGENT": 1.0}
            for m_key in ["LGBM_AGENT", "DL_FUSION", "DQN_AGENT"]:
                if wrs[m_key] > 0.50 and pfs[m_key] >= 1.15:
                    rel_edge = min(1.0, (wrs[m_key] - 0.50) / xgb_edge)
                    weights[m_key] = float(rel_edge ** 2.0) * 0.50
                else:
                    weights[m_key] = 0.0  # Decayed to 0

            tot = sum(weights.values())
            weights = {k: v / tot for k, v in weights.items()}

            for _, row in chunk_df.iterrows():
                p_xgb = row["p_xgb"]
                p_lgb = row["p_lgbm"]
                p_dl = row["p_dl"]
                p_dqn = row["p_dqn"]
                ret = row["ret"]
                d = row["date"]

                # Track independent calls
                for m_key, p_arr in [
                    ("XGB_AGENT", p_xgb),
                    ("LGBM_AGENT", p_lgb),
                    ("DL_FUSION", p_dl),
                    ("DQN_AGENT", p_dqn),
                ]:
                    if np.max(p_arr) >= (threshold / 100.0):
                        a = np.argmax(p_arr)
                        if a == 2:
                            model_trades[m_key].append({"date": d, "ret": ret, "correct": ret > 0})
                        elif a == 0:
                            model_trades[m_key].append({"date": d, "ret": -ret, "correct": ret < 0})

                base_probs = {
                    "XGB_AGENT": p_xgb,
                    "LGBM_AGENT": p_lgb,
                    "DL_FUSION": p_dl,
                    "DQN_AGENT": p_dqn,
                }
                cons = consensus_engine.compute_agreement(base_probs, weights)
                score = cons["agreement_score"]
                direction = cons["dominant_direction"]

                if score >= threshold:
                    if direction == "BUY":
                        results.append({"ret": ret, "correct": ret > 0})
                    elif direction == "SELL":
                        results.append({"ret": -ret, "correct": ret < 0})

    return compute_strategy_metrics(results)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Walk-Forward Multi-Model Audit & Dynamic Consensus Ablation"
    )
    parser.add_argument(
        "--ablation",
        choices=[
            "all",
            "pure_xgb",
            "pure_lgbm",
            "pure_dl",
            "pure_dqn",
            "static_consensus",
            "dynamic_consensus",
            "xgb_zero",
            "lgbm_zero",
        ],
        default="all",
        help="Ablation mode (default: all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Consensus agreement threshold (default: 60.0)",
    )
    args = parser.parse_args()

    # Map legacy aliases
    ablation_mode = args.ablation
    if ablation_mode == "xgb_zero":
        ablation_mode = "pure_lgbm"
    elif ablation_mode == "lgbm_zero":
        ablation_mode = "pure_xgb"

    print("\n" + "=" * 90)
    print("=== QUANTITATIVE STABILITY AUDIT (WALK-FORWARD MULTI-MODEL ENSEMBLE) ===")
    print("=" * 90)

    # Check for pre-computed history cache or generate from live market data
    cache_path = BACKEND_DIR / "artifacts" / "walk_forward_history.pkl"
    if cache_path.exists():
        print(f"Loading cached walk-forward evaluation stream from {cache_path.name}...")
        hist_df = pd.read_pickle(str(cache_path))
    else:
        print("Executing live walk-forward simulation across test folds...")
        hist_df = run_walk_forward_simulation()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        hist_df.to_pickle(str(cache_path))

    if ablation_mode == "all":
        modes = [
            ("Pure XGBoost Benchmark", "pure_xgb"),
            ("Pure Regularized LightGBM", "pure_lgbm"),
            ("Pre-trained DL Fusion", "pure_dl"),
            ("Pre-trained DQN Agent", "pure_dqn"),
            ("Static 50/50 Ensemble", "static_consensus"),
            ("Dynamic Performance Consensus", "dynamic_consensus"),
        ]

        print("\n" + "=" * 95)
        print(
            f"{'Strategy / Ablation':<32} | {'Signals':<7} | {'Win Rate':<9} | {'PF':<5} | {'Sharpe':<6} | {'MaxDD':<7} | {'95% CI PF':<13} | {'P(PF>1.2)':<9}"
        )
        print("-" * 95)

        for label, m_key in modes:
            m = evaluate_ablation(hist_df, m_key, threshold=args.threshold)
            ci_str = f"[{m['ci_lower']:.2f}, {m['ci_upper']:.2f}]"
            print(
                f"{label:<32} | {m['signals']:<7d} | {m['win_rate']:<8.1f}% | {m['profit_factor']:<5.2f} | {m['sharpe']:<6.2f} | {m['max_dd']:<6.1f}% | {ci_str:<13} | {m['prob_pf_12']:<8.1f}%"
            )
        print("=" * 95 + "\n")

    else:
        labels = {
            "pure_xgb": "Pure XGBoost Benchmark",
            "pure_lgbm": "Pure Regularized LightGBM",
            "pure_dl": "Pre-trained DL Fusion",
            "pure_dqn": "Pre-trained DQN Agent",
            "static_consensus": "Static Equal 50/50 Ensemble",
            "dynamic_consensus": "Multi-Model Dynamic Performance Consensus",
        }
        name = labels.get(ablation_mode, ablation_mode)
        print(f"\nEvaluating Configuration: {name} (Threshold: {args.threshold:.1f}%)\n")
        m = evaluate_ablation(hist_df, ablation_mode, threshold=args.threshold)

        print("AUDIT RESULTS:")
        print(f"Total Signals: {m['signals']}")
        print(f"Win Rate:      {m['win_rate']:.1f}%")
        print(f"Profit Factor: {m['profit_factor']:.2f}")
        print(f"Sharpe Ratio:  {m['sharpe']:.2f}")
        print(f"Max Drawdown:  {m['max_dd']:.1f}%")
        print("\n--- PHASE 6: MONTE CARLO BOOTSTRAP (1000 trials) ---")
        print(f"PF 95% Confidence Interval: [{m['ci_lower']:.2f}, {m['ci_upper']:.2f}]")
        print(f"Probability PF > 1.2:       {m['prob_pf_12']:.1f}%\n")


if __name__ == "__main__":
    main()
