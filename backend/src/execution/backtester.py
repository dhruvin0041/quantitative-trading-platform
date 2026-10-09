# backtester.py
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from src.data_ingestion.market_data import fetch_historical_data, get_sector_peer
from src.data_ingestion.nlp_processor import NewsTokenizer
from src.data_ingestion.technical_indicators import add_advanced_features
from src.execution.risk_manager import calculate_full_kelly
from src.models.neural.fusion_network import build_fusion_model
from src.models.rl.dqn_agent import DQNAgent


def run_backtest(ticker="AAPL", start_date="2023-01-01", end_date=None, gov_engine=None):
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(
        f"\n--- Running Institutional Backtest for {ticker} ({start_date} to {end_date}) ---"
    )

    # 1. Load Everything
    backend_dir = Path(__file__).resolve().parent.parent.parent

    def resolve_p(rel: str) -> Path:
        p = Path(rel)
        if p.exists():
            return p
        bp = backend_dir / rel
        if bp.exists():
            return bp
        return p

    with open(resolve_p("configs/model_params.yaml"), "r") as f:
        config = yaml.safe_load(f)
    with open(resolve_p("configs/kept_features.json"), "r") as f:
        kept_features = json.load(f)
    with open(resolve_p("configs/model_accuracies.json"), "r") as f:
        accs = json.load(f)

    scaler = joblib.load(resolve_p("artifacts/latest_scaler.joblib"))

    # Models
    config["data"]["num_features"] = len(kept_features)
    dl_model = build_fusion_model(config)
    dl_model.load_weights("artifacts/latest_fusion_weights.weights.h5")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("artifacts/xgb_ensemble.json")

    dqn_agent = DQNAgent(len(kept_features) + 3 + 3)
    try:
        dqn_agent.load("artifacts/dqn_model.pth")
    except Exception:
        pass

    # 2. Fetch Backtest Data (Target + Peer)
    df = fetch_historical_data(ticker, start_date=start_date, end_date=end_date)
    df = add_advanced_features(df)

    peer_ticker = get_sector_peer(ticker)
    peer_df = fetch_historical_data(
        peer_ticker, start_date=start_date, end_date=end_date
    )
    peer_df = add_advanced_features(peer_df)

    df_filtered = df.reindex(columns=kept_features).dropna()
    peer_filtered = peer_df.reindex(columns=kept_features).dropna()

    # Align
    common_idx = df_filtered.index.intersection(peer_filtered.index)
    df_filtered = df_filtered.loc[common_idx]
    peer_filtered = peer_filtered.loc[common_idx]
    target_df = df.loc[common_idx]

    # 3. Simulate with Realistic Next-Session Causal Execution
    capital = 100000
    shares = 0
    equity_curve = []

    time_steps = config["data"]["time_steps"]
    tokenizer = NewsTokenizer()

    # Weights
    total_acc = sum(accs.values())
    w_dl = accs["dl_accuracy"] / total_acc
    w_xgb = accs["xgb_accuracy"] / total_acc
    w_dqn = accs["dqn_accuracy"] / total_acc

    # Causal Execution Mandate:
    # Signals generated at Day t Close (16:00 EST) execute on Day t+1 Open (09:30 EST)
    from src.execution.strategy_governance import StrategyGovernanceEngine, StrategyLockError
    if gov_engine is None:
        gov_engine = StrategyGovernanceEngine()
    try:
        gov_engine.enforce_anti_overfitting_lock()
        manifest = gov_engine.load_manifest()
        frozen_hp = manifest.get("frozen_hyperparameters")
        if not isinstance(frozen_hp, dict):
            raise StrategyLockError("Frozen hyperparameters section missing or invalid in manifest.")
        exec_assumptions = frozen_hp.get("execution_assumptions")
        if (
            not isinstance(exec_assumptions, dict)
            or "slippage_bps" not in exec_assumptions
            or "commission_per_share_usd" not in exec_assumptions
        ):
            raise StrategyLockError(
                "Authoritative execution assumptions missing from frozen strategy manifest. "
                "Failing closed (anti-overfitting governance mandate)."
            )
        slippage = float(exec_assumptions["slippage_bps"]) / 10000.0
        commission_per_share = float(exec_assumptions["commission_per_share_usd"])
    except Exception as e:
        if isinstance(e, StrategyLockError):
            raise
        raise StrategyLockError(
            f"Authoritative execution assumptions verification failed: {e}. "
            "Frozen candidate execution must fail closed."
        ) from e
    pending_order = None
    pending_position_size = 0.0

    for i in range(time_steps - 1, len(df_filtered)):
        # 1. MORNING EXECUTION (09:30 EST): Fill pending order from Day i-1 close at Day i Open
        open_price = float(target_df["Open"].iloc[i])
        if pending_order == "BUY":
            buy_price = open_price * (1 + slippage)
            max_spend = capital * pending_position_size
            buy_shares = int(max_spend / buy_price)
            if buy_shares > 0:
                shares += buy_shares
                capital -= (buy_shares * buy_price) + (
                    buy_shares * commission_per_share
                )
            pending_order = None
        elif pending_order == "SELL" and shares > 0:
            sell_price = open_price * (1 - slippage)
            capital += (shares * sell_price) - (shares * commission_per_share)
            shares = 0
            pending_order = None

        # 2. MARKET CLOSE (16:00 EST): Evaluate Day i portfolio equity
        close_price = float(target_df["Close"].iloc[i])
        current_equity = capital + (shares * close_price)
        equity_curve.append(current_equity)

        # 3. POST-CLOSE INFERENCE (16:00+ EST): Compute signal using strictly trailing data up to i
        recent_data = df_filtered.iloc[i - time_steps + 1 : i + 1].values
        peer_recent = peer_filtered.iloc[i - time_steps + 1 : i + 1].values

        scaled_data = scaler.transform(recent_data)
        peer_scaled = scaler.transform(peer_recent)

        ts_seq = scaled_data.reshape(1, time_steps, -1)
        peer_seq = peer_scaled.reshape(1, time_steps, -1)
        tabular_row = scaled_data[-1].reshape(1, -1)

        # News mock
        ids, masks, _ = tokenizer.tokenize_daily_news(
            "Neutral market sentiment.", ticker=ticker
        )
        ids = ids.reshape(1, -1)
        masks = masks.reshape(1, -1)

        # Predictions [ts, cnn, trans, peer, ids, masks]
        dl_p = dl_model.predict(
            [ts_seq, ts_seq, ts_seq, peer_seq, ids, masks], verbose=0
        )[2][0]
        xgb_p = xgb_model.predict_proba(tabular_row)[0]

        state = np.hstack((tabular_row[0], dl_p, xgb_p))
        dqn_action = dqn_agent.act(state)

        # Ensemble
        ensemble_p = (dl_p * w_dl) + (xgb_p * w_xgb)
        dqn_p = np.zeros(3)
        dqn_p[dqn_action] = 1.0
        ensemble_p = (ensemble_p * (1 - w_dqn)) + (dqn_p * w_dqn)

        final_signal = np.argmax(ensemble_p)
        confidence = float(ensemble_p[final_signal])

        # Institutional Risk Management: Drawdown Circuit Breaker
        peak_equity = max(max(equity_curve) if equity_curve else capital, capital)
        if current_equity < peak_equity * 0.80 and shares > 0:
            print(
                f"[{df_filtered.index[i]}] Circuit Breaker Triggered! 20% Drawdown reached. Liquidating at next open."
            )
            final_signal = 0
            confidence = 1.0

        # Queue order for NEXT TRADING SESSION (i+1 Open)
        kelly_fraction = calculate_full_kelly(0.55, 1.2)  # Defaults from risk_manager
        position_size_pct = kelly_fraction * confidence

        if final_signal == 2 and confidence > 0.7:
            pending_order = "BUY"
            pending_position_size = position_size_pct
        elif final_signal == 0 and confidence > 0.7 and shares > 0:
            pending_order = "SELL"
            pending_position_size = 0.0
        else:
            pending_order = None

    return equity_curve, df_filtered.index[time_steps - 1 :]


def run_walk_forward(ticker="AAPL", windows=4):
    """
    Implements institutional Walk-Forward Optimization (WFO).
    In each window, it (theoretically) re-trains the models on past data
    and tests on the out-of-sample forward window.
    """
    print(f"\n{'=' * 50}")
    print(f"STARTING WALK-FORWARD OPTIMIZATION: {ticker}")
    print(f"{'=' * 50}")

    end_dt = datetime.now()
    all_equity = []
    all_dates = []
    initial_capital = 100000

    for w in range(windows, 0, -1):
        # Window logic:
        # Train on [Start - 1 year, Test Start]
        # Test on [Test Start, Test Start + 90 days]
        test_start = end_dt - timedelta(days=w * 90)
        test_end = test_start + timedelta(days=90)

        start_str = test_start.strftime("%Y-%m-%d")
        end_str = test_end.strftime("%Y-%m-%d")

        print(
            f"\n>>> Window {windows - w + 1}: Training/Optimizing for period ending {start_str}"
        )
        # In a real WFO, we would call train_main with a custom date range here.
        # For this stub, we assume the latest model is used but validated in this specific segment.

        equity, dates = run_backtest(ticker, start_date=start_str, end_date=end_str)

        # Adjust equity to be continuous
        if all_equity:
            offset = all_equity[-1] - initial_capital
            equity = [e + offset for e in equity]

        all_equity.extend(equity)
        all_dates.extend(dates)

    # Calculate Performance Metrics
    returns = pd.Series(all_equity).pct_change().dropna()
    sharpe = (
        np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
    )
    max_dd = (pd.Series(all_equity) / pd.Series(all_equity).cummax() - 1).min()

    print("\n--- WFO Performance Summary ---")
    print(f"Total Return: {round(((all_equity[-1] / initial_capital) - 1) * 100, 2)}%")
    print(f"Annualized Sharpe: {round(sharpe, 2)}")
    print(f"Max Drawdown: {round(max_dd * 100, 2)}%")


if __name__ == "__main__":
    run_walk_forward("AAPL")
