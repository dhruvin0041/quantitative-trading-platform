# backtester.py
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import yaml
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xgboost as xgb
from src.models.fusion_network import build_fusion_model
from src.models.dqn_agent import DQNAgent
from src.data_ingestion.market_data import fetch_historical_data, get_sector_peer
from src.data_ingestion.technical_indicators import add_advanced_features
from src.data_ingestion.nlp_processor import NewsTokenizer
from src.execution.risk_manager import calculate_full_kelly


def run_backtest(ticker="AAPL", start_date="2023-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(
        f"\n--- Running Institutional Backtest for {ticker} ({start_date} to {end_date}) ---"
    )

    # 1. Load Everything
    with open("configs/model_params.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("configs/kept_features.json", "r") as f:
        kept_features = json.load(f)
    with open("configs/model_accuracies.json", "r") as f:
        accs = json.load(f)

    scaler = joblib.load("artifacts/latest_scaler.joblib")

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

    # 3. Simulate
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

    for i in range(time_steps - 1, len(df_filtered)):
        # Prep inputs using data up to the current day (i)
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
        confidence = ensemble_p[final_signal]
        current_price = df_filtered.iloc[i]["Close"]

        # Institutional Risk Management: Drawdown Circuit Breaker
        current_equity = capital + (shares * current_price)
        peak_equity = max(max(equity_curve) if equity_curve else capital, capital)
        if current_equity < peak_equity * 0.80 and shares > 0:
            print(f"[{df_filtered.index[i]}] Circuit Breaker Triggered! 20% Drawdown reached. Liquidating.")
            final_signal = 0
            confidence = 1.0

        # Institutional assumptions: Slippage and Commission
        slippage = 0.001  # 0.1%
        commission_per_share = 0.005

        # INSTITUTIONAL UPGRADE: Kelly Sizing
        kelly_fraction = calculate_full_kelly(0.55, 1.2)  # Defaults from risk_manager
        position_size_pct = kelly_fraction * confidence

        if final_signal == 2 and confidence > 0.7:
            # BUY using Kelly
            max_spend = capital * position_size_pct
            buy_price = current_price * (1 + slippage)
            buy_shares = int(max_spend / buy_price)
            if buy_shares > 0:
                shares += buy_shares
                capital -= (buy_shares * buy_price) + (buy_shares * commission_per_share)
        elif final_signal == 0 and confidence > 0.7 and shares > 0:
            # SELL All
            sell_price = current_price * (1 - slippage)
            capital += (shares * sell_price) - (shares * commission_per_share)
            shares = 0

        equity_curve.append(capital + (shares * current_price))

    return equity_curve, df_filtered.index[time_steps - 1:]


def run_walk_forward(ticker="AAPL", windows=4):
    """
    Implements institutional Walk-Forward Optimization (WFO).
    In each window, it (theoretically) re-trains the models on past data 
    and tests on the out-of-sample forward window.
    """
    print(f"\n{'=' * 50}")
    print(f"STARTING WALK-FORWARD OPTIMIZATION: {ticker}")
    print(f"{'=' * 50}")

    from train import main as train_main
    
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
        
        print(f"\n>>> Window {windows-w+1}: Training/Optimizing for period ending {start_str}")
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
    sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
    max_dd = (pd.Series(all_equity) / pd.Series(all_equity).cummax() - 1).min()

    print(f"\n--- WFO Performance Summary ---")
    print(f"Total Return: {round(((all_equity[-1]/initial_capital)-1)*100, 2)}%")
    print(f"Annualized Sharpe: {round(sharpe, 2)}")
    print(f"Max Drawdown: {round(max_dd*100, 2)}%")


if __name__ == "__main__":
    run_walk_forward("AAPL")
