# train_lgbm.py
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from lightgbm import LGBMClassifier
from live_inference import add_upgraded_features, FEATURE_COLUMNS
from src.data_ingestion.market_data import fetch_historical_data, apply_dynamic_triple_barrier

def train():
    print("Training LightGBM model on upgraded features...")
    # Diverse set for the base model
    tickers = ["MSFT", "AAPL", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "BRK-B", "JPM", "AMD"]
    all_X = []
    all_y = []
    
    spy_df = yf.download('SPY', period='3y', interval='1d', progress=False)
    vix_df = yf.download('^VIX', period='3y', interval='1d', progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.droplevel(1)

    try:
        scaler = joblib.load("latest_scaler.joblib")
    except:
        print("Error: latest_scaler.joblib not found. Run train.py or regenerate_scaler.py first.")
        return

    for t in tickers:
        try:
            print(f"Fetching and processing {t}...")
            df = fetch_historical_data(t, "2021-01-01", pd.Timestamp.now().strftime("%Y-%m-%d"))
            df = add_upgraded_features(df, spy_df, vix_df)
            df = apply_dynamic_triple_barrier(df)
            
            X = df[FEATURE_COLUMNS]
            y = df['target_signal']
            
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Error processing {t}: {e}")

    if not all_X:
        print("No training data collected.")
        return

    X_train = pd.concat(all_X)
    y_train = pd.concat(all_y)

    X_train_scaled = scaler.transform(X_train)

    print(f"Fitting LightGBM on {len(X_train)} samples with {X_train.shape[1]} features...")
    lgbm = LGBMClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        objective='multiclass', 
        num_class=3,
        random_state=42,
        verbose=-1
    )
    lgbm.fit(X_train_scaled, y_train)
    
    joblib.dump(lgbm, "lgbm_agent.joblib")
    print("LightGBM model saved to lgbm_agent.joblib")

if __name__ == "__main__":
    train()
