import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.isotonic import IsotonicRegression
from src.execution.live_inference import add_upgraded_features, FEATURE_COLUMNS
from src.data_ingestion.market_data import fetch_historical_data

def calibrate():
    print("Fetching validation data (2024-01-01 to 2024-06-30)...")
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "XOM"]
    
    spy_df = yf.download('SPY', start="2023-01-01", end="2024-07-01", interval='1d', progress=False)
    vix_df = yf.download('^VIX', start="2023-01-01", end="2024-07-01", interval='1d', progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.droplevel(1)
    
    val_data = []
    val_targets_buy = []
    val_targets_sell = []
    
    for t in tickers:
        try:
            df = fetch_historical_data(t, start_date="2023-01-01", end_date="2024-06-30")
            df = add_upgraded_features(df, spy_df, vix_df)
            
            # Using 5-day future returns for labels
            future_ret = df['Close'].shift(-5) / df['Close'] - 1
            # Class 2: BUY (> 2%)
            df['target_buy'] = (future_ret > 0.02).astype(int)
            # Class 0: SELL (< -2%)
            df['target_sell'] = (future_ret < -0.02).astype(int)
            
            df = df.iloc[:-10] # drop lookahead NaNs
            
            # Filter just the validation period to avoid training leakage
            df_val = df[(df.index >= "2024-01-01") & (df.index <= "2024-06-30")]
            
            if len(df_val) > 0:
                val_data.append(df_val[FEATURE_COLUMNS])
                val_targets_buy.append(df_val['target_buy'])
                val_targets_sell.append(df_val['target_sell'])
        except Exception as e:
            print(f"Error on {t}: {e}")
            
    if not val_data:
        print("No validation data found.")
        return
        
    X_val_df = pd.concat(val_data)
    y_val_buy = pd.concat(val_targets_buy).values
    y_val_sell = pd.concat(val_targets_sell).values
    
    scaler = joblib.load("artifacts/latest_scaler.joblib")
    X_val = scaler.transform(X_val_df)
    
    # Load Models
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("artifacts/xgb_ensemble.json")
    xgb_probs_sell = xgb_model.predict_proba(X_val)[:, 0]
    xgb_probs_buy = xgb_model.predict_proba(X_val)[:, 2]
    
    lgbm_model = joblib.load("artifacts/lgbm_agent.joblib")
    lgbm_probs_sell = lgbm_model.predict_proba(X_val)[:, 0]
    lgbm_probs_buy = lgbm_model.predict_proba(X_val)[:, 2]
    
    # Calibrate BUY
    xgb_calibrator_buy = IsotonicRegression(out_of_bounds='clip')
    xgb_calibrator_buy.fit(xgb_probs_buy, y_val_buy)
    
    lgbm_calibrator_buy = IsotonicRegression(out_of_bounds='clip')
    lgbm_calibrator_buy.fit(lgbm_probs_buy, y_val_buy)
    
    # Calibrate SELL
    xgb_calibrator_sell = IsotonicRegression(out_of_bounds='clip')
    xgb_calibrator_sell.fit(xgb_probs_sell, y_val_sell)
    
    lgbm_calibrator_sell = IsotonicRegression(out_of_bounds='clip')
    lgbm_calibrator_sell.fit(lgbm_probs_sell, y_val_sell)
    
    # Save
    joblib.dump({"buy": xgb_calibrator_buy, "sell": xgb_calibrator_sell}, 'artifacts/xgb_calibrator.joblib')
    joblib.dump({"buy": lgbm_calibrator_buy, "sell": lgbm_calibrator_sell}, 'artifacts/lgbm_calibrator.joblib')
    
    print(f"XGB before calibration: avg_prob_buy={np.mean(xgb_probs_buy):.2f}")
    print(f"XGB after calibration:  avg_prob_buy={np.mean(xgb_calibrator_buy.predict(xgb_probs_buy)):.2f}")
    print(f"LGBM before calibration: avg_prob_buy={np.mean(lgbm_probs_buy):.2f}")
    print(f"LGBM after calibration:  avg_prob_buy={np.mean(lgbm_calibrator_buy.predict(lgbm_probs_buy)):.2f}")

if __name__ == '__main__':
    calibrate()
