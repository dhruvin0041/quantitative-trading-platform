# train_universal.py
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.data_ingestion.technical_indicators import add_advanced_features
from src.data_ingestion.market_data import apply_dynamic_triple_barrier

# --- 1. CONFIGURATION LOADING ---
def load_universal_params():
    """Loads the universal AI-discovered parameters."""
    param_path = 'configs/optimized_params_UNIVERSAL.json'
    if os.path.exists(param_path):
        with open(param_path, 'r') as file:
            print(">>> Loading UNIVERSAL optimized parameters...")
            return json.load(file)['best_params']
    else:
        print(">>> No universal parameters found. Using robust defaults.")
        return {
            'tp_atr_multiplier': 2.5, 'sl_atr_multiplier': 1.5, 'horizon': 10,
            'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 300,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 1.0, 'min_child_weight': 5
        }

# --- 2. THE DATA INGESTOR ---
def get_sp500_tickers(limit=50):
    """Scrapes the S&P 500 and returns a subset."""
    print(f"Fetching Top {limit} S&P 500 Tickers...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        html_content = requests.get(url, headers=headers).text
        table = pd.read_html(html_content)[0]
        tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers[:limit]
    except Exception as e:
        print(f"Failed to scrape S&P 500: {e}")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]

def build_panel_dataset(tickers, opt_params, start="2018-01-01", end=None):
    if end is None:
        from datetime import datetime
        end = datetime.now().strftime('%Y-%m-%d')
        
    print(f"Building Universal Panel Dataset for {len(tickers)} assets...")
    all_data = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Processing {ticker}...")
        try:
            # 1. Download
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty or len(df) < 500:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 2. Add Features
            df = add_advanced_features(df.copy())
            
            # 3. Dynamic Triple Barrier Labeling (Using Universal Params)
            df = apply_dynamic_triple_barrier(
                df,
                tp_atr_multiplier=opt_params['tp_atr_multiplier'],
                sl_atr_multiplier=opt_params['sl_atr_multiplier'],
                horizon=opt_params['horizon']
            )
            
            # 4. Add Ticker label and store
            df['Ticker'] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Failed on {ticker}: {e}")
            
    if not all_data:
        raise ValueError("No data collected for any tickers.")
        
    master_df = pd.concat(all_data)
    print(f"Panel Dataset Complete! Total Rows: {len(master_df)}")
    return master_df

# --- 3. THE TRAINING SEQUENCE ---
def train_universal_engine():
    # 1. Load Params and Build Data
    opt_params = load_universal_params()
    tickers = get_sp500_tickers(limit=50) 
    master_df = build_panel_dataset(tickers, opt_params)
    
    # Load golden features
    with open('configs/kept_features.json', 'r') as f:
        kept_features = json.load(f)
        
    X = master_df[kept_features]
    y = master_df['target_signal']
    
    # 2. Global Scaling (Data Leakage Fix)
    print("Fitting Global Scaler on Training Data...")
    scaler = StandardScaler()
    
    # Temporal split is better: Split by index since concat preserved order per ticker
    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx].astype(int), y.iloc[split_idx:].astype(int)
    
    # Fit scaler ONLY on training data to prevent future data leakage
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 3. Train Universal XGBoost with AI-optimized params
    print("Calculating Class Weights for Balance...")
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([class_weight_dict[y] for y in y_train])
    print(f"Weights Applied: {class_weight_dict}")

    print("Training Universal Technical Brain (XGBoost)...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=opt_params.get('n_estimators', 300),
        max_depth=opt_params.get('max_depth', 6),
        learning_rate=opt_params.get('learning_rate', 0.05),
        subsample=opt_params.get('subsample', 0.8),
        colsample_bytree=opt_params.get('colsample_bytree', 0.8),
        gamma=opt_params.get('gamma', 1.0),
        min_child_weight=opt_params.get('min_child_weight', 5),
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    print(f"XGBoost Accuracy -> Train: {xgb_model.score(X_train, y_train)*100:.1f}% | Test: {xgb_model.score(X_test, y_test)*100:.1f}%")
    
    # 4. Train Universal Meta-Learner
    print("Training Universal Meta-Learner (Risk Desk)...")
    train_probs = xgb_model.predict_proba(X_train)
    X_meta_train = np.hstack((X_train, train_probs))
    
    # Meta-Target: Did XGBoost get it right?
    xgb_train_preds = xgb_model.predict(X_train)
    meta_targets = (xgb_train_preds == y_train).astype(int)
    
    meta_model = RandomForestClassifier(n_estimators=100, max_depth=4, n_jobs=-1, class_weight='balanced')
    meta_model.fit(X_meta_train, meta_targets)
    
    # 5. Save the Universal Architecture
    print("Saving Universal Architecture...")
    xgb_model.save_model("xgb_ensemble.json")
    joblib.dump(meta_model, 'meta_model.joblib')
    joblib.dump(scaler, 'latest_scaler.joblib')
    
    # 6. Train Macro Regime Detector (Kill-Switch)
    print("Calibrating Macro Kill-Switch (VIX Regimes)...")
    from src.models.regime_detector import train_macro_regime_model
    train_macro_regime_model()
    
    print("Project Hydra: Universal Brain Deployment Complete.")

if __name__ == "__main__":
    train_universal_engine()
