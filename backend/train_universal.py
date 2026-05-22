# train_universal.py
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import requests
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from live_inference import add_upgraded_features, FEATURE_COLUMNS
from src.data_ingestion.market_data import apply_dynamic_triple_barrier, fetch_historical_data
from src.models.lgbm_agent import train_lgbm_agent


# --- 1. CONFIGURATION LOADING ---
def load_universal_params():
    """Loads the universal AI-discovered parameters."""
    param_path = "configs/optimized_params_UNIVERSAL.json"
    if os.path.exists(param_path):
        with open(param_path, "r") as file:
            print(">>> Loading UNIVERSAL optimized parameters...")
            return json.load(file)["best_params"]
    else:
        print(">>> No universal parameters found. Using robust defaults.")
        return {
            "tp_atr_multiplier": 2.5,
            "sl_atr_multiplier": 1.5,
            "horizon": 10,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 1.0,
            "min_child_weight": 5,
        }


# --- 2. THE DATA INGESTOR ---
def get_sp500_tickers(limit=15): # Further reduced for universal speed verification
    """Scrapes the S&P 500 and returns a subset."""
    print(f"Fetching Top {limit} S&P 500 Tickers...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        html_content = requests.get(url, headers=headers).text
        table = pd.read_html(html_content)[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers[:limit]
    except Exception as e:
        print(f"Failed to scrape S&P 500: {e}")
        return ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]


def build_panel_dataset(tickers, opt_params, start="2018-01-01", end=None):
    if end is None:
        from datetime import datetime
        end = datetime.now().strftime("%Y-%m-%d")

    print(f"Building Universal Panel Dataset for {len(tickers)} assets...")
    
    spy_df = yf.download('SPY', start=start, end=end, progress=False)
    vix_df = yf.download('^VIX', start=start, end=end, progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.droplevel(1)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.droplevel(1)

    all_data = []
    for i, ticker in enumerate(tickers):
        print(f"[{i + 1}/{len(tickers)}] Processing {ticker}...")
        try:
            df = fetch_historical_data(ticker, start_date=start, end_date=end)
            df = add_upgraded_features(df, spy_df, vix_df)
            df = apply_dynamic_triple_barrier(
                df,
                tp_atr_multiplier=opt_params["tp_atr_multiplier"],
                sl_atr_multiplier=opt_params["sl_atr_multiplier"],
                horizon=opt_params["horizon"],
            )
            df["Ticker"] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Failed on {ticker}: {e}")

    if not all_data:
        raise ValueError("No data collected for any tickers.")

    master_df = pd.concat(all_data)
    print(f"Panel Dataset Complete! Total Rows: {master_df.shape[0]}")
    return master_df


# --- 3. THE TRAINING SEQUENCE ---
def train_universal_engine():
    # 1. Load Params and Build Data
    opt_params = load_universal_params()
    tickers = get_sp500_tickers(limit=15)
    master_df = build_panel_dataset(tickers, opt_params)

    # Load golden features
    kept_features = FEATURE_COLUMNS
    with open("configs/kept_features.json", "w") as f:
        json.dump(kept_features, f)

    # 2. Global Scaling
    print("Fitting Global Scaler on Training Data...")
    scaler = StandardScaler()

    # Data Cleaning
    master_df = master_df.replace([np.inf, -np.inf], np.nan).dropna(subset=kept_features + ["target_signal"])
    X = master_df[kept_features]
    y = master_df["target_signal"].astype(int)

    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 3. Train Universal XGBoost
    print("Calculating Class Weights for Balance...")
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([class_weight_dict[yi] for yi in y_train])

    print("Training Universal Technical Brain (XGBoost)...")
    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=opt_params.get("n_estimators", 300),
        max_depth=opt_params.get("max_depth", 6),
        learning_rate=opt_params.get("learning_rate", 0.05),
        subsample=opt_params.get("subsample", 0.8),
        colsample_bytree=opt_params.get("colsample_bytree", 0.8),
        gamma=opt_params.get("gamma", 1.0),
        min_child_weight=opt_params.get("min_child_weight", 5),
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    print(f"XGBoost Accuracy -> Train: {xgb_model.score(X_train, y_train) * 100:.1f}% | Test: {xgb_model.score(X_test, y_test) * 100:.1f}%")

    # 4. Train Universal LightGBM
    print("Training Universal Technical Brain (LightGBM)...")
    train_lgbm_agent(X_train, y_train)

    # 5. Save the Universal Architecture
    print("Saving Universal Architecture...")
    xgb_model.save_model("xgb_ensemble.json")
    joblib.dump(scaler, "latest_scaler.joblib")

    # 6. Train Macro Regime Detector
    print("Calibrating Macro Kill-Switch (VIX Regimes)...")
    from src.models.regime_detector import train_macro_regime_model
    train_macro_regime_model()

    print("Project Hydra: Universal Brain Deployment Complete.")


if __name__ == "__main__":
    train_universal_engine()
