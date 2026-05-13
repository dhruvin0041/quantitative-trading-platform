# api.py
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import pandas as pd
import requests
import logging
from datetime import datetime

# --- Setup Logging ---
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/api_{datetime.now().strftime('%Y%m')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import your inference functions
from live_inference import fetch_live_data, fetch_live_news, synthesize_report, load_config
from src.models.fusion_network import build_fusion_model
from src.models.dqn_agent import DQNAgent
import xgboost as xgb
import joblib
import numpy as np
import torch

app = FastAPI()

# Allow Next.js (usually localhost:3000) to fetch data from Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Loading System Architectures into RAM...")
config = load_config()

with open('configs/kept_features.json', 'r') as f:
    kept_features_list = json.load(f)
    actual_num_features = len(kept_features_list)

# Load Accuracies for weighting
try:
    with open('configs/model_accuracies.json', 'r') as f:
        accs = json.load(f)
except:
    accs = {"dl_accuracy": 0.5, "xgb_accuracy": 0.5, "dqn_accuracy": 0.5}

config['data']['num_features'] = actual_num_features
logger.info(f"Initializing 5-Model Ensemble with {actual_num_features} features...")

# Load Models
lstm_model = build_fusion_model(config)
lstm_model.load_weights('latest_fusion_weights.weights.h5')

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_ensemble.json")

dqn_agent = DQNAgent(actual_num_features + 3 + 3)
try: dqn_agent.load("dqn_model.pth")
except: logger.warning("DQN model not found. Using random agent.")

# Load Macro Kill-Switch
kill_switch_data = joblib.load('macro_kill_switch.joblib')
regime_model = kill_switch_data['model']
panic_id = kill_switch_data['panic_cluster']

# ==========================================
# 1. MULTI-ASSET UNIVERSE ENDPOINT
# ==========================================
@app.get("/universe")
def get_stock_universe():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        html_content = requests.get(url, headers=headers).text
        table = pd.read_html(html_content)[0]
        tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
        companies = table['Security'].tolist()
        return {"universe": [{"ticker": t, "name": c} for t, c in zip(tickers, companies)]}
    except:
        return {"universe": [{"ticker": "AAPL", "name": "Apple Inc."}]}

# ==========================================
# 2. HYBRID INFERENCE ENGINE
# ==========================================
@app.get("/predict")
def get_prediction(ticker: str = "AAPL"):
    # 1. Fetch live data (Now includes Peer Sequence)
    ts_sequence, peer_sequence, tabular_row, current_price, updated_config = fetch_live_data(ticker, config)
    
    # Macro check
    import yfinance as yf
    vix_df = yf.download("^VIX", period="5d", progress=False)
    regime = regime_model.predict([[vix_df['Close'].iloc[-1], (vix_df['Close'].iloc[-1]/vix_df['Close'].iloc[-5])-1]])[0]
    is_panic = (regime == panic_id)

    from src.data_ingestion.nlp_processor import NewsTokenizer
    tokenizer = NewsTokenizer(max_length=updated_config['data']['max_seq_length'])
    input_ids, attention_masks, news_text = fetch_live_news(ticker, tokenizer, updated_config) 
    
    # 2. Get Predictions from all 5 models
    # dl_preds contains [direction, range, signal]
    # Updated model input signature: [ts, cnn, trans, peer, ids, masks]
    dl_preds = lstm_model.predict(
        x=[ts_sequence, ts_sequence, ts_sequence, peer_sequence, input_ids, attention_masks], 
        verbose=0
    )[2][0]
    
    xgb_preds = xgb_model.predict_proba(tabular_row)[0]
    
    # DQN State: Features + Predictions
    state = np.hstack((tabular_row[0], dl_preds, xgb_preds))
    dqn_action = dqn_agent.act(state)
    dqn_p = np.zeros(3)
    dqn_p[dqn_action] = 1.0
    
    # 3. Dynamic Weighted Voting
    total_acc = sum(accs.values())
    w_dl = accs['dl_accuracy'] / total_acc
    w_xgb = accs['xgb_accuracy'] / total_acc
    w_dqn = accs['dqn_accuracy'] / total_acc
    
    ensemble_p = (dl_preds * w_dl) + (xgb_preds * w_xgb)
    # Blend in DQN
    ensemble_p = (ensemble_p * (1 - w_dqn)) + (dqn_p * w_dqn)
    
    final_signal_idx = int(np.argmax(ensemble_p))
    confidence = float(ensemble_p[final_signal_idx])
    
    # NEW: Institutional Risk Management (Kelly Sizing)
    risk_metrics = get_position_sizing(confidence)
    
    # Dynamic Confidence threshold fallback
    threshold = 0.7
    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    
    action = signal_map[final_signal_idx] if confidence > threshold else "HOLD"
    if is_panic: action = "SKIP (PANIC)"

    return {
        "ticker": ticker,
        "action": action,
        "confidence": f"{round(confidence * 100, 1)}%",
        "risk_management": risk_metrics,
        "breakdown": {
            "dl_suggests": signal_map[int(np.argmax(dl_preds))],
            "xgb_suggests": signal_map[int(np.argmax(xgb_preds))],
            "dqn_suggests": signal_map[dqn_action]
        },
        "price": current_price,
        "news": news_text
    }

    # ==========================================
    # 3. GENERATE HISTORICAL MARKERS (Batch Inference)
    # ==========================================
    from src.data_ingestion.market_data import fetch_historical_data
    from src.data_ingestion.technical_indicators import add_advanced_features
    from datetime import datetime, timedelta
    
    # NEW: Fetch 5 years of data
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    df = fetch_historical_data(ticker, start_date=start_date, end_date=datetime.now().strftime('%Y-%m-%d'))
    df_features = add_advanced_features(df.copy())
    
    # Fetch VIX for batch markers
    vix_batch = yf.download("^VIX", start=start_date, end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if isinstance(vix_batch.columns, pd.MultiIndex): vix_batch.columns = vix_batch.columns.get_level_values(0)
    vix_batch['VIX_ROC'] = vix_batch['Close'].pct_change(periods=5)
    vix_batch = vix_batch.dropna()

    # --- THE CLEAN FILTER ---
    # We must explicitly add the new visual columns to the list so they aren't dropped
    visual_cols = ['Ribbon_Fast', 'Ribbon_Slow', 'BB_120_Upper', 'BB_120_Lower']
    extended_features_list = kept_features_list + [col for col in visual_cols if col not in kept_features_list]
    
    df_features = df_features.reindex(columns=extended_features_list)
    df_features = df_features.dropna() 
    
    # Align
    common_idx = df_features.index.intersection(vix_batch.index)
    df_features = df_features.loc[common_idx]
    vix_batch = vix_batch.loc[common_idx]

    scaler = joblib.load('latest_scaler.joblib')
    # Use ONLY the original kept_features_list for scaling to match the model
    scaled_data = scaler.transform(df_features[kept_features_list]) 
    
    # --- EXPANDED CHART VIEW (5 YEARS) ---
    view_window = len(common_idx) # Show everything we fetched
    
    # We need the raw prices and bands for "Peak/Valley" detection
    df_recent_raw = df.loc[common_idx].tail(view_window)
    df_recent_features = add_advanced_features(df.loc[common_idx]).tail(view_window)
    
    last_window_scaled = scaled_data[-view_window:]
    dates_window = common_idx[-view_window:].strftime('%Y-%m-%d').tolist()
    
    historical_markers = []
    
    prices = df_recent_raw['Close'].values
    highs = df_recent_raw['High'].values
    lows = df_recent_raw['Low'].values
    
    # logic: SWING HIGH / SWING LOW DETECTION (Pivots)
    # A pivot is a point that is the extreme of its local neighborhood.
    window = 10 # Slightly wider window for a 5-year view to avoid clutter
    
    for i in range(window, len(prices) - window):
        action = "SKIP"
        
        # Check for Valley (Swing Low)
        if lows[i] == np.min(lows[i-window : i+window+1]):
            action = "BUY"
            
        # Check for Peak (Swing High)
        elif highs[i] == np.max(highs[i-window : i+window+1]):
            action = "SELL"
            
        if action != "SKIP":
            historical_markers.append({
                "time": dates_window[i],
                "action": action,
                "label": action,
                "probability": 100 
            })
    
    # ==========================================
    # 4. Package the Final JSON
    # ==========================================
    # Show the full 5 years on the chart
    df_chart = df.copy()
    
    # DATA ALIGNMENT FIX: Use index-based assignment instead of .values
    # df_chart and df_features might have different lengths due to dropna() in technical indicators
    df_chart['ribbon_upper'] = df_features['Ribbon_Fast']
    df_chart['ribbon_lower'] = df_features['Ribbon_Slow']
    df_chart['bb_upper'] = df_features['BB_120_Upper']
    df_chart['bb_lower'] = df_features['BB_120_Lower']
    
    df_chart = df_chart.reset_index()
    date_col = 'Date' if 'Date' in df_chart.columns else 'index'
    df_chart['time'] = df_chart[date_col].dt.strftime('%Y-%m-%d')
    df_chart = df_chart.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    
    # Drop rows where we don't have cloud data for a cleaner UI if desired, 
    # but lightweight-charts handles missing values if we are careful.
    # Here we'll keep all candles but cloud data will be NaN where missing.
    candles = df_chart[['time', 'open', 'high', 'low', 'close']].to_dict(orient='records')
    
    # Filter for clouds to only include where we have data
    df_cloud_json = df_chart.dropna(subset=['ribbon_upper', 'ribbon_lower', 'bb_upper', 'bb_lower'])
    clouds = df_cloud_json[['time', 'ribbon_upper', 'ribbon_lower', 'bb_upper', 'bb_lower']].to_dict(orient='records')
    
    return {
        "candles": candles,
        "clouds": clouds,
        "ai_report": report_dict["AI_QUANT_SYSTEM_REPORT"],
        "historical_markers": historical_markers
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)