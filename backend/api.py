# api.py
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from dotenv import load_dotenv
load_dotenv()
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
from src.execution.risk_manager import get_position_sizing, calculate_beta, calculate_jensens_alpha, detect_stampede_risk
from src.data_ingestion.nlp_processor import NewsTokenizer, GeminiAnalyzer
from src.data_ingestion.alternative_data import PhysicalEdgeAnalyzer
from src.data_ingestion.supply_chain_graph import SupplyChainGraph
from src.agents.orchestrator import InstitutionalOrchestrator
from src.execution.smart_router import PredictiveSmartRouter
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

# NEW: Institutional Qualitative Analyzer
gemini_analyzer = GeminiAnalyzer()

# NEW: Elite SOTA 2026 Components
physical_edge = PhysicalEdgeAnalyzer()
dependency_graph = SupplyChainGraph()
orchestrator = InstitutionalOrchestrator()
smart_router = PredictiveSmartRouter()

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
    spy_df = yf.download("SPY", period="60d", progress=False)
    ticker_df = yf.download(ticker, period="60d", progress=False)
    beta = calculate_beta(ticker_df['Close'], spy_df['Close'])
    
    # 2. Elite Data Layer
    physical_alpha = physical_edge.get_physical_alpha_vector(ticker)
    dependency_graph.build_proxy_graph(ticker)
    propagation_risk = dependency_graph.calculate_propagation_risk(ticker)
    
    # 3. Alpha Generation
    tokenizer = NewsTokenizer(max_length=updated_config['data']['max_seq_length'])
    input_ids, attention_masks, news_text = fetch_live_news(ticker, tokenizer, updated_config) 
    qual_score, qual_reason = gemini_analyzer.analyze_fundamental_alpha(news_text, ticker)
    
    dl_preds = lstm_model.predict(
        x=[ts_sequence, ts_sequence, ts_sequence, peer_sequence, input_ids, attention_masks], 
        verbose=0
    )[2][0]
    xgb_preds = xgb_model.predict_proba(tabular_row)[0]
    
    # 4. Multi-Agent Consensus Logic
    total_acc = sum(accs.values())
    w_dl = accs['dl_accuracy'] / total_acc
    w_xgb = accs['xgb_accuracy'] / total_acc
    
    ensemble_p = (dl_preds * w_dl) + (xgb_preds * w_xgb)
    # Blend in Qualitative Score (10% weight)
    ensemble_p[2] = (ensemble_p[2] * 0.9) + (max(0, qual_score) * 0.1) if qual_score > 0 else ensemble_p[2]
    ensemble_p[0] = (ensemble_p[0] * 0.9) + (abs(min(0, qual_score)) * 0.1) if qual_score < 0 else ensemble_p[0]
    
    # Institutional Risk & Alpha Decomposition
    risk_metrics = get_position_sizing(float(np.max(ensemble_p)))
    alpha_metric = calculate_jensens_alpha(ticker_df['Close'].pct_change(), spy_df['Close'].pct_change(), beta)
    stampede = detect_stampede_risk(np.std(dl_preds), float(np.max(ensemble_p)))
    
    # Run Agentic Orchestration
    mesh_response = orchestrator.run_consensus(ensemble_p, {
        "beta": beta, "suggested_allocation": risk_metrics['suggested_allocation'],
        "hedge_ratio_spy": f"-{round(beta * 100, 1)}%"
    })
    
    # 5. Execution Optimization
    routing = smart_router.predict_venue_liquidity(ticker)
    
    # 6. Explainable AI (XAI) Log Generation
    xai_log = f"Decision primarily driven by {orchestrator.consensus_status}. "
    xai_log += f"Physical supply risk at {round(physical_alpha['supply_chain_disruption_index']*100)}%. "
    xai_log += f"Qualitative Alpha: {qual_reason}"

    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    final_action = signal_map[mesh_response['final_action_idx']]
    if stampede['is_crowded']: final_action = f"SCALE_BACK ({final_action})"

    return {
        "ticker": ticker,
        "action": final_action,
        "confidence": f"{round(float(np.max(ensemble_p)) * 100, 1)}%",
        "agent_consensus": mesh_response,
        "institutional_metrics": {
            "jensens_alpha": round(alpha_metric, 4),
            "beta": round(beta, 2),
            "propagation_risk": round(propagation_risk, 3),
            "stampede_risk": stampede
        },
        "physical_edge": physical_alpha,
        "smart_routing": routing,
        "xai_reasoning": xai_log,
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