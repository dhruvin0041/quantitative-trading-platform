import os
import json
import logging
import joblib
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
import asyncio
import time
import re
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Import inference functions
from live_inference import fetch_live_data, fetch_live_news, load_config, is_near_earnings
from src.models.fusion_network import build_fusion_model
from src.models.dqn_agent import DQNAgent
from src.execution.risk_manager import (
    get_position_sizing,
    calculate_beta,
    calculate_jensens_alpha,
    detect_stampede_risk,
)
from src.data_ingestion.nlp_processor import NewsTokenizer, GeminiAnalyzer
from src.data_ingestion.alternative_data import PhysicalEdgeAnalyzer
from src.data_ingestion.supply_chain_graph import SupplyChainGraph
from src.agents.orchestrator import InstitutionalOrchestrator
from src.execution.smart_router import PredictiveSmartRouter
from src.execution.reporting import ReportGenerator
from src.execution.paper_trading import PaperTradingEngine
from src.execution.broker import AlpacaBroker
from src.models.drift_monitor import DriftMonitor
from src.schemas import PredictResponse
from src.execution.performance_analyzer import PerformanceAnalyzer
from src.execution.alerts import AlertSystem
from src.data_ingestion.sector_mapper import SectorMapper
import yfinance as yf

# --- Structured Logging ---
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
        }
        return json.dumps(log_record)

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(f"logs/api_{datetime.now().strftime('%Y%m')}.log")
file_handler.setFormatter(JSONFormatter())
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(JSONFormatter())

logger.handlers = [file_handler, stream_handler]

# --- Environment & Config ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
load_dotenv()

# --- Security: API Key ---
API_KEY = os.getenv("API_KEY", "dev-secret-key-1234")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        logger.warning(f"Unauthorized access attempt with key: {api_key[:4]}...")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# --- Observability: Prometheus Metrics ---
from prometheus_client import REGISTRY
try:
    REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status_code'])
    REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'API request latency', ['endpoint'])
except ValueError:
    REQUEST_COUNT = REGISTRY._names_to_collectors['api_requests_total']
    REQUEST_LATENCY = REGISTRY._names_to_collectors['api_request_latency_seconds']

app = FastAPI(title="Hydra Terminal API", version="2.0.0")

# --- Security: Restricted CORS ---
FRONTEND_URLS = os.getenv("FRONTEND_URL", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# --- Simple In-Memory Rate Limiting ---
from collections import defaultdict
rate_limit_store = defaultdict(list)
RATE_LIMIT = 50 # requests
RATE_WINDOW = 60 # seconds

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()

    # Clean up old requests
    rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if now - t < RATE_WINDOW]

    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return Response(content="Rate limit exceeded", status_code=429)

    rate_limit_store[client_ip].append(now)

    # Track Metrics
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=response.status_code).inc()
    if request.url.path == "/predict":
        REQUEST_LATENCY.labels(endpoint="/predict").observe(duration)

    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class SimpleCache:
    def __init__(self):
        self._cache = {}
        self._lock = asyncio.Lock()
    async def set(self, key, value):
        async with self._lock:
            self._cache[key] = value
    async def get(self, key):
        async with self._lock:
            return self._cache.get(key)

api_cache = SimpleCache()
paper_engine = PaperTradingEngine()
broker = AlpacaBroker(
    api_key=os.getenv("ALPACA_API_KEY", "STUB"), 
    api_secret=os.getenv("ALPACA_SECRET_KEY", "STUB"), 
    base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
)
drift_monitor = DriftMonitor()
perf_analyzer = PerformanceAnalyzer()
alert_system = AlertSystem()
sector_mapper = SectorMapper()

def sanitize_ticker(ticker: str) -> str:
    """Institutional Sanitization: Only allow 1-5 alphanumeric chars or hyphens."""
    if not re.match(r"^[A-Z0-9-]{1,5}$", ticker.upper()):
        raise HTTPException(status_code=400, detail="Invalid Ticker Format")
    return ticker.upper()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"GLOBAL ERROR: {str(exc)}", exc_info=True)
    return Response(
        content=json.dumps({"detail": "Internal Institutional Error", "type": type(exc).__name__}),
        status_code=500,
        media_type="application/json"
    )

@app.get("/health")
async def health_check():
    return {
        "status": "HEALTHY",
        "models_loaded": lstm_model is not None and xgb_model is not None,
        "regime_model_ready": True,
        "timestamp": datetime.now().isoformat()
    }

# --- System Initialization ---
logger.info("Loading System Architectures into RAM...")
config = load_config()

with open("configs/kept_features.json", "r") as f:
    kept_features_list = json.load(f)
    actual_num_features = len(kept_features_list)

try:
    with open("configs/model_accuracies.json", "r") as f:
        accs = json.load(f)
except Exception:
    accs = {"dl_accuracy": 0.5, "xgb_accuracy": 0.5, "dqn_accuracy": 0.5}

config["data"]["num_features"] = actual_num_features

# Load Models
lstm_model = build_fusion_model(config)
try:
    lstm_model.load_weights("latest_fusion_weights.weights.h5")
except Exception as e:
    logger.warning(f"Could not load LSTM weights: {e}")

xgb_model = xgb.XGBClassifier()
try:
    xgb_model.load_model("xgb_ensemble.json")
except Exception as e:
    logger.warning(f"Could not load XGB ensemble: {e}")

lgbm_model = None
try:
    lgbm_model = joblib.load("lgbm_agent.joblib")
except Exception as e:
    logger.warning(f"Could not load LightGBM agent: {e}")

gemini_analyzer = GeminiAnalyzer()
physical_edge = PhysicalEdgeAnalyzer()
dependency_graph = SupplyChainGraph()
orchestrator = InstitutionalOrchestrator()
smart_router = PredictiveSmartRouter()
report_gen = ReportGenerator(kept_features_list)

try:
    calibrator = joblib.load("calibrator.joblib")
except Exception:
    calibrator = None


# ==========================================
# 1. MULTI-ASSET UNIVERSE ENDPOINT
# ==========================================
@app.get("/universe", dependencies=[Depends(verify_api_key)])
async def get_stock_universe():
    def fetch_universe():
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {"User-Agent": "Mozilla/5.0"}
            html_content = requests.get(url, headers=headers, timeout=10).text
            table = pd.read_html(html_content)[0]
            tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
            companies = table["Security"].tolist()
            return {
                "universe": [{"ticker": t, "name": c} for t, c in zip(tickers, companies)]
            }
        except Exception as e:
            logger.error(f"Failed to fetch universe: {str(e)}")
            return {"universe": [{"ticker": "AAPL", "name": "Apple Inc."}]}

    return await asyncio.to_thread(fetch_universe)


# ==========================================
# 2. HYBRID INFERENCE ENGINE
# ==========================================
@app.get("/predict", dependencies=[Depends(verify_api_key)], response_model=PredictResponse)
async def get_prediction(ticker: str = "AAPL"):
    ticker = sanitize_ticker(ticker)
    
    async def run_inference():
        logger.info(f"Running inference for {ticker}")
        
        # Check cache
        cached = await api_cache.get(f"predict_{ticker}")
        if cached:
            return cached

        # 1. Fetch live data
        ts_sequence, peer_sequence, tabular_row, current_price, updated_config, market_regime, req_conf, vol_ratio, tech_snapshot = await asyncio.to_thread(fetch_live_data, ticker, config)

        # 2. Model Predictions
        dl_outputs = lstm_model.predict(
            x=[ts_sequence, ts_sequence, ts_sequence, peer_sequence],
            verbose=0,
        )
        dl_preds_raw = dl_outputs[2][0]
        dl_prob = float(dl_preds_raw[2])
        out_range = dl_outputs[1][0]
        forecast_low = current_price + out_range[0]
        forecast_high = current_price + out_range[1]

        xgb_preds_raw = xgb_model.predict_proba(tabular_row)[0]
        xgb_prob = float(xgb_preds_raw[2])

        lgbm_prob = 0.5
        lgbm_preds_raw = np.array([0.33, 0.33, 0.33])
        if lgbm_model:
            lgbm_preds_raw = lgbm_model.predict_proba(tabular_row)[0]
            lgbm_prob = float(lgbm_preds_raw[2])

        # 3. Dynamic Thresholds Based on Volatility
        atr_pct = tech_snapshot["ATR"] / (current_price + 1e-9)
        volatility_state = 'LOW'
        if atr_pct > 0.04:
            req_conf = max(req_conf, 0.72)
            volatility_state = 'HIGH'
        elif atr_pct > 0.02:
            volatility_state = 'MEDIUM'

        # 4. Consensus Logic (Majority Voting 2/3)
        votes = []
        def get_vote(p_buy):
            if p_buy > req_conf: return 'BUY'
            elif p_buy < (1 - req_conf): return 'SELL'
            else: return 'HOLD'
        
        votes.append(get_vote(dl_prob))
        votes.append(get_vote(xgb_prob))
        votes.append(get_vote(lgbm_prob))

        buy_votes = votes.count('BUY')
        sell_votes = votes.count('SELL')

        if buy_votes >= 2:
            final_signal = 'BUY'
        elif sell_votes >= 2:
            final_signal = 'SELL'
        else:
            final_signal = 'VETOED'

        # 5. Filters
        signal_note = None
        if await asyncio.to_thread(is_near_earnings, ticker):
            final_signal = 'HOLD'
            signal_note = 'Suppressed: Earnings window'
        elif vol_ratio < 0.7:
            final_signal = 'HOLD'
            signal_note = 'Suppressed: Low volume (ratio: {:.2f})'.format(vol_ratio)

        confidence_score = (sum([dl_prob, xgb_prob, lgbm_prob]) / 3) * 100

        # Qualitative Alpha
        tokenizer = NewsTokenizer(max_length=updated_config["data"]["max_seq_length"])
        _, _, news_text = fetch_live_news(ticker, tokenizer, updated_config)
        _, qual_reason = gemini_analyzer.analyze_fundamental_alpha(news_text, ticker)

        response_data = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "signal": final_signal,
            "confidence_score": round(confidence_score, 1),
            "signal_note": signal_note,
            "market_regime": market_regime,
            "volatility_state": volatility_state,
            "volume_ratio": round(vol_ratio, 2),
            "models": {
                "DL_LSTM_V4": {"signal": get_vote(dl_prob), "probability": round(dl_prob, 3)},
                "XGB_AGENT":  {"signal": get_vote(xgb_prob), "probability": round(xgb_prob, 3)},
                "LGBM_AGENT": {"signal": get_vote(lgbm_prob), "probability": round(lgbm_prob, 3)}
            },
            "projections": {
                "floor": round(float(forecast_low), 2),
                "ceiling": round(float(forecast_high), 2)
            },
            "technical_snapshot": tech_snapshot,
            "qualitative_alpha": qual_reason
        }

        # Legacy fields for paper trading and charts (preserving functionality)
        ticker_df = await asyncio.to_thread(yf.download, ticker, period="250d", progress=False)
        historical_markers, df_full = report_gen.generate_historical_markers(ticker, ticker_df)
        ai_report_stub = {
            "Models": {
                "Primary_Deep_Learning": {"Suggested_Action": get_vote(dl_prob), "Confidence": f"{round(dl_prob*100,1)}%"},
                "Secondary_XGBoost": {"Suggested_Action": get_vote(xgb_prob), "Confidence": f"{round(xgb_prob*100,1)}%"}
            },
            "Risk_Management": {"Meta_Model_Status": "Consensus", "Dynamic_10_Day_Range": {"Low": forecast_low, "High": forecast_high}},
            "Context": {"Top_Headline_Processed": news_text}
        }
        reporting_data = report_gen.package_chart_data(ticker, df_full, ai_report_stub, historical_markers)
        response_data.update(reporting_data)
        
        # Paper Trading
        conf_frac = confidence_score / 100.0
        if final_signal == 'HOLD': conf_frac = 0.0
        trade = paper_engine.execute_trade(ticker=ticker, action=final_signal, price=current_price, 
                                           confidence_fraction=conf_frac, regime=market_regime)
        if trade: response_data["paper_trade"] = trade
        response_data["portfolio"] = paper_engine.get_portfolio_summary({ticker: current_price})

        await api_cache.set(f"predict_{ticker}", response_data)
        return response_data

    return await run_inference()

# ==========================================
# 3. PERFORMANCE & ALERTS ENDPOINTS
# ==========================================
@app.get("/performance", dependencies=[Depends(verify_api_key)])
async def get_performance():
    analysis = perf_analyzer.analyze(
        paper_engine.portfolio_snapshots, 
        paper_engine.history, 
        paper_engine.initial_capital
    )
    if analysis.get("summary"):
        alert_system.check_performance(analysis["summary"])
        
    return analysis

@app.get("/alerts", dependencies=[Depends(verify_api_key)])
async def get_alerts():
    return {"alerts": alert_system.get_recent_alerts()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
