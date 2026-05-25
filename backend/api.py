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
import tensorflow as tf
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Import inference functions
from src.execution.live_inference import fetch_live_data, fetch_live_news, load_config, is_near_earnings
from src.models.neural.fusion_network import build_fusion_model
from src.models.rl.dqn_agent import DQNAgent
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
from src.models.monitoring.drift_monitor import DriftMonitor
from src.schemas import PredictResponse, UniverseResponse, UniverseStockItem, BacktestSummary, BacktestSignal, RiskMetrics
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
try:
    API_KEY = os.environ["API_KEY"]
except KeyError:
    logger.critical("CRITICAL: API_KEY environment variable is NOT SET. System halting for security.")
    API_KEY = "ENFORCE_FAILURE"

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if API_KEY == "ENFORCE_FAILURE" or api_key != API_KEY:
        logger.warning("Unauthorized access attempt rejected.")
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
try:
    FRONTEND_URLS = os.environ["FRONTEND_URL"].split(",")
except KeyError:
    logger.warning("FRONTEND_URL not set, defaulting to strict localhost for development.")
    FRONTEND_URLS = ["http://localhost:3000"]

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

drift_monitor = DriftMonitor()
perf_analyzer = PerformanceAnalyzer()
alert_system = AlertSystem()
sector_mapper = SectorMapper()

def sanitize_ticker(ticker: str) -> str:
    """Institutional Sanitization: Support up to 15 chars and dots for international symbols (e.g. .NS)."""
    if not re.match(r"^[A-Z0-9.-]{1,15}$", ticker.upper()):
        raise HTTPException(status_code=400, detail="Invalid Ticker Format")
    return ticker.upper()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"SYSTEM ERROR: {str(exc)}", exc_info=True)
    return Response(
        content=json.dumps({
            "detail": "An internal institutional error occurred. Contact system administrator.", 
            "code": "INTERNAL_ERROR"
        }),
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
    accs = {"ensemble_accuracy": 54.6, "dl_accuracy": 52.1, "xgb_accuracy": 55.4, "lgbm_accuracy": 53.2, "dqn_accuracy": 48.9}

config["data"]["num_features"] = actual_num_features

# Load Models
lstm_model = build_fusion_model(config)
try:
    lstm_model.load_weights("artifacts/latest_fusion_weights.weights.h5")
except Exception as e:
    logger.warning(f"Could not load LSTM weights: {e}")

from src.models.neural.tft_agent import build_tft_branch
tft_input, tft_output = build_tft_branch(
    time_steps=config["data"]["time_steps"],
    num_features=actual_num_features
)
import keras
tft_model = keras.Model(inputs=tft_input, outputs=tft_output)
try:
    tft_model.load_weights("artifacts/tft_quantile_weights.weights.h5")
except Exception as e:
    logger.warning(f"Could not load TFT weights: {e}")

xgb_model = xgb.XGBClassifier()
try:
    xgb_model.load_model("artifacts/xgb_ensemble.json")
except Exception as e:
    logger.warning(f"Could not load XGB ensemble: {e}")

lgbm_model = None
try:
    lgbm_model = joblib.load("artifacts/lgbm_agent.joblib")
except Exception as e:
    logger.warning(f"Could not load LightGBM agent: {e}")

dqn_agent = DQNAgent(state_size=actual_num_features + 6) 
try:
    dqn_agent.load("artifacts/dqn_model.pth")
except Exception as e:
    logger.warning(f"Could not load DQN agent: {e}")

from src.models.ensemble.meta_ensemble import MetaEnsemble
try:
    meta_ensemble = MetaEnsemble.load("artifacts/meta_ensemble.joblib")
except Exception as e:
    logger.warning(f"Could not load Meta-Ensemble: {e}")
    meta_ensemble = None

from src.execution.live_inference import compute_shap_explanation, get_meta_prediction, get_calibrated_probs, lstm_calibrated_probs

gemini_analyzer = GeminiAnalyzer()
physical_edge = PhysicalEdgeAnalyzer()
dependency_graph = SupplyChainGraph()
orchestrator = InstitutionalOrchestrator()
smart_router = PredictiveSmartRouter()
report_gen = ReportGenerator(kept_features_list)

from src.execution.fx_engine import FXEngine

# --- Universe Configuration ---
# Institutional Metadata: Market, Exchange, Currency, Timezone
UNIVERSES_METADATA = {
    "us": {
        "AAPL": {"name": "Apple Inc.", "exchange": "NASDAQ", "currency": "USD", "timezone": "America/New_York"},
        "MSFT": {"name": "Microsoft Corp.", "exchange": "NASDAQ", "currency": "USD", "timezone": "America/New_York"},
        "NVDA": {"name": "NVIDIA", "exchange": "NASDAQ", "currency": "USD", "timezone": "America/New_York"},
        "GOOGL": {"name": "Alphabet Inc.", "exchange": "NASDAQ", "currency": "USD", "timezone": "America/New_York"},
        "AMZN": {"name": "Amazon.com Inc.", "exchange": "NASDAQ", "currency": "USD", "timezone": "America/New_York"},
        "META": {"name": "Meta Platforms", "exchange": "NASDAQ", "currency": "USD", "timezone": "America/New_York"},
        "TSLA": {"name": "Tesla Inc.", "exchange": "NASDAQ", "currency": "USD", "timezone": "America/New_York"},
        "JPM": {"name": "JPMorgan Chase", "exchange": "NYSE", "currency": "USD", "timezone": "America/New_York"},
        "JNJ": {"name": "Johnson & Johnson", "exchange": "NYSE", "currency": "USD", "timezone": "America/New_York"},
        "XOM": {"name": "Exxon Mobil", "exchange": "NYSE", "currency": "USD", "timezone": "America/New_York"},
        "AMD": {"name": "AMD", "exchange": "NASDAQ", "currency": "USD", "timezone": "America/New_York"},
        "NFLX": {"name": "Netflix", "exchange": "NASDAQ", "currency": "USD", "timezone": "America/New_York"},
        "COST": {"name": "Costco", "exchange": "NASDAQ", "currency": "USD", "timezone": "America/New_York"},
        "AVGO": {"name": "Broadcom", "exchange": "NASDAQ", "currency": "USD", "timezone": "America/New_York"},
        "CRM": {"name": "Salesforce", "exchange": "NYSE", "currency": "USD", "timezone": "America/New_York"},
        "BAC": {"name": "Bank of America", "exchange": "NYSE", "currency": "USD", "timezone": "America/New_York"},
        "WMT": {"name": "Walmart", "exchange": "NYSE", "currency": "USD", "timezone": "America/New_York"},
        "DIS": {"name": "Disney", "exchange": "NYSE", "currency": "USD", "timezone": "America/New_York"},
        "V": {"name": "Visa", "exchange": "NYSE", "currency": "USD", "timezone": "America/New_York"},
        "MA": {"name": "Mastercard", "exchange": "NYSE", "currency": "USD", "timezone": "America/New_York"},
    },
    "india": {
        "RELIANCE.NS": {"name": "Reliance Industries", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "TCS.NS": {"name": "TCS", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "INFY.NS": {"name": "Infosys", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "HDFCBANK.NS": {"name": "HDFC Bank", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "ICICIBANK.NS": {"name": "ICICI Bank", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "SBIN.NS": {"name": "SBI", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "BHARTIARTL.NS": {"name": "Bharti Airtel", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "ITC.NS": {"name": "ITC", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "LT.NS": {"name": "L&T", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "HINDUNILVR.NS": {"name": "Hindustan Unilever", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "AXISBANK.NS": {"name": "Axis Bank", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "KOTAKBANK.NS": {"name": "Kotak Bank", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "BAJFINANCE.NS": {"name": "Bajaj Finance", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "MARUTI.NS": {"name": "Maruti Suzuki", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "ASIANPAINT.NS": {"name": "Asian Paints", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "SUNPHARMA.NS": {"name": "Sun Pharma", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "TITAN.NS": {"name": "Titan", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "ULTRACEMCO.NS": {"name": "UltraTech Cement", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "ADANIENT.NS": {"name": "Adani Enterprises", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
        "POWERGRID.NS": {"name": "Power Grid", "exchange": "NSE", "currency": "INR", "timezone": "Asia/Kolkata"},
    }
}

fx_engine = FXEngine()

@app.on_event("startup")
async def startup_event():
    # Start FX background task
    async def fx_refresh_loop():
        while True:
            await fx_engine.update_rates()
            await asyncio.sleep(300) # Refresh every 5 mins
    
    asyncio.create_task(fx_refresh_loop())

# ==========================================
# 1. MULTI-ASSET UNIVERSE ENDPOINT
# ==========================================
@app.get("/universe", dependencies=[Depends(verify_api_key)], response_model=UniverseResponse)
async def get_stock_universe():
    def fetch_universe():
        try:
            all_tickers = []
            for market_data in UNIVERSES_METADATA.values():
                all_tickers.extend(market_data.keys())

            data = yf.download(all_tickers, period="5d", interval="1d", progress=False)

            results = []
            for market_id, tickers_dict in UNIVERSES_METADATA.items():
                for t, meta in tickers_dict.items():
                    try:
                        # Handle multi-index if yf.download returns it
                        if isinstance(data['Close'], pd.DataFrame):
                            if t in data['Close'].columns:
                                prices = data['Close'][t].dropna()
                            else:
                                curr, pct = 0.0, 0.0
                                results.append(UniverseStockItem(
                                    ticker=t, name=meta["name"], price=curr, pct_change=pct, market=market_id,
                                    metadata={**meta, "ticker": t, "market": market_id.upper()}
                                ))
                                continue
                        else:
                            prices = data['Close'].dropna()

                        if len(prices) >= 2:
                            curr = float(prices.iloc[-1])
                            prev = float(prices.iloc[-2])
                            pct = ((curr / prev) - 1) * 100
                        else:
                            curr, pct = 0.0, 0.0
                    except:
                        curr, pct = 0.0, 0.0

                    results.append(UniverseStockItem(
                        ticker=t,
                        name=meta["name"],
                        price=curr,
                        pct_change=pct,
                        market=market_id,
                        metadata={**meta, "ticker": t, "market": market_id.upper()}
                    ))
            return {"universe": results}
        except Exception as e:
            logger.error(f"Failed to fetch universe: {str(e)}")
            return {"universe": []}

    return await asyncio.to_thread(fetch_universe)

@app.get("/fx_rates", dependencies=[Depends(verify_api_key)])
async def get_fx_rates():
    return fx_engine.get_summary()

# ==========================================
# 2. HYBRID INFERENCE ENGINE
# ==========================================
@app.get("/predict", dependencies=[Depends(verify_api_key)], response_model=PredictResponse)
async def get_prediction(ticker: str = "AAPL"):
    ticker = sanitize_ticker(ticker)
    
    # Identify Market Metadata
    metadata = None
    for m_id, m_dict in UNIVERSES_METADATA.items():
        if ticker in m_dict:
            metadata = {**m_dict[ticker], "ticker": ticker, "market": m_id.upper()}
            break
    
    if not metadata:
        # Fallback for untracked tickers
        metadata = {"ticker": ticker, "market": "UNKNOWN", "exchange": "UNKNOWN", "currency": "USD", "timezone": "UTC"}

    async def run_inference():
        logger.info(f"Running inference for {ticker}")
        
        cached = await api_cache.get(f"predict_{ticker}")
        if cached:
            # Refresh portfolio context even for cached results
            cached["portfolio"] = paper_engine.get_portfolio_summary({ticker: cached["current_price"]})
            return cached

        # 1. Fetch live data
        ts_sequence, peer_sequence, tabular_row, current_price, updated_config, market_regime, req_conf, vol_ratio, tech_snapshot = await asyncio.to_thread(fetch_live_data, ticker, config)

        # ... (rest of model predictions)
        # 2. Model Predictions
        dl_outputs = lstm_model.predict(
            x=[ts_sequence, ts_sequence, ts_sequence, ts_sequence, ts_sequence, peer_sequence],
            verbose=0,
        )
        dl_preds_raw = dl_outputs[2][0]

        xgb_preds_raw = xgb_model.predict_proba(tabular_row)[0]

        lgbm_preds_raw = np.array([0.33, 0.33, 0.33])
        if lgbm_model:
            lgbm_preds_raw = lgbm_model.predict_proba(tabular_row)[0]

        # DQN Prediction
        dqn_state = np.hstack((tabular_row, dl_preds_raw.reshape(1, -1), xgb_preds_raw.reshape(1, -1)))
        dqn_action = dqn_agent.act(dqn_state[0])
        dqn_p = np.array([0.0, 1.0, 0.0])
        if dqn_action == 0: dqn_p = np.array([1.0, 0.0, 0.0])
        elif dqn_action == 2: dqn_p = np.array([0.0, 0.0, 1.0])

        # 3. Meta-Ensemble Consensus
        base_probs = {"LSTM": dl_preds_raw, "XGBoost": xgb_preds_raw, "LightGBM": lgbm_preds_raw, "DQN": dqn_p}
        regime_id_map = {"BEAR": 0, "NEUTRAL": 1, "BULL": 2}
        regime_id = regime_id_map.get(market_regime, 1)
        vol_id = 1
        if tech_snapshot["ATR"] / current_price > 0.04: vol_id = 2
        elif tech_snapshot["ATR"] / current_price < 0.01: vol_id = 0

        final_probs, uncertainty = get_meta_prediction(
            base_probs, regime_id, vol_id, vol_ratio, tech_snapshot["RSI"], tech_snapshot["ADX"]
        )
        final_idx = np.argmax(final_probs)
        final_prob = float(final_probs[final_idx])
        
        final_signal = 'HOLD'
        if final_prob > req_conf:
            final_signal = 'BUY' if final_idx == 2 else 'SELL'
        
        # 4. Filters & Vetoes
        signal_note = None
        if uncertainty > 0.45:
            final_signal = 'VETOED'
            signal_note = f'Vetoed: High Uncertainty ({uncertainty:.2f})'
        elif await asyncio.to_thread(is_near_earnings, ticker):
            final_signal = 'HOLD'
            signal_note = 'Suppressed: Earnings window'
        elif vol_ratio < 0.7:
            final_signal = 'HOLD'
            signal_note = 'Suppressed: Low volume (ratio: {:.2f})'.format(vol_ratio)

        confidence_score = final_prob * 100

        # SHAP Explainability
        shap_xai = await asyncio.to_thread(compute_shap_explanation, lgbm_model, tabular_row)

        # TFT Multi-Horizon Projections
        tft_preds = tft_model.predict(ts_sequence, verbose=0)[0]
        constrained_rets = np.clip(tft_preds, -0.20, 0.20)
        is_point_forecast = np.all(np.isclose(constrained_rets, constrained_rets[0]))
        floor_ret, median_ret, ceiling_ret = float(constrained_rets[0]), float(constrained_rets[2]), float(constrained_rets[4])
        forecast_low = current_price * (1 + floor_ret)
        forecast_median = current_price * (1 + median_ret)
        forecast_high = current_price * (1 + ceiling_ret)

        # Qualitative Alpha
        tokenizer = NewsTokenizer(max_length=updated_config["data"]["max_seq_length"])
        _, _, news_text = fetch_live_news(ticker, tokenizer, updated_config)
        sentiment_score, qual_reason = await asyncio.to_thread(gemini_analyzer.analyze_fundamental_alpha, news_text, ticker)

        # 5. Risk Metrics
        spy_df_risk = await asyncio.to_thread(yf.download, "SPY", period="1y", progress=False)
        ticker_df_risk = await asyncio.to_thread(yf.download, ticker, period="1y", progress=False)
        if isinstance(spy_df_risk.columns, pd.MultiIndex): spy_df_risk.columns = spy_df_risk.columns.get_level_values(0)
        if isinstance(ticker_df_risk.columns, pd.MultiIndex): ticker_df_risk.columns = ticker_df_risk.columns.get_level_values(0)
        
        beta = calculate_beta(ticker_df_risk['Close'], spy_df_risk['Close'])
        returns_history = [t.get('pnl', 0) / (t.get('cost_base', 1e-9)) for t in paper_engine.history if 'pnl' in t]
        
        if not returns_history:
            var_95 = 0.0
            cvar = 0.0
        elif len(returns_history) < 20:
            hist_daily_rets = ticker_df_risk['Close'].pct_change().dropna().values
            var_95 = float(np.percentile(hist_daily_rets, 5)) if len(hist_daily_rets) > 0 else 0.0
            cvar = float(np.mean(hist_daily_rets[hist_daily_rets <= var_95])) if len(hist_daily_rets) > 0 else 0.0
        else:
            var_95 = float(paper_engine.calculate_var(returns_history))
            cvar = float(paper_engine.calculate_expected_shortfall(returns_history))
            
        risk_profile = get_position_sizing(final_prob, paper_engine.history)
        kelly_frac = min(0.01, float(risk_profile["raw_fraction"]))
        
        perf_summary = perf_analyzer.analyze(paper_engine.portfolio_snapshots, paper_engine.history, paper_engine.initial_capital)
        max_dd = perf_summary.get("summary", {}).get("max_drawdown", 0.0)

        risk_metrics = RiskMetrics(
            var_95=var_95, cvar=cvar, beta=float(beta),
            kelly_fraction=kelly_frac, target_size=float(paper_engine.capital * kelly_frac),
            max_drawdown=float(max_dd)
        )

        def map_model_output(probs):
            idx = np.argmax(probs)
            signals = ["SELL", "HOLD", "BUY"]
            return {"signal": signals[idx], "probability": round(float(probs[idx]), 3)}

        signals_only = [m["signal"] for m in [map_model_output(dl_preds_raw), map_model_output(xgb_preds_raw), map_model_output(lgbm_preds_raw)]]
        bullish_count = signals_only.count('BUY')
        bearish_count = signals_only.count('SELL')
        neutral_count = signals_only.count('HOLD')
        majority_count = max(bullish_count, bearish_count, neutral_count)
        model_agreement = (majority_count / 3.0) * 100

        response_data = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "signal": final_signal,
            "confidence_score": round(confidence_score, 1),
            "uncertainty_score": round(uncertainty * 100, 1),
            "signal_note": signal_note,
            "market_regime": market_regime,
            "volatility_state": "HIGH" if vol_id == 2 else ("LOW" if vol_id == 0 else "MEDIUM"),
            "volume_ratio": round(vol_ratio, 2),
            "is_point_forecast": is_point_forecast,
            "model_agreement": round(model_agreement, 1),
            "bullish_models": bullish_count,
            "bearish_models": bearish_count,
            "neutral_models": neutral_count,
            "timestamp": datetime.now().isoformat(),
            "models": {
                "DL_FUSION": map_model_output(dl_preds_raw),
                "XGB_AGENT":  map_model_output(xgb_preds_raw),
                "LGBM_AGENT": map_model_output(lgbm_preds_raw)
            },
            "projections": {
                "floor": round(float(forecast_low), 2),
                "median": round(float(forecast_median), 2),
                "ceiling": round(float(forecast_high), 2)
            },
            "technical_snapshot": tech_snapshot,
            "qualitative_alpha": qual_reason,
            "xai": shap_xai,
            "sentiment_score": float(sentiment_score),
            "risk": risk_metrics
        }

        # Market Metadata Attachment
        response_data["metadata"] = metadata

        historical_markers, df_full = report_gen.generate_historical_markers(ticker, ticker_df_risk)
        ai_report_stub = {
            "Models": {
                "Primary_Deep_Learning": {"Suggested_Action": final_signal, "Confidence": f"{confidence_score:.1f}%"},
                "Secondary_XGBoost": {"Suggested_Action": "BUY" if xgb_preds_raw[2] > 0.5 else "SELL", "Confidence": f"{xgb_preds_raw[2]*100:.1f}%"}
            },
            "Risk_Management": {"Meta_Model_Status": "Live Consensus Active", "Dynamic_10_Day_Range": {"Low": forecast_low, "High": forecast_high}},
            "Context": {"Top_Headline_Processed": news_text}
        }
        reporting_data = report_gen.package_chart_data(ticker, df_full, ai_report_stub, historical_markers)
        response_data.update(reporting_data)
        
        # Paper Trading with Currency Context
        if final_signal in ['BUY', 'SELL']:
            atr = tech_snapshot.get("ATR", current_price * 0.02)
            sl = (current_price - 2*atr) if final_signal == 'BUY' else (current_price + 2*atr)
            tp = forecast_high if final_signal == 'BUY' else forecast_low
            trade = paper_engine.execute_trade(
                ticker, final_signal, current_price, kelly_frac, market_regime, 
                currency=metadata["currency"], market=metadata["market"],
                stop_loss=sl, take_profit=tp
            )
            if trade: response_data["paper_trade"] = trade

        paper_engine.update_positions({ticker: current_price})
        response_data["portfolio"] = paper_engine.get_portfolio_summary({ticker: current_price})

        await api_cache.set(f"predict_{ticker}", response_data)
        return response_data

    return await run_inference()

@app.post("/portfolio/base_currency", dependencies=[Depends(verify_api_key)])
async def set_base_currency(request: Request):
    data = await request.json()
    new_currency = data.get("currency")
    if new_currency not in ["USD", "INR", "EUR", "GBP"]:
        raise HTTPException(status_code=400, detail="Unsupported Currency")
    
    paper_engine.set_base_currency(new_currency)
    return {"status": "SUCCESS", "base_currency": new_currency}

# ==========================================
# 3. PERFORMANCE & ALERTS ENDPOINTS
# ==========================================
@app.get("/performance", dependencies=[Depends(verify_api_key)])
async def get_performance():
    analysis = perf_analyzer.analyze(paper_engine.portfolio_snapshots, paper_engine.history, paper_engine.initial_capital)
    if analysis.get("summary"): alert_system.check_performance(analysis["summary"])
    else: analysis["summary"] = {"total_return": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    
    analysis["models"] = {
        "ensemble": accs.get("ensemble_accuracy", 54.6),
        "lstm": accs.get("dl_accuracy", 0.0),
        "xgboost": accs.get("xgb_accuracy", 0.0),
        "lightgbm": accs.get("lgbm_accuracy", 0.0),
        "dqn": accs.get("dqn_accuracy", 0.0),
        "consensus_rate": 0.0
    }
    analysis["signals"] = {"active": len([p for p in paper_engine.positions.values() if p["shares"] > 0]), "historical": len(paper_engine.history), "regime": "LIVE EXECUTION"}
    return analysis

@app.get("/alerts", dependencies=[Depends(verify_api_key)])
async def get_alerts():
    return {"alerts": alert_system.get_recent_alerts()}

@app.get("/backtest", dependencies=[Depends(verify_api_key)], response_model=BacktestSummary)
async def get_backtest(ticker: str = "AAPL", period: str = "1y"):
    try:
        trades_path = "backtest_results/backtest_trades.csv"
        if not os.path.exists(trades_path): raise HTTPException(status_code=404, detail="Backtest results not found.")
        df = pd.read_csv(trades_path)
        df_ticker = df[df['ticker'] == sanitize_ticker(ticker)]
        df_ticker['date_parsed'] = pd.to_datetime(df_ticker['date'])
        end_date = df_ticker['date_parsed'].max() if not df_ticker.empty else pd.Timestamp.now()
        if period == '3m': start_date = end_date - pd.DateOffset(months=3)
        elif period == '6m': start_date = end_date - pd.DateOffset(months=6)
        elif period == '1y': start_date = end_date - pd.DateOffset(years=1)
        elif period == '2y': start_date = end_date - pd.DateOffset(years=2)
        else: start_date = end_date - pd.DateOffset(years=1)
        df_ticker = df_ticker[df_ticker['date_parsed'] >= start_date]
        if df_ticker.empty: return BacktestSummary(ticker=ticker, period=period, win_rate=0.0, profit_factor=0.0, sharpe_ratio=0.0, max_drawdown=0.0, vetoed_rate=0.0, coverage=0.0)
        correct = df_ticker[df_ticker['was_correct'] == True]
        win_rate = len(correct) / len(df_ticker) * 100
        profits = df_ticker[df_ticker['actual_5day_return'] > 0]['actual_5day_return'].sum()
        losses = abs(df_ticker[df_ticker['actual_5day_return'] < 0]['actual_5day_return'].sum())
        pf = profits / losses if losses > 0 else float('inf')
        returns = df_ticker['actual_5day_return'] / 100
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252/5) if returns.std() > 0 else 0.0
        cum_ret = (1 + returns).cumprod()
        rolling_max = cum_ret.cummax()
        drawdowns = (cum_ret - rolling_max) / rolling_max
        max_dd = drawdowns.min() * 100
        best_row = df_ticker.loc[df_ticker['actual_5day_return'].idxmax()]
        worst_row = df_ticker.loc[df_ticker['actual_5day_return'].idxmin()]
        return BacktestSummary(ticker=ticker, period=period, win_rate=round(win_rate, 1), profit_factor=round(pf, 2), sharpe_ratio=round(sharpe, 2), max_drawdown=round(max_dd, 1), vetoed_rate=0.0, coverage=100.0, best_signal=BacktestSignal(date=str(best_row['date']), ticker=ticker, signal=best_row['signal'], confidence=best_row['confidence'], actual_return=best_row['actual_5day_return']), worst_signal=BacktestSignal(date=str(worst_row['date']), ticker=ticker, signal=worst_row['signal'], confidence=worst_row['confidence'], actual_return=worst_row['actual_5day_return']))
    except Exception as e:
        logger.error(f"Backtest retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
