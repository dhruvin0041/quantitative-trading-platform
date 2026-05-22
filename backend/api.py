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
from live_inference import fetch_live_data, fetch_live_news, load_config
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
        "regime_model_ready": regime_model is not None,
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
logger.info(f"Initializing 5-Model Ensemble with {actual_num_features} features...")

# Load Models
lstm_model = build_fusion_model(config)
try:
    lstm_model.load_weights("latest_fusion_weights.weights.h5")
except Exception as e:
    logger.warning(f"Could not load LSTM weights: {e}. System will run with uninitialized neural paths.")

gemini_analyzer = GeminiAnalyzer()
physical_edge = PhysicalEdgeAnalyzer()
dependency_graph = SupplyChainGraph()
orchestrator = InstitutionalOrchestrator()
smart_router = PredictiveSmartRouter()
report_gen = ReportGenerator(kept_features_list)

xgb_model = xgb.XGBClassifier()
try:
    xgb_model.load_model("xgb_ensemble.json")
except Exception as e:
    logger.warning(f"Could not load XGB ensemble: {e}. XGB branch will be inactive.")


try:
    calibrator = joblib.load("calibrator.joblib")
except Exception:
    calibrator = None

dqn_agent = DQNAgent(actual_num_features + 3 + 3)
try:
    dqn_agent.load("dqn_model.pth")
except Exception:
    logger.warning("DQN model not found. Using random agent.")

try:
    kill_switch_data = joblib.load("macro_kill_switch.joblib")
    regime_model = kill_switch_data["model"]
    panic_id = kill_switch_data["panic_cluster"]
except FileNotFoundError:
    logger.warning("macro_kill_switch.joblib not found. Using dummy regime model.")
    class DummyRegimeModel:
        def predict(self, X):
            return [0]
    regime_model = DummyRegimeModel()
    panic_id = -1



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
        ts_sequence, peer_sequence, tabular_row, current_price, updated_config = await asyncio.to_thread(fetch_live_data, ticker, config)

        # Macro check
        import yfinance as yf
        spy_df = await asyncio.to_thread(yf.download, "SPY", period="250d", progress=False)
        ticker_df = await asyncio.to_thread(yf.download, ticker, period="250d", progress=False)
        
        # Ensure we get Series instead of DataFrames for single tickers (yfinance v0.2.40+ behavior)
        spy_close = spy_df["Close"].squeeze()
        ticker_close = ticker_df["Close"].squeeze()
        
        beta = calculate_beta(ticker_close, spy_close)

        # 2. Elite Data Layer
        physical_alpha = physical_edge.get_physical_alpha_vector(ticker)
        dependency_graph.build_proxy_graph(ticker)
        propagation_risk = dependency_graph.calculate_propagation_risk(ticker)

        # Check Data Drift
        try:
            drift_results = drift_monitor.check_covariate_drift(tabular_row, tabular_row)
            alert_system.check_drift(drift_results)
        except Exception:
            drift_results = {"is_drifting": False}

        # Check Market Regime
        current_vix = await asyncio.to_thread(yf.download, "^VIX", period="5d", progress=False)
        if not current_vix.empty and len(current_vix) >= 5:
            vix_close = current_vix["Close"].squeeze()
            current_vix["VIX_ROC"] = vix_close.pct_change(periods=4)
            vix_features = current_vix[["Close", "VIX_ROC"]].iloc[-1].values.reshape(1, -1)
            try:
                current_regime_idx = regime_model.predict(vix_features)[0]
                is_panic_regime = (current_regime_idx == panic_id)
            except Exception:
                is_panic_regime = False
        else:
            is_panic_regime = False
        
        regime_label = "PANIC" if is_panic_regime else "NORMAL"

        # 3. Alpha Generation
        tokenizer = NewsTokenizer(max_length=updated_config["data"]["max_seq_length"])
        input_ids, attention_masks, news_text = fetch_live_news(ticker, tokenizer, updated_config)
        qual_score, qual_reason = gemini_analyzer.analyze_fundamental_alpha(news_text, ticker)

        dl_outputs = lstm_model.predict(
            x=[ts_sequence, ts_sequence, ts_sequence, peer_sequence],
            verbose=0,
        )
        dl_preds_raw = dl_outputs[2][0]
        out_range = dl_outputs[1][0]
        forecast_low = current_price + out_range[0]
        forecast_high = current_price + out_range[1]

        xgb_preds_raw = xgb_model.predict_proba(tabular_row)[0]
        
        if calibrator:
            try:
                # Attempt to calibrate only the max probability to avoid shape mismatch in IsotonicRegression
                dl_max_idx = np.argmax(dl_preds_raw)
                xgb_max_idx = np.argmax(xgb_preds_raw)
                
                calibrated_dl_max = calibrator.calibrate("dl_model", np.array([dl_preds_raw[dl_max_idx]]))[0]
                calibrated_xgb_max = calibrator.calibrate("xgb_model", np.array([xgb_preds_raw[xgb_max_idx]]))[0]
                
                dl_preds = dl_preds_raw.copy()
                dl_preds[dl_max_idx] = np.clip(calibrated_dl_max, 0, 1) # Ensure valid prob range
                
                xgb_preds = xgb_preds_raw.copy()
                xgb_preds[xgb_max_idx] = np.clip(calibrated_xgb_max, 0, 1)
            except Exception as e:
                logger.warning(f"Calibration failed: {e}. Using raw predictions.")
                dl_preds = dl_preds_raw
                xgb_preds = xgb_preds_raw
        else:
            dl_preds = dl_preds_raw
            xgb_preds = xgb_preds_raw

        # 4. Multi-Agent Consensus Logic
        total_acc = sum(accs.values())
        w_dl = accs["dl_accuracy"] / total_acc
        w_xgb = accs["xgb_accuracy"] / total_acc

        ensemble_p = (dl_preds * w_dl) + (xgb_preds * w_xgb)
        ensemble_p[2] = (ensemble_p[2] * 0.9) + (max(0, qual_score) * 0.1) if qual_score > 0 else ensemble_p[2]
        ensemble_p[0] = (ensemble_p[0] * 0.9) + (abs(min(0, qual_score)) * 0.1) if qual_score < 0 else ensemble_p[0]

        # Institutional Risk & Alpha Decomposition
        if is_panic_regime:
            risk_metrics = get_position_sizing(0.0, paper_engine.history)
        else:
            risk_metrics = get_position_sizing(float(np.max(ensemble_p)), paper_engine.history)

        alpha_metric = calculate_jensens_alpha(ticker_close.pct_change(), spy_close.pct_change(), beta)
        stampede = detect_stampede_risk(np.std(dl_preds), float(np.max(ensemble_p)))
        if is_panic_regime:
             stampede["is_crowded"] = True

        # Run Agentic Orchestration
        mesh_response = orchestrator.run_consensus(
            ensemble_p,
            {
                "beta": beta,
                "suggested_allocation": risk_metrics["suggested_allocation"],
                "hedge_ratio_spy": f"-{round(beta * 100, 1)}%",
                "stampede_risk": stampede
            },
        )

        # 5. Execution Optimization
        routing = smart_router.predict_venue_liquidity(ticker)

        # 6. Explainable AI (XAI) Log Generation
        xai_log = f"Decision primarily driven by {mesh_response['consensus_status']}. "
        if is_panic_regime:
             xai_log += "MACRO REGIME: PANIC (Risk-Off). "
        xai_log += f"Physical supply risk at {round(physical_alpha['supply_chain_disruption_index'] * 100)}%. "
        xai_log += f"Qualitative Alpha: {qual_reason}"

        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        final_action = signal_map[mesh_response["final_action_idx"]]
        if stampede["is_crowded"]:
            final_action = f"SCALE_BACK ({final_action})"

        ai_report_dict = {
            "Models": {
                "Primary_Deep_Learning": {
                    "Suggested_Action": signal_map[np.argmax(dl_preds)],
                    "Confidence": f"{round(float(np.max(dl_preds)) * 100, 1)}%",
                },
                "Secondary_XGBoost": {
                    "Suggested_Action": signal_map[np.argmax(xgb_preds)],
                    "Confidence": f"{round(float(np.max(xgb_preds)) * 100, 1)}%",
                }
            },
            "Risk_Management": {
                "Meta_Model_Status": f"{mesh_response['consensus_status']}. {xai_log}",
                "Dynamic_10_Day_Range": {
                    "Low": round(float(forecast_low), 2),
                    "High": round(float(forecast_high), 2)
                }
            },
            "Context": {
                "Top_Headline_Processed": news_text
            }
        }

        result_dict = {
            "ticker": ticker,
            "action": final_action,
            "confidence": f"{round(float(np.max(ensemble_p)) * 100, 1)}%",
            "agent_consensus": mesh_response,
            "institutional_metrics": {
                "jensens_alpha": round(alpha_metric, 4),
                "beta": round(beta, 2),
                "propagation_risk": round(propagation_risk, 3),
                "stampede_risk": stampede,
                "data_drift": drift_results
            },
            "physical_edge": physical_alpha,
            "smart_routing": routing,
            "xai_reasoning": xai_log,
            "price": current_price,
            "news": news_text,
        }

        # Reporting
        historical_markers, df_full = report_gen.generate_historical_markers(ticker, ticker_df)
        reporting_data = report_gen.package_chart_data(ticker, df_full, ai_report_dict, historical_markers)
        result_dict.update(reporting_data)

        # Paper Trading
        confidence_fraction = float(np.max(ensemble_p))
        if "SCALE_BACK" in final_action: confidence_fraction *= 0.5
        if is_panic_regime: confidence_fraction = 0.0

        ticker_sector = sector_mapper.get_sector(ticker)
        trade = paper_engine.execute_trade(ticker=ticker, action=signal_map[mesh_response["final_action_idx"]], 
                                           price=current_price, confidence_fraction=confidence_fraction,
                                           regime=regime_label, sector=ticker_sector)
        if trade:
            result_dict["paper_trade"] = trade
            broker.submit_order(trade["ticker"], trade["action"], trade.get("shares", 1))

        current_prices = {ticker: current_price}
        result_dict["portfolio"] = paper_engine.get_portfolio_summary(current_prices)

        await api_cache.set(f"predict_{ticker}", result_dict)
        logger.info(f"Inference complete for {ticker}")
        return result_dict

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
    # Check for performance degradation alerts
    if analysis.get("summary"):
        alert_system.check_performance(analysis["summary"])
        
    return analysis

@app.get("/alerts", dependencies=[Depends(verify_api_key)])
async def get_alerts():
    return {"alerts": alert_system.get_recent_alerts()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
