# api.py
import os
import json
import logging
import joblib
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your inference functions
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

# --- Setup Logging ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/api_{datetime.now().strftime('%Y%m')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

os.environ["TF_USE_LEGACY_KERAS"] = "1"
load_dotenv()

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

with open("configs/kept_features.json", "r") as f:
    kept_features_list = json.load(f)
    actual_num_features = len(kept_features_list)

# Load Accuracies for weighting
try:
    with open("configs/model_accuracies.json", "r") as f:
        accs = json.load(f)
except Exception:
    accs = {"dl_accuracy": 0.5, "xgb_accuracy": 0.5, "dqn_accuracy": 0.5}

config["data"]["num_features"] = actual_num_features
logger.info(f"Initializing 5-Model Ensemble with {actual_num_features} features...")

# Load Models
lstm_model = build_fusion_model(config)
lstm_model.load_weights("latest_fusion_weights.weights.h5")

# NEW: Institutional Qualitative Analyzer
gemini_analyzer = GeminiAnalyzer()

# NEW: Elite SOTA 2026 Components
physical_edge = PhysicalEdgeAnalyzer()
dependency_graph = SupplyChainGraph()
orchestrator = InstitutionalOrchestrator()
smart_router = PredictiveSmartRouter()
report_gen = ReportGenerator(kept_features_list)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_ensemble.json")

dqn_agent = DQNAgent(actual_num_features + 3 + 3)
try:
    dqn_agent.load("dqn_model.pth")
except Exception:
    logger.warning("DQN model not found. Using random agent.")

# Load Macro Kill-Switch
kill_switch_data = joblib.load("macro_kill_switch.joblib")
regime_model = kill_switch_data["model"]
panic_id = kill_switch_data["panic_cluster"]


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
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Import your inference functions
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

# --- Structured Logging ---
import json
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
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'API request latency', ['endpoint'])

app = FastAPI(title="Hydra Terminal API", version="2.0.0")

# --- Security: Restricted CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# --- Simple In-Memory Rate Limiting (Phase 10) ---
# For production, Redis is preferred, but this meets the requirement.
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
lstm_model.load_weights("latest_fusion_weights.weights.h5")

gemini_analyzer = GeminiAnalyzer()
physical_edge = PhysicalEdgeAnalyzer()
dependency_graph = SupplyChainGraph()
orchestrator = InstitutionalOrchestrator()
smart_router = PredictiveSmartRouter()
report_gen = ReportGenerator(kept_features_list)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_ensemble.json")

dqn_agent = DQNAgent(actual_num_features + 3 + 3)
try:
    dqn_agent.load("dqn_model.pth")
except Exception:
    logger.warning("DQN model not found. Using random agent.")

kill_switch_data = joblib.load("macro_kill_switch.joblib")
regime_model = kill_switch_data["model"]
panic_id = kill_switch_data["panic_cluster"]


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
@app.get("/predict", dependencies=[Depends(verify_api_key)])
async def get_prediction(ticker: str = "AAPL"):
    def run_inference():
        logger.info(f"Running inference for {ticker}")
        # 1. Fetch live data
        ts_sequence, peer_sequence, tabular_row, current_price, updated_config = (
            fetch_live_data(ticker, config)
        )

        # Macro check
        import yfinance as yf
        spy_df = yf.download("SPY", period="60d", progress=False)
        ticker_df = yf.download(ticker, period="60d", progress=False)
        beta = calculate_beta(ticker_df["Close"], spy_df["Close"])

        # 2. Elite Data Layer
        physical_alpha = physical_edge.get_physical_alpha_vector(ticker)
        dependency_graph.build_proxy_graph(ticker)
        propagation_risk = dependency_graph.calculate_propagation_risk(ticker)

        # 3. Alpha Generation
        tokenizer = NewsTokenizer(max_length=updated_config["data"]["max_seq_length"])
        input_ids, attention_masks, news_text = fetch_live_news(
            ticker, tokenizer, updated_config
        )
        qual_score, qual_reason = gemini_analyzer.analyze_fundamental_alpha(
            news_text, ticker
        )

        dl_outputs = lstm_model.predict(
            x=[
                ts_sequence,
                ts_sequence,
                ts_sequence,
                peer_sequence,
                input_ids,
                attention_masks,
            ],
            verbose=0,
        )
        dl_preds = dl_outputs[2][0]
        out_range = dl_outputs[1][0]
        forecast_low = current_price + out_range[0]
        forecast_high = current_price + out_range[1]
        
        xgb_preds = xgb_model.predict_proba(tabular_row)[0]

        # 4. Multi-Agent Consensus Logic
        total_acc = sum(accs.values())
        w_dl = accs["dl_accuracy"] / total_acc
        w_xgb = accs["xgb_accuracy"] / total_acc

        ensemble_p = (dl_preds * w_dl) + (xgb_preds * w_xgb)
        ensemble_p[2] = (
            (ensemble_p[2] * 0.9) + (max(0, qual_score) * 0.1)
            if qual_score > 0
            else ensemble_p[2]
        )
        ensemble_p[0] = (
            (ensemble_p[0] * 0.9) + (abs(min(0, qual_score)) * 0.1)
            if qual_score < 0
            else ensemble_p[0]
        )

        # Institutional Risk & Alpha Decomposition
        risk_metrics = get_position_sizing(float(np.max(ensemble_p)))
        alpha_metric = calculate_jensens_alpha(
            ticker_df["Close"].pct_change(), spy_df["Close"].pct_change(), beta
        )
        stampede = detect_stampede_risk(np.std(dl_preds), float(np.max(ensemble_p)))

        # Run Agentic Orchestration
        mesh_response = orchestrator.run_consensus(
            ensemble_p,
            {
                "beta": beta,
                "suggested_allocation": risk_metrics["suggested_allocation"],
                "hedge_ratio_spy": f"-{round(beta * 100, 1)}%",
            },
        )

        # 5. Execution Optimization
        routing = smart_router.predict_venue_liquidity(ticker)

        # 6. Explainable AI (XAI) Log Generation & Schema Alignment
        xai_log = f"Decision primarily driven by {orchestrator.consensus_status}. "
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
                "Meta_Model_Status": f"{orchestrator.consensus_status}. {xai_log}",
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
            },
            "physical_edge": physical_alpha,
            "smart_routing": routing,
            "xai_reasoning": xai_log,
            "price": current_price,
            "news": news_text,
        }

        # --- Refactored Reporting Logic ---
        historical_markers, df_full = report_gen.generate_historical_markers(ticker, ticker_df)
        reporting_data = report_gen.package_chart_data(ticker, df_full, ai_report_dict, historical_markers)
        result_dict.update(reporting_data)

        # --- Paper Trading Execution ---
        confidence_fraction = float(np.max(ensemble_p))
        if "SCALE_BACK" in final_action:
            confidence_fraction *= 0.5
            
        trade = paper_engine.execute_trade(
            ticker=ticker,
            action=signal_map[mesh_response["final_action_idx"]],
            price=current_price,
            confidence_fraction=confidence_fraction
        )
        
        if trade:
            result_dict["paper_trade"] = trade
            
        current_prices = {ticker: current_price}
        result_dict["portfolio"] = paper_engine.get_portfolio_summary(current_prices)

        api_cache.set(f"predict_{ticker}", result_dict)

        logger.info(f"Inference complete for {ticker}")
        return result_dict
        
    return await asyncio.to_thread(run_inference)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
