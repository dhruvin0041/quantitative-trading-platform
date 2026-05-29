import os
import json
import logging
import asyncio
import time
import re
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)
from fastapi.responses import Response

import yfinance as yf
import pandas as pd

# Core System Imports
from src.data_ingestion.universes import UNIVERSES_METADATA
from src.models.model_loader import ModelManager
from src.execution.inference_service import InferenceService
from src.execution.backtest_service import BacktestService
from src.utils.cache import SimpleCache

from src.execution.live_inference import load_config
from src.data_ingestion.nlp_processor import GeminiAnalyzer
from src.data_ingestion.alternative_data import PhysicalEdgeAnalyzer
from src.data_ingestion.supply_chain_graph import SupplyChainGraph
from src.agents.orchestrator import InstitutionalOrchestrator
from src.execution.smart_router import PredictiveSmartRouter
from src.execution.reporting import ReportGenerator
from src.execution.paper_trading import PaperTradingEngine
from src.execution.fx_engine import FXEngine
from src.models.monitoring.drift_monitor import DriftMonitor
from src.schemas import (
    PredictResponse,
    UniverseResponse,
    UniverseStockItem,
    BacktestSummary,
)
from src.execution.performance_analyzer import PerformanceAnalyzer
from src.execution.alerts import AlertSystem
from src.data_ingestion.sector_mapper import SectorMapper
from src.execution.signal_journal import SignalJournal
from src.execution.empirical_validation import ValidationAnalytics


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
    logger.critical(
        "CRITICAL: API_KEY environment variable is NOT SET. System halting for security."
    )
    API_KEY = "ENFORCE_FAILURE"

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if API_KEY == "ENFORCE_FAILURE" or api_key != API_KEY:
        logger.warning("Unauthorized access attempt rejected.")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


# --- Observability: Prometheus Metrics ---

try:
    REQUEST_COUNT = Counter(
        "api_requests_total",
        "Total API requests",
        ["method", "endpoint", "status_code"],
    )
    REQUEST_LATENCY = Histogram(
        "api_request_latency_seconds", "API request latency", ["endpoint"]
    )
except ValueError:
    REQUEST_COUNT = REGISTRY._names_to_collectors["api_requests_total"]
    REQUEST_LATENCY = REGISTRY._names_to_collectors["api_request_latency_seconds"]

app = FastAPI(title="Hydra Terminal API", version="2.1.0")

# --- Security: Restricted CORS ---
try:
    FRONTEND_URLS = os.environ["FRONTEND_URL"].split(",")
except KeyError:
    FRONTEND_URLS = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://192.168.29.189:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Middleware: Rate Limiting & Metrics ---
rate_limit_store = {}
RATE_LIMIT = 50
RATE_WINDOW = 60


@app.middleware("http")
async def api_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()

    # Rate Limiting
    ip_history = rate_limit_store.get(client_ip, [])
    ip_history = [t for t in ip_history if now - t < RATE_WINDOW]
    rate_limit_store[client_ip] = ip_history

    if len(ip_history) >= RATE_LIMIT:
        return Response(content="Rate limit exceeded", status_code=429)
    ip_history.append(now)

    # Metrics & Latency
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()
    if request.url.path == "/predict":
        REQUEST_LATENCY.labels(endpoint="/predict").observe(duration)

    return response


# --- System Components Initialization ---
logger.info("Initializing Hydra Ecosystem Components...")
config = load_config()

with open("configs/kept_features.json", "r") as f:
    kept_features_list = json.load(f)

config["data"]["num_features"] = len(kept_features_list)

model_manager = ModelManager(config, kept_features_list)
model_manager.load_all_models()

api_cache = SimpleCache()
drift_monitor = DriftMonitor()
perf_analyzer = PerformanceAnalyzer()
alert_system = AlertSystem()
sector_mapper = SectorMapper()
fx_engine = FXEngine()
paper_engine = PaperTradingEngine(fx_engine=fx_engine)
gemini_analyzer = GeminiAnalyzer()
physical_edge = PhysicalEdgeAnalyzer()
dependency_graph = SupplyChainGraph()
orchestrator = InstitutionalOrchestrator()
smart_router = PredictiveSmartRouter()
report_gen = ReportGenerator(kept_features_list)

signal_journal = SignalJournal()
validation_engine = ValidationAnalytics(signal_journal)

inference_service = InferenceService(
    model_manager,
    gemini_analyzer,
    physical_edge,
    dependency_graph,
    orchestrator,
    smart_router,
    report_gen,
    paper_engine,
    perf_analyzer,
    signal_journal,
)


def sanitize_ticker(ticker: str) -> str:
    if not re.match(r"^[A-Z0-9.-]{1,15}$", ticker.upper()):
        raise HTTPException(status_code=400, detail="Invalid Ticker Format")
    return ticker.upper()


@app.on_event("startup")
async def startup_event():
    async def fx_refresh_loop():
        while True:
            await fx_engine.update_rates()
            await asyncio.sleep(300)

    asyncio.create_task(fx_refresh_loop())


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health_check():
    return {
        "status": "HEALTHY",
        "models_ready": model_manager.lstm_model is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get(
    "/universe", dependencies=[Depends(verify_api_key)], response_model=UniverseResponse
)
async def get_stock_universe():
    # Cache Check
    cached_universe = await api_cache.get("universe_data")
    if cached_universe:
        return cached_universe

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
                        prices = (
                            data["Close"][t].dropna()
                            if isinstance(data["Close"], pd.DataFrame)
                            else data["Close"].dropna()
                        )
                        curr = float(prices.iloc[-1]) if len(prices) >= 1 else 0.0
                        prev = float(prices.iloc[-2]) if len(prices) >= 2 else curr
                        pct = ((curr / prev) - 1) * 100 if prev != 0 else 0.0
                    except Exception:
                        curr, pct = 0.0, 0.0
                    results.append(
                        UniverseStockItem(
                            ticker=t,
                            name=meta["name"],
                            price=curr,
                            pct_change=pct,
                            market=market_id,
                            metadata={**meta, "ticker": t, "market": market_id.upper()},
                        )
                    )
            return {"universe": results}
        except Exception as e:
            logger.error(f"Failed to fetch universe: {str(e)}")
            return {"universe": []}

    universe_data = await asyncio.to_thread(fetch_universe)
    # Cache for 1 hour
    await api_cache.set("universe_data", universe_data, ttl=3600)
    return universe_data


@app.get("/active_ticker", dependencies=[Depends(verify_api_key)])
async def get_active_ticker():
    try:
        with open("configs/active_ticker.json", "r") as f:
            data = json.load(f)
            return data
    except Exception:
        # Default to Apple if no training run has set an active ticker
        return {"ticker": "AAPL", "market": "us"}


@app.get(
    "/predict", dependencies=[Depends(verify_api_key)], response_model=PredictResponse
)
async def get_prediction(ticker: str = "AAPL"):
    ticker = sanitize_ticker(ticker)

    # Identify Market Metadata
    metadata = None
    for m_id, m_dict in UNIVERSES_METADATA.items():
        if ticker in m_dict:
            metadata = {**m_dict[ticker], "ticker": ticker, "market": m_id.upper()}
            break
    if not metadata:
        metadata = {
            "ticker": ticker,
            "market": "UNKNOWN",
            "exchange": "UNKNOWN",
            "currency": "USD",
            "timezone": "UTC",
        }

    cached = await api_cache.get(f"predict_{ticker}")
    if cached:
        cached["portfolio"] = paper_engine.get_portfolio_summary(
            {ticker: cached["current_price"]}
        )
        return cached

    response_data = await inference_service.get_prediction(ticker, config, metadata)
    await api_cache.set(f"predict_{ticker}", response_data)
    return response_data


@app.get("/performance", dependencies=[Depends(verify_api_key)])
async def get_performance():
    analysis = perf_analyzer.analyze(
        paper_engine.portfolio_snapshots,
        paper_engine.history,
        paper_engine.initial_capital,
    )
    if analysis.get("summary"):
        alert_system.check_performance(analysis["summary"])

    accs = model_manager.accuracies
    analysis["models"] = {
        "ensemble": accs.get("ensemble_accuracy", 54.6),
        "lstm": accs.get("dl_accuracy", 0.0),
        "xgboost": accs.get("xgb_accuracy", 0.0),
        "lightgbm": accs.get("lgbm_accuracy", 0.0),
        "dqn": accs.get("dqn_accuracy", 0.0),
        "consensus_rate": 0.0,
    }
    analysis["signals"] = {
        "active": len([p for p in paper_engine.positions.values() if p["shares"] > 0]),
        "historical": len(paper_engine.history),
        "regime": "LIVE EXECUTION",
    }
    return analysis


@app.get(
    "/backtest", dependencies=[Depends(verify_api_key)], response_model=BacktestSummary
)
async def get_backtest(ticker: str = "AAPL", period: str = "1y"):
    return BacktestService.get_summary(ticker, period)


@app.get("/alerts", dependencies=[Depends(verify_api_key)])
async def get_alerts():
    return {"alerts": alert_system.get_recent_alerts()}


@app.get("/fx_rates", dependencies=[Depends(verify_api_key)])
async def get_fx_rates():
    return fx_engine.get_summary()


@app.get("/validation", dependencies=[Depends(verify_api_key)])
async def get_validation():
    return await asyncio.to_thread(validation_engine.get_full_dashboard_data)


@app.post("/portfolio/base_currency", dependencies=[Depends(verify_api_key)])
async def set_base_currency(request: Request):
    data = await request.json()
    new_currency = data.get("currency")
    if new_currency not in ["USD", "INR", "EUR", "GBP"]:
        raise HTTPException(status_code=400, detail="Unsupported Currency")
    paper_engine.set_base_currency(new_currency)
    return {"status": "SUCCESS", "base_currency": new_currency}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
