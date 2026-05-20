# Institutional Quant Platform: Project Hydra

Hydra is a state-of-the-art (SOTA) 2026 multi-modal, multi-agent hedge fund execution and research platform. It autonomously generates trading signals, assesses risk dynamically, and executes paper trades.

## Key Features

- **Multi-Modal Data Ingestion**: Processes technical (OHLCV), macro (VIX regimes), alternative (Supply chain maps, port weather), and sentiment (SEC EDGAR, News via BERT/Gemini).
- **Hybrid Ensemble Models**: Fuses temporal signals from LSTMs, pattern recognition from CNNs, and long-range dependencies from Transformers with tabular XGBoost predictions.
- **Multi-Agent Orchestration**: 
  - *Alpha Agent*: Synthesizes ensemble predictions into buy/sell signals.
  - *Risk Agent*: Validates trades against VaR, beta neutrality, and stampede/crowding risks with absolute veto power.
  - *Execution Agent*: Optimizes venue routing through a simulated Smart Order Router (SOR).
- **Institutional Risk Management**: Incorporates Full Kelly criterion sizing, dynamic drawdown circuit breakers, cross-sectional factor modeling, and **Mean-Variance Portfolio Optimization**.
- **Institutional Validation**:
  - **Walk-Forward Optimization (WFO)**: Rigorous out-of-sample testing with rolling parameter re-optimization.
  - **Generative Stress Testing**: Monte Carlo simulations using GANs to estimate Max Drawdown boundaries.
  - **Probability Calibration**: Isotonic Regression to ensure model confidence maps to empirical success rates.
- **Production-Grade Infrastructure**:
  - **Experiment Tracking**: Systematic logging of hyperparameters and results for strategy reproducibility.
  - **Model Drift Monitoring**: Kolmogorov-Smirnov statistical tests to detect feature distribution shifts in real-time.
  - **Market Regime Detection**: GMM-based macro kill-switch (Normal vs. Panic) to protect capital during volatility spikes.
  - **Broker Interoperability**: Pluggable architecture supporting Alpaca, Interactive Brokers, and custom fix-adapters.
- **Explainable AI (XAI)**: Generates human-readable rationales behind every signal.

## Tech Stack

- **Language**: Python 3.11+, TypeScript
- **Framework**: FastAPI (Async Backend), Next.js 14+ (App Router)
- **Machine Learning**: TensorFlow, Keras, XGBoost, Scikit-Learn
- **Data Engineering**: Pandas, Numba, YFinance, Open-Meteo
- **Orchestration**: Custom Agentic Mesh
- **Frontend**: Lightweight Charts (TradingView), Tailwind CSS v4
- **Observability**: Prometheus, Structured JSON Logging
- **Deployment**: Docker, Uvicorn

## Prerequisites

- Python 3.11+
- Node.js 20+
- pnpm or npm
- API Keys: Set up a `.env` in `backend/` with `API_KEY=dev-secret-key-1234` and `FRONTEND_URL=http://localhost:3000`

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/dhruvin0041/stock-indicator-buy-sell.git
cd stock-indicator-buy-sell
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
```

### 4. Running the Platform

**Start the Backend (FastAPI):**
```bash
cd backend
python api.py
```
*API will run on http://localhost:8000*

**Start the Frontend (Next.js):**
```bash
cd frontend
npm run dev
```
*Dashboard will run on http://localhost:3000*

## Architecture

### Directory Structure

```
├── backend/
│   ├── api.py                   # Async FastAPI server & Rate Limiter
│   ├── live_inference.py        # Scaler-aligned data pipeline
│   ├── train.py                 # Out-of-sample DQN and Ensemble Training
│   ├── src/
│   │   ├── agents/              # Multi-Agent Orchestrator (Alpha, Risk, Exec)
│   │   ├── data_ingestion/      # Market, Alt-data, and NLP ingestion
│   │   ├── execution/           # Smart Routing, Kelly Sizing, Paper Trading
│   │   ├── features/            # Sequence Builders
│   │   └── models/              # Fusion Network, XGBoost, GANs, Calibrators
├── frontend/
│   ├── app/                     # Next.js App Router (Dashboard)
│   ├── components/              # TradingView charts & UI
│   └── public/
```

### Request Lifecycle

1. Client (Next.js) requests `/predict?ticker=AAPL` with `X-API-Key`.
2. `api.py` checks rate limits and in-memory caches.
3. `live_inference.py` pulls historical market data and applies dynamic triple-barrier labeling and feature deflation.
4. `nlp_processor.py` tokenizes live news/SEC data using FinBERT.
5. The 5-Model Ensemble generates predictions, which are calibrated and weighted by historical accuracy.
6. The `InstitutionalOrchestrator` passes predictions through the Alpha, Risk, and Execution agents.
7. The `RiskAgent` computes VaR, Kelly fraction, and stampede risk, applying a veto if conditions breach strict hedge fund parameters.
8. The `PaperTradingEngine` logs the executed trade and updates the portfolio.
9. A validated `PredictResponse` is returned to the frontend.

## Advanced Quant Components

### Generative Stress Testing
The `MarketTimeGAN` (`synthetic_gan.py`) simulates non-historical black swan events (fat tails, volatility clustering). The `DQNAgent` is stress-tested across these synthetic paths to estimate Maximum Drawdown limits under unprecedented market chaos.

### Regime Detection
`regime_detector.py` uses Gaussian Mixture Models on the VIX to separate the market into "Normal" and "Panic" clusters. If the market enters a Panic regime, aggressive scaling-back is triggered.

### Cross-Sectional Factors
`factor_model.py` uses PCA to extract statistical factors from the S&P 500, isolating systematic risk (beta) from idiosyncratic alpha.

## Security & Observability

- **API Hardening**: All API endpoints use Pydantic models (`src/schemas.py`) for strict contract enforcement.
- **Authentication**: Custom API key header (`X-API-Key`) validation.
- **Rate Limiting**: Custom token-bucket middleware (50 req / 60s).
- **Metrics**: Prometheus instrumentation exposes `/metrics` with `api_requests_total` and `api_request_latency_seconds`.
- **Structured Logging**: All backend logs are output as JSON for ingestion into Datadog/ELK.

## Disclaimer

This software is for research and educational purposes only. Do not use these signals for live financial trading without extensive paper-trading validation and legal compliance.
