# 🐍 Hydra Terminal: Institutional-Grade Quantitative Trading System

<div align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success" alt="Status" />
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="Python" />
  <img src="https://img.shields.io/badge/Next.js-14+-black" alt="Next.js" />
  <img src="https://img.shields.io/badge/FastAPI-0.109+-teal" alt="FastAPI" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</div>

## 📌 Executive Summary

**Hydra Terminal** is a State-of-the-Art (SOTA) multi-agent quantitative trading system. It leverages a multi-modal machine learning fusion network (CNN, LSTM, Transformer, FinBERT, XGBoost) and advanced alternative data (weather, supply chain dependency graphs) to generate non-repainting, highly calibrated trading signals.

Built for hedge-fund operations, Hydra Terminal enforces strict risk controls, beta-neutral hedging logic, and stampede (crowding) risk detection. It includes a built-in Paper Trading engine for simulated execution, minimizing slippage via a predictive smart router.

---

## 🏛️ System Architecture

### 1. Data Ingestion & Engineering
- **Market Data:** Live OHLCV fetching via yfinance with dynamic feature scaling and technical indicator generation (Bollinger Bands, MACD, RSI, ATR).
- **Alternative Data:**
  - **Supply Chain Graph:** GNN-based propagation risk assessment across N-tier corporate relationships.
  - **Physical Edge:** Weather and port congestion tracking for supply chain disruption forecasting.
  - **NLP Sentiment:** FinBERT and Gemini-powered analysis of SEC EDGAR filings and real-time news headlines.

### 2. Multi-Modal Fusion Network
The brain of the system is a deeply ensembled neural network with 5 specialized branches:
- **LSTM Branch:** Captures long-term sequential momentum.
- **CNN Branch:** Extracts local spatial chart patterns.
- **Transformer Branch:** Employs multi-head attention for regime shifting.
- **Peer-Context Branch:** Lead-lag correlation analysis against sector peers.
- **FinBERT Branch:** Analyzes unstructured sentiment data.
*(Outputs are further ensembled with an XGBoost classifier and a DQN Reinforcement Learning agent).*

### 3. Agentic Orchestration & Risk Management
- **Institutional Orchestrator:** Resolves conflicting model signals using a strict consensus algorithm.
- **Risk Manager:** Implements the Kelly Criterion for dynamic position sizing, Beta tracking against the SPY, and Jensen's Alpha scoring.
- **Stampede Risk:** Detects retail crowding and enforces automatic position scale-backs.
- **Paper Trading Engine:** Simulates real-world execution with configurable slippage, maintaining portfolio cash, equity, and position state.

---

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 20+
- Gemini API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dhruvin0041/stock-indicator-buy-sell.git
   cd stock-indicator-buy-sell
   ```

2. **Environment Configuration**
   Create a `.env` file in the `backend/` directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   API_KEY=dev-secret-key-1234
   FRONTEND_URL=http://localhost:3000
   ```

3. **Run via Docker Compose (Recommended)**
   ```bash
   docker-compose up --build -d
   ```

### Manual Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 🔒 Security & Observability

Hydra Terminal implements production-grade backend hardening:
- **API Key Authentication:** All endpoints are protected by `X-API-Key` headers.
- **Rate Limiting:** Protects against API abuse and DDoS attacks.
- **CORS Restrictions:** Strictly bound to the frontend origin.
- **Prometheus Metrics:** Available at `/metrics` for latency and request tracking.
- **Structured JSON Logging:** Ready for Datadog or ELK stack ingestion.

---

## 🧪 Testing & Validation

The system relies on rigorous quantitative validation:
- **Cross-Sectional Factor Modeling**
- **Walk-Forward Validation** (WFA)
- **Generative Stress Testing:** Uses TimeGAN to simulate non-historical market crashes.

---

## 📄 License
This project is licensed under the MIT License. Strictly intended for educational and research purposes. Do not use for live trading without thorough financial vetting.
