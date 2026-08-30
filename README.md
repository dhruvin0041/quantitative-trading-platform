# Multi-Agent Quantitative Trading Platform

A State-of-the-Art (SOTA) financial codebase combining traditional Machine Learning predictive models with a **Decentralized Multi-Agent Large Language Model (LLM) Mesh**. The system is designed to provide comprehensive, institutional-grade quantitative consensus by fusing technical price action with qualitative analysis, macroeconomic data, and rigorous risk management.

## 🌟 System Architecture

The platform is split into two primary interfaces, backed by a high-performance Python engine.

### 1. The Main Dashboard (`/`)
The traditional quantitative command center.
*   **Price Action Canvas:** High-performance TradingView lightweight-charts displaying real-time price data, volume histograms, and dynamic swing markers.
*   **Signal Consensus:** Visualizes the mathematical agreement across multiple ML models (XGBoost, SVM, LSTM).
*   **Risk Engine:** Displays real-time Value-at-Risk (VaR), Beta-neutral hedging ratios, and portfolio exposure.
*   **Theme Engine:** Seamlessly switches between Dark Mode (Institutional Night) and Light Mode (Warm Sepia) via CSS semantic tokens.

### 2. The Agentic Engine Dashboard (`/agents`)
A real-time window into the "brains" of the operation. This dashboard uses Server-Sent Events (SSE) to stream the LangGraph-orchestrated thoughts of 9 distinct AI agents as they debate and formulate a trading decision.

#### The LangGraph Agent Pipeline:
**Phase 1: Analyst Briefings**
*   **Fundamentals Analyst:** Evaluates P/E ratios, revenue growth, and earnings reports.
*   **Sentiment Analyst:** Gauges social media and market narrative momentum.
*   **News Analyst:** Analyzes macroeconomic indicators, interest rates, and geopolitical risk.
*   **Technical Analyst:** Integrates directly with the `technical_prediction_tool` to fetch XGBoost/SVM probabilities and chart structures.

**Phase 2: The Researcher Debate**
*   **Bullish Researcher & Bearish Researcher:** These agents engage in a multi-turn, stateful debate. They read the Phase 1 briefings and argue directly against each other's points to uncover hidden risks or asymmetric upside.

**Phase 3: Risk & Execution**
*   **Lead Trader:** Synthesizes the entire debate and formulates a concrete execution plan (LONG, SHORT, or PASS).
*   **Risk Manager:** The ultimate safeguard. Uses the `timegan_stress_test_tool` to simulate 10,000 synthetic market paths. If the maximum drawdown exceeds safe parameters, this agent has absolute **VETO** power over the Lead Trader.
*   **Portfolio Manager:** Finalizes the trade if it passes the Risk Manager's guardrails.

---

## 🛠 Tech Stack

**Frontend (Next.js 16.2 Institutional Hub):**
*   **Framework:** Next.js (App Router), React 18, TypeScript.
*   **Styling:** Tailwind CSS v4, Semantic CSS Variables (for flawless Light/Dark mode).
*   **Charting:** TradingView `lightweight-charts`.
*   **Streaming:** Custom React Hooks for consuming Server-Sent Events (SSE).
*   **Icons:** Lucide React.

**Backend (FastAPI & Agent Mesh):**
*   **Framework:** FastAPI (Uvicorn), Python 3.12+.
*   **Orchestration:** LangGraph (Stateful Multi-Agent Graphs).
*   **LLM Provider:** Google Gemini API (`gemini-3.6-flash` / `gemini-pro`).
*   **ML & Data:** XGBoost, Scikit-Learn (SVM), TimeGAN architectures, Pandas, NumPy.
*   **Artifacts:** File-system registry for `.joblib` and `.h5` model weights.

---

## 🚀 Environment Setup & Installation

### 1. Prerequisites
*   Python 3.10+
*   Node.js 20+

### 2. Backend Setup
Navigate to the backend directory and set up the Python environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory:
```env
API_KEY=your_institutional_key
FRONTEND_URL=http://localhost:3000
GOOGLE_API_KEY=your_gemini_api_key
```

### 3. Frontend Setup
Navigate to the frontend directory and install dependencies:
```bash
cd frontend
npm install
```

Create a `.env.local` file in the `frontend/` directory:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Running the System
**Terminal 1 (Backend):**
```bash
cd backend
source venv/bin/activate
python api.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

Access the command center at `http://localhost:3000`. Access the Agentic Engine at `http://localhost:3000/agents`.

---

## 📜 System Mandates & Operational Rules (The "Zero-State" Protocol)

As defined by the core system architecture:
1.  **Zero-Hardcoding Policy:** The engine is built to dynamically handle various tickers. Ticker-specific logic is never hardcoded.
2.  **The "Zero-State" Reset:** Before running the ML pipeline on a new ticker, the system must wipe previous memory pathways to prevent neural contamination. (See `backend/scripts/ops/clean_artifacts.py`).
3.  **Local State Integrity:** Files like `GEMINI.md` and `graphify-out/` are local-only assets and should never be committed to source control.

---
*Developed by the Quantitative Research Team. Built for survival in unprecedented markets.*
