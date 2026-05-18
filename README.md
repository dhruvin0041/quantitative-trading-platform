# Hydra Terminal: Institutional-Grade Quantitative Trading & Signal Intelligence

## Executive Summary
Hydra Terminal is a high-performance quantitative research and systematic trading platform designed for institutional-grade signal generation. The system moves beyond traditional technical analysis by employing a **Decentralized Multi-Agent Mesh** that fuses deep learning, reinforcement learning, and alternative data streams. By integrating physical intelligence—including global supply chain mapping via Graph Neural Networks (GNN) and weather-proxy risk assessment—Hydra provides a comprehensive Alpha research environment and production-ready signal engine.

The platform is engineered with a "Zero-State" protocol to ensure maximum quantitative rigor, eliminating look-ahead bias and data leakage through strict temporal alignment and non-repainting signal logic.

---

## Key Features
- **Multi-Agent Consensus Engine:** Decentralized orchestration where Alpha, Risk, and Execution agents negotiate trades.
- **Advanced Model Fusion:** Hybrid architecture combining CNN, LSTM, Transformers, and XGBoost for robust feature extraction.
- **Physical Intelligence Layer:** Supply chain disruption detection via N-tier corporate dependency mapping and weather coordinates.
- **Generative Stress Testing:** Market TimeGAN module for simulating 10,000+ non-historical black swan scenarios.
- **Non-Repainting Signal Logic:** Mathematically guaranteed signal stability with zero look-ahead bias.
- **Institutional Risk Management:** Full Kelly Criterion sizing, Beta-neutral hedging, and Stampede (Crowding) risk detection.
- **Explainable AI (XAI):** Integration of SHAP for full transparency into the mathematical reasoning behind every decision.
- **Next.js Command Center:** Institutional monitoring hub for real-time agent consensus and supply chain risk visualization.

---

## Trading Strategy Overview
Hydra Terminal operates on a multi-modal hypothesis that alpha is found at the intersection of technical momentum, cross-asset lead-lag relationships, and qualitative supply chain health.

- **Market Hypothesis:** Price action is a lagging indicator of physical supply chain shifts and retail sentiment trends.
- **Indicator Framework:** A composite of technical (OHLCV), fundamental (SEC EDGAR), and alternative (Google Trends, Weather) data.
- **Entry Logic:** Multi-agent consensus requiring Alpha Agent probability thresholds and Risk Agent volatility-adjusted validation.
- **Exit Logic:** Dynamic Triple Barrier labeling adjusted by ATR-based volatility regimes.

---

## Signal Generation Methodology
Signals are generated through a rigorous pipeline designed to capture transient market anomalies:
1. **Data Ingestion:** Parallel fetching of financial and physical data layers.
2. **Feature Engineering:** Vectorized computation of technical indicators and GNN-based centrality scores.
3. **Model Fusion:** Simultaneous evaluation across CNN (spatial), LSTM (temporal), and Transformer (attention-based) branches.
4. **Agentic Negotiation:** The Alpha Agent proposes a trade; the Risk Agent performs a "Veto" check against VaR and crowding metrics.

---

## Non-Repainting and Anti-Lookahead Design
To ensure the integrity of quantitative research, Hydra Terminal enforces strict architectural boundaries:
- **Point-in-Time Data:** All features are computed using only information available at time *t*.
- **Temporal Anchoring:** Backtesting execution logic mirrors real-world fills, ensuring that signals are locked at the close of the candle.
- **Data Leakage Audits:** Automated checks prevent future data from influencing historical model training.

---

## Risk Management Framework
Risk is not an afterthought; it is the core arbiter of the system:
- **Risk Agent Veto:** Absolute authority to block signals failing safety guardrails.
- **Position Sizing:** Scaled Full Kelly formula based on empirical win rates and model confidence.
- **Drawdown Protection:** Fully vectorized circuit breakers and daily loss limits.
- **Hedging:** Beta-neutral logic with automated short SPY correlation balancing.
- **Crowding Detection:** Monitoring for "Stampede Risk" to avoid over-crowded trade entries.

---

## Backtesting Methodology
Our backtesting engine provides a realistic simulation of institutional execution:
- **Slippage & Commission Modeling:** Precise accounting for transaction costs and liquidity constraints.
- **Execution Latency:** Simulated delays to model real-world fill prices.
- **Equity Curve Analysis:** Detailed reporting on drawdown, recovery factors, and underwater periods.

---

## Walk-Forward and Robustness Testing
Hydra utilizes Walk-Forward Analysis (WFA) to ensure out-of-sample validity:
- **Sliding Window Optimization:** Models are periodically retrained to adapt to changing market regimes.
- **Monte Carlo Simulations:** 10,000 synthetic paths generated via TimeGAN to test survival during non-historical events.
- **Jensen’s Alpha:** Continuous measurement of skill-based returns versus market beta.

---

## Supported Markets and Timeframes
The system is ticker-agnostic and designed for flexibility:
- **Asset Classes:** Equities (Primary), with support for cross-asset peer analysis.
- **Timeframes:** Configurable from Intraday (M15/H1) to Daily (D1) resolutions.
- **Data Sources:** YFinance, SEC EDGAR, Google Trends, and custom weather/port proxies.

---

## System Architecture
Hydra Terminal follows a decoupled, micro-agent architecture:
- **Backend (Python 3.10+):** FastAPI-powered ML framework.
- **Frontend (Next.js 16.2):** Institutional monitoring command center.
- **Agents:** Decentralized nodes (Alpha, Risk, Execution) communicating via an internal orchestration mesh.
- **Data Layer:** Multi-modal ingestion pipelines with GNN dependency mapping.

---

## Technology Stack
- **Languages:** Python (ML/Backend), TypeScript (Frontend).
- **ML Frameworks:** TensorFlow/PyTorch (Deep Learning), XGBoost (Gradient Boosting), Optuna (Optimization).
- **Performance:** Numba (JIT compilation), Vectorized Pandas/NumPy operations.
- **API/Web:** FastAPI, Uvicorn, Next.js, Tailwind CSS.
- **Visualization:** TradingView Lightweight-Charts v5+.
- **Monitoring:** SHAP (XAI), Structured Logging.

---

## Repository Structure
```text
.
├── backend/
│   ├── src/
│   │   ├── agents/          # Multi-Agent Mesh (Orchestrator, Risk, Alpha)
│   │   ├── data_ingestion/  # Market & Alternative data pipelines
│   │   ├── execution/       # Signal generation & Smart routing
│   │   ├── features/        # Sequence builders & TA-Lib integration
│   │   └── models/          # DL Branches (CNN, LSTM, Transformer, GAN)
│   ├── configs/             # Hyperparameters & Model definitions
│   ├── tests/               # Unit and integration tests
│   ├── train.py             # Ensemble training pipeline
│   └── backtester.py        # Institutional WFA engine
├── frontend/
│   ├── app/                 # Next.js 16.2 App Router
│   ├── components/          # Institutional chart components
│   └── public/              # Static assets
└── .github/workflows/       # CI/CD pipelines
```

---

## Installation

### Backend Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/dhruvin0041/stock-indicator-buy-sell.git
   cd stock-indicator-buy-sell/backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (e.g., GOOGLE_API_KEY for NLP)
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install Node.js dependencies:
   ```bash
   npm install
   ```

---

## Configuration
Hyperparameters are managed via `backend/configs/model_params.yaml`. This allows for granular control over:
- Sequence lengths (`time_steps`)
- Model architecture (layer counts, dropout rates)
- Risk parameters (stop-loss, take-profit, Kelly fraction)
- Ticker-specific overrides

---

## Usage

### 1. Alpha Research & Training
Train the ensemble model (CNN/LSTM/Transformer/XGBoost/DQN) on a specific ticker:
```bash
python train.py --ticker AAPL
```

### 2. Institutional Backtesting
Run a rigorous backtest with realistic slippage and transaction costs:
```bash
python backtester.py --ticker AAPL --mode walk-forward
```

### 3. Real-Time Inference (Simulation)
Start the live inference agent for real-time signal monitoring:
```bash
python live_inference.py
```

### 4. Launch Command Center
Start the Next.js dashboard:
```bash
npm run dev
```

---

## Example Workflow
1. **Clean Slate:** Run `python clean_artifacts.py` to reset the neural pathway for a new ticker.
2. **Ingestion:** Fetch multi-modal data including weather and Google Trends.
3. **Training:** Execute `train.py` to optimize the ensemble on historical sequences.
4. **Validation:** Run `backtester.py` to confirm out-of-sample robustness.
5. **Monitoring:** Open `http://localhost:3000` to track agent consensus and risk levels.

---

## Performance Metrics Explained
The system evaluates performance using institutional-grade metrics:
- **Sharpe Ratio:** Risk-adjusted return relative to volatility.
- **Sortino Ratio:** Risk-adjusted return focusing on downside volatility.
- **Calmar Ratio:** CAGR divided by Maximum Drawdown.
- **Maximum Drawdown (MDD):** The largest peak-to-trough decline.
- **Profit Factor:** Gross profit divided by gross loss.
- **Jensen’s Alpha:** Excess return generated above the market benchmark (Beta).

---

## Security Considerations
- **Credential Management:** Use `.env` files for all API keys. Never commit sensitive credentials.
- **Data Integrity:** All ingestion pipelines use checksums and validation schemas to prevent data corruption.
- **Process Isolation:** Backend agents run in isolated threads to prevent cascading failures.

---

## Limitations and Risk Disclosures
- **Black Swan Events:** While TimeGAN models extreme scenarios, unforeseen market dislocations can exceed model parameters.
- **Liquidity Assumptions:** Backtesting assumes a level of liquidity that may not be available during market stress.
- **Hardware Requirements:** Training complex ensemble models requires significant CPU/GPU resources.

---

## Testing
Hydra Terminal maintains a rigorous testing suite:
- **Unit Tests:** Located in `backend/tests/`, covering indicator accuracy and model logic.
- **Integration Tests:** Validating the multi-agent mesh communication.
- **Frontend Linting:** Enforced via ESLint and TypeScript type-checking.

---

## CI/CD Pipeline
Automated workflows are defined in `.github/workflows/ci.yml`:
- **Python Linting:** `ruff check .` for PEP 8 compliance.
- **Next.js Validation:** `npm run lint` and `tsc --noEmit`.
- **Automated Testing:** Execution of core mathematical unit tests on every push.

---

## Docker Usage
Deploy the entire stack using Docker Compose:
```bash
docker-compose up --build
```
This orchestrates the FastAPI backend, Next.js frontend, and necessary proxies.

---

## Infrastructure and Deployment
- **Local Dev:** FastAPI (8000) and Next.js (3000).
- **Production:** Recommended deployment via Kubernetes for agent scaling and GPU-accelerated inference nodes.
- **Data Storage:** Artifacts and models are stored locally with support for S3-compatible cloud storage.

---

## Monitoring and Observability
- **Structured Logging:** All agent decisions and risk vetos are logged with mathematical justification.
- **XAI Integration:** SHAP values are computed for every signal to provide human-readable reasoning.
- **Health Checks:** Real-time monitoring of agent latency and API connectivity.

---

## Development Workflow
1. Feature branch creation.
2. Implementation with strict PEP 8 and TypeScript typing.
3. Unit test addition.
4. `ruff` and `npm run lint` validation.
5. Pull request with documentation update.

---

## Roadmap
- [ ] **Interactive Brokers Integration:** Full bridge for live paper-trading execution.
- [ ] **Advanced Options Hedging:** Automated delta-neutral SPY hedging within the Execution Agent.
- [ ] **Global Macro Expansion:** Integration of interest rate and inflation (CPI/PCE) data streams.

---

## Contributing
We welcome contributions from quantitative researchers and engineers. Please ensure all pull requests follow the rigorous testing and linting standards of the repository.

---

## License
Distributed under the MIT License. See `LICENSE` for more information.

---

## Disclaimer
**Trading financial instruments involves significant risk.** This software is for educational and research purposes only. The authors assume no responsibility for financial losses incurred through the use of this system. Always perform your own due diligence and consult with a certified financial advisor before trading.
