# Hydra Terminal
### Institutional-Grade Quantitative Trading Research and Signal Intelligence Platform

Hydra Terminal is a high-performance framework designed for systematic alpha research and production-grade trading signal generation. It integrates decentralized multi-agent orchestration with multi-modal deep learning and physical supply chain intelligence.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CI Status](https://github.com/dhruvin0041/stock-indicator-buy-sell/actions/workflows/ci.yml/badge.svg)](https://github.com/dhruvin0041/stock-indicator-buy-sell/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker Support](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)

---

## 1. Executive Summary
Hydra Terminal is engineered to provide a robust environment for developing, testing, and deploying quantitative trading strategies. Unlike conventional platforms, Hydra employs a **Decentralized Multi-Agent Mesh** where specialized Alpha, Risk, and Execution agents collaborate to validate anomalies. The system leverages Graph Neural Networks (GNN) to map N-tier corporate dependencies and weather-based physical risks, ensuring that signals are grounded in both statistical and physical reality. The platform is architected with a strict "Zero-State" protocol, mathematically eliminating look-ahead bias and ensuring all research is reproducible and production-ready.

## 2. Key Features
- **Signal Generation:** High-fidelity BUY/SELL signals derived from multi-modal fusion.
- **Non-Repainting Logic:** Guaranteed signal stability through strict temporal anchoring.
- **Look-ahead Bias Prevention:** Point-in-time data ingestion and audited feature engineering.
- **Multi-Agent Orchestration:** Decentralized consensus engine for trade validation and risk oversight.
- **Institutional Risk Engine:** Dynamic Kelly sizing, Beta-neutral hedging, and VaR-based veto logic.
- **Advanced Backtesting:** Walk-forward analysis (WFA) with realistic slippage and liquidity modeling.
- **Generative Stress Testing:** Market TimeGAN for simulating non-historical black swan scenarios.
- **Explainable AI (XAI):** Integrated SHAP analysis for transparent mathematical reasoning.
- **Production Architecture:** FastAPI-powered backend with a Next.js 16.2 institutional command center.

## 3. Trading Strategy Overview
The core strategy operates on the hypothesis that alpha resides at the confluence of technical momentum, cross-asset lead-lag effects, and supply chain health.
- **Market Hypothesis:** Price action is a lagging indicator of structural supply chain disruptions and retail sentiment shifts.
- **Indicator Categories:** Technical (OHLCV), Fundamental (SEC EDGAR), and Alternative (Weather, Google Trends, GNN Centrality).
- **Signal Confirmation:** Requires consensus between the Alpha Agent (directional probability) and the Risk Agent (regime-adjusted safety).
- **Entry/Exit Criteria:** Dynamic Triple Barrier method adjusted by ATR-based volatility regimes.

## 4. Signal Generation Methodology
The signal pipeline is a four-stage process designed for institutional precision:
1. **Multi-Modal Ingestion:** Parallelized fetching of financial and physical data layers.
2. **Feature Propagation:** Vectorized computation of indicators and GNN-based dependency scores.
3. **Fusion Inference:** Simultaneous evaluation across CNN, LSTM, and Transformer branches.
4. **Agentic Negotiation:** The Execution Agent optimizes entry/exit timing only after the Risk Agent clears the trade for VaR compliance.

## 5. Non-Repainting and Anti-Lookahead Design
Quantitative integrity is maintained through:
- **Temporal Alignment:** Features at time *t* are strictly computed using information available prior to the close of candle *t*.
- **Locked Signals:** Once a signal is generated at a candle close, it is immutable and non-repainting.
- **Audited Pipelines:** Systematic checks prevent "future-leakage" during rolling correlation and normalization steps.

## 6. Risk Management Framework
Hydra treats risk as the primary constraint on capital allocation:
- **Stop-Loss/Take-Profit:** Volatility-adjusted barriers derived from rolling ATR.
- **Position Sizing:** Scaled Full Kelly Criterion based on empirical win-rates and model confidence scores.
- **Risk Per Trade:** Hard-coded maximum exposure limits per ticker and asset class.
- **Drawdown Protection:** Fully vectorized circuit breakers and real-time Beta-neutral hedging (Short SPY).
- **Crowding Detection:** Monitoring for "Stampede Risk" to avoid over-crowded trade entries.

## 7. Backtesting Methodology
The backtesting engine simulates realistic market conditions:
- **Data Assumptions:** Survival-bias-free data pipelines with point-in-time adjustments.
- **Execution Logic:** End-of-day (EOD) or Next-Bar-Open (NBO) fill simulation.
- **Slippage & Commission:** Configurable basis-point (bps) costs and liquidity-based slippage models.
- **Accounting:** Full margin and interest rate modeling for leveraged positions.

## 8. Walk-Forward and Robustness Testing
Robustness is verified through iterative validation:
- **Walk-Forward Analysis (WFA):** Sliding window retraining to ensure model adaptability across regimes.
- **Monte Carlo Simulations:** 10,000+ synthetic paths generated via TimeGAN to test survival during tail-risk events.
- **Expectancy Analysis:** Rigorous evaluation of the system's mathematical edge (Profit Factor vs. Win Rate).

## 9. Supported Markets and Timeframes
- **Asset Classes:** Global Equities (Primary), with cross-asset correlation support.
- **Resolutions:** M15, H1, D1.
- **Data Integration:** YFinance, SEC EDGAR (8-K/10-Q), Weather Proxies, Google Trends.

## 10. System Architecture
Hydra Terminal utilizes a decoupled micro-service architecture:
- **Data Layer:** Parallel ingestion and GNN mapping service.
- **Model Layer:** Multi-branch fusion network (CNN/LSTM/Transformer).
- **Agent Layer:** Decentralized orchestrator for Alpha, Risk, and Execution.
- **Interface Layer:** Next.js institutional monitoring hub.

## 11. Technology Stack
- **Backend:** Python 3.10+, FastAPI, Numba, Vectorized Pandas/NumPy.
- **Machine Learning:** TensorFlow, PyTorch, XGBoost, Optuna, SHAP.
- **Frontend:** Next.js 16.2, TypeScript, Tailwind CSS, Lightweight-Charts v5+.
- **DevOps:** GitHub Actions (CI), Docker Compose, Ruff, MyPy.

## 12. Repository Structure
```text
.
├── backend/
│   ├── src/
│   │   ├── agents/          # Multi-Agent Mesh (Alpha, Risk, Execution)
│   │   ├── data_ingestion/  # Physical & Financial data pipelines
│   │   ├── execution/       # Signal generation & Smart routing
│   │   ├── features/        # Non-leaking sequence builders
│   │   └── models/          # Deep Learning fusion branches
│   ├── configs/             # Hyperparameters & Regime definitions
│   ├── tests/               # Quantitative validation suite
│   ├── train.py             # Ensemble training pipeline
│   └── backtester.py        # Institutional WFA engine
├── frontend/
│   ├── app/                 # Next.js institutional dashboard
│   ├── components/          # TradingView integration
│   └── public/              # Analytics assets
└── .github/workflows/       # CI/CD pipelines (Lint, Test, Docker)
```

## 13. Installation
### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (Optional)

### Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/dhruvin0041/stock-indicator-buy-sell.git
   cd stock-indicator-buy-sell
   ```
2. Backend installation:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Frontend installation:
   ```bash
   cd ../frontend
   npm install
   ```

## 14. Configuration
System behavior is defined in `backend/configs/model_params.yaml`. Key parameters include:
- `time_steps`: Length of input sequences for temporal models.
- `risk_params`: Stop-loss thresholds, Kelly fraction, and Max Drawdown limits.
- `tickers`: List of monitored assets and their GNN dependency nodes.

## 15. Usage
### Training
Execute the multi-modal training pipeline:
```bash
python train.py --ticker TSLA --mode ensemble
```
### Backtesting
Run a walk-forward analysis with transaction costs:
```bash
python backtester.py --ticker TSLA --start 2020-01-01 --wfa
```
### Live Monitoring
Launch the institutional command center:
```bash
# Terminal 1
python api.py
# Terminal 2 (frontend directory)
npm run dev
```

## 16. Example Workflow
1. **Ingestion:** Fetch OHLCV, Weather, and Sentiment data.
2. **Analysis:** Generate GNN-based corporate dependency map.
3. **Training:** Optimize Alpha Agent on non-leaking sequences.
4. **Validation:** Perform TimeGAN stress test and WFA backtest.
5. **Execution:** Monitor real-time agent consensus via the Dashboard.

## 17. Performance Metrics Explained
- **CAGR:** Compound Annual Growth Rate.
- **Sharpe Ratio:** Risk-adjusted return (Excess Return / Volatility).
- **Sortino Ratio:** Downside-risk-adjusted return.
- **Calmar Ratio:** CAGR / Maximum Drawdown.
- **Profit Factor:** Gross Profit / Gross Loss.
- **Expectancy:** Average profit per trade in dollar or percentage terms.

## 18. Security Considerations
- **API Protection:** Use `.env` for all sensitive credentials.
- **Input Validation:** Strict schema enforcement for all alternative data streams.
- **Process Isolation:** Agents run in isolated threads to prevent resource starvation.

## 19. Limitations and Risk Disclosures
- **Black Swans:** Models are trained on historical and synthetic data; unforeseen market regime shifts can lead to significant drawdowns.
- **Liquidity:** Signals assume sufficient liquidity; execution in illiquid assets may result in worse-than-modeled slippage.
- **Slippage:** Real-world slippage can vary significantly during high-volatility events.

## 20. Testing
Hydra maintains a comprehensive test suite:
- **Unit Tests:** `pytest backend/tests/`
- **Typing:** `mypy backend/src/`
- **Linting:** `ruff check .`

## 21. CI/CD Pipeline
Automated workflows ensure code quality:
- **Linting:** Enforces PEP 8 and Ruff standards.
- **Testing:** Runs mathematical validation on every pull request.
- **Docker:** Automated image builds for backend and frontend services.

## 22. Docker Usage
Orchestrate the full stack:
```bash
docker-compose up --build
```

## 23. Infrastructure and Deployment
- **Local:** Optimized for high-thread CPU and NVIDIA GPU acceleration.
- **Cloud:** Ready for deployment via Kubernetes (EKS/GKE) for agent scaling.

## 24. Monitoring and Observability
- **Logging:** Structured JSON logs for all agentic decisions.
- **Metrics:** Real-time tracking of agent latency and inference confidence.
- **Health Checks:** Automated service-level monitoring for API nodes.

## 25. Development Workflow
1. Feature branch creation.
2. Implementation with type-safety and docstrings.
3. Unit testing and regression validation.
4. CI-passed PR for review and merge.

## 26. Roadmap
- [ ] Integration with Interactive Brokers TWS API.
- [ ] Multi-asset class expansion (FX, Futures).
- [ ] Real-time sentiment analysis via LLM-based news scrapers.

## 27. Contributing
Contributions from quantitative researchers and engineers are welcome. Please refer to `CONTRIBUTING.md` for coding standards and PR protocols.

## 28. License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## 29. Disclaimer
Trading financial markets involves significant risk of loss. This software is provided for research and educational purposes only. Past performance is not indicative of future results. The authors assume no liability for financial losses incurred through the use of this system. Always perform your own due diligence.
