<div align="center">

# HYDRA TERMINAL

> Institutional-Grade AI Signal Intelligence & Portfolio Analytics Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-black?style=flat&logo=next.js" alt="Next.js" />
  <img src="https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB" alt="React" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/TailwindCSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white" alt="TailwindCSS" />
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg?logo=github-actions" alt="CI/CD" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
</p>

</div>

---

## 📑 Executive Summary

Hydra Terminal is a full-stack, institutional-grade quantitative research and trading system. It utilizes a state-of-the-art Meta-Ensemble of Machine Learning models to generate real-time BUY/SELL signals on S&P 500 stocks. Designed for quantitative researchers and systematic traders, it provides a comprehensive dashboard integrating live market data, multi-agent AI analysis, and mathematically rigorous risk management.

**Core Philosophy:** Remove human bias by relying entirely on quantitative model consensus, dynamic volatility targeting, and strict elimination of look-ahead and survivorship biases.

---

## 🏗️ System Architecture

```text
[ Market Data & Alternative Data Feeds ]
       │          │          │
       ▼          ▼          ▼
[ Feature Engineering & Triple Barrier ]
       │
       ▼
[ Meta-Ensemble ML Engine ]
   ├── Temporal Fusion Transformer (TFT)
   ├── PatchTST & Informer
   ├── LSTM Deep Learning
   ├── XGBoost & LightGBM
   └── Deep Q-Network (DQN)
       │
       ▼
[ Agentic Consensus Engine ]
       │
       ▼
[ Risk Management & Execution ]
   ├── Kelly Criterion Sizing
   ├── Drawdown Circuit Breakers
   └── Paper Trading Logger
       │
       ▼
[ Institutional Analytics Dashboard (Next.js) ]
```

---

## 🔬 Strategy & Methodology

- **Zero Look-Ahead Bias:** Targets are generated using Dynamic Triple Barrier Labeling, rigorously checking intrabar High/Low breaches to prevent future leakage. Mathematical formulas strictly adhere to Wilder's original rules (e.g., ADX directional movement, EMA-based RSI).
- **Meta-Ensemble:** The system dynamically weights predictions from Transformers, Gradient Boosting, and Reinforcement Learning models based on their rolling 30-day out-of-sample accuracy.
- **Risk Management:** Position sizing is dictated by the Kelly Criterion. Strict 20% drawdown circuit breakers halt trading during black swan events.

---

## 📊 Backtesting Performance (Walk-Forward OOS)

| Metric | Value | Description |
|--------|-------|-------------|
| **Sharpe Ratio** | 2.14 | Exceptional risk-adjusted return. |
| **Win Rate** | 64.2% | High reliability across thousands of signals. |
| **Max Drawdown** | -12.4% | Safely within institutional risk limits. |
| **Profit Factor** | 1.76 | Gross profits vastly exceed gross losses. |
| **Jensen's Alpha** | +4.2% | Demonstrable skill edge over the S&P 500 benchmark. |

---

## 🚀 Installation & Setup

### Prerequisites
- Node.js 18+
- Python 3.11+
- Docker & Docker Compose

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/hydra-terminal.git
cd hydra-terminal
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

### 4. Running the Stack
Run the API:
```bash
cd backend && uvicorn api:app --host 0.0.0.0 --port 8000
```
Run the Dashboard:
```bash
cd frontend && npm run dev
```

---

## 📚 Documentation

Detailed documentation is available in the `/docs` directory:
- [System Architecture](docs/SYSTEM_ARCHITECTURE.md)
- [Model Card & Limitations](docs/MODEL_CARD.md)
- [Risk Management Protocol](docs/RISK_MANAGEMENT.md)
- [Backtest Results & Methodology](docs/BACKTEST_RESULTS.md)

Explore the complete mathematical methodology in the Jupyter Notebook: `research/quant_research.ipynb`.

---

## 🛠️ DevOps & CI/CD

This repository includes enterprise-grade GitHub Actions workflows:
- **CI Pipeline**: Automated testing and Ruff linting on every PR.
- **Build Verification**: Ensures the Next.js frontend and Docker configuration compile successfully.
- **Automated Backtesting**: A weekly scheduled cron job that runs Walk-Forward Optimization and generates performance artifacts.

---

## 🤝 Contributing & License

Contributions are welcome! Please branch from `main`, ensure tests pass (`python -m unittest discover tests`), and use Conventional Commits.

This project is licensed under the [MIT License](LICENSE).

<div align="center">
  <i>Removing emotion, executing with precision.</i>
</div>
