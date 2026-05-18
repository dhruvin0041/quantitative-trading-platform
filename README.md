# Hydra Terminal: Institutional-Grade Quantitative Trading System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

## Executive Summary
Hydra Terminal is a State-of-the-Art (SOTA) quantitative trading and backtesting framework designed for institutional-grade robustness. Moving beyond simplistic technical indicators, Hydra employs a **Decentralized Multi-Agent Mesh** that fuses Deep Learning (CNN/LSTM/Transformer), Reinforcement Learning (DQN), and Gradient Boosting (XGBoost) to evaluate market anomalies and generate high-fidelity signals. 

This repository has been strictly audited to eliminate look-ahead bias, prevent data leakage, and ensure rigorous out-of-sample validity through advanced walk-forward analysis.

## Strategy Overview
Hydra does not rely on a single monolithic model. Instead, it utilizes an ensemble architecture:
*   **Alpha Agent:** Processes multi-modal inputs (technical sequences, cross-asset peer momentum, and SEC/News qualitative sentiment via NLP) to forecast directional probabilities.
*   **Risk Agent:** Acts as the strict final arbiter. It calculates Jensen's Alpha, Beta, Stampede (Crowding) Risk, and enforces fully vectorized Drawdown Circuit Breakers.
*   **Execution Agent:** Applies simulated Smart Order Routing constraints, modelling slippage and transaction costs precisely as they would occur in Dark Pools or Lit Exchanges.

## Signal Methodology
1. **Dynamic Triple Barrier Labeling:** Labels are generated dynamically based on ongoing volatility (ATR), effectively adjusting profit targets and stop losses to the current market regime.
2. **Deflated Feature Space:** Highly collinear features are surgically removed on the training set to prevent multicollinearity and network confusion while mathematically protecting critical mean-reversion indicators (e.g., Bollinger Bands, RSI).
3. **Kelly Criterion Sizing:** Capital allocation is strictly dictated by a scaled Full Kelly formula, sizing bets based on empirical win rates and current model confidence.

## Architecture
```text
backend/
├── src/
│   ├── agents/            # Multi-Agent Mesh (Alpha, Risk, Execution Orchestrator)
│   ├── data_ingestion/    # YFinance, NLP tokenization, and TA-Lib indicators
│   ├── execution/         # Kelly sizing, Signal deduplication, Risk controls
│   ├── features/          # Non-leaking sequence generation for deep learning
│   └── models/            # CNN, LSTM, Transformers, DQN, Fusion logic
├── backtester.py          # Institutional Walk-Forward Analysis engine
├── train.py               # Ensemble training pipeline
└── tests/                 # Automated validation of mathematical models
```

## Installation
### Prerequisites
* Python 3.10+
* Node.js (for the Next.js frontend, if used)

```bash
git clone https://github.com/dhruvin0041/stock-indicator-buy-sell.git
cd stock-indicator-buy-sell/backend
pip install -r requirements.txt
cp .env.example .env
```
*Note: Add your `GOOGLE_API_KEY` to the `.env` file if qualitative NLP modeling is required.*

## Usage
**1. Train the Ensemble:**
```bash
python train.py
```
This trains the Deep Learning fusion model, the XGBoost tree, and the DQN agent simultaneously on dynamically generated sequences without look-ahead bias.

**2. Institutional Backtesting:**
```bash
python backtester.py
```
Executes a highly rigorous backtest incorporating transaction costs, slippage, and a 20% drawdown circuit breaker. It outputs realistic equity curves reflecting end-of-day execution logic.

## Configuration
Hyperparameters are centralized. You can define model architecture sizes, sequence lengths (`time_steps`), and dropout rates inside `configs/model_params.yaml`. The system dynamically saves optimal parameters to `configs/optimized_params_{TICKER}.json` via Optuna (if configured).

## Backtesting Methodology
Our backtesting framework guarantees integrity:
*   **Strict Temporal Alignment:** Features at `t` are explicitly paired with execution prices at `t`, mirroring real-world end-of-day execution logic perfectly.
*   **No Repainting or Leakage:** Rolling correlation drops and forward-filling imputations are strictly confined to avoid contaminating historical paths with future data.
*   **Walk-Forward Robustness:** `backtester.py` includes a `run_walk_forward` execution mode that trains and tests in sliding windows to ensure the strategy is robust across changing regimes.

## Risk Disclosures
Algorithmic trading involves substantial risk of loss and is not suitable for all investors. The models and methodologies presented in this codebase are for educational and research purposes only. Past performance (even in simulated Walk-Forward Analysis) is not necessarily indicative of future results. 

## Limitations
*   **Intraday Volatility:** End-of-day models do not account for extreme intraday drawdowns unless explicitly modelled.
*   **Cost Assumptions:** While slippage and commissions are modelled, severe liquidity crises may result in significantly worse fills.

## Roadmap
*   **Integration with Interactive Brokers API** for live paper-trading execution.
*   **Market TimeGAN Module:** Synthetic market data generation to train the DQN on non-historical Black Swan events.
*   **Advanced Options Hedging:** Automated delta-neutral SPY hedging integration within the Execution Agent.
