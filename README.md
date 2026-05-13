# 🐉 Hydra Terminal: SOTA 2025 AI-Powered Trading System

![Hydra Terminal](https://img.shields.io/badge/Status-Elite-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![Next.js](https://img.shields.io/badge/Next.js-16.2-black) ![Google Gemini](https://img.shields.io/badge/LLM-Gemini_1.5_Flash-purple) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

Hydra Terminal is a State-of-the-Art (SOTA 2025) institutional-grade trading system. It moves beyond simple numeric prediction by employing a **Multi-Modal Agentic Architecture** that combines numeric deep learning with qualitative economic reasoning.

---

## 🚀 Tier 2 Institutional Upgrades (SOTA 2025)

The system has been upgraded to match elite hedge fund standards with three primary new capabilities:

### 1. Qualitative LLM Gatekeeper (Google Gemini)
The system now integrates **Google Gemini 1.5 Flash** to act as a qualitative qualitative analyst. It parses real-time **SEC EDGAR filings (8-K/10-Q)** and news to detect:
* **Moving Targets:** Shifts in corporate performance metrics.
* **Litigious Risk:** Hidden uncertainty in regulatory language.
* **Economic Reasoning:** Blending narrative alpha with numeric signals.

### 2. Cross-Asset Lead-Lag (Contextual Alpha)
The model no longer treats stocks as islands. It dynamically identifies a **High-Correlation Sector Peer** (e.g., MSFT for AAPL) and feeds its real-time behavior into a dedicated **Context Branch**, allowing the model to learn sector-wide trends before they manifest in a single ticker.

### 3. Beta-Neutral Risk Management (Institutional Hedging)
Hydra Terminal now calculates the rolling **Beta** of every asset relative to the S&P 500 (SPY).
* **Full Kelly Sizing:** Growth-optimized position sizing based on historical edge.
* **Beta-Neutral Hedge:** The API outputs a suggested **Short SPY ratio** to neutralize market risk, ensuring the portfolio is "Market Neutral."

---

## 🧠 The Hybrid Ensemble Architecture

The system's intelligence relies on a multi-modal fusion of six independent neural and agentic architectures:

1. **Qualitative Gatekeeper (Gemini LLM):** Economic reasoning and fundamental alpha extraction.
2. **Transformer (Attention):** Assigns mathematical weight to critical historical events.
3. **LSTM (Memory):** Captures temporal price momentum and sequence dependencies.
4. **CNN (Visual):** Recognizes chart patterns via price/indicator matrix analysis.
5. **XGBoost (Tabular):** Optimizes for technical indicator feature importance.
6. **DQN (Reinforcement Learning):** Learns optimal entry/exit policies through self-correction.

---

## 🛡️ Verification: Walk-Forward Analysis (WFA)

To ensure institutional robustness, Hydra Terminal uses **Walk-Forward Analysis** instead of standard backtests.
* **Sliding Windows:** Iteratively trains on 1-year windows and tests on 90-day "unseen" periods.
* **Regime Agnostic:** Ensures the model remains profitable across Bull, Bear, and Sideways markets, preventing the common "overfitting" trap of retail bots.

## ⚙️ Feature Engineering & Technical Indicators

The data pipeline enriches standard OHLCV data with over 20 advanced technical and macro indicators:

* **Momentum & Trend:** RSI (14), MACD (12, 26, 9), EMA (9, 21, 50, 200)
* **Volatility:** ATR (14), Bollinger Bands (20 & 120 period)
* **Volume Analysis:** VWAP, OBV, Money Flow Index (MFI), Chaikin Money Flow (CMF), Volume Anomaly %
* **Advanced Visuals:** Ichimoku Cloud components, Stochastic Oscillator
* **Macro Environment:** VIX (Volatility Index) for regime detection, 10-Year Treasury Yields

---

## 🛡️ Risk Management & Meta-Modeling

* **Macro Kill-Switch:** A Gaussian Mixture Model continuously monitors the VIX. If a high-panic market regime is detected, the system automatically overrides all BUY signals to protect capital.
* **Dynamic Confidence Thresholds:** The ensemble requires a weighted consensus (default > 0.70 probability) before issuing a strong signal.
* **High-Sensitivity Pivots:** Chart markers accurately pinpoint local Swing Highs (Peaks) and Swing Lows (Valleys).

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.10+
* Node.js & npm (for the frontend)

### 1. Clone the Repository
```bash
git clone https://github.com/dhruvin0041/stock-indicator-buy-sell.git
cd stock-indicator-buy-sell
```

### 2. Backend Setup (Python)
```bash
cd backend
pip install -r requirements.txt
```
*(If `requirements.txt` is missing, ensure you install `tensorflow`, `torch`, `xgboost`, `pandas`, `numpy`, `scikit-learn`, `ta`, `yfinance`, `fastapi`, `uvicorn`, `transformers`)*

### 3. Frontend Setup (Next.js)
```bash
cd frontend
npm install
```

---

## 💻 Usage Guide

The system is designed to be operated in a specific lifecycle: **Optimize -> Train -> Backtest -> Execute**.

### Phase 1: Hyperparameter Optimization
Use Optuna to find the best neural network architectures and ensemble weights for a specific stock.
```bash
cd backend
python optimize.py --ticker AAPL --trials 30
```
*(For the entire market, run `python optimize_universal.py`)*

### Phase 2: System Training
Train all 5 models using the optimized parameters.
```bash
python train.py
```

### Phase 3: Backtesting
Simulate trading over the last year to verify the model's edge. This generates an equity curve chart (`backtest_results.png`).
```bash
python backtester.py
```

### Phase 4: Live Inference & Dashboard
Start the FastAPI backend:
```bash
python api.py
```
Start the Next.js Hydra Terminal frontend:
```bash
cd frontend
npm run dev
```
Open `http://localhost:3000` in your browser to view the live AI dashboard.

---

## 🛠️ Tech Stack

* **Machine Learning:** TensorFlow/Keras, PyTorch, XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy, TA (Technical Analysis), yfinance
* **Optimization:** Optuna
* **Backend API:** FastAPI, Uvicorn
* **Frontend UI:** Next.js, React, TailwindCSS, Lightweight-Charts

---
*Disclaimer: This software is for educational and research purposes only. Do not use it to trade real money without understanding the risks involved in algorithmic trading.*
