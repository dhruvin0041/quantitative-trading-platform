# 🐉 Hydra Terminal: AI-Powered Stock Trading System

![Hydra Terminal](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![Next.js](https://img.shields.io/badge/Next.js-16.2-black) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

Hydra Terminal is an institutional-grade, AI-driven stock analysis and prediction system. It utilizes a **5-Model Hybrid Ensemble** (Deep Learning, Tree-based, and Reinforcement Learning) to analyze historical data, technical indicators, and live market regimes to generate highly accurate **BUY**, **SELL**, and **HOLD** signals.

---

## 🧠 The 5-Model Ensemble Architecture

The system's core intelligence relies on five independent neural architectures, dynamically weighted based on their recent predictive accuracy.

1. **XGBoost (Tree-Based Classifier):** Analyzes the latest technical indicators and tabular data to classify the current market state.
2. **LSTM (Time Series Memory):** A dual-layer Long Short-Term Memory network that processes 60-day sequence windows to understand temporal price momentum.
3. **CNN (Pattern Recognition):** Treats the last 60 days of OHLCV and indicator data as a 1D "image" to recognize visual chart patterns (e.g., head and shoulders, flags).
4. **Transformer (Advanced Sequence Attention):** Uses Multi-Head Attention to assign weight to specific historical days that have the most mathematical impact on the current price.
5. **DQN (Reinforcement Learning):** A PyTorch-based Deep Q-Network that takes the predictions of the other 4 models as its "State" and learns to optimize for maximum portfolio profit, penalizing drawdowns.

---

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
