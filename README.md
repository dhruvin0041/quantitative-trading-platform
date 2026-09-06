# Quantitative Trading Platform

A backtested research platform that combines a multi-branch deep learning ensemble with an LLM-powered agent pipeline. The system generates directional trading signals for equities using 27 engineered features, probability-calibrated model outputs, and a multi-layer risk veto system.

> **Scope**: This is a solo-built research project with strict temporal isolation throughout the ML pipeline. It is not a live-trading system, not a hedge fund, and not financial advice. All backtest results come from out-of-sample evaluation with purged time-series splits and embargo gaps to prevent look-ahead bias.

---

## Backtest Results (RELIANCE.NS, Jan–Apr 2026)

Evaluated on 21 executed signals (out of 25 total; 4 were vetoed by the risk layer). Source: [`backtest_results/backtest_summary.json`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/backtest_results/backtest_summary.json).

| Metric | Value | Note |
|---|---|---|
| Win Rate | 61.9% | 13/21 correct directional calls (5-day forward return) |
| Profit Factor | 2.59 | Sum of winning returns / sum of losing returns |
| Sharpe Ratio | 6.12 | ⚠️ Based on only 21 trades — insufficient for statistical confidence |
| Max Drawdown | −2.0% | Half-Kelly sizing capped at 10%, with 2% per-trade risk cap |
| Vetoed Rate | 16.0% | 4 signals blocked by the risk engine |
| Avg Confidence | 72.8% | Weighted consensus across calibrated model outputs |

**Caveat**: These results are from a single ticker over ~4 months. The Sharpe of 6.12 is computed from 21 trade returns and is not statistically robust. The system's own `StatisticalValidityEngine` suppresses Sharpe/Sortino when N < 60 returns. Individual model validation accuracies (DL Fusion: ~49.7%, XGBoost: ~45.6%, DQN: 50.0%) reflect a genuinely hard 3-class prediction problem (Sell/Hold/Buy via Dynamic Triple Barrier labels), where the ensemble consensus and risk veto provide the actual edge.

---

## ML Ensemble Architecture

The prediction pipeline uses a 4-tier ensemble:

### Tier 1: Multi-Branch Deep Learning Fusion

Six neural branches fused via cross-modal multi-head attention ([`fusion_network.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/neural/fusion_network.py)):

- **LSTM** — 2-layer stacked LSTM with BatchNorm and Dropout
- **1D-CNN** — Conv1D → MaxPool → Dense for local pattern extraction
- **Transformer** — Multi-head attention encoder with sinusoidal positional encoding
- **TCN** — Temporal Convolutional Network with causal dilations `[1, 2, 4, 8, 16]`
- **PatchTST** — Channel-independent patch embedding with attention
- **Peer Context** — LSTM branch processing a sector peer's features for lead-lag signals

A `CrossModalAttention` layer (4-head, key_dim=64) dynamically weights branch outputs, with residual connections and LayerNorm. Three output heads: direction (sigmoid), price range (Huber loss), and signal class (softmax over Sell/Hold/Buy).

Trained weights: [`latest_fusion_weights.weights.h5`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/artifacts/latest_fusion_weights.weights.h5) (41 MB).

### Tier 1b: Temporal Fusion Transformer (TFT) Quantile Forecaster

A separate TFT branch ([`tft_agent.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/neural/tft_agent.py)) with Gated Residual Networks, LSTM local context, and multi-head temporal attention. Outputs 5 price quantiles (p10, p25, p50, p75, p90) used for stop-loss and take-profit calibration.

### Tier 2: Gradient-Boosted Tree Models

- **XGBoost** — Multi-class soft-probability classifier with Optuna-tuned hyperparameters
- **LightGBM** — Multi-class classifier with GPU acceleration
- **CatBoost** — Multi-class classifier (training code present; may not always have a saved artifact)

### Tier 3: Meta-Ensemble (Stacked Generalization)

An ElasticNet Logistic Regression meta-learner ([`meta_ensemble.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/ensemble/meta_ensemble.py)) takes the 3-class probability distribution from each base model plus a one-hot market regime vector, and outputs the final consensus signal. It also provides model contribution analysis and prediction uncertainty (inter-model dispersion).

### Tier 4: DQN Reinforcement Learning Agent

A Dueling Double DQN ([`dqn_agent.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/rl/dqn_agent.py)) with prioritized experience replay and reward shaping that penalizes drawdown and excessive hold time. Acts on a state vector combining tabular features with DL and XGB probability outputs.

### Probability Calibration

Raw model outputs are calibrated before entering the consensus engine:

- **Isotonic regression** per class per model (DL, XGB, LGBM) — fitted strictly on held-out validation predictions ([`model_calibrator.joblib`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/artifacts/model_calibrator.joblib))
- **Temperature scaling** (T=2.5) for deep learning softmax outputs in high-uncertainty regimes
- Calibration audit (Brier scores: XGB 0.218, LGBM 0.202, Consensus 0.209) in [`calibration_audit.json`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/backtest_results/calibration_audit.json)

---

## Feature Engineering

27 stationary features survive the feature selection pipeline ([`kept_features.json`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/configs/kept_features.json)). Categories:

- **Trend spreads**: MA20 vs MA50, EMA9 vs EMA21, Price vs EMA9/EMA21
- **Volatility**: Bollinger Band width, ATR regime ratio, VIX level
- **Momentum**: RSI, ADX, MACD histogram, relative strength vs SPY
- **Volume**: OBV change, volume ratio (current / 20-day SMA)
- **Stationarized Z-scores**: Rolling Z-scores at 20/50/120-day windows for RSI, BB Position, MACD Hist, Return, and Volume Ratio — these transform raw indicators into mean-reverting, stationary signals

The full technical indicator library ([`technical_indicators.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/data_ingestion/technical_indicators.py)) computes 50+ raw indicators including Ichimoku Cloud, Shannon entropy, Hurst exponent proxy, and Amihud illiquidity — most are deflated during feature selection.

### Data Leakage Prevention

The codebase includes multiple explicit leakage safeguards:

- **Purged time-series split with 10-day embargo** — prevents Triple Barrier label horizon from bleeding into training ([`model_loader.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/model_loader.py#L20-L48))
- **Target column exclusion** — sequence builder explicitly drops `target_*` and `future_*` columns ([`sequence_builder.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/features/sequence_builder.py#L19-L23))
- **Scaler fitted only on training data** — `StandardScaler.fit()` is called on pre-2024 data and never refitted on validation or live data
- **Automated leakage detection** — [`leakage_proof.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/scripts/evaluation/leakage_proof.py) compares features from truncated vs full datasets and flags discrepancies > 1e-6

---

## Risk Management

### Computed Metrics (Live)

- **Value-at-Risk (VaR)** — Historical 95th-percentile VaR computed from return distributions in [`portfolio_analytics.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/portfolio_analytics.py) and [`paper_trading.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/paper_trading.py). The `RiskAgent` aggregates position-weighted VaR and vetoes if the portfolio exceeds 5%.
- **Beta** — Empirical covariance/variance calculation vs SPY in [`risk_manager.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/risk_manager.py#L5-L27). Displayed as a hedge ratio but not automatically executed (no short-SPY orders are placed).
- **Kelly sizing** — Full Kelly formula with 25% institutional cap. Backtest evaluation uses **Half-Kelly capped at 10%** per position. A safety audit found negative raw Kelly on out-of-sample data, so the live cap is set to 1% in [`risk_params.json`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/configs/risk_params.json).
- **Crowding / Stampede risk** — Detects when retail sentiment volatility × signal confidence exceeds 0.8, triggering scale-back.
- **Jensen's Alpha** — Measures skill vs beta exposure (annualized excess return).
- **Market regime detection** — HMM or GMM on log returns and realized volatility, classifying into Bear/Neutral/Bull states ([`regime_detector.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/regime_detector.py)).
- **Feature drift monitoring** — KS test and PSI across all features, saved to [`drift_report.json`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/backtest_results/drift_report.json).

### Veto System

The `RiskAgent` ([`orchestrator.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/agents/orchestrator.py)) enforces:

1. Portfolio VaR > 5% → entire batch vetoed
2. Sector exposure > 15% → lowest-conviction signals in that sector dropped
3. Confidence < 65% → signal filtered pre-evaluation
4. Institutional Risk Index > 80 → execution vetoed (Critical/Panic regime)

---

## LLM Agent Pipeline (Agentic Dashboard)

A 9-node LangGraph pipeline ([`langgraph_orchestrator.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/agents/langgraph_orchestrator.py)) streamed to the frontend at `/agents` via Server-Sent Events:

**Phase 1 — Analyst Briefings** (4 agents, sequential):
Fundamentals, Sentiment, News, and Technical analysts. Each prompts the Gemini API with the ticker name. The Technical Analyst invokes a `technical_prediction_tool` that wraps the ML ensemble.

**Phase 2 — Researcher Debate** (2 agents, 2-turn cycle):
Bullish and Bearish researchers read all Phase 1 reports and argue against each other for 2 rounds.

**Phase 3 — Risk & Execution** (3 agents):
Lead Trader synthesizes the debate into a LONG/SHORT/PASS decision. Risk Manager invokes a `timegan_stress_test_tool` and can VETO. Portfolio Manager auto-approves if no veto.

**Current limitations** (marked for improvement):
- Analyst nodes prompt the LLM with only the ticker symbol — they do not inject real SEC filings, social media data, or news feeds into the context. The LLM relies on its parametric knowledge.
- `technical_prediction_tool` returns **hardcoded mock probabilities** rather than querying the live ensemble (the live feature array is not wired through).
- `timegan_stress_test_tool` returns **hardcoded drawdown values** (−24.5% / −12.3%). The TimeGAN architecture classes exist ([`timegan.py`](file:///D:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/models/generative/timegan.py)) but inference is bypassed in favor of `np.random.normal`. No trained TimeGAN weights exist.
- Portfolio Manager unconditionally returns `APPROVED` — human-in-the-loop review is planned but not implemented.

The pipeline requires a `GOOGLE_API_KEY` for the Gemini API. Model fallback cascades across multiple Gemini model variants on 429/503 errors.

---

## Frontend

A Next.js dashboard with 4 routes:

| Route | Purpose |
|---|---|
| `/` | Main trading dashboard: lightweight-charts price canvas, signal consensus panel, risk metrics, paper trading portfolio |
| `/agents` | LLM agent pipeline viewer with SSE streaming |
| `/performance` | Portfolio performance analytics (Sharpe, Sortino, Calmar, equity curve, drawdown) |
| `/validation` | Empirical signal validation and model accuracy tracking |

---

## Tech Stack

### Backend

| Component | Library | Version |
|---|---|---|
| API Framework | FastAPI + Uvicorn | — |
| Deep Learning | TensorFlow/Keras, PyTorch | — |
| Gradient Boosting | XGBoost, LightGBM, CatBoost | — |
| Feature Engineering | ta, pandas, NumPy, SciPy, statsmodels | — |
| Hyperparameter Search | Optuna | — |
| Experiment Tracking | MLflow | — |
| Explainability | SHAP (offline), feature-importance heuristic (live) | — |
| LLM Integration | google-genai, LangChain, LangGraph | — |
| Supply Chain Graph | NetworkX | — |
| Market Data | yfinance | — |
| NLP | HuggingFace Transformers (FinBERT) | — |
| Observability | prometheus-client (metrics endpoint) | — |

> **Note on requirements.txt**: The file lists 30 core dependencies. Some imports used in the codebase (`langgraph`, `langchain-core`, `langchain-google-genai`, `hmmlearn`, `prometheus-client`) are not listed and must be installed separately. This is a known gap.

### Frontend

| Component | Library | Version |
|---|---|---|
| Framework | Next.js (App Router) | 16.2.4 |
| UI Library | React | 19.2.4 |
| Language | TypeScript | 5.x |
| Styling | Tailwind CSS | 4.x |
| Charting | lightweight-charts | 5.1.0 |
| UI Components | Radix UI, shadcn/ui, cmdk | — |
| Animation | Framer Motion | 12.x |
| Icons | Lucide React | — |
| Theme | next-themes (Dark/Light mode via CSS tokens) | — |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 20+
- A Google Gemini API key (required for the agent pipeline; the ML ensemble runs without it)

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `backend/.env`:
```env
API_KEY=your_api_key_here
FRONTEND_URL=http://localhost:3000
GOOGLE_API_KEY=your_gemini_api_key
```

### Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:
```env
NEXT_PUBLIC_API_KEY=your_api_key_here
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Running

**Terminal 1 (Backend):**
```bash
cd backend
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
python api.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

Dashboard: `http://localhost:3000` · Agent Pipeline: `http://localhost:3000/agents`

---

## Artifact Reset (Zero-State Protocol)

Before training the ML pipeline on a new ticker, run:

```bash
python backend/scripts/ops/clean_artifacts.py
```

This clears cached model weights, scalers, and feature artifacts from the previous ticker to prevent stale state from leaking between runs. The system is designed to handle any ticker dynamically — no ticker-specific logic is hardcoded in the ML backend.

---

## Project Structure

```
├── backend/
│   ├── api.py                          # FastAPI entry point
│   ├── artifacts/                      # Trained model weights (.h5, .joblib, .pth, .json)
│   ├── backtest_results/               # Backtest trades, metrics, calibration audit
│   ├── configs/                        # Feature lists, model params, Optuna results
│   ├── scripts/
│   │   ├── training/                   # train.py, optimize.py, calibrate_models.py
│   │   ├── evaluation/                 # backtest.py, final_audit.py, leakage_proof.py
│   │   ├── ops/                        # clean_artifacts.py, maintenance scripts
│   │   └── research/                   # SHAP analysis, calibration audits
│   └── src/
│       ├── agents/                     # LangGraph orchestrator, institutional agent mesh
│       ├── data_ingestion/             # Market data, technical indicators, NLP, alt data
│       ├── execution/                  # Inference, backtesting, risk management, paper trading
│       ├── features/                   # Sequence builder with leakage prevention
│       ├── models/                     # Neural branches, boosting, ensemble, generative, RL
│       ├── optimization/              # Optuna search with purged time-series splits
│       └── utils/                      # GPU config, caching
├── frontend/                           # Next.js 16.2 dashboard
├── notebooks/                          # Research notebook (outline only, not executed)
├── docs/                               # Implementation reports, backtest methodology
└── mlflow.db                           # MLflow experiment tracking database
```

---

## License

MIT
