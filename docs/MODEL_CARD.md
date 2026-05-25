# MODEL CARD

## 1. Model Details
- **Architecture**: Meta-Ensemble (LSTM Fusion, XGBoost, LightGBM, DQN).
- **Task**: 3-Class Time-Series Classification (SELL, HOLD, BUY) via Triple Barrier Labeling.
- **Ensemble Logic**: Weighted consensus with dynamic uncertainty quantification.
- **Version**: 5.1.0 (2026 Paradigm)

## 2. Model Pipeline
Hydra utilizes a 4-branch intelligence pipeline:
- **DL Fusion**: Dual-branch LSTM architecture analyzing ticker-specific time-steps and cross-sector peer correlations.
- **GBDT Agents**: Gradient Boosted Decision Trees (XGBoost/LightGBM) optimized for high-dimensional tabular feature sets.
- **RL Policy**: Deep Q-Network (DQN) agent trained for sequential action optimization (Execution timing).
- **Qualitative Alpha**: **Gemini 2.0 Flash** performing zero-shot fundamental analysis on SEC 8-K filings and news context.

## 3. Training Data
- **Financial OHLCV**: S&P 500 and Nifty 50 constituents (2022-Present).
- **Alternative Signals**: Global port weather coordinates (Open-Meteo) and search interest proxies.
- **Fundamental Context**: Real-time SEC EDGAR RSS feed.
- **Preprocessing**: Robust scaling, forward-filling, and cross-sectional factor neutralization (PCA-based).

## 4. Evaluation & Verification
- **Validation Strategy**: Walk-Forward Optimization (WFO) with 252-day training windows and 90-day out-of-sample segments.
- **Explainability**: SHAP (SHapley Additive exPlanations) values logged for every decision to provide mathematical transparency of feature impact.
- **Data Integrity**: Systematic "Zero-State" protocol enforced via `clean_artifacts.py` to prevent neural pathway contamination across tickers.

## 5. Limitations & Operational Risks
- **Black Swan Events**: Rapid, non-linear regime shifts (e.g., flash crashes) may exceed the Meta-Ensemble's recalibration latency.
- **API Dependency**: Heavy reliance on Yahoo Finance and Open-Meteo availability; system fallback defaults to `HOLD` on data interruption.
- **Slippage**: Simulation assumes 0.05% slippage; high-volatility environments or illiquid tickers may realize higher execution costs.

## 6. Biases
- **Survivorship Bias**: Mitigated by ticker-neutral architecture, though historical backtests are limited to current constituents.
- **Recency Bias**: The Meta-Ensemble dynamically weights models based on recent performance, which may underperform during sudden mean-reversion at cycle peaks.
