# SYSTEM ARCHITECTURE

## Overview
Hydra Terminal is an institutional-grade, multi-agent quantitative trading system designed for scalability, zero look-ahead bias, and absolute determinism.

## 1. Data Flow Pipeline
1. **Ingestion**: Raw OHLCV data is fetched via Yahoo Finance.
2. **Alternative Data**: Options Flow, Insider Trading, and Analyst Revisions are fetched and merged into the primary pipeline.
3. **Feature Engineering**: 40+ quantitative indicators (RSI, ADX, Bollinger Bands, ATR) are computed.
4. **Triple Barrier Labeling**: Intrabar High/Low prices are checked against ATR-based dynamic barriers to generate labels without look-ahead bias.

## 2. Model Flow
1. **Tabular & Sequential Split**: Data is bifurcated into cross-sectional rows for boosting models (XGBoost, LightGBM) and sequences for deep models.
2. **Deep Learning Core**: Temporal Fusion Transformer (TFT), PatchTST, and LSTM analyze sequences.
3. **Boosting Core**: XGBoost and LightGBM analyze the raw feature snapshot.
4. **Reinforcement Learning**: DQN evaluates the output of all other models.
5. **Meta-Ensemble**: Dynamically weights the models based on rolling 30-day out-of-sample accuracy to generate a final probability matrix.

## 3. Consensus Engine
The consensus engine requires a super-majority (e.g., 2/3 agreement) across the dynamically weighted ensemble. 
If models disagree or confidence is low, the signal is `VETOED` and defaults to `HOLD`.

## 4. Execution Engine & Risk Management
1. **Sizing**: Uses the Kelly Criterion adjusted by the ensemble confidence score.
2. **Circuit Breakers**: A hard 20% drawdown limit halts all execution logic.
3. **Paper Trading**: Executes simulated trades, recording fills, slippage, and commissions in SQLite.

## 5. Deployment Flow
- **FastAPI**: Serves inference endpoints and Prometheus metrics.
- **Next.js**: Provides the real-time institutional dashboard.
- **Docker**: The entire stack is containerized for deterministic deployment.
- **GitHub Actions**: CI/CD pipelines automate testing, linting, and weekly Walk-Forward Optimization backtests.
