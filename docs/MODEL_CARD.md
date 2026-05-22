# MODEL CARD

## 1. Model Details
- **Architecture**: Meta-Ensemble (LSTM, XGBoost, LightGBM, DQN, TFT, Informer, PatchTST).
- **Task**: 3-Class Time-Series Classification (BUY, HOLD, SELL) via Triple Barrier Labeling.
- **Version**: 4.0.0

## 2. Intended Use
- **Primary Use**: Generating systematic trading signals for S&P 500 equities.
- **Out of Scope**: High-Frequency Trading (HFT), illiquid micro-caps, automated execution without human oversight.

## 3. Training Data
- **Sources**: Yahoo Finance OHLCV, SEC EDGAR (mocked), Options Flow (mocked).
- **Timeframe**: 2020-01-01 to Present.
- **Preprocessing**: Robust scaling, missing value forward-filling, dynamic feature deflation to remove multicollinearity while protecting mean-reverting indicators.

## 4. Evaluation & Metrics
- **Validation**: Walk-Forward Optimization (WFO) utilizing a 1-year training window and a 90-day out-of-sample testing window.
- **Primary Metrics**: Jensen's Alpha, Sharpe Ratio, Maximum Drawdown, Precision, and Recall on minority classes (BUY/SELL).

## 5. Limitations & Risks
- **Regime Change**: The models assume future market dynamics will vaguely resemble historical data. Extreme black swan events (e.g., COVID-19 crash) may degrade performance temporarily until the Meta-Ensemble recalibrates weights.
- **Slippage**: Simulated at 0.1%. In fast-moving markets, true slippage may exceed this, degrading the realized Sharpe ratio.

## 6. Biases
- **Survivorship Bias**: Currently mitigated by tracking delisted equities in the paper-trading environment, but historical data fetching may still suffer if external APIs overwrite delisted tickers.
