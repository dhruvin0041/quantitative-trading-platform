# BACKTEST RESULTS

## 1. Methodology
Backtesting is performed using rigorous **Walk-Forward Optimization (WFO)** to eliminate look-ahead bias and overfitting.
- **Training Window**: 252 trading days (1 Year).
- **Out-of-Sample (OOS) Test**: 90 Days.
- **Normalization**: Rolling Standard Scaler fit only on the training window to prevent leakage.

## 2. Institutional Execution Assumptions
- **Starting Capital**: $1,000,000 (Base Currency).
- **Slippage**: 0.05% per execution (Simulated for Buy and Sell).
- **Sizing**: Half-Kelly (50% of the raw Kelly fraction) capped at 10% per ticker.
- **Risk Cap**: 2% equity risk per trade based on ATR stop-loss.
- **Risk-Free Rate**: 4.0% (Annualized).

## 3. Core Metrics Performance (Aggregated OOS)
*Note: Realized metrics depend on the latest model weights and regime calibration.*

| Metric | Target | Interpretation |
| :--- | :--- | :--- |
| **Win Rate** | > 52% | Reliability of the Agentic Consensus Engine. |
| **Profit Factor** | > 1.25 | Ratio of gross profits to gross losses. |
| **Sharpe Ratio** | > 1.50 | Risk-adjusted excess return. |
| **Max Drawdown** | < 15% | Robustness against Black Swan events. |
| **Jensen's Alpha** | > 0 | Ability to outperform the S&P 500/Nifty benchmarks. |

## 4. Regime-Aware Analysis
The Meta-Ensemble effectively navigates shifts by dynamically weighting models:
- **BULL Regime**: High reliance on DL Fusion and LSTM for trend-following.
- **BEAR Regime**: Increased weight on XGBoost/LightGBM for precise mean-reversion detection.
- **VETO Rate**: Approximately 30-40% of signals are vetoed by the Risk Agent to preserve capital during high uncertainty.

## 5. Limitations
The current engine executes at daily close prices. While ATR-based stops are simulated, intraday breaches may occur before the close in live environments. Future iterations will support minute-level granularity for high-fidelity slippage simulation.
