# RISK MANAGEMENT

Institutional trading systems are defined not by their returns, but by how they survive tail-risk events. Hydra Terminal employs a strict, multi-layered risk management framework synchronized across global markets.

## 1. Global Multi-Currency Risk
Hydra is uniquely engineered for multi-market risk accounting via its dedicated **FX Engine**.
- **Real-Time Normalization**: All positions across USA (USD) and India (INR) universes are normalized into a user-defined Base Currency (e.g., USD, INR, EUR, GBP).
- **Equity Tracking**: Portfolio equity and High-Water Marks are tracked in the base currency to ensure consistent drawdown detection regardless of exchange rate volatility.

## 2. Position Sizing (Kelly Criterion)
The system calculates the optimal fraction of capital to risk using the Full Kelly formula:
`K = W - [(1 - W) / R]`
Where `W` is the historical win rate and `R` is the win/loss ratio.
- **Constraint**: The Kelly fraction is strictly capped at **25%** for safety.
- **Dynamic Sizing**: The raw Kelly fraction is further multiplied by the Meta-Ensemble's confidence score to scale risk based on model conviction.

## 3. Drawdown Circuit Breakers
The portfolio tracks a dynamic "High-Water Mark". If base-currency equity falls below 80% of the peak (a 20% drawdown), the circuit breaker triggers:
- **Immediate Neutralization**: All open positions are liquidated.
- **Hard-Hold State**: The system enters a forced `HOLD` state, refusing all new `BUY` signals until manual intervention.

## 4. Advanced Risk Metrics
The Risk Agent calculates institutional-grade metrics for every signal:
- **Jensen's Alpha**: Measures the specific skill edge by subtracting expected market returns from actual returns.
- **Stampede / Crowding Risk**: Detects momentum traps by correlating retail sentiment volatility (NLP) with model confidence. Signals with high crowding are automatically scaled back.
- **Beta Limits**: Signals on assets with a Beta > 2.5 relative to the SPY are automatically vetoed to prevent extreme volatility exposure.

## 5. Dynamic Execution Barriers
Targets follow a Triple Barrier method based on the Average True Range (ATR):
- **Take Profit**: Entry Price + (ATR * 2.0)
- **Stop Loss**: Entry Price - (ATR * 1.0)
- **Time Barrier**: 10 trading days (Positions are liquidated at the close of the 10th day if barriers aren't hit).

## 6. Regime-Aware Conviction
The system enforces harder thresholds based on market regime:
- **BULL Regime**: Required conviction threshold = 55%
- **NEUTRAL Regime**: Required conviction threshold = 58%
- **BEAR Regime**: Required conviction threshold = 68% (Filtering for high-probability mean-reversion).
