# RISK MANAGEMENT

Institutional trading systems are defined not by their returns, but by how they survive tail-risk events. Hydra Terminal employs a strict, multi-layered risk management framework.

## 1. Position Sizing (Kelly Criterion)
The system calculates the optimal fraction of capital to risk using the Full Kelly formula:
`K = W - [(1 - W) / R]`
Where `W` is the historical win rate and `R` is the win/loss ratio.
**Constraint**: The Kelly fraction is strictly capped at 25% to prevent over-leverage, and further multiplied by the Meta-Ensemble's confidence score.

## 2. Drawdown Circuit Breakers
The portfolio tracks a "High Water Mark". If current equity falls below 80% of the peak equity (a 20% drawdown), the circuit breaker triggers:
- All open positions are immediately liquidated.
- The system enters a hard `HOLD` state and refuses all `BUY` signals until manual intervention.

## 3. Dynamic Stop Loss & Take Profit (Triple Barrier)
Targets are not fixed percentages. They are dynamically calculated based on the Average True Range (ATR):
- **Take Profit**: Current Price + (ATR * 2.0)
- **Stop Loss**: Current Price - (ATR * 1.0)
- **Time Horizon**: 10 trading days.

## 4. Market Regime Filters
If the SPY drops below its 200-day moving average, the system classifies the regime as `BEAR`.
In a `BEAR` regime, the required conviction threshold for `BUY` signals is increased from 55% to 68%, filtering out low-conviction mean-reversion trades.

## 5. Stampede / Crowding Risk
If retail sentiment volatility (derived from NLP analysis) is extremely high while the model is also highly confident, the trade is flagged for "Crowding Risk". Position sizing is automatically scaled back to avoid momentum traps.
