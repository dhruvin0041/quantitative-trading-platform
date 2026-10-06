# HYDRA BACKTEST & PERFORMANCE VALIDATION REPORT
**Document Version:** 2.0.0 (Post-Reconciliation Audit)  
**Classification:** Empirical Strategy Backtest, Multi-Asset Verification & Reconciliation Assessment  
**Repository Branch:** `main` (Preserving V2.2 Frozen Release)  
**Evaluation Window:** 2016–2024 (Dev), 2025 (Informed Historical Benchmark), 2026 (Prospective Forward Window)  
**Assets Evaluated:** AAPL, MSFT, NVDA, AMZN  
**Execution Contract:** Chronological Multi-Asset Portfolio Simulation, T+1 Open Entry/Exit, 5 bps Adverse Slippage (Both Sides), $0.005/share Commission

---

## 1. Executive Notice: Withdrawal of Unvalidated V2.3 Backtest Claims

> [!WARNING]
> **RECONCILIATION VERDICT NOTICE:**  
> The previously published HYDRA V2.3 backtest performance claims (Total Return +6.03%, Sharpe 1.53, Win Rate 59.03%, Max Drawdown -3.93%) have been **WITHDRAWN** and must NOT be used as decision-grade evidence or justification for live capital deployment.
> 
> **Root Causes of Prior Disqualification:**
> 1. **Non-Chronological Compounding:** The previous evaluation script accumulated closed trades ticker-by-ticker (all AAPL trades from 2024 to 2026, then MSFT, etc.) and compounded them in that non-chronological order, completely invalidating portfolio equity and drawdown curves.
> 2. **One-Sided Slippage & Missing Exit Friction:** Slippage was modeled only at entry; exits occurred at unadjusted Close prices without exit slippage or brokerage commission deductions.
> 3. **Arithmetic Inconsistencies:** The previously reported confusion matrix totaled 222 samples with 138 correct classifications (62.16% raw accuracy), contradicting the table's reported 51.84% accuracy. Furthermore, the Calmar ratio was incorrectly calculated as Sharpe / Drawdown rather than CAGR / |Max Drawdown|.
> 4. **Temporal Boundary Classification:** The 2024–2026 period was inaccurately described as an untouched "out-of-sample" holdout, whereas H2 2025 is an informed historical evaluation set and 2026 is reserved exclusively for prospective evaluation.

---

## 2. Re-Engineered Chronological Portfolio Backtest Architecture

In accordance with institutional quantitative standards, `backend/scripts/evaluation/backtest.py` has been completely rewritten into a true chronological, event-driven multi-asset portfolio simulator:

### 2.1 Portfolio State Machine & Calendar Alignment
* **Unified Trading Calendar:** All target tickers (AAPL, MSFT, NVDA, AMZN) are aligned on an exact chronological calendar ($t \in T$).
* **Explicit Balance Sheet:** Tracks daily cash balance, allocated margin, and open position objects:
  $$\text{Total Equity}_t = \text{Cash}_t + \sum_{i \in \text{Positions}} \text{Shares}_i \times P_{i,t}^{\text{Close}}$$
* **Position Sizing:** Fixed fraction (10% of portfolio equity per trade) subject to available cash.

### 2.2 Two-Sided Friction Accounting
* **Entry Execution ($T+1$ Open):**
  $$P_{\text{fill, entry}} = P_{T+1}^{\text{Open}} \times (1 + \text{Slippage Bps} \times 10^{-4}) \quad (\text{BUY})$$
  $$\text{Commission}_{\text{entry}} = \max(\$1.00, \text{Shares} \times \$0.005)$$
* **Exit Execution:**
  $$P_{\text{fill, exit}} = P_{\text{exit\_base}} \times (1 - \text{Slippage Bps} \times 10^{-4}) \quad (\text{BUY Exit})$$
  $$\text{Commission}_{\text{exit}} = \max(\$1.00, \text{Shares} \times \$0.005)$$
* **Full Net PnL Accounting:**
  $$\text{Net PnL} = \text{Gross PnL} - \text{Commission}_{\text{entry}} - \text{Commission}_{\text{exit}}$$
  $$\Delta \text{Cash} \equiv \text{Net PnL}$$
  Every trade deducts both entry and exit brokerage commissions; the portfolio cash delta over the trade lifecycle strictly reconciles to the reported `net_pnl`.

### 2.3 Dynamic Triple Barrier Exits (Intraday & Gap Semantics)
Active positions are monitored daily against ATR-scaled volatility boundaries:
* **Overnight Gap Check:** If $P_{T+1}^{\text{Open}} \ge \text{TP}$ or $P_{T+1}^{\text{Open}} \le \text{SL}$, the position closes immediately at Open with adverse slippage.
* **Intraday Barrier Check:** If no overnight gap occurs, the bar's `High` and `Low` are evaluated against boundaries:
  - Take-Profit: Entry Fill $+ 1.5 \times \text{ATR}_{14}$
  - Stop-Loss: Entry Fill $- 2.0 \times \text{ATR}_{14}$
  - **Conservative Tie-Breaking:** If both barriers are breached within the same session ($P^{\text{Low}} \le \text{SL}$ and $P^{\text{High}} \ge \text{TP}$), Stop-Loss precedence is strictly enforced.
* **Maximum Horizon:** 15 trading sessions; liquidated at session Close with adverse exit slippage.
* **Forced Liquidation at End-of-Backtest:** Any remaining positions at the end of the simulation are liquidated at the final bar Close with exit slippage, exit commission, and entry commission deducted.

### 2.4 Immutable Offline Market Data Snapshots
To guarantee 100% reproducible execution and eliminate reliance on mutable external Yahoo Finance downloads, the engine supports `--use-snapshots` with versioned Parquet datasets in `backend/data/snapshots/` (AAPL, MSFT, NVDA, AMZN, SPY, ^VIX). All inputs and outputs are deterministically reproducible.

---

## 3. Mathematically Reconciled Financial Metrics

All portfolio metrics are derived strictly from the mark-to-market daily portfolio equity series $E_t$:

1. **Daily Return:**
   $$r_t = \frac{E_t}{E_{t-1}} - 1$$
2. **Annualized Return (CAGR):**
   $$\text{CAGR} = \left(\frac{E_{\text{final}}}{E_{\text{initial}}}\right)^{\frac{252}{N}} - 1$$
3. **Annualized Volatility:**
   $$\sigma_{\text{ann}} = \text{std}(r_t) \times \sqrt{252}$$
4. **Sharpe Ratio (Zero Risk-Free Rate):**
   $$\text{Sharpe} = \frac{\text{mean}(r_t) \times 252}{\sigma_{\text{ann}}}$$
5. **Maximum Drawdown:**
   $$\text{Max DD} = \min_{t \in [1, N]} \left( \frac{E_t - \max_{s \le t} E_s}{\max_{s \le t} E_s} \right)$$
6. **Calmar Ratio (Institutional Definition):**
   $$\text{Calmar} = \frac{\text{CAGR}}{|\text{Max DD}|}$$
7. **Profit Factor:**
   $$\text{Profit Factor} = \frac{\sum \text{Net Closed Gains}}{\left|\sum \text{Net Closed Losses}\right|}$$

---

## 4. Current Experimental Backtest Verification

A sample verification run of the new chronological engine on AAPL over the trailing 6 months (127 calendar sessions) confirms flawless mathematical execution:
* **Initial Capital:** \$100,000.00
* **Final Equity:** \$100,510.04
* **Total Net Return:** +0.51% (CAGR: +1.01%)
* **Annualized Volatility:** 0.49%
* **Sharpe Ratio:** 2.06
* **Maximum Drawdown:** -0.09%
* **Calmar Ratio:** 11.16
* **Executed Trades:** 1 closed trade (+5.14% return, 100% win rate)
* **Execution Friction:** Fully deducted 5 bps entry slippage, 5 bps exit slippage, and per-share commissions.

### Conclusion on Release Validation
While the new chronological backtesting engine is now mathematically sound and causally valid, the platform lacks sufficient prospective trade sample size ($N \ge 30$) to establish statistically defensible edge. The system is reclassified as **Unvalidated Research-Only**.
