# HYDRA V2.4.3 Institutional Research Report — Hierarchical Regime-Switching Strategy

**Date:** October 16, 2026  
**Auditor / Principal Quantitative Researcher:** HYDRA Research Mesh  
**Corpus / Repository:** `dhruvin0041/quantitative-trading-platform`  
**Target Asset:** AAPL (Warmup: 2015-01-01, Out-of-Fold Validation: 2017-01-01 to 2024-12-31, Out-of-Sample Forward Window: 2025-10-13 to 2026-10-14)  
**Execution Convention:** Next-bar open execution ($Open[t+1]$), 5 bps adverse slippage per side, $0.005/share brokerage ($1.00 min ticket), non-negative cash, whole shares only  
**Primary Finding:** The Two-Tier Hierarchical Regime-Switching Strategy achieves **+$188.34 (+18.83%)** net profit ($1,188.34 final equity) with 18 trades (50.0% win rate, Sharpe 1.082, max drawdown -13.39%) on the forward evaluation window. This outperforms the pure ML candidate (+15.50%, $1,155.00) by eliminating whipsaw churn during secular trends. However, it **trails the simple 200-day SMA benchmark (+26.57%, $1,265.66)** and **Buy & Hold (+38.08%, $1,380.82)**, fails multiclass Brier calibration ($BSS = -2.45\%$, $ECE = 0.0816$), and registers only 18 forward closed trades against the 50-trade institutional minimum.  
**Production Verdict:** **STRICT NO-GO FOR REAL CAPITAL. Paper / Shadow Validation Active.**

---

## 1. Executive Summary & Controlled Strategy Comparison

| Strategy Candidate | Initial Capital | Final Cash | Final Equity | Net Profit | Net Return | Total Trades | Win Rate | Profit Factor | Max DD (%) | Sharpe | Sortino | Exposure | Slippage | Open Position |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Buy & Hold** | $1,000.00 | $15.74 | **$1,380.82** | +$380.82 | **+38.08%** | 0 | 0.0% | 0.00 | -13.61% | 1.463 | 1.966 | 99.60% | 5 bps | 4 shs @ $245.82 |
| **200-Day SMA Trend** | $1,000.00 | $241.85 | **$1,265.66** | +$265.66 | **+26.57%** | 1 | 100.0% | 999.0 | -13.61% | 1.233 | 1.655 | 99.21% | 5 bps | 3 shs @ $253.75 |
| **Hierarchical Regime Switching (V2.4.3)** | $1,000.00 | $164.53 | **$1,188.34** | +$188.34 | **+18.83%** | 18 | 50.0% | 2.12 | -13.39% | 1.082 | 1.283 | 70.63% | 5 bps | 3 shs @ $333.43 |
| **Current Locked LightGBM (Pure ML)** | $1,000.00 | $131.19 | **$1,155.00** | +$155.00 | **+15.50%** | 27 | 44.4% | 2.08 | -12.27% | 0.995 | 1.065 | 60.32% | 5 bps | 3 shs @ $331.87 |
| **Hierarchical with Volatility Veto** | $1,000.00 | $252.78 | **$935.32** | -$64.68 | **-6.47%** | 8 | 25.0% | 0.59 | -14.23% | -0.696 | -0.622 | 28.17% | 5 bps | 2 shs @ $333.43 |

```mermaid
graph TD
    A["Market Data Bar Close[t]"] --> B{"Tier 1: Macro Trend Filter"}
    B -->|"Close >= 1.07 * SMA200 & Slope >= 0"| C["Strong Uptrend Regime"]
    B -->|"Close < SMA200"| D["Downtrend Regime (Bear Market)"]
    B -->|"ATR / Close > 0.02"| E["Elevated Volatility Regime"]
    B -->|"Hovering near SMA200 (Consolidation)"| F["Neutral / Consolidation Regime"]
    
    C --> G["Macro Trend Participation (Full Long, Zero Churn)"]
    D --> H["Capital Preservation (100% Cash / Strict Exit)"]
    E --> I{"Candidate Variant"}
    I -->|"Candidate 4: Standard Hierarchical"| J["Tactical LightGBM Timing (theta=0.45)"]
    I -->|"Candidate 5: Risk Volatility Veto"| K["Veto BUYs; Exit Long on 5% Drawdown"]
    F --> J
```

---

## 2. Baseline Reproduction & Forensic Reconciliations

### 2.1 Target Reproduction Audit

Before implementing new strategies, all 8 baseline targets from the Master Research Sprint report were audited and verified from archived artifacts:
1. **Buy and Hold (+38.08%):** Verified at +38.08% ($1,380.82 final equity, 0 closed trades, 4 shares open).
2. **200-day SMA Trend (+26.57%):** Verified at +26.57% ($1,265.66 final equity, 1 closed trade, 3 shares open).
3. **Tuned LightGBM (+14.66% reported):** Re-simulated at +15.50% ($1,155.00 final equity, 27 closed trades, 12 winners, 15 losers).
4. **Frozen Baseline ML (0.00%):** Verified at $1,000.00 cash due to 100% abstention ($\theta = 0.50$ unbreached).
5. **Tuned LightGBM Brier Score (0.5917):** Reproduced at exactly **0.5917** on outer walk-forward validation folds.
6. **Tuned LightGBM Brier Skill Score (−2.46%):** Reproduced at **−2.45%** (pooled over 1,656 OOF bars).
7. **Tuned LightGBM ECE (0.0858):** Reproduced at **0.0816** (10-bin equal-width partition).
8. **Forward Closed Trades (28 reported):** Re-simulated at 27 closed trades + 1 open position (28 total trading actions).

### 2.2 Mathematical Proof of the Constant-Prior BSS (-0.07%) Discrepancy

**The Audit Question:** Why did the constant-prior baseline in the previous sprint report a nonzero negative Brier Skill Score ($-0.07\%$) when a reference model evaluated against itself should theoretically yield $BSS \equiv 0.00\%$?

**Mathematical Proof & Provenance:**
1. Let $y \in \{0, 1, 2\}$ be the ternary target (0: SELL, 1: HOLD, 2: BUY).
2. In `master_research_validation_sprint.py`, the constant-prior vector $\mathbf{p}_{\text{train}}$ was parameterized over the training prefix $\mathcal{D}_{\text{train}} = y_{\text{dev}}[0:1656]$:
   $$\mathbf{p}_{\text{train}} = [P(0)=0.4275, P(1)=0.1021, P(2)=0.4704]$$
3. It was then evaluated out-of-sample on the outer test folds $\mathcal{D}_{\text{eval}} = y_{\text{dev}}[414:2070]$:
   $$\mathbf{p}_{\text{eval}} = [P(0)=0.4215, P(1)=0.0924, P(2)=0.4861]$$
4. By definition, the reference Brier score $Brier_{\text{ref}}$ of the evaluation set is the empirical cross-entropy lower bound achieved by $\mathbf{p}_{\text{eval}}$ on $\mathcal{D}_{\text{eval}}$:
   $$Brier_{\text{ref}} = \frac{1}{N} \sum_{i=1}^N \sum_{c=0}^2 \left(p_{\text{eval}, c} - \mathbf{1}_{\{y_i = c\}}\right)^2 = 0.57753$$
5. Evaluating the constant predictor $\mathbf{p}_{\text{train}}$ on $\mathcal{D}_{\text{eval}}$ yields:
   $$Brier(\mathbf{p}_{\text{train}}, \mathcal{D}_{\text{eval}}) = 0.57794$$
6. Computing the Brier Skill Score:
   $$BSS = \left(1 - \frac{Brier(\mathbf{p}_{\text{train}}, \mathcal{D}_{\text{eval}})}{Brier_{\text{ref}}}\right) \times 100\% = \left(1 - \frac{0.57794}{0.57753}\right) \times 100\% = -0.071\% \approx -0.07\%$$
7. **Conclusion:** The $-0.07\%$ score is not an arithmetic bug; it is the exact penalty of **out-of-sample class frequency drift** between the training prefix and outer validation folds. If evaluated in-sample against its own distribution, $BSS$ is identically $0.00\%$.

### 2.3 Shadow Signal Cutoff Timestamp & Timezone Availability Semantics

**The Audit Question:** A reported shadow signal cutoff of `2026-10-13 00:00:00` must not be treated as a regular-session closing observation without proving its timezone and data-availability semantics.

**Forensic Finding:**
- **Data Source:** Daily historical equity bars from Yahoo Finance (`yf.download('AAPL')`).
- **Timestamp Normalization:** Yahoo Finance daily bars are normalized to `00:00:00 UTC` of the trade date.
- **Regular Session Windows:** The trading day corresponding to `2026-10-13 00:00:00` represents trading on NASDAQ from 09:30:00 to 16:00:00 Eastern Daylight Time (EDT, UTC-4), which spans 13:30:00 to 20:00:00 UTC on October 13, 2026.
- **Data Availability:** The close price of `$341.27` is strictly unavailable until the closing auction completes at 16:00:00 EDT (20:00:00 UTC).
- **Execution Convention:** Under HYDRA's point-in-time causality rules, a signal generated using `Close[2026-10-13]` is published after 16:00:00 EDT and targets execution at the market open of the next regular session: **2026-10-14 09:30:00 EDT** (+5 bps adverse slippage). It never executes retroactively at midnight or during the October 13 session.

---

## 3. Architecture of the Two-Tier Regime-Switching Strategy

### 3.1 Tier 1: Macro Trend Regime Classifier (Causal & Point-in-Time)

Inputs available at $Close[t]$:
1. $Price\_vs\_SMA200[t] = \frac{Close[t] - SMA_{200}[t]}{SMA_{200}[t]}$ (Dimensionless distance)
2. $Normalized\_SMA\_Slope\_20[t] = \frac{SMA_{50}[t] - SMA_{50}[t-20]}{SMA_{50}[t-20]}$ (Dimensionless momentum)
3. $ATR\_over\_Close[t] = \frac{ATR_{14}[t]}{Close[t]}$ (Daily normalized volatility)

Candidate Decision Boundaries:
- **Strong Uptrend:** $Price\_vs\_SMA200 \ge \delta$ and $Normalized\_SMA\_Slope\_20 \ge 0$.
- **Downtrend:** $Price\_vs\_SMA200 < 0$ (Price below 200-day SMA).
- **Elevated Volatility:** $ATR\_over\_Close > \sigma_{\text{threshold}}$.
- **Neutral / Consolidation:** Hovering near 200-day SMA ($0 \le Price\_vs\_SMA200 < \delta$), or $Price\_vs\_SMA200 \ge \delta$ with negative slope.

### 3.2 Tier 2: Tactical ML Timing (Optuna-Tuned LightGBM)

- **Feature Space:** Set B Scale-Invariant (13 features: RSI, ADX, Stoch_K, Stoch_D, Volatility_Ratio, Price_vs_SMA200, BB_Position, BB_Width, MACD_Norm, Return_1d, Return_5d, Return_20d, Volume_Ratio).
- **Locked Decision Threshold:** $\theta = 0.45$.
- **Signal Priority:**
  - In **Strong Uptrend**: Trend participation dominates. If FLAT, emit BUY; if LONG, emit HOLD. Eliminates turnover drag.
  - In **Neutral Consolidation**: Tactical LightGBM dominates. If FLAT and $P(BUY) \ge 0.45$, emit BUY; if LONG and $P(SELL) \ge 0.45$, emit SELL (tactical profit taking).
  - In **Downtrend**: Capital preservation filter dominates. If LONG, emit SELL (regime break exit); if FLAT, emit HOLD (stay in cash).
  - In **Elevated Volatility**: Standard Hierarchical (Candidate 4) allows tactical ML timing; Candidate 5 applies a risk veto (no BUYs, strict exit on 5% drawdown).

### 3.3 Position State Invariants

State transitions strictly enforce:
- States: $\mathcal{S} \in \{\text{FLAT}, \text{LONG}, \text{ENTRY\_PENDING}, \text{EXIT\_PENDING}\}$.
- **No Duplicate Entries:** A BUY signal emitted while in LONG state produces `HOLD_MAINTAINED` without altering position size or incurring broker tickets.
- **No Illegal Short Exits:** A SELL signal emitted while in FLAT state produces `HOLD_MAINTAINED`.
- **Regime Shift Continuity:** If LONG when transitioning from Strong Uptrend to Neutral Consolidation, the position is **never silently liquidated**. It remains LONG, allowing LightGBM to evaluate tactical exit conditions.

---

## 4. Pre-2025 Sensitivity Analysis (Validation Folds 2017–2024 Only)

To prevent lookahead bias and guard against data snooping on the forward test period, candidate parameters $\delta \in \{0.02, 0.03, 0.05, 0.07\}$ and $\sigma_{\text{threshold}} \in \{0.020, 0.025, 0.030\}$ were swept strictly over the 1,656 out-of-fold validation bars (2017–2024):

| SMA Distance $\delta$ | Volatility Threshold $\sigma$ | Final Equity ($) | Net Return (%) | Total Trades | Win Rate (%) | Profit Factor | Max DD (%) | Sharpe | Sortino | Exposure (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.07** | **0.020** | **$3,272.44** | **+227.24%** | **110** | **50.00%** | **1.99** | **-24.57%** | **1.024** | **1.254** | **64.37%** |
| 0.07 | 0.030 | $3,315.60 | +231.56% | 90 | 50.00% | 1.91 | -26.47% | 0.962 | 1.223 | 69.87% |
| 0.05 | 0.020 | $3,018.03 | +201.80% | 113 | 49.56% | 1.93 | -24.67% | 0.960 | 1.185 | 64.92% |
| 0.03 | 0.020 | $2,954.07 | +195.41% | 111 | 50.45% | 1.90 | -25.10% | 0.940 | 1.153 | 65.28% |
| 0.02 | 0.020 | $2,947.40 | +194.74% | 110 | 50.00% | 1.84 | -25.10% | 0.939 | 1.151 | 65.34% |
| 0.05 | 0.030 | $3,189.09 | +218.91% | 90 | 50.00% | 1.89 | -26.18% | 0.927 | 1.190 | 71.26% |
| 0.03 | 0.030 | $3,099.70 | +209.97% | 87 | 51.72% | 1.86 | -27.01% | 0.898 | 1.147 | 72.46% |
| 0.02 | 0.030 | $2,965.99 | +196.60% | 85 | 49.41% | 1.78 | -27.94% | 0.868 | 1.113 | 72.95% |
| 0.07 | 0.025 | $2,431.87 | +143.19% | 99 | 48.48% | 1.62 | -29.12% | 0.781 | 0.957 | 68.24% |
| 0.05 | 0.025 | $2,246.64 | +124.66% | 100 | 46.00% | 1.57 | -30.22% | 0.716 | 0.886 | 69.50% |
| 0.03 | 0.025 | $2,124.71 | +112.47% | 97 | 47.42% | 1.53 | -28.82% | 0.667 | 0.819 | 70.53% |
| 0.02 | 0.025 | $1,979.99 | +98.00% | 96 | 44.79% | 1.46 | -28.93% | 0.614 | 0.756 | 70.95% |

**Selection Decision:** $\delta = 0.07$ and $\sigma_{\text{threshold}} = 0.020$ were selected and locked based on the highest risk-adjusted Sharpe ratio (1.024), highest profit factor (1.99), and lowest max drawdown (-24.57%) before running any forward evaluation.

---

## 5. Execution Slippage Sensitivity Analysis

The Hierarchical candidates were evaluated across 0, 5, 10, and 15 basis points of adverse slippage on the forward period:

| Strategy | Slippage (bps) | Final Equity ($) | Net Profit ($) | Net Return (%) | Total Trades | Win Rate (%) | Sharpe Ratio | Max DD (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hierarchical Regime Switching** | 0.0 | $1,205.12 | +$205.12 | +20.51% | 18 | 50.0% | 1.171 | -12.84% |
| **Hierarchical Regime Switching** | **5.0** | **$1,188.34** | **+$188.34** | **+18.83%** | **18** | **50.0%** | **1.082** | **-13.39%** |
| **Hierarchical Regime Switching** | 10.0 | $1,160.41 | +$160.41 | +16.04% | 18 | 50.0% | 0.942 | -13.95% |
| **Hierarchical Regime Switching** | 15.0 | $1,136.90 | +$136.90 | +13.69% | 18 | 50.0% | 0.841 | -14.87% |
| **Hierarchical with Vol Veto** | 0.0 | $942.10 | -$57.90 | -5.79% | 8 | 25.0% | -0.619 | -13.76% |
| **Hierarchical with Vol Veto** | 5.0 | $935.32 | -$64.68 | -6.47% | 8 | 25.0% | -0.696 | -14.23% |
| **Hierarchical with Vol Veto** | 10.0 | $928.55 | -$71.45 | -7.14% | 8 | 25.0% | -0.772 | -14.71% |
| **Hierarchical with Vol Veto** | 15.0 | $921.77 | -$78.23 | -7.82% | 8 | 25.0% | -0.848 | -15.18% |

**Observations:**
1. Hierarchical Regime Switching exhibits strictly monotonic decay under adverse slippage: $+20.51\% \to +18.83\% \to +16.04\% \to +13.69\%$.
2. Even under severe 15 bps slippage, the strategy maintains a positive net return (+13.69%) and a Sharpe of 0.841, demonstrating substantial structural buffer against retail execution drag.

---

## 6. Regime Distribution & Forensic Analysis of the Volatility Veto

On the forward period (October 13, 2025 to October 13, 2026, 252 bars):
- **Elevated Volatility Regime ($ATR / Close > 0.020$):** 180 bars (71.4% of period)
- **Strong Uptrend Regime ($Close \ge 1.07 \times SMA_{200}$):** 71 bars (28.2% of period)
- **Neutral / Consolidation Regime:** 1 bar (0.4% of period)
- **Downtrend Regime ($Close < SMA_{200}$):** 0 bars (AAPL remained above 200-day SMA throughout 2025–2026)

### Why Did the Volatility Veto Overlay (Candidate 5) Fail (-6.47%)?
During 2025–2026, AAPL exhibited an annualized realized volatility of ~24%, causing normalized ATR/Close to hover above 0.020 for 180 bars. In Candidate 5:
1. New BUY entries were vetoed whenever volatility spiked, preventing the model from entering right before large momentum rallies.
2. The 5% stop-loss in elevated volatility chopped the portfolio out of 6 out of 8 trades during normal intraday pullbacks, converting what were otherwise profitable multi-week holdings into small realized losses.
3. **Quantitative Takeaway:** A rigid volatility veto that forces abstention during secular bull markets degrades risk-adjusted return because volatility and trend often co-exist during large institutional markups.

---

## 7. Institutional Acceptance Gate Matrix

| Gate | Required Criterion | Observed Metric | Verdict |
| :--- | :--- | :--- | :---: |
| **1. Automated Test Suite** | 100% pass across all unit, causal, and regression tests | 301 passed, 0 failed in pytest suite | **PASS** |
| **2. Cryptographic Dataset Invariance** | Byte-for-byte SHA-256 match on pinned dataset & manifest | `d580b50a...` and `e764c60c...` match exactly | **PASS** |
| **3. Causal Execution Firewall** | Signal at $Close[t]$ executes at $Open[t+1]$ + slippage | Causal bar-boundary verified across all 5 candidates | **PASS** |
| **4. Penny-Exact Accounting** | Ending Equity == Cash + Position Value; zero cent drift | 100% cent reconciliation verified in trade ledgers | **PASS** |
| **5. Multiclass Brier Score** | $\le 0.2000$ | $0.5917$ | **FAIL** |
| **6. Brier Skill Score (BSS)** | $> 0.00\%$ | **$-2.45\%$** | **FAIL** |
| **7. Expected Calibration Error** | $\le 0.0800$ | **$0.0816$** | **FAIL** |
| **8. Cost-Adjusted Expectancy** | Positive net return after slippage and commissions | **+$18.83% ($1,188.34)** | **PASS** |
| **9. Benchmark Outperformance** | Net return exceeding 200-day SMA trend (+26.57%) | Hierarchical +18.83% vs SMA +26.57% (trailed by 7.74%) | **FAIL** |
| **10. Prospective Paper Trades** | $\ge 50$ independent closed prospective paper trades | 18 closed forward trades | **PENDING** |
| **11. Live-Money Trading Gate** | All 10 gates cleared | Fails calibration, skill score, and benchmark comparison | **STRICT NO-GO** |

---

## 8. Defect Status & Next Research Recommendations

### 8.1 Confirmed Progress
1. **Turnover & Churn Reduction:** The Hierarchical strategy reduced forward trade count from 27 trades (pure LightGBM) to 18 trades, increasing win rate from 44.4% to 50.0% and lifting net profit from $1,155.00 (+15.50%) to $1,188.34 (+18.83%).
2. **Reconciliation Complete:** The -0.07% constant-prior BSS result and the 2026-10-13 00:00:00 shadow signal timezone semantics have been mathematically proven and documented.

### 8.2 Remaining Obstacles
1. **Underperforming Passive Beta:** In a persistent secular bull market (+38.08% Buy & Hold), any strategy that allocates to cash or incurs transaction fees lags the buy-and-hold index. The simple 200-day SMA (+26.57%) only traded once and stayed 99.2% invested, outperforming the Hierarchical strategy (+18.83%).
2. **Underlying ML Model Calibration:** Optuna tuning and scale-invariant features improved Brier score from 0.8111 to 0.5917 and ECE from 0.3207 to 0.0816, but BSS remains negative ($-2.45\%$). The ML agent's probabilities still do not beat an empirical uninformative prior in multi-class cross-entropy.

### 8.3 Recommended Next Experiment
- **Regime-Conditioned Target Rescaling:** Instead of applying fixed ATR multipliers (2.0 ATR TP / 1.5 ATR SL) uniformly across all market regimes, investigate target labels dynamically scaled by market volatility and trend strength.
- **Asymmetric Sizing:** Allow the Hierarchical strategy to scale exposure between 50% and 100% rather than binary all-in/all-out, capturing trend beta while tactically harvesting volatility.
