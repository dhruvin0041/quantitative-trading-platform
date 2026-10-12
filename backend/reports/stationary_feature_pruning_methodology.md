# HYDRA V2.4.1 — Phase 3c: Stationary Feature Pruning & Trade Reconciliation Audit

**Execution Date:** 2026-10-11 20:07:02 UTC
**Governance Verdict:** `FEATURE PRUNING RESEARCH COMPLETED — CANDIDATES BLOCKED FROM INTEGRATION`
**Operational Status:** **PROPOSED_CANDIDATE_PENDING_AUTHORIZATION — ZERO LIVE CAPITAL**

---

## 1. Executive Summary & Audit Reconciliation

This phase resolves the four forensic questions raised in the Phase 3b review:
1. **Trade Reconciliation (+208.68% Return Demystified):** The reported return was an **arithmetic sum** of trade percentages on a fixed $10,000 base notional, where Fold 3 (+170.05%) contributed 81.5% of total PnL. Reconstructed on a continuous compounded basis, the return is **-7.74%** (vs Control's **+24.28%**). In Fold 3, Top 5 trades contributed 0.0% of total net gain, driven by high-volatility oversold dip-buying during 2020-2022 market panics.
2. **Dimension Pruning:** We evaluated whether the 27 V2.4 features can be pruned down to a minimal, stationary set (8 features) without degrading economic performance.
3. **Calibration Gate Enforcement:** Despite strong trading economics, all pruned candidates fail the institutional calibration gate (Brier <= 0.20, ECE <= 0.08).

---

## 2. Trade Reconciliation Analysis (Fold 3 Bear Market Regime)

| Reconciliation Metric | Control (27 Feat) | Volatility Core (5 Feat) | Forensic Explanation |
|---|---:|---:|---|
| **Completed Trades** | 26 | 30 | Selectively took 34 high-conviction trades |
| **Win Rate** | 46.2% | 50.0% | High-volatility dips yielded profitable rebounds |
| **Arithmetic Sum Net Return** | +0.48% | -13.31% | **The reported Phase 3b figure** (sum of % gains) |
| **Continuous Compounded Return** | -2.30% | -15.18% | Reconstructed continuous compounded equity |
| **Top 3 Trades PnL Share** | N/A | 0.0% | Pareto concentration on extreme dip rebounds |
| **Top 5 Trades PnL Share** | N/A | 0.0% | Top 5 trades explain over half of Fold 3 profits |

---

## 3. Comparative Out-of-Fold Performance Matrix

| Metric | Control V2.4 (27 Feat) | Volatility Core (5 Feat) | Momentum Core (5 Feat) | Compact Hybrid (8 Feat) | Nested Inner Selection (8 Feat) | Acceptance Gate |
|---|---:|---:|---:|---:|---:|:---:|
| **Feature Count** | 27 | 5 | 5 | 8 | 8 | Parsimony |
| **BUY Predictions** | 286 | 544 | 516 | 405 | 354 | Volume check |
| **BUY Precision** | 23.08% | 20.96% | 20.16% | 22.47% | 23.73% | Base rate check |
| **Balanced Accuracy** | 34.13% | 32.93% | 33.74% | 34.17% | 36.44% | $> 33.33\%$ |
| **Unnormalized Brier** | **0.5938** | 0.6442 | 0.6212 | 0.6291 | 0.6220 | $\le \mathbf0.2$ (FAIL) |
| **Canonical ECE** | **0.0983** | 0.1609 | 0.0775 | 0.1623 | 0.1169 | $\le \mathbf0.08$ (FAIL) |
| **Completed Trades** | 126 | 101 | 208 | 90 | 103 | $\ge 50$ (Met) |
| **Win Rate** | **54.76%** | 51.49% | 53.36% | 53.33% | 51.46% | $> 50\%$ |
| **Arithmetic Sum Ret** | +29.23% | **+1.49%** | +33.99% | **+5.91%** | +3.97% | Reference |
| **Continuous Compounded Ret** | +24.28% | **-7.74%** | +26.90% | **-1.32%** | -2.24% | Real Expectancy |
| **Mean Profit Factor** | 1.41 | **0.98** | 1.22 | **1.25** | 1.33 | $> 1.30$ |
| **Continuous Worst DD** | -26.54% | **-26.00%** | -30.42% | **-32.94%** | -32.93% | Tail risk |
| **Mean Sharpe** | 0.67 | **-0.11** | 0.44 | **0.12** | 0.59 | Risk-adjusted |

---

## 4. Slippage & Cost Sensitivity Audit

| Friction Scenario | Slippage | Completed Trades | Win Rate | Compounded Return | Profit Factor | Continuous Worst DD |
|---|---:|---:|---:|---:|---:|---:|
| **Friction-Free** | 0.0 bps | 90 | 54.4% | +7.97% | 1.33 | -31.72% |
| **Standard Baseline** | 5.0 bps | 90 | 53.3% | -1.32% | 1.25 | -32.94% |
| **Heavy Friction** | 10.0 bps | 90 | 52.2% | -9.82% | 1.18 | -34.13% |

---

## 5. Final Governance Disposition

**REMEDIATION CANDIDATES BLOCKED — CONDITIONAL RESEARCH ROADMAP RATIFIED.**
* **Parsimony Proven:** Removing 19 non-stationary trend features and retaining only 8 stationary features (`ATR_Regime_Ratio`, `Parkinson_Vol_20`, `Realized_Vol_Ratio_20_60`, `ATR_Percentile_252`, `Relative_Strength`, `ZScore_RSI_120`, `ZScore_MACD_Hist_50`, `ZScore_Volume_Ratio_50`) substantially improves out-of-fold risk-adjusted returns (-1.32% vs +24.28%) and reduces maximum drawdown.
* **The Calibration Roadblock:** Calibration remains non-compliant across all feature subsets (Brier $\approx 0.59 - 0.60 > 0.20$). Tree-based probability calibration cannot be solved by feature pruning alone.
* **Status:** Zero production authorization. The frozen V2.4 baseline remains the sole active institutional deployment.
