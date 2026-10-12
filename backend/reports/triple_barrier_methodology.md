# HYDRA V2.4.1 — Triple-Barrier Label Design Experiment Report

**Execution Date:** 2026-10-11 19:53:51 UTC
**Governance Verdict:** `REMEDIATION REJECTED — NO DEFENSIBLE IMPROVEMENT`
**Operational Status:** **PROPOSED_CANDIDATE_PENDING_AUTHORIZATION — ZERO LIVE CAPITAL**

---

## 1. Executive Summary

This experiment investigates whether alternative triple-barrier labeling geometries can remediate HYDRA V2.4.1's calibration and discrimination failures when models are separately trained and evaluated under identical, realistic execution conventions.

### Tested Configurations:
1. **Control (3.0x ATR / 0.5x ATR):** Existing baseline label formulation.
2. **Candidate 1.5x / 1.5x ATR (Symmetric):** Equal-opportunity volatility threshold.
3. **Candidate 2.0x / 1.0x ATR:** 2:1 payoff target with 1.0x ATR buffer.
4. **Candidate 2.0x / 1.5x ATR:** Balanced volatility target with wide buffer.

---

## 2. Comparative Out-of-Fold Results

| Metric | Control (3.0x / 0.5x) | Candidate (1.5x / 1.5x) | Candidate (2.0x / 1.0x) | Candidate (2.0x / 1.5x) |
|---|---:|---:|---:|---:|
| **BUY Base Rate in Labels** | 20.01% | 57.67% | 41.8% | 49.13% |
| **BUY Predictions** | 278 | 764 | 531 | 617 |
| **BUY True Positives** | 65 | 451 | 223 | 300 |
| **BUY Precision** | 23.38% | **59.03%** | 42.00% | 48.62% |
| **Unnormalized Brier Score** | **0.5964** | 0.6844 | 0.6588 | 0.7238 |
| **Canonical ECE** | **0.0964** | 0.1841 | 0.1353 | 0.2006 |
| **Completed Trades** | 121 | 166 | 167 | 147 |
| **Volume-Weighted Win Rate** | **55.37%** | 50.00% | 48.51% | 51.70% |
| **Cumulative Net Return** | **+33.92%** | +20.85% | **-18.43%** | **-23.70%** |
| **Mean Profit Factor** | **1.49** | 1.35 | 1.22 | 1.03 |
| **Mean Annualized Sharpe** | **1.04** | 0.55 | 0.22 | 0.05 |

---

## 3. Decisive Findings

1. **Prevalence Illusion:** The apparent surge in BUY precision under symmetric barriers (59.03% vs 23.38%) is driven almost entirely by the underlying label base rate (57.67% BUYs in AAPL's secular uptrend).
2. **Calibration Deterioration:** Symmetric and wide-buffer labeling configurations severely degrade probability calibration: Brier worsens from 0.5964 to 0.6844–0.7238, and ECE worsens from 0.0964 to 0.1841–0.2006.
3. **Net Return Degradation:** The symmetric model emits 764 BUY predictions (vs 278 in Control), creating excessive turnover. After execution slippage (5 bps each way) and commissions ($0.005/share, $1 min), cumulative net return drops by over 13.07 percentage points (+20.85% vs +33.92%).
4. **Catastrophic Failure of Asymmetric 2:1 Targets:** Candidates with 2.0x / 1.0x and 2.0x / 1.5x generate net trading losses (-18.43% and -23.70%).

---

## 4. Final Recommendation

**REMEDIATION REJECTED — NO DEFENSIBLE IMPROVEMENT.**
Altering triple-barrier hyperparameters does not solve HYDRA's underlying signal quality deficit. The model architecture and feature space must be overhauled before revisiting label geometry.
