# HYDRA V2.4.1 — Regime Discrimination Audit & Institutional Acceptance Report

**Audit Date:** 2026-10-11 19:46:15 UTC
**Repository Branch:** `research/v2.3_phase_3a_remediation`
**Starting Commit:** `88447a6c3c4fd8dfd818cb8c2a9eebde4f437e01`
**Governance Verdict:** `REMEDIATION REJECTED — NO DEFENSIBLE IMPROVEMENT`
**Operational Status:** **PROPOSED_CANDIDATE_PENDING_AUTHORIZATION — ZERO LIVE CAPITAL**

---

## 1. Executive Summary

This forensic research audit investigates whether regime-conditioned signal filtering (asset-trend and macro-market moving average confirmations) can remediate the BUY-signal discrimination failure documented in HYDRA V2.4.1.

### Decisive Findings:
1. **Hypothesis Falsification:** The hypothesis that oversold oscillators trigger false BUY signals primarily during falling markets is **falsified**. Direct feature measurement proves that True BUYs and False BUYs on downward moves have **statistically identical oscillator distributions** (mean RSI 55.73 vs 56.78, BB Position 0.61 vs 0.59).
2. **False BUYs Pervade Bull Regimes:** 72.3% (136 / 188) of all false BUY signals on downward moves occur during **confirmed dual-bull regimes** (AAPL >= SMA200 and SPY >= SMA200).
3. **Regime Filter Inefficacy:** Applying the combined regime filter (Candidate D) produces a negligible precision increase (+1.94 percentage points, from 27.05% to 28.99%, well within the 95% confidence interval [23.1%, 35.7%]) while reducing total net trading return from **+50.55% down to +46.94%**.
4. **Root Cause Identified in Label Barrier Mechanics:** Direct measurement of the triple-barrier labeling engine shows that the lower stop-loss barrier ($0.5 \times \text{ATR}$) is touched on **74.87% of all bars** with a median latency of only **2.0 trading days**. The classifier is forced to predict an extreme majority class (72.8% SELL), while ordinary daily price fluctuations trip the razor-thin stop-loss barrier before the $3.0 \times \text{ATR}$ profit target can be reached.

---

## 2. Comparative Candidate Performance

| Metric | Candidate A (Baseline) | Candidate B (AAPL >= SMA200) | Candidate C (SPY >= SMA200) | Candidate D (Combined Bull) |
|---|---:|---:|---:|---:|
| **BUY Predictions** | 281 | 218 | 247 | 207 |
| **True Positive BUYs** | 76 | 62 | 67 | 60 |
| **BUY Precision** | **27.05%** | **28.44%** | **27.13%** | **28.99%** |
| **95% Wilson CI** | [22.1%, 32.6%] | [22.8%, 34.8%] | [21.9%, 33.0%] | [23.1%, 35.7%] |
| **BUY Recall** | 22.75% | 18.56% | 20.06% | 17.96% |
| **Completed Trades** | 120 | 101 | 109 | 96 |
| **Win Rate** | 57.82% | 58.48% | 61.21% | 62.65% |
| **Cumulative Net Return** | **+50.55%** | **+30.98%** | **+49.34%** | **+46.94%** |
| **Mean Profit Factor** | 1.84 | 1.52 | 1.81 | 1.76 |
| **Mean Annualized Sharpe** | 1.08 | 0.82 | 1.12 | 1.05 |
| **Multiclass Brier Score** | 0.5876 | 0.5876 | 0.5876 | 0.5876 |

---

## 3. Institutional Governance Matrix

| Gate Requirement | Threshold | Measured Result | Disposition |
|---|---|---|---|
| **Unnormalized Multiclass Brier** | $\le 0.20$ | 0.5876 | **FAIL** |
| **Canonical Max-Confidence ECE** | $\le 0.08$ | 0.0342 (H2), 0.0521 (Dev) | **PASS** |
| **BUY Discrimination Edge** | Economically Meaningful | +1.94% Prec Gain, Net Ret Degrades | **FAIL** |
| **Prospective Closed Trades** | $\ge 50$ | 0 (Isolated ledger preserved) | **NOT MET** |
| **Dual Regime Coverage** | Bull and Bear | Pending prospective trade accumulation | **PENDING** |
| **V2.4 Baseline Immutability** | Byte-for-byte identical | Verified SHA-256 `d580b50a441c...` | **PASS** |
| **V2.4.1 Main Integration** | Authorized | Strictly Quarantined | **BLOCKED** |

---

## 4. Final Recommendation

**REMEDIATION REJECTED — NO DEFENSIBLE IMPROVEMENT.**
The proposed moving-average trend filters do not resolve HYDRA V2.4.1's directional discrimination deficit. Future research should not pursue simple post-inference trend gates, but must address the asymmetric triple-barrier formulation ($3.0 \times \text{ATR}$ vs $0.5 \times \text{ATR}$) that distorts the supervised objective.
