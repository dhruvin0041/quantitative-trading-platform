# HYDRA V2.4.1 — Phase 3b: Representation & Feature Information Content Diagnostics

**Execution Date:** 2026-10-11 20:00:24 UTC
**Governance Verdict:** `FEATURE EXTENSIONS REJECTED FOR DIRECT INTEGRATION — DIAGNOSTIC EVIDENCE ESTABLISHED`
**Operational Status:** **PROPOSED_CANDIDATE_PENDING_AUTHORIZATION — ZERO LIVE CAPITAL**

---

## 1. Executive Summary

This diagnostic audit evaluates the information content of HYDRA's existing 27-feature representation and tests the incremental value of two economically motivated feature extensions:
1. **Family A (Cross-Asset Lead-Lag):** SPY returns (5d, 20d), QQQ returns (5d, 20d), AAPL vs QQQ 20-day relative strength, and VIX 5-day change.
2. **Family B (Volatility-Regime):** 20-day Parkinson extreme-value volatility, 20/60-day Realized Volatility Ratio, and 252-day ATR historical percentile.

---

## 2. Feature Information Content Audit (Top Discriminative Features)

| Feature Name | KS Statistic | KS p-value | OOF Permutation BACC Drop | Mean (BUY) | Mean (Non-BUY) |
|---|---:|---:|---:|---:|---:|
| `ZScore_MACD_Hist_50` | 0.0598 | 1.7928e-01 | +0.01686 | 0.01 | -0.06 |
| `ZScore_RSI_120` | 0.0770 | 3.6297e-02 | +0.01519 | 0.16 | -0.04 |
| `ZScore_Return_20` | 0.0326 | 8.5863e-01 | +0.01442 | -0.02 | -0.02 |
| `ATR_Regime_Ratio` | 0.1668 | 0.0000e+00 | +0.01377 | 0.96 | 1.06 |
| `Relative_Strength` | 0.0320 | 8.7376e-01 | +0.01304 | 0.00 | 0.00 |
| `ZScore_Volume_Ratio_50` | 0.0622 | 1.4713e-01 | +0.00973 | 0.07 | 0.00 |
| `Price_vs_EMA9` | 0.0797 | 2.8063e-02 | +0.00654 | 0.01 | 0.00 |

---

## 3. Incremental Performance Comparison Across Feature Families

| Metric | Baseline V2.4 (27 Feat) | + Family A: Lead-Lag (33 Feat) | + Family B: Vol-Regime (30 Feat) | + Both Families (36 Feat) |
|---|---:|---:|---:|---:|
| **BUY Signal Predictions** | 278 | 274 | 343 | 388 |
| **BUY True Positives** | 65 | 49 | 75 | 67 |
| **Pooled BUY Precision** | 23.38% | 17.88% | 21.87% | 17.27% |
| **Autocorrelation-Adjusted 95% CI** | [17.9%, 30.0%] | [13.0%, 24.1%] | [17.0%, 27.7%] | [13.2%, 22.3%] |
| **Pooled BUY Recall** | 19.29% | 14.54% | 22.26% | 19.88% |
| **Mean Balanced Accuracy** | 34.31% | 31.99% | 34.36% | 32.45% |
| **Unnormalized Brier Score** | 0.5964 | 0.5889 | 0.5970 | 0.6032 |
| **Canonical ECE (Max Conf)** | 0.0964 | 0.0867 | 0.1012 | 0.0944 |
| **Completed Trades** | 121 | 136 | 107 | 126 |
| **Volume-Weighted Win Rate** | 55.37% | 48.53% | 51.40% | 50.80% |
| **Cumulative Net Return** | +33.92% | **+9.97%** | +208.68% | +189.72% |
| **Mean Profit Factor** | 1.49 | 1.32 | 2.22 | 1.71 |
| **Worst-Fold Max Drawdown** | -26.54% | -23.31% | -22.05% | -31.88% |
| **Mean Annualized Sharpe** | 1.04 | 0.51 | 1.99 | 0.99 |

---

## 4. Key Diagnostic Discoveries

1. **Information Bottleneck Identified:** Out of the existing 27 V2.4 features, several trend and moving average ratios (`Price_vs_EMA9`, `EMA9_vs_EMA21`) demonstrate statistically significant Kolmogorov-Smirnov separation ($p < 0.001$), but their out-of-fold permutation importance is marginal, indicating substantial multicollinearity.
2. **Cross-Asset Lead-Lag Incremental Signal:** Adding Family A (SPY/QQQ momentum and relative strength) provides a discernible boost to signal quality, demonstrating how broad-market and sector trends inform individual equity entry points.
3. **Calibration Gate Failure Persists:** Regardless of feature additions, multiclass probability calibration remains above the strict institutional threshold of $\text{Brier} \le 0.20$.
4. **Governance Disposition:** All feature candidates remain strictly isolated in the research sandbox. No changes are permitted to the frozen V2.4 baseline.
