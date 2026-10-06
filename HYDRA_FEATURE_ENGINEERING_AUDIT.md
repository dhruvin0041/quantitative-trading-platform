# HYDRA FEATURE ENGINEERING & SIGNAL INTELLIGENCE AUDIT
**Document Version:** 2.0.0 (Post-Reconciliation Audit)  
**Classification:** Quantitative Signal Representation & Feature Pipeline Verification  
**Repository Branch:** `main` (Preserving V2.2 Frozen Release)  
**Date:** October 2026

---

## 1. Executive Summary & Schema Reconciliation Notice

> [!IMPORTANT]
> **SCHEMA RECONCILIATION NOTICE:**  
> A previous draft of this audit described an experimental or legacy feature set (including `weather_disruption_index`, `retail_interest_ratio`, `returns_1d`, etc.) that did NOT correspond to the deployed models.
> 
> This document audits the **actual authoritative 27-feature schema** strictly enforced by [backend/configs/kept_features.json](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/configs/kept_features.json), [backend/src/execution/live_inference.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/live_inference.py), and [backend/artifacts/latest_scaler.joblib](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/artifacts/latest_scaler.joblib).

---

## 2. Canonical 27-Feature Schema & Mathematical Definitions

The authoritative 27 deployed features are structured into four quantitative families:

### 2.1 Trend & Moving Average Spreads (4 Features)
1. **`MA20_vs_MA50`:** Relative spread between 20-day and 50-day simple moving averages:
   $$\text{MA20\_vs\_MA50}_t = \frac{\text{SMA}_{20}(P)_t - \text{SMA}_{50}(P)_t}{\text{SMA}_{50}(P)_t}$$
2. **`EMA9_vs_EMA21`:** Short-term trend momentum spread:
   $$\text{EMA9\_vs\_EMA21}_t = \frac{\text{EMA}_{9}(P)_t - \text{EMA}_{21}(P)_t}{\text{EMA}_{21}(P)_t}$$
3. **`Price_vs_EMA9`:** Price divergence from the 9-day fast exponential moving average:
   $$\text{Price\_vs\_EMA9}_t = \frac{P_t - \text{EMA}_{9}(P)_t}{\text{EMA}_{9}(P)_t}$$
4. **`Price_vs_EMA21`:** Price divergence from the 21-day medium exponential moving average:
   $$\text{Price\_vs\_EMA21}_t = \frac{P_t - \text{EMA}_{21}(P)_t}{\text{EMA}_{21}(P)_t}$$

### 2.2 Core Technical Oscillators & Volatility (10 Features)
5. **`VIX_Level`:** Point-in-time level of the CBOE Volatility Index ($VIX$).
6. **`BB_Width`:** Normalized Bollinger Band width ($20, 2\sigma$):
   $$\text{BB\_Width}_t = \frac{\text{Upper}_t - \text{Lower}_t}{\text{SMA}_{20}(P)_t}$$
7. **`BB_Position`:** Percentile rank within the Bollinger Band envelope:
   $$\text{BB\_Position}_t = \frac{P_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t + \epsilon}$$
8. **`RSI`:** 14-period Relative Strength Index bounded in $[0, 100]$.
9. **`ADX`:** 14-period Average Directional Index measuring trend strength.
10. **`MACD_Hist`:** MACD histogram ($\text{MACD Line} - \text{Signal Line}$).
11. **`Relative_Strength`:** 20-day return of ticker minus 20-day return of SPY benchmark.
12. **`OBV_Change`:** 5-day percentage change in On-Balance Volume.
13. **`Return`:** 1-day logarithmic return $\ln(P_t / P_{t-1})$.
14. **`Volume_Ratio`:** Ratio of current volume to 20-day average volume: $V_t / \text{SMA}_{20}(V)_t$.

### 2.3 Multi-Horizon Rolling Z-Scores (12 Stationarized Features)
To enforce mean-reverting stationarity across regimes, multi-window rolling Z-scores are computed as:
$$Z_w(X)_t = \frac{X_t - \mu_w(X)_t}{\sigma_w(X)_t + \epsilon}$$

15. **`ZScore_RSI_20`:** 20-bar rolling Z-score of RSI.
16. **`ZScore_RSI_50`:** 50-bar rolling Z-score of RSI.
17. **`ZScore_RSI_120`:** 120-bar rolling Z-score of RSI.
18. **`ZScore_BB_Position_20`:** 20-bar rolling Z-score of Bollinger Band position.
19. **`ZScore_BB_Position_50`:** 50-bar rolling Z-score of Bollinger Band position.
20. **`ZScore_MACD_Hist_20`:** 20-bar rolling Z-score of MACD Histogram.
21. **`ZScore_MACD_Hist_50`:** 50-bar rolling Z-score of MACD Histogram.
22. **`ZScore_Return_20`:** 20-bar rolling Z-score of 1-day return.
23. **`ZScore_Return_50`:** 50-bar rolling Z-score of 1-day return.
24. **`ZScore_Return_120`:** 120-bar rolling Z-score of 1-day return.
25. **`ZScore_Volume_Ratio_20`:** 20-bar rolling Z-score of Volume Ratio.
26. **`ZScore_Volume_Ratio_50`:** 50-bar rolling Z-score of Volume Ratio.

### 2.4 Regime Ratio (1 Feature)
27. **`ATR_Regime_Ratio`:** Ratio of short-term ATR (5-bar) to long-term ATR (50-bar):
    $$\text{ATR\_Regime\_Ratio}_t = \frac{\text{ATR}_5(t)}{\text{ATR}_{50}(t) + \epsilon}$$

---

## 3. Causal Timing & Lookahead Immunity

Every feature in the 27-feature vector satisfies strict causal availability:
1. **Observation Point:** All technical indicators, moving averages, and rolling Z-scores use data strictly through bar $t$ Close (16:00 ET).
2. **Execution Target:** Planned orders target bar $t+1$ Open (09:30 ET).
3. **No Future Leakage:** No centered rolling windows, negative shifts, or forward metrics exist in the pipeline.
4. **Scaler Isolation:** Production `latest_scaler.joblib` was fitted strictly on historical observations without prospective data access.

---

## 4. Conclusion & Audit Status
The actual deployed 27-feature schema is mathematically consistent, stationarized, and causally valid across all production pipelines. Prior descriptions of alternative feature sets are retired.
