# HYDRA DATA INTEGRITY AUDIT
**Document Version:** 1.1.0  
**Classification:** Time-Series, Market Data & Feature Data Integrity Inspection (Reconciled Research Baseline)  
**Repository Branch:** `main`  
**Date:** October 2026

---

## 1. Executive Summary

This forensic audit examines data pipelines, market feeds, historical storage, and database persistence across HYDRA. Data integrity is the prerequisite for all quantitative research; an inaccurate market feed or misaligned temporal timestamp invalidates downstream predictive signals regardless of model sophistication.

> [!WARNING]
> **RECONCILIATION NOTICE:** In accordance with the Reconciliation Verdict, all previous performance claims are withdrawn. V2.3 is reclassified as **Unvalidated Research-Only**.

### Key Audit Findings
1. **Stationarity Enforcement:** Complete transition away from raw non-stationary price levels (Close, High, Low) to log returns, normalized volume, and relative volatility metrics.
2. **Corporate Action Accounting:** Yahoo Finance feeds fetch split- and dividend-adjusted closing prices (`Close` is split-adjusted; `Adj Close` accounts for dividends).
3. **Calendar & Trading Holiday Alignment:** Forward-filling handles weekend gaps; trading holidays are respected via exchange calendar validation.
4. **Prospective Ledger Immutability:** SQLite ledger records are protected by cryptographic SHA-256 signatures, preventing post-hoc manipulation or overwriting of forward paper signals.
5. **Data Source Single Point of Failure:** Direct reliance on Yahoo Finance without a redundant secondary market data vendor poses availability risk during rate-limiting events. Versioned historical snapshots are required for reproducible backtests.

---

## 2. Market Data Ingestion & Sanitization

### 2.1 OHLCV Time-Series Integrity
- **Module:** [backend/src/data/data_loader.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/data/data_loader.py)
- **Ingestion Validation:**
  - Multi-level checks ensure `High >= Low`, `High >= Open`, `High >= Close`, `Low <= Open`, `Low <= Close`.
  - Non-positive prices or zero/negative volumes trigger immediate validation exceptions.
  - Timestamps are normalized to UTC timezone to eliminate daylight savings time discrepancies across exchanges.
- **Handling of Missing Bars:**
  - Zero-volume trading holidays are identified and pruned.
  - Intra-day missing ticks or bars are forward-filled for exogenous series (e.g. macro proxies, weather), while price bars require contiguous trading days.

### 2.2 Macro & Physical Exogenous Data Feeds
- **SEC EDGAR Client:** [backend/src/data/edgar_client.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/data/edgar_client.py)
  - Fetches 8-K and 10-Q corporate disclosures using the official SEC API with institutional User-Agent headers.
  - Strict publication timestamp enforcement prevents using information released after 16:00 EST on day $T$ for signals evaluated at close $T$.
- **Weather Proxy Client:** [backend/src/data/weather_client.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/data/weather_client.py)
  - Ingests regional weather telemetry across supply chain hubs.
  - All metrics are normalized with rolling 30-day Z-scores.
- **Google Trends Client:** [backend/src/data/trends_client.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/data/trends_client.py)
  - Tracks retail interest proxies; values are lagged by 1 day ($T-1$) to guarantee causal publication availability.

---

## 3. Stationarity & Transform Integrity

Machine learning models, particularly tree ensembles and neural networks, fail catastrophically when presented with non-stationary time series characterized by stochastic drift.

### 3.1 Verification of the 27 Stationarized Features
The canonical feature list in [backend/configs/kept_features.json](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/configs/kept_features.json) contains exactly 27 stationarized features:

| Index | Feature Column | Definition / Formula | Stationarization Mechanism |
|---|---|---|---|
| 0 | `Return_1d` | $(P_t / P_{t-1}) - 1$ | 1-day percentage change |
| 1 | `Return_5d` | $(P_t / P_{t-5}) - 1$ | 5-day percentage change |
| 2 | `Return_20d` | $(P_t / P_{t-20}) - 1$ | 20-day percentage change |
| 3 | `Return_60d` | $(P_t / P_{t-60}) - 1$ | 60-day percentage change |
| 4 | `Vol_5d` | $\text{std}(R_{1d}, 5)$ | 5-day rolling volatility |
| 5 | `Vol_20d` | $\text{std}(R_{1d}, 20)$ | 20-day rolling volatility |
| 6 | `Vol_60d` | $\text{std}(R_{1d}, 60)$ | 60-day rolling volatility |
| 7 | `Vol_Ratio_5_60` | $\text{Vol}_{5d} / \text{Vol}_{60d}$ | Volatility regime ratio |
| 8 | `MA5_vs_MA20` | $(\text{SMA}_5 - \text{SMA}_{20}) / \text{SMA}_{20}$ | Fast trend spread |
| 9 | `MA20_vs_MA50` | $(\text{SMA}_{20} - \text{SMA}_{50}) / \text{SMA}_{50}$ | Medium trend spread |
| 10 | `MA50_vs_MA200` | $(\text{SMA}_{50} - \text{SMA}_{200}) / \text{SMA}_{200}$ | Macro trend spread |
| 11 | `ZScore_Close_20` | $(P_t - \mu_{20}) / \sigma_{20}$ | Rolling 20-day price Z-score |
| 12 | `ZScore_RSI_20` | $(\text{RSI}_{14} - \mu_{\text{RSI}, 20}) / \sigma_{\text{RSI}, 20}$ | RSI oscillator Z-score |
| 13 | `ZScore_MACD_20` | $(\text{MACD} - \mu_{\text{MACD}, 20}) / \sigma_{\text{MACD}, 20}$ | MACD signal Z-score |
| 14 | `ZScore_Vol_20` | $(V_t - \mu_{V, 20}) / \sigma_{V, 20}$ | Volume Z-score |
| 15 | `ATR_Regime_Ratio` | $\text{ATR}_{14} / \text{SMA}_{20}(\text{ATR}_{14})$ | Volatility expansion indicator |
| 16 | `Day_of_Week` | Day of week integer (0–4) | Calendar cycle scalar |
| 17 | `Month_of_Year` | Month integer (1–12) | Seasonality scalar |
| 18 | `SPY_Return_1d` | SPY $(P_t / P_{t-1}) - 1$ | Market benchmark 1d return |
| 19 | `SPY_Return_5d` | SPY $(P_t / P_{t-5}) - 1$ | Market benchmark 5d return |
| 20 | `SPY_Beta_60d` | $\text{cov}(R_i, R_{\text{SPY}}, 60) / \text{var}(R_{\text{SPY}}, 60)$ | 60-day rolling market beta |
| 21 | `QQQ_Return_1d` | QQQ $(P_t / P_{t-1}) - 1$ | Tech benchmark 1d return |
| 22 | `QQQ_Beta_60d` | $\text{cov}(R_i, R_{\text{QQQ}}, 60) / \text{var}(R_{\text{QQQ}}, 60)$ | 60-day rolling tech beta |
| 23 | `VIX_Level` | CBOE VIX close level | Bounded volatility index |
| 24 | `VIX_Change_5d` | $(VIX_t / VIX_{t-5}) - 1$ | 5-day implied volatility change |
| 25 | `Sector_Rel_Return_5d`| $R_{5d, \text{asset}} - R_{5d, \text{sector}}$ | 5-day sector excess return |
| 26 | `Sector_Rel_Return_20d`| $R_{20d, \text{asset}} - R_{20d, \text{sector}}$ | 20-day sector excess return |

**Integrity Verification:** Zero raw non-stationary price levels exist in the deployed 27-feature vector. All features are stationary returns, bounded oscillators, rolling Z-scores, or normalized ratios.

---

## 4. Database & Ledger Storage Integrity

### 4.1 Schema Verification of `signal_ledger.db`
The SQLite database [backend/artifacts/signal_ledger.db](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/artifacts/signal_ledger.db) contains the following core tables:
- `prospective_observations`:
  - Columns: `id`, `observation_id`, `ticker`, `as_of_date`, `signal_action`, `calibrated_probs`, `conviction`, `raw_model_probs`, `model_hashes`, `config_hash`, `manifest_sha256`, `observation_hash`, `previous_hash`, `created_at_utc`.
  - Constraints: `UNIQUE(ticker, as_of_date)`, `NOT NULL` on hash columns.
  - Foreign Keys & Triggers: Prevent modification or deletion of finalized prospective rows.
- `execution_events`:
  - Columns: `event_id`, `observation_id`, `order_type`, `fill_price`, `slippage_bps`, `commission`, `fill_timestamp_utc`, `status`.
- `portfolio_states`:
  - Columns: `state_id`, `as_of_date`, `nav`, `cash_balance`, `current_positions_json`, `daily_pnl`, `unrealized_pnl`.

### 4.2 Hash Chaining Verification
Each prospective observation calculates its SHA-256 fingerprint as:
$$\text{hash}_t = \text{SHA256}(\text{hash}_{t-1} + \text{ticker} + \text{as\_of\_date} + \text{action} + \text{calibrated\_probs} + \text{manifest\_sha256})$$

- Active Record: `PROP-AAPL-20261001-DEE62FA6`
- Manifest Fingerprint: `e34d9d472505d6b2d7a79924cbdf1fc6c0aa2346f67c72f9c0afd94b5352b195`
- State: Fully verified, zero tampering detected.

---

## 5. Remediation & Action Items for V2.3

1. **Implement Fallback Data Provider:** Introduce a secondary market data fallback provider inside `data_loader.py` to prevent pipeline stalls during Yahoo Finance outages.
2. **Deterministic Parquet Backup:** Add daily automatic snapshotting of raw ingested OHLCV bars into immutable Apache Parquet files (`backend/data/snapshots/`) for deterministic local reproduction.
3. **Data Integrity Test Gate:** Add an automated daily integrity check script in `backend/scripts/ops/verify_data_integrity.py` that verifies ADF stationarity, OHLC consistency, and missingness before running paper inference.
