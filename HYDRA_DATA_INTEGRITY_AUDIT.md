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
| 0 | `MA20_vs_MA50` | $(\text{SMA}_{20} - \text{SMA}_{50}) / \text{SMA}_{50}$ | Medium trend spread |
| 1 | `EMA9_vs_EMA21` | $(\text{EMA}_{9} - \text{EMA}_{21}) / \text{EMA}_{21}$ | Fast trend momentum spread |
| 2 | `Price_vs_EMA9` | $(P_t - \text{EMA}_{9}) / \text{EMA}_{9}$ | Fast moving average divergence |
| 3 | `Price_vs_EMA21` | $(P_t - \text{EMA}_{21}) / \text{EMA}_{21}$ | Medium moving average divergence |
| 4 | `VIX_Level` | CBOE VIX close level | Bounded volatility index |
| 5 | `BB_Width` | $(\text{Upper}_{20} - \text{Lower}_{20}) / \text{SMA}_{20}$ | Normalized Bollinger Band width |
| 6 | `BB_Position` | $(P_t - \text{Lower}) / (\text{Upper} - \text{Lower} + \epsilon)$ | Percentile rank within Bollinger envelope |
| 7 | `RSI` | 14-period Relative Strength Index | Bounded momentum oscillator $[0, 100]$ |
| 8 | `ADX` | 14-period Average Directional Index | Bounded trend strength metric |
| 9 | `MACD_Hist` | $\text{MACD Line} - \text{Signal Line}$ | Momentum convergence spread |
| 10 | `Relative_Strength`| $R_{20d, \text{ticker}} - R_{20d, \text{SPY}}$ | 20-day excess benchmark return |
| 11 | `OBV_Change` | $(OBV_t - OBV_{t-5}) / |OBV_{t-5}|$ | 5-day volume flow momentum |
| 12 | `Return` | $\ln(P_t / P_{t-1})$ | 1-day logarithmic return |
| 13 | `Volume_Ratio` | $V_t / \text{SMA}_{20}(V)$ | Normalized volume activity ratio |
| 14 | `ZScore_RSI_20` | $(\text{RSI}_t - \mu_{20}) / \sigma_{20}$ | 20-bar rolling RSI Z-score |
| 15 | `ZScore_RSI_50` | $(\text{RSI}_t - \mu_{50}) / \sigma_{50}$ | 50-bar rolling RSI Z-score |
| 16 | `ZScore_RSI_120` | $(\text{RSI}_t - \mu_{120}) / \sigma_{120}$ | 120-bar rolling RSI Z-score |
| 17 | `ZScore_BB_Position_20` | $(\text{BBP}_t - \mu_{20}) / \sigma_{20}$ | 20-bar rolling Bollinger position Z-score |
| 18 | `ZScore_BB_Position_50` | $(\text{BBP}_t - \mu_{50}) / \sigma_{50}$ | 50-bar rolling Bollinger position Z-score |
| 19 | `ZScore_MACD_Hist_20` | $(\text{MACDH}_t - \mu_{20}) / \sigma_{20}$ | 20-bar rolling MACD histogram Z-score |
| 20 | `ZScore_MACD_Hist_50` | $(\text{MACDH}_t - \mu_{50}) / \sigma_{50}$ | 50-bar rolling MACD histogram Z-score |
| 21 | `ZScore_Return_20` | $(R_t - \mu_{20}) / \sigma_{20}$ | 20-bar rolling return Z-score |
| 22 | `ZScore_Return_50` | $(R_t - \mu_{50}) / \sigma_{50}$ | 50-bar rolling return Z-score |
| 23 | `ZScore_Return_120` | $(R_t - \mu_{120}) / \sigma_{120}$ | 120-bar rolling return Z-score |
| 24 | `ZScore_Volume_Ratio_20` | $(\text{VR}_t - \mu_{20}) / \sigma_{20}$ | 20-bar rolling volume ratio Z-score |
| 25 | `ZScore_Volume_Ratio_50` | $(\text{VR}_t - \mu_{50}) / \sigma_{50}$ | 50-bar rolling volume ratio Z-score |
| 26 | `ATR_Regime_Ratio` | $\text{ATR}_{5} / (\text{ATR}_{50} + \epsilon)$ | Volatility expansion / contraction ratio |

**Integrity Verification:** Zero raw non-stationary price levels exist in the deployed 27-feature vector. All features are stationary returns, bounded oscillators, rolling Z-scores, or normalized ratios matching [backend/configs/kept_features.json](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/configs/kept_features.json) and [HYDRA_FEATURE_ENGINEERING_AUDIT.md](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/HYDRA_FEATURE_ENGINEERING_AUDIT.md).

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
2. **Deterministic Parquet Backup (COMPLETED in V2.3):** Implemented committed immutable Parquet snapshot datasets in `backend/data/snapshots/` (`AAPL_features.parquet`, `MSFT_features.parquet`, `NVDA_features.parquet`, `AMZN_features.parquet`, `SPY_benchmark.parquet`, `VIX_benchmark.parquet`), strictly validated via `snapshot_manifest.json` with fail-closed SHA-256 integrity verification in `backtest.py` (`--use-snapshots`).
3. **Data Integrity Test Gate:** Add an automated daily integrity check script in `backend/scripts/ops/verify_data_integrity.py` that verifies ADF stationarity, OHLC consistency, and missingness before running paper inference.
