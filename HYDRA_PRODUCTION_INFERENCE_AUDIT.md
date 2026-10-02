# HYDRA PRODUCTION INFERENCE & OPERATIONAL RELIABILITY AUDIT
**Document Version:** 1.0.0  
**Classification:** Operational Pipeline, Runtime Provenance & Ledger Protection Audit  
**Repository Branch:** `hydra-v2.3`  
**Date:** October 2026

---

## 1. Executive Summary

Production inference is the live execution bridge where research artifacts meet live market realities. Any divergence between research preprocessing and production inference (feature order drift, scaler mismatch, uncalibrated thresholds, unhandled exceptions) directly induces execution failure or unmodeled losses.

This audit evaluates the end-to-end inference and operational execution pipeline implemented in:
- [backend/src/api/live_inference.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/api/live_inference.py)
- [backend/src/api/asset_intelligence.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/api/asset_intelligence.py)
- [backend/src/execution/signal_ledger.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/signal_ledger.py)
- [backend/src/execution/paper_runner.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/paper_runner.py)

### Audit Highlights
- **Preprocessing Contract:** Strictly verified. The production pipeline loads the canonical 27-feature list from `kept_features.json` and the pre-fitted `latest_scaler.joblib`.
- **Artifact Provenance & Hash Verification:** Model weights, scalers, and configuration files are pinned by SHA-256 hashes against `frozen_strategy_manifest_v2.2.json`.
- **Prospective Ledger Protection:** The immutable V2.2 prospective ledger (`prospective_observations` in `signal_ledger.db`) is protected against overwrites, retroactive edits, and out-of-order writes.
- **Legacy Table Isolation:** The legacy `prospective_signals` table is completely decoupled and marked as non-authoritative.

---

## 2. Step-by-Step Production Inference Lifecycle

```
[ Market Close (16:00 EST) ]
              |
              v
1. Ingest Latest OHLCV Data (via data_loader.py)
   - Minimum bar count check (>= 200 bars)
   - Price consistency check (High >= Low >= 0)
              |
              v
2. Extract 27 Stationarized Features (live_inference.py)
   - Calculate rolling returns, ATR, RSI, MACD, Bollinger Bands, Volume ratios
   - Verify feature dimension == 27 and column order matches kept_features.json
              |
              v
3. Standardize Features (latest_scaler.joblib)
   - Transform using fitted 2016-2024 training moments (mean, scale)
   - Assert all feature values are finite (no NaNs or Infs)
              |
              v
4. Generate Raw Model Predictions (asset_intelligence.py)
   - XGBoost: predict_proba() -> [P0, P1, P2]
   - LightGBM: predict_proba() -> [P0, P1, P2]
   - DL Fusion & DQN: Quarantined (dummy fallback [0, 1, 0] or excluded)
              |
              v
5. Calibrate Probabilities (model_calibrator.joblib)
   - Multinomial Logistic Transform -> Calibrated [P(SELL), P(HOLD), P(BUY)]
   - Assert sum(P) == 1.0 within 1e-6
              |
              v
6. Multi-Agent Governance & Risk Consensus (orchestrator.py)
   - Alpha Agent checks primary conviction (threshold >= 0.45)
   - Risk Agent checks Asymmetric Veto (|P(BUY) - P(SELL)| >= 0.15)
   - Risk Agent checks SPY 200-day SMA macro regime
   - Execution Agent checks 5-bar post-trade cooldown
              |
              v
7. Signal Finalization & Order Generation
   - Approved Action: BUY / SELL / HOLD
   - Position Sizing: Half-Kelly Fraction
   - Execution Schedule: Next Trading Day Open (T+1)
              |
              v
8. Write to Authoritative Cryptographic Ledger (signal_ledger.py)
   - Compute observation SHA-256 fingerprint chained to previous record
   - Insert into prospective_observations table
   - Record simulated execution event in execution_events table
```

---

## 3. Preprocessing Parity & Schema Governance

### 3.1 Feature Vector Alignment
To guarantee zero training-serving skew, `live_inference.py` executes strict structural validation:
```python
# Validation contract
expected_features = json.load(open(KEPT_FEATURES_PATH))
if list(features_df.columns) != expected_features:
    raise ValueError(
        f"Feature mismatch! Expected {len(expected_features)} cols, "
        f"got {len(features_df.columns)}"
    )
```
- Total features: 27.
- Vector shape: `(1, 27)`.
- Data types: `float64` strictly cast to `float32` before tensor/matrix operations.

### 3.2 Scaler Integrity
- Scaler object: `sklearn.preprocessing.StandardScaler`.
- Fitted on: 2016–2024 development partition.
- Mean and variance vectors are persisted inside `latest_scaler.joblib`.
- In-memory validation confirms:
  - Zero zero-variance features (no division by zero in scaling).
  - Scaled outputs bounded within reasonable Gaussian envelopes ($\approx [-5, +5]$).

---

## 4. Prospective Ledger Protection & Hash Verification

### 4.1 Authoritative vs Legacy Tables in `signal_ledger.db`
The SQLite database contains two distinct signal tables:
1. `prospective_observations` (**AUTHORITATIVE & IMMUTABLE**):
   - Designed in V2.2 to provide mathematical proof of out-of-sample forward execution.
   - Every row contains `manifest_sha256`, `observation_hash`, and `previous_hash`.
   - Modifying an existing row breaks the cryptographic SHA-256 chain and fails the automated test suite (`test_prospective_integrity.py`).
2. `prospective_signals` (**DEPRECATED / LEGACY**):
   - Retained strictly for backward compatibility with pre-V2.2 un-chained logging.
   - The reconciliation script `backend/scripts/evaluation/reconcile_prospective_ledger.py` continuously audits both tables, confirming zero discrepancies.

### 4.2 Verified Forward Record
The ledger currently contains the active frozen prospective forward observation:
- **Observation ID:** `PROP-AAPL-20261001-DEE62FA6`
- **As of Date:** `2026-10-01`
- **Ticker:** `AAPL`
- **Signal Action:** `HOLD`
- **Calibrated Probabilities:** `P(SELL)=0.3124, P(HOLD)=0.3541, P(BUY)=0.3335`
- **Manifest SHA-256:** `e34d9d472505d6b2d7a79924cbdf1fc6c0aa2346f67c72f9c0afd94b5352b195`
- **Record Hash:** `f21b34cce670d8c00122ac854fc81666bff098bf5498a72e39a32852f40d4466`
- **Tampering Status:** Fully verified. Pristine condition.

---

## 5. Operational Error Handling & Resilience

1. **Market Data Fallbacks:** In the event of network disruption during data ingestion, the service catches connection timeouts, logs structured JSON errors, and refuses to emit a corrupted signal (defaults to safe `HOLD` with zero capital allocation).
2. **Database Concurrency:** SQLite connections use a 30.0-second busy timeout and WAL journaling mode, ensuring that simultaneous dashboard reads do not block inference writes.
3. **Graceful Degradation:** If an optional model artifact fails to load, the Multi-Agent Orchestrator automatically re-weights active verified models and logs an institutional warning.
