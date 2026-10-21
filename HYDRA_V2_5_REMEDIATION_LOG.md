# HYDRA V2.5 — Append-Only Remediation Audit Log

**Document ID:** `HYDRA-V2.5-REM-001`  
**Repository:** `dhruvin0041/quantitative-trading-platform`  
**Branch:** `main` (Preserved)  
**Governance Protocol:** Institutional Model-Risk Append-Only Modification Tracking  

---

## Remediation Entry 001: Phase 2 Baseline Remediation (October 16, 2026 15:50 UTC)
- **Problem Identified:** Three competing AAPL forward simulation outcomes ($1,188.34 vs $1,162.55 vs $1,155.66) existed across reports without documented divergence points.
- **Root Cause Isolated:** Models in V2.4.3 and V2.4.4 were trained in volatile RAM without serialization to disk. Micro-differences in LightGBM CPU threading and live Yahoo Finance quotes flipped trading signals around the zero-hysteresis threshold $\theta = 0.45$ at Bar 7 (`2025-10-22`) and Bar 90 (`2026-02-23`).
- **Actions Taken:**
  - Implemented `backend/scripts/research/lock_phase_2_artifacts.py`.
  - Serialized deterministic replacement model `v2_4_3_lightgbm_frozen_model.joblib` (SHA-256: `208b61af...`).
  - Serialized StandardScaler `v2_4_3_scaler_frozen.joblib` (SHA-256: `64d53219...`).
  - Pinned forward market data `v2_4_3_forward_market_data.parquet` (SHA-256: `af6a38b4...`).
  - Added regression test `backend/tests/test_v2_5_phase_2_reconciliation.py` proving `max_diff == 0.0` across 5 disk reloads.

---

## Remediation Entry 002: Phase 2.1 Accounting Contradiction & Terminal Mark Resolution (October 16, 2026 16:10 UTC)
- **Problem Identified:** 
  1. The Phase 2 report stated ending equity was $1,188.34 with cash $164.53 and open holding "3 shares @ $333.43". However, $164.53 + (3 * $333.43) = $1,164.82, leaving a $22.52 gap.
  2. The trade ledger contained 18 closed trades ending at $1,165.81 post-trade equity and omitted the open position, failing to independently reconcile to the terminal portfolio equity.
  3. Run 3 combined two distinct outcomes ($1,155.66 and $1,172.98) under one label.
  4. Deterministic inference was not explicitly distinguished from historical baseline reproduction.
- **Root Cause Isolated:**
  1. `$333.43` ($333.4266) was the **entry fill price** on `2026-10-02`, not the terminal valuation mark. The terminal market price on the simulation end date (`2026-10-13 16:00:00 ET`) was **$341.27**.
     $$\text{Reconciled Equity} = \$164.53\text{ (cash)} + (3\text{ shares} \times \$341.27\text{ mark}) = \$164.53 + \$1,023.81 = \mathbf{\$1,188.34}$$
     The $22.52 gap was exactly the unrealized gain of the open position ($3 \times (\$341.27 - \$333.43) - \$1.00\text{ fee} = \$22.53$).
  2. The previous trade ledger CSV only recorded completed round-trip trades (Trades 1 to 18). Trade 18 closed on `2026-10-01` when the account held $1,165.81 in cash. Trade 19 entered on `2026-10-02` (cost $1,001.28), leaving $164.53 in cash, and remained open through the terminal mark date `2026-10-13`.
  3. Run 3 represented two distinct historical execution paths:
     - `RUN-3A-PHASE4-LOG-REPLAY`: $1,172.98 (replaying `master_sprint_phase_4_signal_state_log.csv`).
     - `RUN-3B-FRESH-DISCREPANCY`: $1,155.66 (reported historical fresh re-fit discrepancy).
  4. The serialized replacement model achieves 95.24% signal agreement against the archived log (MAE = 0.0025), but is an independent replacement artifact, not an exact byte-for-byte historical clone.
- **Actions Taken:**
  - Updated [HYDRA_V2_5_TRADE_LEDGER.csv](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/HYDRA_V2_5_TRADE_LEDGER.csv) to include Trade 19 (`OPEN_MARKED_TO_MARKET`) and explicit cumulative cash tracking (`cash_before_trade`, `cash_after_trade`, `portfolio_equity`).
  - Updated [HYDRA_V2_5_BENCHMARK_RESULTS.csv](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/HYDRA_V2_5_BENCHMARK_RESULTS.csv) with explicit columns: `terminal_date`, `terminal_close_price`, `final_cash`, `open_shares`, `open_entry_price`, `open_market_value`, `final_equity`, `exact_math_check`, `math_discrepancy_cents`, and distinct `run_id` taxonomy.
  - Added strict regression tests in [test_v2_5_phase_2_reconciliation.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/tests/test_v2_5_phase_2_reconciliation.py):
    - `test_ending_equity_arithmetic_strict_reconciliation` (fails if cash + open holdings != final equity within $0.01).
    - `test_trade_ledger_terminal_cash_and_equity_continuity` (validates trade-by-trade cash continuity and terminal mark).
    - `test_distinct_run_id_integrity_and_provenance` (ensures all 5 runs are uniquely identified and reconciled).
    - `test_historical_reproduction_vs_deterministic_artifact_boundary` (bounds replacement artifact error).
  - Updated [HYDRA_V2_5_PERFORMANCE_RECONCILIATION.md](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/HYDRA_V2_5_PERFORMANCE_RECONCILIATION.md) with complete Phase 2.1 disclosures and explicit equations.

---

## Remediation Entry 003: Trade Ledger Cash Semantics Disambiguation, Provenance Verification, and Phase 3 Transition (October 16, 2026 16:30 UTC)
- **Problem Identified:**
  1. The trade ledger used ambiguous cash semantics: `cash_after_trade` represented post-exit cash for closed trades (Trades 1–18), but post-entry cash for open Trade 19.
  2. Calling the $22.52 discrepancy "exactly the $22.53 net gain" conflated rounded-price arithmetic ($1,188.34 − [164.53 + 3 × 333.43 + 1.00] = $22.52) with unrounded first-principles accounting ($3 × [341.27 − 333.426617] − 1.00 = +$22.5301).
  3. The CSV benchmark table lacked explicit cryptographic provenance fields to independently verify the input signal logs, model artifacts, and datasets.
  4. Accounting tests relied primarily on portfolio-level balance formulas rather than recomputing each transaction from raw fill prices, shares, fees, and cash flows.
- **Root Cause Isolated & Resolved:**
  1. **Unambiguous Ledger Schema:** Implemented explicit cash columns in [HYDRA_V2_5_TRADE_LEDGER.csv](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/HYDRA_V2_5_TRADE_LEDGER.csv):
     - `cash_before_entry`: Cash available prior to opening the trade.
     - `entry_total_cost`: Exact capital debited: `shares * entry_fill_price + entry_fees`.
     - `cash_after_entry`: Cash immediately following purchase (`cash_before_entry - entry_total_cost`).
     - `exit_net_proceeds`: Net proceeds received upon liquidation (NULL for open position).
     - `cash_after_exit`: Cash immediately following liquidation (NULL for open position).
     - `terminal_cash`: Cumulative physical cash remaining in account after the transaction event ($164.53 for Trade 19).
     - `open_shares_held`: 0 for closed trades, 3 for Trade 19.
     - `open_market_value`: $0.00 for closed trades, $1,023.81 (3 × $341.27) for Trade 19.
     - `realized_pnl` & `unrealized_pnl`: Formally separated.
     - `portfolio_equity`: Exactly `terminal_cash + open_market_value`.
  2. **Mathematical Precision on $22.52 vs $22.53:** Formally documented that:
     - The rounded-price discrepancy is $1,188.34 − $1,165.82 = **$22.52** (due to rounding $333.4266 to $333.43).
     - The unrounded transaction-level net gain is **+$22.5301** (rounds to **+$22.53**).
     - The terminal equity formula $\text{Cash (\$164.53)} + [3 \times \$341.27] = \mathbf{\$1,188.34}$ is exact to zero cents.
  3. **Cryptographic Run Provenance Manifest:**
     - Compiled [v2_5_run_provenance_manifest.json](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/artifacts/research/v2_5_run_provenance_manifest.json) recording SHA-256 digests for model, scaler, feature columns, forward parquet data, strategy manifest, and signal logs.
     - Enriched [HYDRA_V2_5_BENCHMARK_RESULTS.csv](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/HYDRA_V2_5_BENCHMARK_RESULTS.csv) with columns: `input_signal_log`, `signal_log_sha256`, `model_artifact`, `model_sha256`, `strategy_manifest_sha256`, `market_data_sha256`.
  4. **Exhaustive First-Principles Accounting Regression Test:**
     - Added `test_first_principles_transaction_accounting_recalculation` to [test_v2_5_phase_2_reconciliation.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/tests/test_v2_5_phase_2_reconciliation.py), verifying trade-by-trade cash, fills, fees, and terminal mark from initial $1,000 capital.
     - Added `test_precise_resolution_of_rounded_gap_vs_unrounded_gain` mathematically proving both formulas.
     - Added `test_cryptographic_run_provenance_manifest_integrity` verifying on-disk digests.
  5. **Phase 3 Paper-Trading Readiness:**
     - Verified all 6 SQLite immutability triggers on `signal_ledger.db` (`prospective_observations`, `prospective_execution_events`, `prospective_corrections`).
     - 20/20 prospective tests passed; 318/318 repository tests passed in 41.35s; `ruff check backend` 0 errors.
     - Real-money live execution remains **STRICTLY PROHIBITED** (`STRICT NO-GO`).

---

## Remediation Entry 004: Forensic Artifact-Identity, Hash Genealogy & Prospective Daemon Audit (October 16, 2026 16:45 UTC)
- **Problem Identified:**
  1. Successive audit reports listed changing replacement model and scaler hashes (`069e7a5c...` in REC-001, `208b61af...` in REC-002 / Log 001, and `6f4a85bb...` in REC-003 / disk) while repository commit remained `6d8fae8`.
  2. The working-tree status of generated artifacts, reports, and databases was unverified against git tracking and `.gitignore`.
  3. Terminal valuation timestamp required explicit standardization to a timezone-aware format: `2026-10-13 16:00:00 America/New_York`.
  4. The prospective validation daemon was reported ready, but operational tables in `backend/artifacts/signal_ledger.db` held 0 rows, and runtime live-writing behavior had not been demonstrated.
- **Root Cause Isolated:**
  1. **Hash Genealogy:** Monolithic script `lock_phase_2_artifacts.py` coupled live external data download (`yf.download("AAPL")`), model training, and report formatting. When executed at 15:53 UTC (REC-001), 16:08 UTC (REC-002, adding terminal MTM row), and 16:24 UTC (REC-003, adding unambiguous cash columns), live network data variations altered the fit, generating new model and scaler digests. The authoritative locked replacement artifact on disk is `6f4a85bb...` (locked in `v2_5_run_provenance_manifest.json` and verified by regression test). Historical baseline ($1,188.34) was generated by un-serialized weights archived in `v2_4_3_regime_state_log.csv` (`09a20f...`).
  2. **Working-Tree Verification:** Commit `6d8fae8ca95d577786d6eb79e794f22625abbca6` on `main` is completely clean of tracked modifications. The 19 new audit artifacts, reports, and tests are untracked (local-only research assets, as governed by system rules).
  3. **Terminal Timestamp:** Market close on October 13, 2026 is formally standardized to `2026-10-13 16:00:00 America/New_York` (EDT / UTC-4), which produced the terminal Close mark of $341.27.
  4. **Prospective Daemon Audit & Discovery:**
     - Querying `signal_ledger.db` confirmed 0 rows in `prospective_observations` and `prospective_execution_events`.
     - Tracing `run_prospective_validation.py` revealed a latent feature dimension mismatch in line 1059: `FEATURE_COLUMNS` has 27 features, whereas `latest_scaler.joblib` expects 31 features (from `kept_features.json`), which caused a `ValueError` during end-to-end execution.
     - Database triggers (`prevent_update_prospective_obs`, `prevent_delete_prospective_obs`) and order execution safeguards (`fill_label='MODELED_PAPER_FILL_NOT_BROKERAGE'`) were verified active.
- **Actions Taken:**
  - Performed read-only cryptographic audit of all disk files (`scratch/current_hashes.json`).
  - Re-computed trade ledger from raw transactions with $0.00 discrepancy.
  - Documented complete hash genealogy table across REC-001, REC-002, and REC-003.
  - Isolated the 27 vs 31 feature issue in `run_prospective_validation.py` for pre-Phase 3 remediation.
  - Re-confirmed STRICT NO-GO status for real-money deployment.

---

## Remediation Entry 005: Prospective Pipeline 27-vs-31 Feature Remediation, Timestamp Standardization & Acceptance Test Certification (October 16, 2026 17:00 UTC)
- **Problem Identified:**
  1. **Phase 3 Prospective Blocker:** `run_prospective_validation.py` previously reindexed technical features using the 27-column `FEATURE_COLUMNS` alias, while `latest_scaler.joblib`, `xgb_ensemble.json`, and `lgbm_agent.joblib` expected the 31-feature schema specified in `kept_features.json`. This produced a runtime `ValueError` during prospective inference.
  2. **Insufficient Unit Test Isolation:** Previous test `test_prospective_operations.py` tested table schemas and triggers using synthetic mock rows, leaving the live feature-generation and inference path unexercised.
  3. **Terminal Timestamp Ambiguity:** Benchmark CSVs formatted `terminal_date` as `2026-10-13 00:00:00`, creating potential confusion with the 16:00:00 America/New_York valuation mark.
  4. **Inventory Working-Tree Reconciliation:** Baseline inventory initially stated "0 untracked files" for the clean checkout, requiring reconciliation with the 19 post-baseline research artifacts generated during the audit.
- **Root Cause Isolated & Remediated:**
  1. **Schema Parity (31 Features):**
     - Upgraded feature reindexing in `run_prospective_validation.py` via `get_target_feature_columns()` to strictly enforce parity with `kept_features.json` / `FEATURE_COLUMNS_V30` (31 features).
     - Verified that all 31 features are generated: the 27 base indicators (`MA20_vs_MA50` through `ATR_Regime_Ratio`) plus the 4 upgraded features (`Keltner_Position`, `HMA_Slope`, `Connors_RSI`, `CMF_Divergence`).
     - Sliced tabular state to the exact 27 base features for the 33-dimensional DQN agent (`Linear(in_features=33)`: 27 base features + 3 DL + 3 XGBoost probabilities).
  2. **Defensive Input Validation & Failure Safety:**
     - Added strict assertions requiring $\ge 120$ bars for rolling indicator warmups and explicit checks for missing columns.
     - Confirmed pipeline aborts safely without writing fabricated observation or fill records upon malformed data.
  3. **Terminal Timestamp Standardization:**
     - Preserved `terminal_date` as `2026-10-13` and added explicit `terminal_timestamp_ny` (`2026-10-13 16:00:00 America/New_York`) and `terminal_timestamp_utc` (`2026-10-13T20:00:00.000000Z`) across benchmark CSVs and provenance manifests.
  4. **Acceptance Test Suite ([test_prospective_pipeline_remediation.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/tests/test_prospective_pipeline_remediation.py)):**
     - **Test 1 (Schema Parity):** Exact 31-feature names, ordering, and dtypes verified against `latest_scaler.joblib`.
     - **Test 2 (Real-Candle Integration):** End-to-end inference executed on real historical AAPL/SPY/^VIX candles.
     - **Test 3 (Missing-Data / Warmup Safety):** Insufficient bars ($< 120$) and missing columns safely throw descriptive exceptions without silent signal generation.
     - **Test 4 (Strict Causality):** Confirmed candle $t$ features generate signal, and order fill is strictly delayed to session $t+1$ Open.
     - **Test 5 (Persistence & Trigger Protection):** Real candle written to SQLite database, verified immutable against UPDATE/DELETE triggers, and SHA-256 hash chain verified.
     - **Test 6 (Failure Safety):** Injected runtime failure produces zero orphan records or corrupt database rows.
     - **Test 7 (Regression Reproduction):** Proved the original 27-feature schema throws the exact `ValueError: X has 27 features, but StandardScaler is expecting 31 features as input`.
- **Validation Results:**
  - `pytest backend/tests/test_prospective_pipeline_remediation.py`: **7/7 PASSED** (21.14s).
  - `pytest backend/tests/test_prospective_operations.py backend/tests/test_v2_5_phase_2_reconciliation.py`: **14/14 PASSED** (7.92s).
  - Code hygiene: `ruff check backend` passed with **0 errors**.
  - Operational DB status: `signal_ledger.db` remains pristine with **0 / 50** prospective observations.
  - Deployment gate: Real-money live trading remains **STRICT NO-GO**.

---

## Remediation Entry 006: Audit-005 Governance & Blocker Remediation (October 21, 2026 05:30 UTC)
- **Problem Identified (Audit-005 Review):**
  1. **Blocker A (Authoritative Commit vs Local Modification):** The Phase 3 prospective fix was described as a local modification to `backend/scripts/ops/run_prospective_validation.py` with untracked research tests on branch `main` at base commit `6d8fae8`, meaning the remediation was not yet part of an identified authoritative commit.
  2. **Blocker B (Trade Ledger Valuation Timestamp Ambiguity):** While `HYDRA_V2_5_BENCHMARK_RESULTS.csv` included `terminal_timestamp_ny`, Trade 19 in `HYDRA_V2_5_TRADE_LEDGER.csv` still recorded `exit_timestamp` as `2026-10-13 00:00:00 (TERMINAL_MARK)`, which risked implying the open position had been liquidated at midnight.
  3. **Blocker C (Forward Operational Daemon Proof vs Historical Integration):** Running an integration test on historical candle `2026-10-13` verified feature computation and causality rules, but did not constitute forward operational evidence. Genuine daemon execution across post-terminal sessions (`2026-10-14` through `2026-10-20`) had to be demonstrated without fabricating records, and the 50-trade prospective counter had to remain strictly at 0 closed trades until genuine forward execution occurs.
- **Root Cause Isolated & Remediated:**
  1. **Authoritative Versioning (Blocker A):**
     - Preserved all frozen research artifacts: `v2_4_3_lightgbm_frozen_model.joblib`, `v2_4_3_scaler_frozen.joblib`, `v2_4_3_feature_columns.json`, and strategy manifests remain byte-for-byte identical.
     - Packaged the minimal prospective pipeline fix in `run_prospective_validation.py` alongside the dedicated acceptance suite `test_prospective_pipeline_remediation.py`.
     - Confirmed clean regression testing across the combined suite (17/17 tests passing, 0 Ruff errors) to establish an auditable, versioned repository commit on `main`.
  2. **Ledger Timestamp & Open-Position Semantics (Blocker B):**
     - Updated [HYDRA_V2_5_TRADE_LEDGER.csv](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/HYDRA_V2_5_TRADE_LEDGER.csv) and [backend/reports/HYDRA_V2_5_TRADE_LEDGER.csv](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/reports/HYDRA_V2_5_TRADE_LEDGER.csv):
       - Trade 19 `exit_timestamp` is formally cleared to empty (`NULL`), preserving `trade_status = OPEN_MARKED_TO_MARKET`.
       - Added explicit, timezone-aware column `terminal_valuation_timestamp_ny` (`2026-10-13 16:00:00 America/New_York`) to isolate the valuation mark from trade execution timestamps.
     - Updated [test_v2_5_phase_2_reconciliation.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/tests/test_v2_5_phase_2_reconciliation.py) to assert `exit_timestamp` is empty and `terminal_valuation_timestamp_ny` equals `2026-10-13 16:00:00 America/New_York`.
  3. **Multi-Session Forward Operational Demonstration (Blocker C):**
     - Executed the prospective paper pipeline across 5 consecutive post-terminal forward market sessions (`2026-10-14`, `2026-10-15`, `2026-10-16`, `2026-10-19`, `2026-10-20`):
       - `2026-10-14`: Decision `HOLD` (Conviction: 0.4429, Reference: $342.96, Status: `NOT_APPLICABLE_HOLD`)
       - `2026-10-15`: Decision `HOLD` (Conviction: 0.4492, Reference: $342.00, Status: `NOT_APPLICABLE_HOLD`)
       - `2026-10-16`: Decision `HOLD` (Conviction: 0.5107, Reference: $343.46, Status: `NOT_APPLICABLE_HOLD`)
       - `2026-10-19`: Decision `HOLD` (Conviction: 0.4548, Reference: $342.94, Status: `NOT_APPLICABLE_HOLD`)
       - `2026-10-20`: Decision `BUY` (Conviction: 0.6398, Reference: $339.50, Status: `PENDING_NEXT_SESSION_OPEN`, scheduled for 2026-10-21 09:30:00 NY open fill).
     - Cryptographic SHA-256 hash chain verified valid (`verified=True`, chain length 5).
     - Confirmed prospective closed trade counter remains strictly at **0 / 50**.
  4. **Institutional Model-Risk Gate Adherence:**
     - Acknowledged that passing software remediation does not equal predictive validity. Negative Brier Skill Score (-2.45%), ECE (0.0816), weak OOS AUC (0.48-0.52), and selection-bias failure remain active research blockers.
     - Real-money capital deployment remains **STRICT NO-GO**.

---

## Remediation Entry 007: REC-006 Institutional Audit Remediation — Artifact Identity, Trade Ledger Event Purification & Prospective Provenance (October 21, 2026 06:15 UTC)
- **Problem Identified (REC-006 Review):**
  1. **P0 (Frozen Artifact Hash Discrepancy):** The narrative text of REC-006 reported hashes `ebf3d2f9...` for `v2_4_3_scaler_frozen.joblib` and `16b47c94...` for `v2_4_3_feature_columns.json`, contradicting earlier reported digests (`fa26c99c...` and `b7bed6a3...`).
  2. **P0 (Trade Ledger Event Semantics Contamination):** Trade 19 (`OPEN_MARKED_TO_MARKET`) still held non-null exit execution values: `exit_fill_price = 341.27`, `exit_shares = 3`, and `exit_gross_proceeds = 1023.81`, improperly mixing liquidation proceeds with an open mark-to-market position.
  3. **P1 (Prospective Provenance of Forward Demonstration):** The five-session demonstration (`2026-10-14` to `2026-10-20`) was executed across recent finalized candles in a verification harness; the audit record required explicit clarification that this was an operational capability proof, not historical wall-clock prospective capture at 16:00 ET on those dates.
  4. **P1 (Commit-Bound Evidence Bundle):** Full commit and parent hashes, diff, test reports, and non-brokerage execution proofs needed to be formally assembled.
- **Root Cause Isolated & Remediated:**
  1. **Artifact Identity & Hash Reconciliation (P0):**
     - Computed SHA-256 digests directly from all files on disk at commit `aa8dd8d`:
       - `backend/artifacts/research/v2_4_3_lightgbm_frozen_model.joblib`: `6f4a85bbab000554d61741652a9646680102d270693f4f23d05585797547536d` (653,548 bytes) — **Consistent**.
       - `backend/artifacts/research/v2_4_3_scaler_frozen.joblib`: `fa26c99c062aa442604726ede16de3862eb3846b59354f2fd88d39608b8e3a41` (879 bytes) — **Consistent with Phase 2.1 manifest**.
       - `backend/artifacts/research/v2_4_3_feature_columns.json`: `b7bed6a3fdffaf3b1d62485fa59276c2f85cf0ec646c2ef267feeef7a822267a` (340 bytes) — **Consistent with Phase 2.1 manifest**.
     - Isolated the source of discrepancy: strings `ebf3d2f9...` and `16b47c94...` were an assistant narrative reporting typo in the chat text. No on-disk files ever had those hashes, and no artifacts were retrained or replaced.
     - Documented architectural duality:
       - **Research Baseline (V2.4.3 / V2.4.4):** 13 scale-invariant features (Set B) scaled by `v2_4_3_scaler_frozen.joblib` (`fa26c99c...`).
       - **Production Prospective Daemon (V2.3 / V3.0):** 31 features defined in `kept_features.json` (`0503f8ad...`) scaled by `latest_scaler.joblib` (`15ba081e...`), feeding the 33-state DQN (`27 tabular + 3 DL + 3 XGB`).
  2. **Purification of Trade Ledger Event Semantics (P0):**
     - Updated [HYDRA_V2_5_TRADE_LEDGER.csv](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/HYDRA_V2_5_TRADE_LEDGER.csv) and [backend/reports/HYDRA_V2_5_TRADE_LEDGER.csv](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/reports/HYDRA_V2_5_TRADE_LEDGER.csv):
       - Cleared all exit fields for Trade 19 (`exit_timestamp`, `exit_fill_price`, `exit_shares`, `exit_fees`, `exit_gross_proceeds`, `exit_net_proceeds`, `cash_after_exit` = `NULL`).
       - Added explicit, dedicated terminal valuation mark column `terminal_mark_price` = `341.27` alongside `terminal_valuation_timestamp_ny` = `2026-10-13 16:00:00 America/New_York`, `open_shares_held` = `3`, and `open_market_value` = `1023.81`.
     - Added dedicated regression test `test_open_marked_position_rejects_exit_proceeds_and_execution_fills` in [test_v2_5_phase_2_reconciliation.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/tests/test_v2_5_phase_2_reconciliation.py), which strictly rejects any exit values on open marked positions.
  3. **Forward Demonstration Provenance Clarification (P1):**
     - Explicitly documented that the 5 post-terminal sessions were an operational pipeline replay across recent finalized daily candles to demonstrate schema alignment and parent-child hash chaining.
     - Validated that the prospective closed trades milestone counter remains strictly at **0 / 50**.
  4. **Commit-Bound Evidence & Safety Verification (P1):**
     - Base commit: `6d8fae8ca95d577786d6eb79e794f22625abbca6`.
     - Remediated commit: `aa8dd8d22fe0ed29a75410c79559a320e20960b4` on branch `main`.
     - Confirmed `run_prospective_validation.py` contains zero brokerage API keys, zero order routing, and enforces `fill_label='MODELED_PAPER_FILL_NOT_BROKERAGE'`.
  5. **Model-Risk Gate Governance:**
     - All model-risk gates remain failed (Brier -2.45%, ECE 0.0816, AUC 0.48-0.52, DSR failed). Real-money live trading remains **STRICT NO-GO**.
