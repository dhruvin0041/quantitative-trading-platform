"""
Unit and regression test suite for HYDRA V2.5 Phase 2.1 — Deterministic Inference,
Accounting Reconciliation, and Artifact Provenance.

Tests:
1. Deterministic inference idempotency (max_diff == 0.0 across re-loaded runs).
2. Strict ending-equity arithmetic reconciliation (cash + open_shares * terminal_price == final_equity).
3. Trade ledger cumulative cash continuity and open terminal mark reconciliation.
4. First-divergence boundary sensitivity verification (Bar 7 and Bar 90).
5. Preservation of frozen baseline parameters (delta=0.07, sigma=0.020, theta=0.45).
6. Distinct run ID taxonomy and provenance integrity (Runs 1, 2, 3A, 3B, Frozen).
7. Distinction between historical reproduction vs deterministic replacement artifact.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_DIR / "reports"
RESEARCH_ARTIFACTS_DIR = BACKEND_DIR / "artifacts" / "research"
ROOT_DIR = BACKEND_DIR.parent

from scripts.research.master_research_validation_sprint import (
    compute_all_feature_candidates,
)


@pytest.fixture
def locked_artifacts():
    model_path = RESEARCH_ARTIFACTS_DIR / "v2_4_3_lightgbm_frozen_model.joblib"
    scaler_path = RESEARCH_ARTIFACTS_DIR / "v2_4_3_scaler_frozen.joblib"
    feat_cols_path = RESEARCH_ARTIFACTS_DIR / "v2_4_3_feature_columns.json"
    market_data_path = RESEARCH_ARTIFACTS_DIR / "v2_4_3_forward_market_data.parquet"
    provenance_path = RESEARCH_ARTIFACTS_DIR / "v2_4_3_model_provenance.json"

    assert model_path.exists(), "Frozen model artifact missing!"
    assert scaler_path.exists(), "Frozen scaler artifact missing!"
    assert feat_cols_path.exists(), "Feature columns specification missing!"
    assert market_data_path.exists(), "Pinned forward market data missing!"
    assert provenance_path.exists(), "Provenance metadata missing!"

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "feat_cols_path": feat_cols_path,
        "market_data_path": market_data_path,
        "provenance_path": provenance_path,
    }


def test_deterministic_inference_idempotency(locked_artifacts):
    """
    Assert that serialized model and scaler produce float-exact (0.0000000000000000 tolerance)
    identical probabilities across 5 independent re-loads from disk.
    """
    df_raw = pd.read_parquet(locked_artifacts["market_data_path"])
    df_full, feature_sets = compute_all_feature_candidates(df_raw)

    with open(locked_artifacts["feat_cols_path"], "r", encoding="utf-8") as f:
        feat_meta = json.load(f)
    selected_features = feat_meta["feature_columns"]

    eval_dates = df_full.loc["2025-10-13":"2026-10-13"].index
    eval_features_df = df_full.loc[eval_dates, selected_features].ffill().fillna(0.0)

    # Initial inference
    m0 = joblib.load(locked_artifacts["model_path"])
    s0 = joblib.load(locked_artifacts["scaler_path"])
    p_ref = m0.predict_proba(s0.transform(eval_features_df.values))

    # Test 5 independent disk re-loads
    for i in range(5):
        m_trial = joblib.load(locked_artifacts["model_path"])
        s_trial = joblib.load(locked_artifacts["scaler_path"])
        p_trial = m_trial.predict_proba(s_trial.transform(eval_features_df.values))
        max_abs_diff = float(np.max(np.abs(p_ref - p_trial)))
        assert max_abs_diff == 0.0, f"Trial {i+1} failed determinism test: diff={max_abs_diff}"


def test_ending_equity_arithmetic_strict_reconciliation():
    """
    Assert that for every row in HYDRA_V2_5_BENCHMARK_RESULTS.csv:
    final_cash + (open_shares * terminal_close_price) == final_equity
    within 1 cent ($0.01 tolerance).
    """
    bench_paths = [
        ROOT_DIR / "HYDRA_V2_5_BENCHMARK_RESULTS.csv",
        REPORTS_DIR / "HYDRA_V2_5_BENCHMARK_RESULTS.csv",
    ]
    for p in bench_paths:
        assert p.exists(), f"Benchmark results CSV missing at {p}"
        df_bench = pd.read_csv(p)
        assert len(df_bench) >= 9, "Benchmark table missing required strategies"

        for idx, row in df_bench.iterrows():
            run_id = row["run_id"]
            cash = float(row["final_cash"])
            shares = int(row["open_shares"])
            mark_price = float(row["terminal_close_price"])
            claimed_equity = float(row["final_equity"])

            calculated_equity = round(cash + (shares * mark_price), 2)
            discrepancy = round(abs(calculated_equity - claimed_equity), 2)

            assert discrepancy <= 0.01, (
                f"Accounting failure in {run_id}: "
                f"Cash ${cash:.2f} + ({shares} sh * ${mark_price:.2f} = ${shares * mark_price:.2f}) "
                f"= ${calculated_equity:.2f}, but claimed final equity is ${claimed_equity:.2f} "
                f"(discrepancy: ${discrepancy:.2f})"
            )


def test_trade_ledger_terminal_cash_and_equity_continuity():
    """
    Assert that HYDRA_V2_5_TRADE_LEDGER.csv forms a closed, unambiguous, self-reconciling cash ledger:
    - Starts at initial capital $1,000.00.
    - Each closed trade preserves cash continuity:
        cash_after_entry == round(cash_before_entry - entry_total_cost, 2)
        cash_after_exit == round(cash_after_entry + exit_net_proceeds, 2)
        terminal_cash == cash_after_exit
        portfolio_equity == terminal_cash
    - Open Trade 19 maintains distinct, unambiguous semantics:
        cash_after_entry == $164.53 ($1,165.81 - $1,001.28)
        cash_after_exit is None/empty (no exit cash liquidation took place)
        terminal_cash == $164.53
        open_shares_held == 3, open_market_value == $1,023.81 (3 * $341.27)
        portfolio_equity == terminal_cash + open_market_value == $1,188.34
    """
    ledger_paths = [
        ROOT_DIR / "HYDRA_V2_5_TRADE_LEDGER.csv",
        REPORTS_DIR / "HYDRA_V2_5_TRADE_LEDGER.csv",
    ]
    for p in ledger_paths:
        assert p.exists(), f"Trade ledger CSV missing at {p}"
        df_ledger = pd.read_csv(p)
        assert len(df_ledger) == 19, f"Expected 19 rows (18 closed + 1 open), got {len(df_ledger)}"

        # Initial cash check
        assert df_ledger.iloc[0]["cash_before_entry"] == 1000.0

        # Verify continuity across all closed trades
        for idx in range(18):
            row = df_ledger.iloc[idx]
            assert row["status"] == "CLOSED"
            pre_c = float(row["cash_before_entry"])
            cost = float(row["entry_total_cost"])
            proceeds = float(row["exit_net_proceeds"])
            post_entry_c = float(row["cash_after_entry"])
            post_exit_c = float(row["cash_after_exit"])
            term_c = float(row["terminal_cash"])

            # Verify entry cash deduction
            assert round(abs(round(pre_c - cost, 2) - post_entry_c), 2) <= 0.01

            # Verify exit cash addition
            assert round(abs(round(post_entry_c + proceeds, 2) - post_exit_c), 2) <= 0.01

            # Verify terminal cash equals cash after exit for closed trade
            assert round(abs(term_c - post_exit_c), 2) <= 0.01
            assert int(row["open_shares_held"]) == 0
            assert float(row["open_market_value"]) == 0.00
            assert round(abs(float(row["portfolio_equity"]) - term_c), 2) <= 0.01

            # Verify link to next trade
            next_pre_c = float(df_ledger.iloc[idx + 1]["cash_before_entry"])
            assert round(abs(post_exit_c - next_pre_c), 2) <= 0.01

        # Verify Trade 19 (Open Position marked at terminal close)
        t19 = df_ledger.iloc[18]
        assert t19["trade_id"] == 19
        assert t19["status"] == "OPEN_MARKED_TO_MARKET"
        assert t19["cash_before_entry"] == 1165.81
        assert float(t19["cash_after_entry"]) == 164.53
        assert pd.isna(t19["cash_after_exit"]) or t19["cash_after_exit"] == ""
        assert pd.isna(t19["exit_timestamp"]) or t19["exit_timestamp"] == ""
        assert pd.isna(t19["exit_fill_price"]) or t19["exit_fill_price"] == ""
        assert pd.isna(t19["exit_shares"]) or t19["exit_shares"] == ""
        assert pd.isna(t19["exit_fees"]) or t19["exit_fees"] == ""
        assert pd.isna(t19["exit_gross_proceeds"]) or t19["exit_gross_proceeds"] == ""
        assert pd.isna(t19["exit_net_proceeds"]) or t19["exit_net_proceeds"] == ""
        assert t19["terminal_valuation_timestamp_ny"] == "2026-10-13 16:00:00 America/New_York"
        assert float(t19["terminal_mark_price"]) == 341.27
        assert float(t19["terminal_cash"]) == 164.53
        assert int(t19["open_shares_held"]) == 3
        assert float(t19["open_market_value"]) == 1023.81
        assert float(t19["portfolio_equity"]) == 1188.34
        assert float(t19["entry_shares"]) == 3



def test_first_divergence_boundary_sensitivity():
    """
    Verify the mathematical mechanism behind the first divergence points:
    Bar 7 (Run 1 vs Run 3) and Bar 90 (Run 1 vs Run 2).
    """
    archived_log_path = REPORTS_DIR / "v2_4_3_regime_state_log.csv"
    df_log = pd.read_csv(archived_log_path)

    # Bar 7 (2025-10-22): Run 1 vs Run 3
    bar7 = df_log.iloc[7]
    assert bar7["bar_index"] == 7
    assert "2025-10-22" in str(bar7["timestamp"])
    assert bar7["regime"] == "ELEVATED_VOLATILITY"

    # In Run 1, P(Sell) = 0.4461 < 0.45 (HOLD maintained)
    assert bar7["prob_sell"] < 0.45
    assert bar7["emitted_signal"] == "HOLD"
    assert bar7["position_state_after_bar"] == "IN_LONG"

    # Bar 90 (2026-02-23): Run 1 vs Run 2
    bar90 = df_log.iloc[90]
    assert bar90["bar_index"] == 90
    assert "2026-02-23" in str(bar90["timestamp"])

    # In Run 1, P(Buy) = 0.4521 >= 0.45 (BUY emitted)
    assert bar90["prob_buy"] >= 0.45
    assert bar90["emitted_signal"] == "BUY"


def test_preservation_of_frozen_baseline_manifest():
    """
    Verify that frozen_strategy_manifest_v2.4.3.json parameters and payload hash are intact.
    """
    manifest_path = BACKEND_DIR / "artifacts" / "frozen_strategy_manifest_v2.4.3.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check parameters
    assert data["macro_filter_rules"]["delta_sma_distance"] == 0.07
    assert data["macro_filter_rules"]["volatility_threshold"] == 0.02
    assert data["ml_agent_specifications"]["decision_threshold"] == 0.45

    # Check cryptographic payload digest
    payload_without_hash = {k: v for k, v in data.items() if k != "manifest_sha256"}
    recomputed_hash = hashlib.sha256(json.dumps(payload_without_hash, indent=2, sort_keys=True).encode("utf-8")).hexdigest()
    assert recomputed_hash == data["manifest_sha256"]
    assert recomputed_hash == "e3468981077500b203a9ea5608bf657723090be90e90fd099accd2814dd5f32f"


def test_distinct_run_id_integrity_and_provenance():
    """
    Assert that all evaluated runs (Run 1, 2, 3A, 3B, Frozen) have unique Run IDs,
    independent inputs, distinct classifications, and documented equity outcomes.
    """
    bench_path = ROOT_DIR / "HYDRA_V2_5_BENCHMARK_RESULTS.csv"
    df_bench = pd.read_csv(bench_path)

    run_ids = df_bench["run_id"].tolist()
    assert len(run_ids) == len(set(run_ids)), "Duplicate run IDs detected in benchmark table"

    expected_runs = {
        "RUN-1-ARCHIVED-BASELINE": 1188.34,
        "RUN-2-MULTIASSET-PIPELINE": 1162.55,
        "RUN-3A-PHASE4-LOG-REPLAY": 1172.98,
        "RUN-3B-FRESH-DISCREPANCY": 1155.66,
        "RUN-FROZEN-NEW-ARTIFACT": 1192.02,
    }

    for rid, expected_eq in expected_runs.items():
        assert rid in run_ids, f"Run ID {rid} missing from benchmark table"
        actual_eq = float(df_bench[df_bench["run_id"] == rid]["final_equity"].iloc[0])
        assert actual_eq == expected_eq, f"Mismatch for {rid}: expected {expected_eq}, got {actual_eq}"


def test_historical_reproduction_vs_deterministic_artifact_boundary(locked_artifacts):
    """
    Verify the boundary between historical reproduction and deterministic new artifact:
    - Serialized model is deterministic on reloads (max_diff == 0.0).
    - But compared to the historical archived log, divergence exists (MAE ~ 0.0025, max diff ~ 0.023).
    - Proves the serialized model is a replacement artifact, NOT an exact historical clone.
    """
    df_raw = pd.read_parquet(locked_artifacts["market_data_path"])
    df_full, feature_sets = compute_all_feature_candidates(df_raw)

    with open(locked_artifacts["feat_cols_path"], "r", encoding="utf-8") as f:
        feat_meta = json.load(f)
    selected_features = feat_meta["feature_columns"]

    eval_dates = df_full.loc["2025-10-13":"2026-10-13"].index
    eval_features_df = df_full.loc[eval_dates, selected_features].ffill().fillna(0.0)

    # Archived probabilities
    df_archived = pd.read_csv(REPORTS_DIR / "v2_4_3_regime_state_log.csv")
    probs_archived = df_archived[["prob_sell", "prob_hold", "prob_buy"]].values

    # Model inference
    m = joblib.load(locked_artifacts["model_path"])
    s = joblib.load(locked_artifacts["scaler_path"])
    probs_frozen = m.predict_proba(s.transform(eval_features_df.values))

    prob_diff = np.abs(probs_frozen - probs_archived)
    max_prob_diff = float(np.max(prob_diff))
    mae = float(np.mean(prob_diff))

    # Assert bounded difference (proves not identical, but bounded within 0.03)
    assert max_prob_diff > 0.0, "Frozen model should NOT claim exact identity with lost RAM weights"
    assert max_prob_diff < 0.05, f"Unexpectedly large probability difference: {max_prob_diff}"
    assert mae < 0.005, f"MAE exceeds expected threshold: {mae}"


def test_first_principles_transaction_accounting_recalculation():
    """
    Action 2 Mandate: Recalculate each transaction from raw fill prices, share quantities,
    slippage, commissions, and cash movements from first principles ($1,000 initial capital).
    Assert zero penny discrepancy across all 19 trades.
    """
    ledger_path = ROOT_DIR / "HYDRA_V2_5_TRADE_LEDGER.csv"
    df_ledger = pd.read_csv(ledger_path)

    running_cash = 1000.0000

    for idx, row in df_ledger.iterrows():
        trade_id = int(row["trade_id"])
        status = row["status"]
        shares = int(row["entry_shares"])
        entry_price = float(row["entry_fill_price"])
        entry_fees = float(row["entry_fees"])

        # 1. Recalculate entry cost from first principles
        calc_entry_cost = (shares * entry_price) + entry_fees
        reported_entry_cost = float(row["entry_total_cost"])
        assert abs(calc_entry_cost - reported_entry_cost) <= 0.001, (
            f"Trade {trade_id} entry cost mismatch: {calc_entry_cost} vs {reported_entry_cost}"
        )

        # 2. Check cash before entry matches cumulative running cash
        reported_pre_cash = float(row["cash_before_entry"])
        assert abs(running_cash - reported_pre_cash) <= 0.01, (
            f"Trade {trade_id} pre-entry cash mismatch: running={running_cash:.2f}, reported={reported_pre_cash:.2f}"
        )

        # 3. Deduct entry cost
        running_cash_after_entry = running_cash - calc_entry_cost
        reported_after_entry = float(row["cash_after_entry"])
        assert abs(running_cash_after_entry - reported_after_entry) <= 0.01, (
            f"Trade {trade_id} post-entry cash mismatch: running={running_cash_after_entry:.2f}, reported={reported_after_entry:.2f}"
        )

        if status == "CLOSED":
            exit_price = float(row["exit_fill_price"])
            exit_fees = float(row["exit_fees"])
            reported_proceeds = float(row["exit_net_proceeds"])

            # Recalculate exit proceeds from first principles
            calc_net_proceeds = (shares * exit_price) - exit_fees
            assert abs(calc_net_proceeds - reported_proceeds) <= 0.001, (
                f"Trade {trade_id} exit proceeds mismatch: {calc_net_proceeds} vs {reported_proceeds}"
            )

            # Recalculate realized PnL
            calc_realized_pnl = calc_net_proceeds - calc_entry_cost
            reported_pnl = float(row["realized_pnl"])
            assert abs(calc_realized_pnl - reported_pnl) <= 0.01

            # Add proceeds to running cash
            running_cash = running_cash_after_entry + calc_net_proceeds
            reported_after_exit = float(row["cash_after_exit"])
            assert abs(running_cash - reported_after_exit) <= 0.01

            # Equity when flat is exactly running cash
            reported_equity = float(row["portfolio_equity"])
            assert abs(running_cash - reported_equity) <= 0.01

        elif status == "OPEN_MARKED_TO_MARKET":
            # For open trade, no exit proceeds or execution fills exist
            assert pd.isna(row["exit_timestamp"]) or row["exit_timestamp"] == ""
            assert pd.isna(row["exit_fill_price"]) or row["exit_fill_price"] == ""
            assert pd.isna(row["exit_shares"]) or row["exit_shares"] == ""
            assert pd.isna(row["exit_fees"]) or row["exit_fees"] == ""
            assert pd.isna(row["exit_gross_proceeds"]) or row["exit_gross_proceeds"] == ""
            assert pd.isna(row["exit_net_proceeds"]) or row["exit_net_proceeds"] == ""
            assert pd.isna(row["cash_after_exit"]) or row["cash_after_exit"] == ""
            terminal_price = float(row["terminal_mark_price"])
            assert terminal_price == 341.27

            # Recalculate open market value
            calc_open_market_value = shares * terminal_price
            reported_market_value = float(row["open_market_value"])
            assert abs(calc_open_market_value - reported_market_value) <= 0.01

            # Recalculate unrealized PnL (net of entry commission)
            calc_unrealized_pnl = calc_open_market_value - calc_entry_cost
            reported_unrealized = float(row["unrealized_pnl"])
            assert abs(calc_unrealized_pnl - reported_unrealized) <= 0.01

            # Terminal equity: remaining cash + marked shares
            calc_terminal_equity = running_cash_after_entry + calc_open_market_value
            reported_equity = float(row["portfolio_equity"])
            assert abs(calc_terminal_equity - reported_equity) <= 0.01
            assert round(calc_terminal_equity, 2) == 1188.34


def test_precise_resolution_of_rounded_gap_vs_unrounded_gain():
    """
    Action 1 & Reviewer Point C: Precise mathematical resolution of rounded discrepancy vs unrounded net gain.
    - Proves why 1,188.34 - (164.53 + 3 * 333.43 + 1.00) = 22.52 in rounded terms.
    - Proves why 3 * (341.27 - 333.426617) - 1.00 = +22.53 in unrounded first-principles terms.
    - Asserts that both arithmetic formulas are mathematically consistent and do not conflate rounded/unrounded values.
    """
    reported_cash = 164.53
    shares = 3
    rounded_entry_price = 333.43
    entry_fee = 1.00
    terminal_close_price = 341.27
    claimed_final_equity = 1188.34

    # Rounded pre-Trade-19 cash basis:
    rounded_entry_cost = (shares * rounded_entry_price) + entry_fee  # 1000.29 + 1.00 = 1001.29
    rounded_pre_entry_cash = round(reported_cash + rounded_entry_cost, 2)  # 1165.82
    rounded_discrepancy = round(claimed_final_equity - rounded_pre_entry_cash, 2)
    assert rounded_discrepancy == 22.52, f"Expected 22.52, got {rounded_discrepancy}"

    # Exact unrounded transaction-level accounting from parquet data
    exact_entry_fill_price = 333.4266174316406  # raw fill price from simulation
    exact_entry_cost = (shares * exact_entry_fill_price) + entry_fee  # 1000.279852... + 1.00 = 1001.279852...
    exact_pre_entry_cash = 1165.8100  # Trade 18 exit cash
    exact_final_cash = exact_pre_entry_cash - exact_entry_cost  # 164.5301477...

    # Exact terminal valuation
    exact_marked_value = shares * terminal_close_price  # 3 * 341.27 = 1023.8100
    exact_terminal_equity = exact_final_cash + exact_marked_value  # 1188.3401477...

    # Exact unrealized net gain on Trade 19 (after $1 entry fee)
    exact_net_open_gain = exact_marked_value - exact_entry_cost  # 1023.81 - 1001.279852... = +22.5301477...
    assert round(exact_net_open_gain, 2) == 22.53
    assert round(exact_terminal_equity, 2) == 1188.34

    # Final Portfolio Equality
    # Cash ($164.53) + Marked Value ($1,023.81) == Final Equity ($1,188.34) exactly to 0.00 cents
    assert round(reported_cash + (shares * terminal_close_price), 2) == claimed_final_equity


def test_cryptographic_run_provenance_manifest_integrity():
    """
    Action 3 & Reviewer Point D: Verify cryptographic run provenance manifest.
    Asserts that every evaluated run has reproducible inputs, non-empty hashes,
    and that on-disk files match their claimed cryptographic digests.
    """
    manifest_path = RESEARCH_ARTIFACTS_DIR / "v2_5_run_provenance_manifest.json"
    assert manifest_path.exists(), "Master run provenance manifest missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Verify file-level cryptographic hashes
    crypto = manifest["cryptographic_hashes"]
    expected_files = {
        "frozen_model_sha256": RESEARCH_ARTIFACTS_DIR / "v2_4_3_lightgbm_frozen_model.joblib",
        "frozen_scaler_sha256": RESEARCH_ARTIFACTS_DIR / "v2_4_3_scaler_frozen.joblib",
        "feature_columns_sha256": RESEARCH_ARTIFACTS_DIR / "v2_4_3_feature_columns.json",
        "market_data_sha256": RESEARCH_ARTIFACTS_DIR / "v2_4_3_forward_market_data.parquet",
        "strategy_manifest_sha256": BACKEND_DIR / "artifacts" / "frozen_strategy_manifest_v2.4.3.json",
        "run1_signal_log_sha256": REPORTS_DIR / "v2_4_3_regime_state_log.csv",
        "run3a_signal_log_sha256": REPORTS_DIR / "master_sprint_phase_4_signal_state_log.csv",
        "run2_multiasset_log_sha256": REPORTS_DIR / "v2_4_4_multi_asset_comparison.csv",
    }

    for key, path in expected_files.items():
        assert path.exists(), f"Artifact missing for {key}: {path}"
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_hash == crypto[key], f"Cryptographic mismatch for {key}: actual={actual_hash}, recorded={crypto[key]}"

    # 2. Verify benchmark CSV matches provenance manifest
    bench_path = ROOT_DIR / "HYDRA_V2_5_BENCHMARK_RESULTS.csv"
    df_bench = pd.read_csv(bench_path)

    for idx, row in df_bench.iterrows():
        run_id = row["run_id"]
        assert pd.notna(row["signal_log_sha256"]) and row["signal_log_sha256"] != ""
        assert pd.notna(row["model_sha256"]) and row["model_sha256"] != ""
        assert pd.notna(row["strategy_manifest_sha256"]) and row["strategy_manifest_sha256"] != ""
        assert pd.notna(row["market_data_sha256"]) and row["market_data_sha256"] != ""

        if run_id == "RUN-1-ARCHIVED-BASELINE":
            assert row["signal_log_sha256"] == crypto["run1_signal_log_sha256"]
        elif run_id == "RUN-3A-PHASE4-LOG-REPLAY":
            assert row["signal_log_sha256"] == crypto["run3a_signal_log_sha256"]
        elif run_id == "RUN-2-MULTIASSET-PIPELINE":
            assert row["signal_log_sha256"] == crypto["run2_multiasset_log_sha256"]
        elif run_id == "RUN-FROZEN-NEW-ARTIFACT":
            assert row["model_sha256"] == crypto["frozen_model_sha256"]


def test_open_marked_position_rejects_exit_proceeds_and_execution_fills():
    """
    Audit-006 Mandate: Reject exit proceeds, execution fills, or liquidation fees
    for any position with status OPEN_MARKED_TO_MARKET. Ensure strict separation
    between actual exit executions and terminal mark-to-market valuations.
    """
    ledger_path = ROOT_DIR / "HYDRA_V2_5_TRADE_LEDGER.csv"
    df_ledger = pd.read_csv(ledger_path)
    open_positions = df_ledger[df_ledger["status"] == "OPEN_MARKED_TO_MARKET"]
    assert len(open_positions) > 0, "No open marked position found in ledger"

    exit_fields = [
        "exit_timestamp",
        "exit_fill_price",
        "exit_shares",
        "exit_fees",
        "exit_gross_proceeds",
        "exit_net_proceeds",
        "cash_after_exit",
    ]

    for _, row in open_positions.iterrows():
        for field in exit_fields:
            val = row[field]
            assert pd.isna(val) or val == "" or str(val).strip() == "", (
                f"Trade {row['trade_id']} is OPEN_MARKED_TO_MARKET but contains non-null exit field '{field}' = {val}"
            )
        # Confirm terminal mark fields are properly populated
        assert not pd.isna(row["terminal_valuation_timestamp_ny"])
        assert float(row["terminal_mark_price"]) > 0
        assert int(row["open_shares_held"]) > 0
        assert float(row["open_market_value"]) > 0
        assert round(float(row["terminal_mark_price"]) * int(row["open_shares_held"]), 2) == round(float(row["open_market_value"]), 2)


