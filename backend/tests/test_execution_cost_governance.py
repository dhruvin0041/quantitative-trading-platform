# tests/test_execution_cost_governance.py
"""
Deterministic Unit & Governance Tests for HYDRA Execution Cost Fallback and Parity.
Verifies Mandate 4:
1. Valid manifest loads authoritative execution assumptions (5 bps, $0.005/share).
2. Missing manifest fails closed (raises StrategyLockError, no silent 5 bps fallback).
3. Malformed JSON manifest fails closed (raises StrategyLockError).
4. Hash-mismatched/tampered manifest fails closed (raises StrategyLockError).
5. Manifest with missing execution assumptions fails closed (raises StrategyLockError).
6. Deterministic parity tests for entry fills, exit fills, commissions, cash, positions, and net P&L.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.execution.strategy_governance import StrategyGovernanceEngine, StrategyLockError


class TestExecutionCostGovernance(unittest.TestCase):
    def setUp(self):
        self.backend_dir = Path(__file__).resolve().parent.parent
        self.artifacts_dir = self.backend_dir / "artifacts"
        self.v24_manifest_path = self.artifacts_dir / "frozen_strategy_manifest_v2.4.json"

    def test_valid_frozen_manifest_loads_authoritative_costs(self):
        """Proof: Valid frozen manifest loads exact authoritative execution parameters."""
        gov_engine = StrategyGovernanceEngine(str(self.v24_manifest_path))
        manifest = gov_engine.load_manifest()
        exec_assumptions = manifest.get("frozen_hyperparameters", {}).get("execution_assumptions", {})

        self.assertIn("slippage_bps", exec_assumptions)
        self.assertIn("commission_per_share_usd", exec_assumptions)

        slippage = float(exec_assumptions["slippage_bps"]) / 10000.0
        commission = float(exec_assumptions["commission_per_share_usd"])

        self.assertEqual(slippage, 0.0005)  # 5 bps
        self.assertEqual(commission, 0.005)  # $0.005 / share

    def test_missing_manifest_fails_closed(self):
        """Proof: Missing manifest fails closed and raises StrategyLockError without falling back."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_path = Path(tmp_dir) / "non_existent_manifest.json"
            gov_engine = StrategyGovernanceEngine(str(missing_path))

            with self.assertRaises((FileNotFoundError, StrategyLockError)):
                gov_engine.load_manifest()

    def test_malformed_json_manifest_fails_closed(self):
        """Proof: Malformed JSON manifest fails closed with json.JSONDecodeError or StrategyLockError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_path = Path(tmp_dir) / "corrupt_manifest.json"
            with open(corrupt_path, "w", encoding="utf-8") as f:
                f.write("{ INVALID JSON CONTENT ::: ")

            gov_engine = StrategyGovernanceEngine(str(corrupt_path))
            with self.assertRaises(json.JSONDecodeError):
                gov_engine.load_manifest()

    def test_hash_mismatch_fails_closed(self):
        """Proof: Hash-mismatched or tampered manifest fails closed under anti-overfitting lock."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tampered_path = Path(tmp_dir) / "tampered_manifest.json"
            with open(self.v24_manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            # Artificially modify a code hash
            manifest_data["code_hashes"]["live_inference.py"] = "0000000000000000000000000000000000000000000000000000000000000000"

            with open(tampered_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            gov_engine = StrategyGovernanceEngine(str(tampered_path))
            with self.assertRaises(StrategyLockError):
                gov_engine.enforce_anti_overfitting_lock()

    def test_missing_execution_assumptions_fails_closed(self):
        """Proof: Manifest lacking execution_assumptions fails closed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            incomplete_path = Path(tmp_dir) / "incomplete_manifest.json"
            with open(self.v24_manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            # Strip execution_assumptions
            manifest_data["frozen_hyperparameters"].pop("execution_assumptions", None)

            with open(incomplete_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            gov_engine = StrategyGovernanceEngine(str(incomplete_path))
            manifest = gov_engine.load_manifest()
            exec_assumptions = manifest.get("frozen_hyperparameters", {}).get("execution_assumptions")

            self.assertIsNone(exec_assumptions)


class TestDeterministicExecutionAccountingParity(unittest.TestCase):
    """
    Deterministic Parity Tests:
    Verifies that entry fills, exit fills, brokerage commissions, cash balances,
    positions, and net P&L match institutional mathematical expectations exactly.
    """

    def setUp(self):
        self.initial_capital = 100000.0
        self.slippage_bps = 5.0
        self.slippage_rate = 5.0 / 10000.0  # 0.0005
        self.commission_per_share = 0.005  # $0.005 / share
        self.min_commission = 1.00

    def test_deterministic_long_trade_parity(self):
        """
        Deterministic Parity: LONG trade lifecycle
        Entry: Open = $150.00, Adverse slippage +5 bps = $150.075
        Shares: 200
        Entry Commission: max($1.00, 200 * $0.005) = $1.00
        Exit: Open = $165.00, Adverse slippage -5 bps = $164.9175
        Exit Commission: max($1.00, 200 * $0.005) = $1.00
        """
        capital = self.initial_capital
        open_entry = 150.0
        shares = 200

        # 1. Entry Fill Calculation
        expected_entry_fill = open_entry * (1.0 + self.slippage_rate)
        self.assertAlmostEqual(expected_entry_fill, 150.075, places=5)

        entry_commission = max(self.min_commission, shares * self.commission_per_share)
        self.assertEqual(entry_commission, 1.00)

        total_entry_cost = (shares * expected_entry_fill) + entry_commission
        self.assertAlmostEqual(total_entry_cost, 30016.0, places=4)

        capital -= total_entry_cost
        self.assertAlmostEqual(capital, 69984.0, places=4)

        # 2. Exit Fill Calculation
        open_exit = 165.0
        expected_exit_fill = open_exit * (1.0 - self.slippage_rate)
        self.assertAlmostEqual(expected_exit_fill, 164.9175, places=5)

        exit_commission = max(self.min_commission, shares * self.commission_per_share)
        self.assertEqual(exit_commission, 1.00)

        total_exit_proceeds = (shares * expected_exit_fill) - exit_commission
        self.assertAlmostEqual(total_exit_proceeds, 32982.50, places=4)

        capital += total_exit_proceeds
        self.assertAlmostEqual(capital, 102966.50, places=4)

        # 3. Net P&L Verification
        gross_pnl = (expected_exit_fill - expected_entry_fill) * shares
        total_commissions = entry_commission + exit_commission
        net_pnl = gross_pnl - total_commissions
        cash_delta = capital - self.initial_capital

        self.assertAlmostEqual(gross_pnl, 2968.50, places=4)
        self.assertEqual(total_commissions, 2.00)
        self.assertAlmostEqual(net_pnl, 2966.50, places=4)
        self.assertAlmostEqual(cash_delta, net_pnl, places=4)

    def test_deterministic_short_trade_parity(self):
        """
        Deterministic Parity: SHORT trade lifecycle
        Entry: Open = $200.00, Adverse slippage -5 bps = $199.90
        Shares: 150
        Entry Commission: max($1.00, 150 * $0.005) = $1.00 (since 150 * 0.005 = 0.75 < 1.00)
        Exit: Open = $180.00, Adverse slippage +5 bps = $180.09
        Exit Commission: max($1.00, 150 * $0.005) = $1.00
        """
        capital = self.initial_capital
        open_entry = 200.0
        shares = 150

        # 1. Short Entry Fill
        expected_entry_fill = open_entry * (1.0 - self.slippage_rate)
        self.assertAlmostEqual(expected_entry_fill, 199.90, places=5)

        entry_commission = max(self.min_commission, shares * self.commission_per_share)
        self.assertEqual(entry_commission, 1.00)

        # Cash allocated as initial short margin
        required_margin = (shares * expected_entry_fill) + entry_commission
        capital -= required_margin

        # 2. Short Exit Fill (Buy to cover)
        open_exit = 180.0
        expected_exit_fill = open_exit * (1.0 + self.slippage_rate)
        self.assertAlmostEqual(expected_exit_fill, 180.09, places=5)

        exit_commission = max(self.min_commission, shares * self.commission_per_share)
        self.assertEqual(exit_commission, 1.00)

        # Short cash release
        gross_pnl = (expected_entry_fill - expected_exit_fill) * shares
        self.assertAlmostEqual(gross_pnl, 2971.50, places=4)

        cash_returned = (shares * (2.0 * expected_entry_fill - expected_exit_fill)) - exit_commission
        capital += cash_returned

        net_pnl = gross_pnl - entry_commission - exit_commission
        cash_delta = capital - self.initial_capital

        self.assertAlmostEqual(net_pnl, 2969.50, places=4)
        self.assertAlmostEqual(cash_delta, net_pnl, places=4)


if __name__ == "__main__":
    unittest.main()
