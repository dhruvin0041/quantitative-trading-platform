"""
Unit and Integration Tests for Portfolio Status API Endpoints.
Verifies GET /api/v1/portfolio/status and /portfolio/status:
- API key authentication gating (403 on missing/invalid key).
- Schema integrity of account, positions, and trailing stops payloads.
- Fast read-only query execution against persistent SQLite storage.
- Custom state db_path query parameter handling.
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

# Ensure API_KEY is set prior to importing api/main
os.environ["API_KEY"] = "institutional-test-key-2026"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi.testclient import TestClient

from api import app


class TestPortfolioStatusAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.api_key = "institutional-test-key-2026"
        cls.headers = {"X-API-Key": cls.api_key}

    def setUp(self):
        # Create an isolated temporary test database for parameterized tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = str(Path(self.temp_dir.name) / "test_portfolio.db")
        self._init_test_database(self.test_db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _init_test_database(self, db_path: str):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE broker_account (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        cur.executemany(
            "INSERT INTO broker_account (key, value) VALUES (?, ?);",
            [
                ("cash", "35000.0"),
                ("initial_capital", "100000.0"),
                ("currency", "USD"),
            ],
        )
        cur.execute(
            """
            CREATE TABLE positions (
                symbol TEXT PRIMARY KEY,
                qty REAL NOT NULL,
                side TEXT NOT NULL,
                avg_entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL
            );
            """
        )
        cur.execute(
            """
            INSERT INTO positions (symbol, qty, side, avg_entry_price, current_price, stop_loss, take_profit)
            VALUES ('AAPL', 150.0, 'LONG', 200.0, 210.0, 190.0, 240.0);
            """
        )
        cur.execute(
            """
            CREATE TABLE trailing_stops (
                symbol TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                peak_trough_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                ts_mult REAL NOT NULL,
                trail_mult REAL,
                atr REAL NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            INSERT INTO trailing_stops (symbol, side, entry_price, peak_trough_price, stop_price, ts_mult, trail_mult, atr, updated_at)
            VALUES ('AAPL', 'LONG', 200.0, 215.0, 198.50, 3.20, 3.20, 5.15, '2026-09-25T16:00:00');
            """
        )
        conn.commit()
        conn.close()

    def test_auth_unauthorized_without_key(self):
        """Rejects request with 401/403 when X-API-Key is omitted."""
        resp = self.client.get("/api/v1/portfolio/status")
        self.assertIn(resp.status_code, [401, 403])

    def test_auth_unauthorized_with_invalid_key(self):
        """Rejects request with 403 when X-API-Key is invalid."""
        resp = self.client.get(
            "/api/v1/portfolio/status",
            headers={"X-API-Key": "invalid-hacker-key"},
        )
        self.assertEqual(resp.status_code, 403)

    def test_portfolio_status_v1_endpoint_success(self):
        """Validates /api/v1/portfolio/status response schema and fields."""
        resp = self.client.get("/api/v1/portfolio/status", headers=self.headers)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn("account", data)
        self.assertIn("positions", data)
        self.assertIn("trailing_stops", data)
        self.assertIn("timestamp", data)

        account = data["account"]
        expected_account_keys = [
            "cash",
            "equity",
            "buying_power",
            "currency",
            "initial_capital",
            "invested_capital",
            "allocation_pct",
            "cash_pct",
            "unrealized_pnl",
            "unrealized_pnl_pct",
            "total_pnl",
            "total_pnl_pct",
        ]
        for key in expected_account_keys:
            self.assertIn(key, account)
            self.assertIsInstance(account[key], (int, float, str))

        self.assertIsInstance(data["positions"], list)
        self.assertIsInstance(data["trailing_stops"], list)

    def test_portfolio_status_alias_endpoint_success(self):
        """Validates that /portfolio/status alias returns identical 200 payload."""
        resp = self.client.get("/portfolio/status", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("account", data)
        self.assertIn("positions", data)
        self.assertIn("trailing_stops", data)

    def test_portfolio_status_with_custom_db_path(self):
        """Validates inspecting an isolated custom database path via query parameter."""
        resp = self.client.get(
            f"/api/v1/portfolio/status?db_path={self.test_db_path}",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Account verification
        acc = data["account"]
        self.assertEqual(acc["cash"], 35000.0)
        self.assertEqual(acc["initial_capital"], 100000.0)
        self.assertEqual(acc["currency"], "USD")
        # 150 shares * $210 current price = $31,500 market value
        self.assertEqual(acc["invested_capital"], 31500.0)
        self.assertEqual(acc["equity"], 66500.0)
        # Unrealized: (210 - 200) * 150 = +1500.0
        self.assertEqual(acc["unrealized_pnl"], 1500.0)

        # Positions verification
        pos = data["positions"]
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0]["symbol"], "AAPL")
        self.assertEqual(pos[0]["qty"], 150.0)
        self.assertEqual(pos[0]["side"], "LONG")
        self.assertEqual(pos[0]["avg_entry_price"], 200.0)
        self.assertEqual(pos[0]["current_price"], 210.0)
        self.assertEqual(pos[0]["market_value"], 31500.0)
        self.assertEqual(pos[0]["unrealized_pnl"], 1500.0)
        self.assertEqual(pos[0]["stop_loss"], 190.0)
        self.assertEqual(pos[0]["take_profit"], 240.0)

        # Trailing Stops verification
        ts = data["trailing_stops"]
        self.assertEqual(len(ts), 1)
        self.assertEqual(ts[0]["symbol"], "AAPL")
        self.assertEqual(ts[0]["side"], "LONG")
        self.assertEqual(ts[0]["entry_price"], 200.0)
        self.assertEqual(ts[0]["peak_trough_price"], 215.0)
        self.assertEqual(ts[0]["stop_price"], 198.50)
        self.assertAlmostEqual(ts[0]["multiplier"], 3.20)
        self.assertAlmostEqual(ts[0]["atr"], 5.15)
        # Distance to stop: (198.50 - 210.0) / 210.0 * 100 = -5.476%
        expected_dist = ((198.50 - 210.0) / 210.0) * 100
        self.assertAlmostEqual(ts[0]["distance_to_stop_pct"], expected_dist, places=2)


if __name__ == "__main__":
    unittest.main()
