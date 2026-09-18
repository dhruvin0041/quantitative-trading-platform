import os
import tempfile
import unittest

from execution.broker_interface import BaseBrokerAdapter, MockPaperBroker


class TestBrokerInterface(unittest.TestCase):
    def setUp(self):
        # Create temp dir for persistence tests
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_abstract_base_class(self):
        """Verify BaseBrokerAdapter cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseBrokerAdapter()

    def test_initial_balance_and_positions(self):
        """Verify initial account balance and empty positions list."""
        broker = MockPaperBroker(initial_capital=50000.0, state_file=None)
        bal = broker.get_account_balance()
        self.assertEqual(bal["cash"], 50000.0)
        self.assertEqual(bal["equity"], 50000.0)
        self.assertEqual(bal["buying_power"], 50000.0)
        self.assertEqual(bal["currency"], "USD")
        self.assertEqual(bal["initial_capital"], 50000.0)

        pos = broker.get_positions()
        self.assertEqual(pos, [])

    def test_buy_order_fill_with_slippage(self):
        """Verify BUY order executes with 10 bps upward slippage."""
        broker = MockPaperBroker(initial_capital=100000.0, slippage_bps=10.0, state_file=None)
        order = broker.submit_order(
            symbol="AAPL",
            qty=100,
            side="BUY",
            current_price=150.0,
            stop_loss=140.0,
            take_profit=170.0,
        )

        self.assertEqual(order["status"], "FILLED")
        self.assertEqual(order["symbol"], "AAPL")
        self.assertEqual(order["qty"], 100)
        self.assertEqual(order["side"], "BUY")

        # Expected fill: 150.0 * (1 + 0.0010) = 150.15
        self.assertAlmostEqual(order["fill_price"], 150.15, places=2)
        self.assertAlmostEqual(order["slippage_cost"], 15.0, places=2)

        # Cash reduced by 100 * 150.15 = 15015.0
        bal = broker.get_account_balance()
        self.assertAlmostEqual(bal["cash"], 100000.0 - 15015.0, places=2)

        # Position exists
        positions = broker.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "AAPL")
        self.assertEqual(positions[0]["qty"], 100)
        self.assertEqual(positions[0]["side"], "LONG")
        self.assertAlmostEqual(positions[0]["avg_entry_price"], 150.15, places=2)

    def test_sell_order_fill_with_slippage(self):
        """Verify SELL order executes with 10 bps downward slippage and realizes PnL."""
        broker = MockPaperBroker(initial_capital=100000.0, slippage_bps=10.0, state_file=None)
        # Open long 100 @ 150.0 (filled @ 150.15)
        broker.submit_order(symbol="AAPL", qty=100, side="BUY", current_price=150.0)

        # Sell 100 @ 160.0 (filled @ 160 * (1 - 0.0010) = 159.84)
        sell_order = broker.submit_order(symbol="AAPL", qty=100, side="SELL", current_price=160.0)

        self.assertEqual(sell_order["status"], "FILLED")
        self.assertAlmostEqual(sell_order["fill_price"], 159.84, places=2)

        # Position should be closed
        positions = broker.get_positions()
        self.assertEqual(len(positions), 0)

        # Net PnL = (159.84 - 150.15) * 100 = 969.0
        bal = broker.get_account_balance()
        expected_cash = 100000.0 - 15015.0 + 15984.0
        self.assertAlmostEqual(bal["cash"], expected_cash, places=2)
        self.assertAlmostEqual(bal["equity"], expected_cash, places=2)

    def test_insufficient_buying_power_rejection(self):
        """Verify order is REJECTED when notional exceeds cash."""
        broker = MockPaperBroker(initial_capital=500.0, state_file=None)
        # Trying to buy 100 shares @ $150 = $15,000 > $500
        order = broker.submit_order(symbol="AAPL", qty=100, side="BUY", current_price=150.0)
        self.assertEqual(order["status"], "REJECTED")
        self.assertIn("Insufficient buying power", order["reason"])

        # Cash unchanged
        bal = broker.get_account_balance()
        self.assertEqual(bal["cash"], 500.0)

    def test_input_validation(self):
        """Verify invalid orders throw ValueError."""
        broker = MockPaperBroker(initial_capital=10000.0, state_file=None)
        with self.assertRaises(ValueError):
            broker.submit_order("AAPL", qty=0, side="BUY", current_price=100.0)
        with self.assertRaises(ValueError):
            broker.submit_order("AAPL", qty=10, side="INVALID", current_price=100.0)
        with self.assertRaises(ValueError):
            broker.submit_order("AAPL", qty=10, side="BUY", current_price=-5.0)

    def test_cancel_order(self):
        """Verify cancel_order behavior."""
        broker = MockPaperBroker(initial_capital=10000.0, state_file=None)
        order = broker.submit_order("AAPL", qty=10, side="BUY", current_price=100.0)
        # Already filled orders cannot be cancelled
        self.assertFalse(broker.cancel_order(order["order_id"]))
        # Non-existent orders return False
        self.assertFalse(broker.cancel_order("NON_EXISTENT_ID"))

    def test_json_persistence(self):
        """Verify broker state persists and recovers from a JSON file."""
        state_path = os.path.join(self.test_dir.name, "broker_state.json")
        broker1 = MockPaperBroker(initial_capital=50000.0, state_file=state_path)
        broker1.submit_order("MSFT", qty=50, side="BUY", current_price=200.0)

        # Reopen with new instance
        broker2 = MockPaperBroker(state_file=state_path)
        self.assertAlmostEqual(broker2.cash, broker1.cash, places=2)
        self.assertEqual(len(broker2.get_positions()), 1)
        self.assertEqual(broker2.get_positions()[0]["symbol"], "MSFT")
        self.assertEqual(len(broker2.orders), 1)

    def test_sqlite_persistence(self):
        """Verify broker state persists and recovers from an SQLite database."""
        state_path = os.path.join(self.test_dir.name, "broker_state.db")
        broker1 = MockPaperBroker(initial_capital=75000.0, state_file=state_path)
        broker1.submit_order("NVDA", qty=20, side="BUY", current_price=300.0)

        # Reopen with new instance
        broker2 = MockPaperBroker(state_file=state_path)
        self.assertAlmostEqual(broker2.cash, broker1.cash, places=2)
        self.assertEqual(len(broker2.get_positions()), 1)
        self.assertEqual(broker2.get_positions()[0]["symbol"], "NVDA")
        self.assertEqual(len(broker2.orders), 1)


if __name__ == "__main__":
    unittest.main()
