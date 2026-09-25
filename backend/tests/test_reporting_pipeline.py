import unittest

import numpy as np
import pandas as pd

from src.execution.reporting import ReportGenerator


class TestReportingPipeline(unittest.TestCase):
    def setUp(self):
        self.cooldown = 5
        self.trail_mult = 2.0
        self.atr_period = 14
        self.k = 3
        self.report_gen = ReportGenerator(
            kept_features_list=[],
            k=self.k,
            cooldown_bars=self.cooldown,
            atr_period=self.atr_period,
            trail_mult=self.trail_mult,
        )

    def _create_uptrend_series(self, n=60):
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        p1 = np.ones(30) * 100.0
        p2 = np.linspace(101, 150, n - 30)
        prices = np.concatenate([p1, p2])
        df = pd.DataFrame(
            {
                "Open": prices - 0.5,
                "High": prices + 1.5,
                "Low": prices - 1.5,
                "Close": prices,
                "Volume": 1000000,
            },
            index=dates,
        )
        return df

    def _create_downtrend_series(self, n=60):
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        p1 = np.ones(30) * 150.0
        p2 = np.linspace(149, 100, n - 30)
        prices = np.concatenate([p1, p2])
        df = pd.DataFrame(
            {
                "Open": prices + 0.5,
                "High": prices + 1.5,
                "Low": prices - 1.5,
                "Close": prices,
                "Volume": 1000000,
            },
            index=dates,
        )
        return df

    def test_no_same_bar_reentry(self):
        """
        Verify that when a stop loss is breached on bar t, an EXIT marker is emitted
        and NO immediate BUY marker can be emitted on that exact same timestamp,
        even if bar t closes strongly and meets entry triggers.
        """
        df = self._create_uptrend_series(60)
        dates = df.index.strftime("%Y-%m-%d").tolist()

        # Baseline run to find where the first BUY occurs
        markers_pre, df_res_pre = self.report_gen.generate_historical_markers("TEST", df.copy())
        buy_markers = [m for m in markers_pre if m["action"] == "BUY"]
        self.assertTrue(len(buy_markers) > 0, "Uptrend should have triggered at least one BUY")

        first_buy_time = buy_markers[0]["time"]
        buy_idx = dates.index(first_buy_time)

        # Let position run, then on bar 34 simulate a flash crash and recovery
        exit_bar_idx = buy_idx + 4
        prev_stop = df_res_pre["trailing_stop"].iloc[exit_bar_idx - 1]
        self.assertFalse(np.isnan(prev_stop), "Trailing stop should be active prior to exit bar")

        # Set Low on exit_bar_idx to sharply penetrate below the stop loss
        df.iloc[exit_bar_idx, df.columns.get_loc("Low")] = prev_stop - 10.0
        # Set High and Close on exit_bar_idx to a new high (bullish close meeting trigger)
        df.iloc[exit_bar_idx, df.columns.get_loc("High")] = 160.0
        df.iloc[exit_bar_idx, df.columns.get_loc("Close")] = 158.0

        markers, _ = self.report_gen.generate_historical_markers("TEST", df)
        exit_date = dates[exit_bar_idx]

        same_day_markers = [m for m in markers if m["time"] == exit_date]
        actions = [m["action"] for m in same_day_markers]

        self.assertIn("EXIT", actions, f"Expected an EXIT marker on {exit_date}")
        self.assertNotIn(
            "BUY",
            actions,
            f"Bug detected: Same-bar re-entry occurred! Actions on {exit_date}: {actions}",
        )
        self.assertEqual(
            len(same_day_markers),
            1,
            f"Expected exactly 1 marker (EXIT) on {exit_date}, got {actions}",
        )

    def test_post_exit_cooldown_enforced(self):
        """
        Verify that for bars < cooldown_bars after an exit, subsequent entry triggers
        are suppressed by the post-exit refractory cooldown.
        """
        df = self._create_uptrend_series(60)
        dates = df.index.strftime("%Y-%m-%d").tolist()

        markers_pre, df_res_pre = self.report_gen.generate_historical_markers("TEST", df.copy())
        buy_markers = [m for m in markers_pre if m["action"] == "BUY"]
        buy_idx = dates.index(buy_markers[0]["time"])

        exit_bar_idx = buy_idx + 4
        prev_stop = df_res_pre["trailing_stop"].iloc[exit_bar_idx - 1]

        # Force stop out on exit_bar_idx
        df.iloc[exit_bar_idx, df.columns.get_loc("Low")] = prev_stop - 5.0

        markers, _ = self.report_gen.generate_historical_markers("TEST", df)

        # Check all bars within [exit_bar_idx, exit_bar_idx + cooldown - 1]
        cooldown_dates = [dates[i] for i in range(exit_bar_idx, exit_bar_idx + self.cooldown)]
        forbidden_buys = [m for m in markers if m["time"] in cooldown_dates and m["action"] == "BUY"]

        self.assertEqual(
            len(forbidden_buys),
            0,
            f"Cooldown violation: Found BUY signals during post-exit refractory period: {forbidden_buys}",
        )

    def test_short_trailing_stop_ratchets_down(self):
        """
        Verify that:
        1. Short entry generates an initial trailing stop above entry price.
        2. As price prints lower lows, the short trailing stop ratchets DOWN monotonically.
        3. When High >= current_short_stop, an EXIT marker is emitted and position is closed.
        """
        df = self._create_downtrend_series(60)
        dates = df.index.strftime("%Y-%m-%d").tolist()

        markers, df_res = self.report_gen.generate_historical_markers("TEST", df.copy())
        sell_markers = [m for m in markers if m["action"] == "SELL"]
        self.assertTrue(len(sell_markers) > 0, "Downtrend should have triggered a SELL (short) entry")

        sell_time = sell_markers[0]["time"]
        sell_idx = dates.index(sell_time)

        # Check stops after sell_idx
        short_stops = df_res["trailing_stop"].iloc[sell_idx : sell_idx + 5].values
        valid_stops = [s for s in short_stops if not np.isnan(s)]

        self.assertTrue(len(valid_stops) >= 2, "Expected multiple valid short trailing stops")
        # In a downtrend with declining lows, short stop must be monotonically decreasing or equal
        for i in range(1, len(valid_stops)):
            self.assertLessEqual(
                valid_stops[i],
                valid_stops[i - 1],
                f"Short trailing stop increased! Bar {i}: {valid_stops[i]} > {valid_stops[i-1]}",
            )

        # Now test exit on High spike
        test_exit_idx = sell_idx + 4
        active_stop = df_res["trailing_stop"].iloc[test_exit_idx - 1]

        df_exit = df.copy()
        # High spikes above active short stop
        df_exit.iloc[test_exit_idx, df_exit.columns.get_loc("High")] = active_stop + 10.0
        df_exit.iloc[test_exit_idx, df_exit.columns.get_loc("Close")] = active_stop + 5.0

        markers_exit, _ = self.report_gen.generate_historical_markers("TEST", df_exit)
        exit_date = dates[test_exit_idx]
        exit_markers = [m for m in markers_exit if m["time"] == exit_date and m["action"] == "EXIT"]

        self.assertEqual(
            len(exit_markers),
            1,
            f"Expected 1 short stop EXIT on {exit_date} when High spikes above trailing stop, got {exit_markers}",
        )


if __name__ == "__main__":
    unittest.main()
