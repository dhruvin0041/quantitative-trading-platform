import unittest

import numpy as np
import pandas as pd

from src.execution.reporting import ReportGenerator


class TestReportingPipeline(unittest.TestCase):
    def setUp(self):
        self.cooldown = 7
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

    def test_only_buy_and_sell_markers_emitted(self):
        """
        Chart Marker Cleanliness Mandate:
        Assert that all emitted markers strictly have action in ['BUY', 'SELL']
        and that zero EXIT / STOP markers exist.
        """
        for df_series in [self._create_uptrend_series(60), self._create_downtrend_series(60)]:
            markers, _ = self.report_gen.generate_historical_markers("TEST", df_series.copy())
            for m in markers:
                self.assertIn(
                    m["action"],
                    ["BUY", "SELL"],
                    f"Invalid marker action found: {m['action']}. Strictly BUY and SELL allowed.",
                )
            exit_markers = [m for m in markers if m["action"] in ["EXIT", "STOP", "SHORT_EXIT"]]
            self.assertEqual(
                len(exit_markers),
                0,
                f"Prohibited EXIT/STOP markers found on chart: {exit_markers}",
            )

    def test_cooldown_strictly_enforces_bar_count(self):
        """
        Synthesize an exit at bar t=50; assert no BUY can fire for t in [51, 56].
        The first allowable entry must be t >= 57 (pure integer bar index).
        """
        n = 70
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        p1 = np.ones(30) * 100.0
        p2 = 100.0 + np.arange(1, 41) * 2.0
        prices = np.concatenate([p1, p2])
        df = pd.DataFrame(
            {
                "Open": prices - 0.5,
                "High": prices + 1.0,
                "Low": prices - 1.0,
                "Close": prices,
                "Volume": 1000000,
            },
            index=dates,
        )

        # Baseline to obtain active trailing stop prior to bar 50
        _, df_res_pre = self.report_gen.generate_historical_markers("TEST", df.copy())
        active_stop_49 = df_res_pre["trailing_stop"].iloc[49]
        self.assertFalse(np.isnan(active_stop_49), "Trailing stop should be active at bar 49")

        # Synthesize stop breach at bar 50
        df_mod = df.copy()
        df_mod.iloc[50, df_mod.columns.get_loc("Low")] = active_stop_49 - 10.0
        df_mod.iloc[50, df_mod.columns.get_loc("Close")] = active_stop_49 - 5.0

        # For bars 51..69, provide valid pullback and bullish reversal candles
        for b in range(51, n):
            df_mod.iloc[b, df_mod.columns.get_loc("Low")] = 90.0  # touches value
            prev_h = df_mod.iloc[b - 1]["High"]
            df_mod.iloc[b, df_mod.columns.get_loc("Open")] = prev_h + 0.5
            df_mod.iloc[b, df_mod.columns.get_loc("High")] = prev_h + 3.0
            df_mod.iloc[b, df_mod.columns.get_loc("Close")] = prev_h + 2.0

        markers, _ = self.report_gen.generate_historical_markers("TEST", df_mod)
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]
        buy_indices = [date_strs.index(m["time"]) for m in markers if m["action"] == "BUY"]

        # Assert no BUY fired during lockout window [51, 56]
        forbidden_window = list(range(51, 57))
        for b in forbidden_window:
            self.assertNotIn(
                b,
                buy_indices,
                f"Cooldown leak detected! BUY marker fired at bar {b} within lockout [51, 56]",
            )

        # First allowable entry after exit at 50 must be t >= 57
        post_exit_buys = [b for b in buy_indices if b >= 50]
        self.assertTrue(len(post_exit_buys) > 0, "Expected a BUY entry after cooldown expired")
        self.assertGreaterEqual(
            post_exit_buys[0],
            57,
            f"First entry after exit at bar 50 was at bar {post_exit_buys[0]}, expected >= 57",
        )

    def test_buy_only_at_pullbacks(self):
        """
        Verify that candles expanding at the upper envelope with no pullback
        cannot trigger a BUY until a structural swing bottom pullback occurs.
        """
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        prices = 100.0 + np.arange(n) * 2.0
        # Low is close to Close, well above fast_ma and lower_band (pure upper envelope expansion)
        df_no_pb = pd.DataFrame(
            {
                "Open": prices - 0.5,
                "High": prices + 1.0,
                "Low": prices - 0.1,
                "Close": prices,
                "Volume": 1000000,
            },
            index=dates,
        )

        markers_no_pb, _ = self.report_gen.generate_historical_markers("TEST", df_no_pb.copy())
        buy_markers_no_pb = [m for m in markers_no_pb if m["action"] == "BUY"]
        self.assertEqual(
            len(buy_markers_no_pb),
            0,
            f"Expected zero BUY markers during upper envelope expansion without pullback, got: {buy_markers_no_pb}",
        )

        # Now introduce a pullback on bar 38 dipping into value, followed by bullish reversal on bar 39
        df_pb = df_no_pb.copy()
        df_pb.iloc[38, df_pb.columns.get_loc("Low")] = 85.0  # dip into value
        df_pb.iloc[39, df_pb.columns.get_loc("Open")] = 175.0
        df_pb.iloc[39, df_pb.columns.get_loc("High")] = 185.0
        df_pb.iloc[39, df_pb.columns.get_loc("Close")] = 182.0  # reversal

        markers_pb, _ = self.report_gen.generate_historical_markers("TEST", df_pb)
        buy_markers_pb = [m for m in markers_pb if m["action"] == "BUY"]
        self.assertTrue(
            len(buy_markers_pb) >= 1,
            "Expected a BUY marker once a swing bottom pullback occurred",
        )

    def test_no_same_bar_reentry(self):
        """
        Verify that when a stop loss is breached on bar t,
        NO immediate BUY marker can be emitted on that exact same timestamp,
        even if bar t closes strongly and meets entry triggers.
        """
        df = self._create_uptrend_series(60)
        dates = df.index.strftime("%Y-%m-%d").tolist()

        markers_pre, df_res_pre = self.report_gen.generate_historical_markers("TEST", df.copy())
        buy_markers = [m for m in markers_pre if m["action"] == "BUY"]
        self.assertTrue(len(buy_markers) > 0, "Uptrend should have triggered at least one BUY")

        first_buy_time = buy_markers[0]["time"]
        buy_idx = dates.index(first_buy_time)

        # Force stop out on exit_bar_idx
        exit_bar_idx = buy_idx + 4
        prev_stop = df_res_pre["trailing_stop"].iloc[exit_bar_idx - 1]
        self.assertFalse(np.isnan(prev_stop), "Trailing stop should be active prior to exit bar")

        df.iloc[exit_bar_idx, df.columns.get_loc("Low")] = prev_stop - 10.0
        df.iloc[exit_bar_idx, df.columns.get_loc("High")] = 160.0
        df.iloc[exit_bar_idx, df.columns.get_loc("Close")] = 158.0

        markers, df_res = self.report_gen.generate_historical_markers("TEST", df)
        exit_date = dates[exit_bar_idx]

        same_day_markers = [m for m in markers if m["time"] == exit_date]
        actions = [m["action"] for m in same_day_markers]

        # Stop loss breach must NOT emit an EXIT marker (cleanliness mandate)
        self.assertNotIn("EXIT", actions)
        self.assertNotIn("STOP", actions)
        # And NO same-bar re-entry allowed
        self.assertNotIn(
            "BUY",
            actions,
            f"Bug detected: Same-bar re-entry occurred! Actions on {exit_date}: {actions}",
        )
        self.assertEqual(len(same_day_markers), 0)

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
        3. When High >= current_short_stop, position is closed (trailing stop drops to nan)
           and zero EXIT markers are emitted.
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
        for i in range(1, len(valid_stops)):
            self.assertLessEqual(
                valid_stops[i],
                valid_stops[i - 1],
                f"Short trailing stop increased! Bar {i}: {valid_stops[i]} > {valid_stops[i-1]}",
            )

        # Test silent exit on High spike
        test_exit_idx = sell_idx + 4
        active_stop = df_res["trailing_stop"].iloc[test_exit_idx - 1]

        df_exit = df.copy()
        df_exit.iloc[test_exit_idx, df_exit.columns.get_loc("High")] = active_stop + 10.0
        df_exit.iloc[test_exit_idx, df_exit.columns.get_loc("Close")] = active_stop + 5.0

        markers_exit, _ = self.report_gen.generate_historical_markers("TEST", df_exit)
        exit_date = dates[test_exit_idx]
        exit_markers = [m for m in markers_exit if m["time"] == exit_date and m["action"] in ["EXIT", "STOP"]]

        self.assertEqual(
            len(exit_markers),
            0,
            f"Silent exit mandate violated: Emitted {exit_markers} on stopout",
        )


if __name__ == "__main__":
    unittest.main()
