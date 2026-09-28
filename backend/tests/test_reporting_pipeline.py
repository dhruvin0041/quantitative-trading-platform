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

    def _create_oscillating_series(self, n=120):
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        t = np.arange(n)
        prices = 100.0 + 15.0 * np.sin(t / 6.0) + (t * 0.4)
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

    def _create_swing_series(self):
        prices = [100.0] * 30
        for i in range(1, 16):
            prices.append(100.0 + i * 2.0)
        prices.append(115.0)  # bar 45: stopout candle
        for i in range(1, 8):
            prices.append(115.0 + i * 0.5)  # cooldown
        for i in range(1, 26):
            prices.append(120.0 + i * 2.0)  # second swing rally

        prices = np.array(prices)
        n = len(prices)
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "Open": prices - 0.5,
                "High": prices + 1.5,
                "Low": prices - 1.0,
                "Close": prices,
                "Volume": 1000000,
            },
            index=dates,
        )
        df.iloc[28, df.columns.get_loc("Low")] = 92.0
        df.iloc[29, df.columns.get_loc("Open")] = 98.0
        df.iloc[29, df.columns.get_loc("Close")] = 100.0

        # Force stopout on bar 45
        df.iloc[45, df.columns.get_loc("Low")] = 100.0
        df.iloc[45, df.columns.get_loc("Close")] = 105.0

        # Second dip and reversal on bar 53-54
        df.iloc[53, df.columns.get_loc("Low")] = 105.0
        df.iloc[54, df.columns.get_loc("Open")] = 115.0
        df.iloc[54, df.columns.get_loc("Close")] = 125.0
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

        # Stop loss breach emits a SELL marker placed above bar (take profit/stop)
        self.assertNotIn("EXIT", actions)
        self.assertNotIn("STOP", actions)
        # And NO same-bar re-entry allowed
        self.assertNotIn(
            "BUY",
            actions,
            f"Bug detected: Same-bar re-entry occurred! Actions on {exit_date}: {actions}",
        )
        self.assertIn("SELL", actions)
        self.assertEqual(len(same_day_markers), 1)

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

    def test_strictly_alternating_signals(self):
        """
        Verify that across various market regimes (uptrend, downtrend, oscillating),
        historical_markers strictly alternates BUY -> SELL -> BUY -> SELL.
        Zero consecutive identical actions allowed and no orphan SELL before the first BUY.
        """
        for df_series in [
            self._create_uptrend_series(80),
            self._create_downtrend_series(80),
            self._create_swing_series(),
        ]:
            markers, _ = self.report_gen.generate_historical_markers("TEST", df_series.copy())
            actions = [m["action"] for m in markers]
            if not actions:
                continue

            # First marker must always be BUY (no orphan SELL)
            self.assertEqual(
                actions[0],
                "BUY",
                f"First marker must be BUY, got {actions[0]}",
            )
            # All markers must strictly alternate
            for i in range(1, len(actions)):
                self.assertNotEqual(
                    actions[i],
                    actions[i - 1],
                    f"Duplicate consecutive signal detected at index {i}: {actions[i-1]} followed by {actions[i]}",
                )
                if actions[i - 1] == "BUY":
                    self.assertEqual(actions[i], "SELL")
                else:
                    self.assertEqual(actions[i], "BUY")

    def test_trend_runner_not_cut_on_breakout(self):
        """
        Verify that during a strong trend expansion where price breaks out above
        the upper volatility envelope, the position is NOT cut prematurely on bar 1 or 2.
        The dynamic ATR trailing stop must ratchet upward under the runner.
        """
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        prices = [100.0] * 30
        for i in range(1, 31):
            prices.append(100.0 + i * 3.0)

        prices = np.array(prices)
        df = pd.DataFrame(
            {
                "Open": prices - 0.5,
                "High": prices + 1.5,
                "Low": prices - 1.0,
                "Close": prices,
                "Volume": 1000000,
            },
            index=dates,
        )
        # Bar 28 dips to touch fast MA value
        df.iloc[28, df.columns.get_loc("Low")] = 92.0
        df.iloc[29, df.columns.get_loc("Open")] = 98.0
        df.iloc[29, df.columns.get_loc("Close")] = 100.0
        df.iloc[29, df.columns.get_loc("High")] = 101.0

        markers, df_res = self.report_gen.generate_historical_markers("TEST", df)
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        buy_markers = [m for m in markers if m["action"] == "BUY"]
        self.assertTrue(len(buy_markers) >= 1, "Expected a BUY entry before the breakout run")

        buy_date = buy_markers[0]["time"]
        buy_bar = date_strs.index(buy_date)

        # Immediate breakout bars following the entry
        sell_dates = [m["time"] for m in markers if m["action"] == "SELL"]
        sell_bars = [date_strs.index(s) for s in sell_dates]

        self.assertNotIn(
            buy_bar + 1,
            sell_bars,
            f"Premature exit detected! Position was cut on bar 1 of breakout (bar {buy_bar + 1})",
        )
        self.assertNotIn(
            buy_bar + 2,
            sell_bars,
            f"Premature exit detected! Position was cut on bar 2 of breakout (bar {buy_bar + 2})",
        )

        # Verify that trailing stop ratchets upward under the runner
        stops = df_res["trailing_stop"].iloc[buy_bar : buy_bar + 8].values
        valid_stops = [s for s in stops if not np.isnan(s)]
        self.assertGreaterEqual(len(valid_stops), 4, "Expected active trailing stops during runner")
        for i in range(1, len(valid_stops)):
            self.assertGreaterEqual(
                valid_stops[i],
                valid_stops[i - 1],
                f"Trailing stop failed to ratchet upward at step {i}: {valid_stops[i]} < {valid_stops[i - 1]}",
            )

    def test_api_markers_strictly_alternate(self):
        """
        Integration test verifying that package_chart_data returns markers that
        strictly alternate without duplicate BUYs or SELLs, even if polluted
        system_signals or order journal records are supplied.
        """
        df_series = self._create_swing_series()
        raw_markers, df_res = self.report_gen.generate_historical_markers("TEST", df_series.copy())

        # Synthesize noisy system_signals DataFrame attempting to inject duplicate BUYs
        polluted_journal = pd.DataFrame(
            [
                {"timestamp": "2024-02-01", "signal_type": "BUY", "asset": "TEST", "confidence": 95},
                {"timestamp": "2024-02-02", "signal_type": "BUY", "asset": "TEST", "confidence": 92},
                {"timestamp": "2024-02-03", "signal_type": "BUY", "asset": "TEST", "confidence": 88},
            ]
        )

        response = self.report_gen.package_chart_data(
            "TEST",
            df_res,
            ai_report_dict={"Status": "OK"},
            historical_markers=raw_markers,
            system_signals=polluted_journal,
        )

        self.assertIn("markers", response)
        self.assertIn("historical_markers", response)
        self.assertEqual(response["markers"], response["historical_markers"])

        actions = [m["action"] for m in response["markers"]]
        self.assertTrue(len(actions) >= 2, "Expected at least 2 markers from oscillating series")

        # 1. First signal must be BUY
        self.assertEqual(actions[0], "BUY", f"First signal must be BUY, got: {actions[0]}")

        # 2. Assert that actions strictly alternates: actions[i] != actions[i-1] for all i > 0
        for i in range(1, len(actions)):
            self.assertNotEqual(
                actions[i],
                actions[i - 1],
                f"Consecutive duplicate actions detected at index {i}: {actions[i-1]} followed by {actions[i]}",
            )

        # 3. Assert that actions.count('BUY') and actions.count('SELL') differ by at most 1
        buy_count = actions.count("BUY")
        sell_count = actions.count("SELL")
        self.assertLessEqual(
            abs(buy_count - sell_count),
            1,
            f"Counts of BUY ({buy_count}) and SELL ({sell_count}) differ by more than 1",
        )

        # 4. Assert zero occurrences of consecutive ['BUY', 'BUY'] or ['SELL', 'SELL']
        for i in range(len(actions) - 1):
            pair = [actions[i], actions[i + 1]]
            self.assertNotEqual(
                pair,
                ["BUY", "BUY"],
                f"Found illegal consecutive BUY signals at index {i}: {pair}",
            )
            self.assertNotEqual(
                pair,
                ["SELL", "SELL"],
                f"Found illegal consecutive SELL signals at index {i}: {pair}",
            )


if __name__ == "__main__":
    unittest.main()
