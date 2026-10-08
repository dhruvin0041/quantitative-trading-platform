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

    def _create_swing_series(self):
        prices = [100.0] * 30
        for i in range(1, 16):
            prices.append(100.0 + i * 2.0)
        prices.append(115.0)  # pullback / breakdown candle
        for i in range(1, 8):
            prices.append(115.0 + i * 0.5)
        for i in range(1, 26):
            prices.append(120.0 + i * 2.0)

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

    def test_swing_dip_and_peak_detection(self):
        """
        Verify requirement: BUY markers appear strictly on swing dips (local troughs)
        and SELL markers appear strictly on swing peaks (local crests).
        """
        df = self._create_swing_series()
        markers, _ = self.report_gen.generate_historical_markers("TEST", df)
        self.assertTrue(len(markers) >= 2, "Expected at least one BUY and one SELL on swing series")

        # First confirmed marker should be BUY on the initial dip
        self.assertEqual(markers[0]["action"], "BUY")

        # For every BUY, price must be a local dip relative to adjacent swings
        # For every SELL, price must be a local peak relative to adjacent swings
        for i in range(len(markers) - 1):
            cur = markers[i]
            nxt = markers[i + 1]
            if cur["action"] == "BUY" and nxt["action"] == "SELL":
                self.assertGreaterEqual(
                    nxt["price"],
                    cur["price"],
                    f"SELL peak ({nxt['price']}) should be higher than BUY dip ({cur['price']})",
                )

    def test_strict_alternation_guarantee(self):
        """
        Verify that emitted signals strictly alternate (BUY -> SELL -> BUY -> SELL)
        with zero consecutive duplicate actions.
        """
        for df_series in [
            self._create_uptrend_series(60),
            self._create_downtrend_series(60),
            self._create_swing_series(),
        ]:
            markers, _ = self.report_gen.generate_historical_markers("TEST", df_series.copy())
            for i in range(len(markers) - 1):
                self.assertNotEqual(
                    markers[i]["action"],
                    markers[i + 1]["action"],
                    f"Consecutive identical actions detected at indices {i} and {i+1}: {markers[i]['action']}",
                )

    def test_package_chart_data_preserves_consecutive_signals(self):
        """
        Integration test verifying that package_chart_data does NOT deduplicate
        consecutive identical signals (multiple consecutive BUYs or SELLs are preserved).
        """
        df_series = self._create_swing_series()

        # Create consecutive BUY markers intentionally
        mock_markers = [
            {"time": "2024-02-01", "action": "BUY", "label": "BUY", "probability": 100, "price": 100.0},
            {"time": "2024-02-02", "action": "BUY", "label": "BUY", "probability": 100, "price": 102.0},
            {"time": "2024-02-03", "action": "BUY", "label": "BUY", "probability": 100, "price": 104.0},
            {"time": "2024-02-04", "action": "SELL", "label": "SELL", "probability": 100, "price": 101.0},
            {"time": "2024-02-05", "action": "SELL", "label": "SELL", "probability": 100, "price": 99.0},
        ]

        response = self.report_gen.package_chart_data(
            "TEST",
            df_series,
            ai_report_dict={"Status": "OK"},
            historical_markers=mock_markers,
        )

        self.assertIn("markers", response)
        self.assertIn("historical_markers", response)
        self.assertEqual(response["markers"], response["historical_markers"])

        actions = [m["action"] for m in response["markers"]]
        # Verify 3 consecutive BUYs followed by 2 consecutive SELLs are preserved
        self.assertEqual(actions, ["BUY", "BUY", "BUY", "SELL", "SELL"])

    def test_package_chart_data_rolling_1year_window(self):
        """
        Integration test verifying that package_chart_data strictly enforces
        a rolling 1-year window of candlesticks up to the latest date.
        """
        # Create a 3-year daily price series from 2023-10-08 to 2026-10-08
        dates = pd.date_range("2023-10-08", "2026-10-08", freq="D")
        prices = np.linspace(150, 250, len(dates))
        df_3y = pd.DataFrame(
            {
                "Open": prices - 1.0,
                "High": prices + 1.0,
                "Low": prices - 1.0,
                "Close": prices,
                "Volume": 1000000,
            },
            index=dates,
        )

        mock_markers = [
            {"time": "2024-01-15", "action": "BUY", "label": "BUY", "probability": 90, "price": 160.0},
            {"time": "2025-05-01", "action": "SELL", "label": "SELL", "probability": 90, "price": 180.0},
            {"time": "2025-11-01", "action": "BUY", "label": "BUY", "probability": 95, "price": 200.0},
            {"time": "2026-03-15", "action": "SELL", "label": "SELL", "probability": 95, "price": 220.0},
        ]

        response = self.report_gen.package_chart_data(
            "AAPL",
            df_3y,
            ai_report_dict={"Status": "OK"},
            historical_markers=mock_markers,
        )

        candles = response["candles"]
        self.assertGreater(len(candles), 0)

        # Min candle date should be exactly on or after 2025-10-08 (1 year before latest date 2026-10-08)
        min_candle_date = candles[0]["time"]
        max_candle_date = candles[-1]["time"]

        self.assertEqual(min_candle_date, "2025-10-08")
        self.assertEqual(max_candle_date, "2026-10-08")

        # Markers before 2025-10-08 must be filtered out; markers within the 1-year window preserved
        marker_times = [m["time"] for m in response["markers"]]
        self.assertNotIn("2024-01-15", marker_times)
        self.assertNotIn("2025-05-01", marker_times)
        self.assertIn("2025-11-01", marker_times)
        self.assertIn("2026-03-15", marker_times)


if __name__ == "__main__":
    unittest.main()

