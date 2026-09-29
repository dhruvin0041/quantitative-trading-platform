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

    def test_debounce_anti_spam_filter(self):
        """
        Verify that multiple consecutive BUY/SELL signals are debounced to a minimum 3-bar gap
        and require edge-triggered reversal conditions.
        """
        df = self._create_uptrend_series(60)
        # Induce consecutive pullback bounce candles (Low <= fast_ma, Close > Open, Close > High[t-1])
        # On bars 35, 36, 37, 38
        for b in [35, 36, 37, 38]:
            df.iloc[b-1, df.columns.get_loc("High")] = 100.0 # ensure High[t-1] is low enough
            df.iloc[b, df.columns.get_loc("Low")] = 85.0  # touches/crosses below fast_ma
            df.iloc[b, df.columns.get_loc("Open")] = 110.0
            df.iloc[b, df.columns.get_loc("High")] = 125.0
            df.iloc[b, df.columns.get_loc("Close")] = 120.0  # green candle and Close > High[t-1]

        markers, _ = self.report_gen.generate_historical_markers("TEST", df)
        date_strs = [d.strftime("%Y-%m-%d") for d in df.index]
        marker_dates = {m["time"]: m["action"] for m in markers}

        # Verify that ONLY bar 35 and 38 fired a BUY marker due to the 3-bar debounce rule
        self.assertIn(date_strs[35], marker_dates, f"Expected marker on bar 35 ({date_strs[35]})")
        self.assertNotIn(date_strs[36], marker_dates, f"Expected NO marker on bar 36 ({date_strs[36]})")
        self.assertNotIn(date_strs[37], marker_dates, f"Expected NO marker on bar 37 ({date_strs[37]})")
        self.assertIn(date_strs[38], marker_dates, f"Expected marker on bar 38 ({date_strs[38]})")

        self.assertEqual(marker_dates[date_strs[35]], "BUY")
        self.assertEqual(marker_dates[date_strs[38]], "BUY")

    def test_edge_triggered_sell_conditions(self):
        """
        Verify SELL trigger: (High >= upper_band OR High > fast_ma * 1.02) AND Close < Open AND Close < Low[t-1].
        """
        df = self._create_uptrend_series(60)
        # Bar 40: red candle with breakdown, overextended high, taking out previous low
        df.iloc[39, df.columns.get_loc("Low")] = 110.0
        df.iloc[40, df.columns.get_loc("Open")] = 120.0
        df.iloc[40, df.columns.get_loc("High")] = 200.0 # overextended
        df.iloc[40, df.columns.get_loc("Low")] = 80.0
        df.iloc[40, df.columns.get_loc("Close")] = 85.0  # Close < Open AND Close < Low[t-1]

        markers, _ = self.report_gen.generate_historical_markers("TEST", df)
        date_strs = [d.strftime("%Y-%m-%d") for d in df.index]
        d_40 = date_strs[40]

        matching = [m for m in markers if m["time"] == d_40]
        self.assertTrue(len(matching) >= 1, f"Expected SELL marker on breakdown candle at {d_40}")
        self.assertEqual(matching[0]["action"], "SELL")

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


if __name__ == "__main__":
    unittest.main()
