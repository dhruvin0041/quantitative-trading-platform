import unittest
from unittest.mock import patch

import pandas as pd

from src.data_ingestion.corporate_calendar import CorporateCalendarGate
from src.execution.risk_manager import calculate_dynamic_volatility_sizing


class TestCorporateCalendarGate(unittest.TestCase):
    def setUp(self):
        # Reset cache for isolated testing
        CorporateCalendarGate._earnings_cache.clear()
        CorporateCalendarGate._cache_timestamps.clear()

    def test_dynamic_volatility_sizing_calm_high_conviction(self):
        # VIX < 18, P(BUY) >= 0.60, Bull Market -> 100% allocation
        res = calculate_dynamic_volatility_sizing(
            vix=15.2,
            p_buy=0.68,
            is_dip_reversal=False,
            is_macro_bull=True,
        )
        self.assertEqual(res["target_allocation"], 1.00)
        self.assertEqual(res["allocation_pct"], 100)
        self.assertEqual(res["regime_label"], "CALM_HIGH_CONVICTION")
        self.assertIn("100% Full Sizing", res["sizing_rationale"])

    def test_dynamic_volatility_sizing_turbulent_dip_reversal(self):
        # VIX > 22 or Dip Reversal -> 50% to 70% allocation
        res = calculate_dynamic_volatility_sizing(
            vix=24.5,
            p_buy=0.48,
            is_dip_reversal=True,
            is_macro_bull=True,
        )
        self.assertEqual(res["target_allocation"], 0.70)
        self.assertEqual(res["allocation_pct"], 70)
        self.assertEqual(res["regime_label"], "TURBULENT_DIP_REVERSAL")
        self.assertIn("70% Risk-Managed Sizing", res["sizing_rationale"])

    def test_dynamic_volatility_sizing_moderate_volatility(self):
        # 18 <= VIX <= 22 -> 75% to 85%
        res = calculate_dynamic_volatility_sizing(
            vix=20.0,
            p_buy=0.65,
            is_dip_reversal=False,
            is_macro_bull=True,
        )
        self.assertEqual(res["target_allocation"], 0.85)
        self.assertEqual(res["allocation_pct"], 85)
        self.assertEqual(res["regime_label"], "MODERATE_VOLATILITY_STANDARD")

    def test_dynamic_volatility_sizing_high_turbulence(self):
        # VIX > 28 -> 50% defensive allocation floor
        res = calculate_dynamic_volatility_sizing(
            vix=32.0,
            p_buy=0.65,
            is_dip_reversal=False,
            is_macro_bull=True,
        )
        self.assertEqual(res["target_allocation"], 0.50)
        self.assertEqual(res["allocation_pct"], 50)
        self.assertEqual(res["regime_label"], "HIGH_TURBULENCE_DEFENSIVE")

    def test_earnings_proximity_outside_window(self):
        # Mock earnings dates for AAPL: 2026-11-02
        with patch.object(
            CorporateCalendarGate,
            "get_earnings_dates",
            return_value=[pd.Timestamp("2026-11-02")],
        ):
            # Test on 2026-10-15 (18 days before earnings)
            res = CorporateCalendarGate.check_earnings_proximity(
                ticker="AAPL",
                current_date="2026-10-15",
                blackout_days_before=3,
                blackout_days_after=1,
            )
            self.assertFalse(res["is_blackout"])
            self.assertFalse(res["earnings_shield_active"])
            self.assertEqual(res["action"], "NORMAL")
            self.assertEqual(res["days_to_earnings"], 18)

    def test_earnings_proximity_inside_blackout_window(self):
        with patch.object(
            CorporateCalendarGate,
            "get_earnings_dates",
            return_value=[pd.Timestamp("2026-11-02")],
        ):
            # Test on 2026-10-31 (2 days before earnings -> inside 3-day window)
            res = CorporateCalendarGate.check_earnings_proximity(
                ticker="AAPL",
                current_date="2026-10-31",
                blackout_days_before=3,
                blackout_days_after=1,
            )
            self.assertTrue(res["is_blackout"])
            self.assertTrue(res["earnings_shield_active"])
            self.assertEqual(res["action"], "PRE_EARNINGS_BLACKOUT")
            self.assertEqual(res["days_to_earnings"], 2)
            self.assertIn("PRE-EARNINGS SHIELD ACTIVE", res["message"])

    def test_earnings_proximity_earnings_day(self):
        with patch.object(
            CorporateCalendarGate,
            "get_earnings_dates",
            return_value=[pd.Timestamp("2026-11-02")],
        ):
            # On exact earnings date
            res = CorporateCalendarGate.check_earnings_proximity(
                ticker="AAPL",
                current_date="2026-11-02",
                blackout_days_before=3,
                blackout_days_after=1,
            )
            self.assertTrue(res["is_blackout"])
            self.assertEqual(res["days_to_earnings"], 0)
            self.assertEqual(res["action"], "PRE_EARNINGS_BLACKOUT")


if __name__ == "__main__":
    unittest.main()
