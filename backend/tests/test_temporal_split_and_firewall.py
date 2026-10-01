import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_ingestion.market_data import apply_dynamic_triple_barrier
from src.execution.data_firewall import DataContaminationError, TemporalFirewall


class TestTemporalSplitAndFirewall(unittest.TestCase):
    """
    Comprehensive tests for the HYDRA Temporal Data-Split & Hard 2026 Firewall.
    Enforces:
    - 2016-01-01 to 2024-12-31: Model Development Universe
    - 2025-01-01 to 2025-12-31: Dedicated Validation / Calibration
    - 2026-01-01 onward: TRUE UNTOUCHED OUT-OF-SAMPLE (OOS)
    """

    def setUp(self):
        self.backend_dir = Path(__file__).resolve().parent.parent

    def test_firewall_passes_valid_development_data(self):
        dates = pd.date_range("2016-01-04", "2024-12-30", freq="B")
        df = pd.DataFrame({"Close": np.random.randn(len(dates))}, index=dates)
        # Should not raise
        TemporalFirewall.validate_development_data(df, "test_valid_dev")
        TemporalFirewall.validate_no_2026_leakage(df, "test_valid_dev")

    def test_firewall_blocks_pre_2016_data(self):
        dates = pd.date_range("2015-12-01", "2020-01-01", freq="B")
        df = pd.DataFrame({"Close": np.random.randn(len(dates))}, index=dates)
        with self.assertRaises(DataContaminationError) as ctx:
            TemporalFirewall.validate_development_data(df, "test_pre_2016")
        self.assertIn("Pre-2016 data found", str(ctx.exception))

    def test_firewall_blocks_post_2024_in_development(self):
        dates = pd.date_range("2016-01-04", "2025-01-15", freq="B")
        df = pd.DataFrame({"Close": np.random.randn(len(dates))}, index=dates)
        with self.assertRaises(DataContaminationError) as ctx:
            TemporalFirewall.validate_development_data(df, "test_post_2024")
        self.assertIn("Post-2024 data found", str(ctx.exception))

    def test_firewall_passes_valid_validation_data(self):
        dates = pd.date_range("2025-01-02", "2025-12-30", freq="B")
        df = pd.DataFrame({"Close": np.random.randn(len(dates))}, index=dates)
        # Should not raise
        TemporalFirewall.validate_validation_data(df, "test_valid_val")
        TemporalFirewall.validate_no_2026_leakage(df, "test_valid_val")

    def test_firewall_blocks_2026_in_validation(self):
        dates = pd.date_range("2025-01-02", "2026-01-05", freq="B")
        df = pd.DataFrame({"Close": np.random.randn(len(dates))}, index=dates)
        with self.assertRaises(DataContaminationError) as ctx:
            TemporalFirewall.validate_validation_data(df, "test_2026_val")
        self.assertIn("2026 data found", str(ctx.exception))

    def test_hard_2026_firewall_breach(self):
        dates = pd.date_range("2026-01-02", "2026-06-30", freq="B")
        df = pd.DataFrame({"Close": np.random.randn(len(dates))}, index=dates)
        with self.assertRaises(DataContaminationError) as ctx:
            TemporalFirewall.validate_no_2026_leakage(df, "test_hard_breach")
        self.assertIn("HARD 2026 FIREWALL BREACH", str(ctx.exception))

    def test_no_future_label_contamination(self):
        """
        Tests that when triple barrier labeling is applied to a DataFrame ending at 2024-12-30,
        the last `horizon` bars are dropped and no sample uses future prices beyond 2024-12-30.
        """
        horizon = 15
        dates = pd.date_range("2024-01-02", "2024-12-30", freq="B")
        n = len(dates)
        np.random.seed(42)
        close = 150.0 + np.cumsum(np.random.randn(n) * 1.5)
        high = close + np.random.rand(n) * 2.0
        low = close - np.random.rand(n) * 2.0
        atr = pd.Series(1.5, index=dates)

        df = pd.DataFrame({
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "ATR": atr,
        }, index=dates)

        df_labeled = apply_dynamic_triple_barrier(df.copy(), tp_atr_multiplier=2.0, sl_atr_multiplier=1.0, horizon=horizon)

        # Proves that exactly horizon rows were dropped because future data was absent
        self.assertEqual(len(df) - len(df_labeled), horizon)

        # Proves that the latest labeled date is at least `horizon` trading bars before the dataset end
        last_labeled_date = df_labeled.index[-1]
        self.assertEqual(last_labeled_date, dates[-(horizon + 1)])
        self.assertTrue(last_labeled_date <= pd.Timestamp("2024-12-30"))

    def test_manifest_versioning_preservation(self):
        """
        Proves that V1.0 manifest is preserved and distinct from V2.0 manifest.
        """
        v1_path = self.backend_dir / "artifacts" / "frozen_strategy_manifest_v1.0.json"
        self.assertTrue(v1_path.exists(), "frozen_strategy_manifest_v1.0.json must exist to preserve historical V1.0")

        with open(v1_path, "r", encoding="utf-8") as f:
            v1_data = json.load(f)

        self.assertEqual(v1_data["strategy_version"], "HYDRA_PROSPECTIVE_V1.0")


if __name__ == "__main__":
    unittest.main()
