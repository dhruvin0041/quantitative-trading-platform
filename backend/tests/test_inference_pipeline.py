import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.execution.asset_intelligence import MODEL_REGISTRY, ModelRole
from src.execution.consensus_engine import WeightedConsensusEngine
from src.execution.inference_service import InferenceService
from src.models.model_loader import ModelManager


class TestInferencePipelineIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.consensus_engine = WeightedConsensusEngine()

        # Build mock model manager
        self.mock_mm = MagicMock(spec=ModelManager)
        self.mock_mm.accuracies = {
            "ensemble_accuracy": 55.0,
            "dl_accuracy": 50.0,
            "xgb_accuracy": 58.0,
            "lgbm_accuracy": 54.0,
            "dqn_accuracy": 52.0,
        }
        self.mock_mm.lstm_model = None  # Quarantined

        # Mock XGBoost model
        self.mock_xgb = MagicMock()
        self.mock_xgb.feature_importances_ = np.ones(27) / 27.0
        self.mock_mm.xgb_model = self.mock_xgb

        # Mock LightGBM model
        self.mock_lgbm = MagicMock()
        self.mock_mm.lgbm_model = self.mock_lgbm

        # Mock DQN agent
        self.mock_dqn = MagicMock()
        self.mock_mm.dqn_agent = self.mock_dqn

        # Mock TFT model
        self.mock_tft = MagicMock()
        self.mock_tft.predict.return_value = np.zeros((1, 10, 3))
        self.mock_mm.tft_model = self.mock_tft

        # Build other mock dependencies for InferenceService
        self.mock_gemini = MagicMock()
        self.mock_gemini.analyze_fundamental_alpha.return_value = (0.5, "Strong fundamentals")
        self.mock_physical = MagicMock()
        self.mock_graph = MagicMock()

        self.mock_orchestrator = MagicMock()
        self.mock_orchestrator.run_consensus.return_value = {
            "final_action_idx": 2,
            "consensus_status": "APPROVED",
            "veto_reason": None,
            "veto_code": None,
            "decision_tree": [],
        }

        self.mock_router = MagicMock()
        self.mock_report_gen = MagicMock()
        self.mock_report_gen.generate_historical_markers.return_value = ([], pd.DataFrame())
        self.mock_report_gen.package_chart_data.return_value = {
            "candles": [],
            "clouds": [],
            "ai_report": {},
        }

        self.mock_paper = MagicMock()
        self.mock_paper.history = []
        self.mock_paper.portfolio_snapshots = []
        self.mock_paper.initial_capital = 100000.0
        self.mock_paper.get_portfolio_summary.return_value = {
            "peak_equity": 100000.0,
            "trough_equity": 100000.0,
        }
        self.mock_paper.execute_trade.return_value = None

        self.mock_perf = MagicMock()
        self.mock_journal = MagicMock()
        self.mock_journal.get_all_signals.return_value = None

        # Instantiate InferenceService with mocked dependencies
        self.service = InferenceService(
            self.mock_mm,
            self.mock_gemini,
            self.mock_physical,
            self.mock_graph,
            self.mock_orchestrator,
            self.mock_router,
            self.mock_report_gen,
            self.mock_paper,
            self.mock_perf,
            self.mock_journal,
        )
        # Disable external isotonic calibrator for deterministic test assertions
        self.service.model_calibrator = None

    def _generate_mock_market_data(self, curr_close=160.0, sma_200=140.0, spy_close=450.0, spy_sma_50=420.0):
        """Helper to generate realistic market DataFrames."""
        # Underlying: 200 rows of sma_200, then latest row at curr_close
        close_vals = [sma_200] * 200 + [curr_close]
        ticker_df = pd.DataFrame(
            {
                "Close": close_vals,
                "Open": close_vals,
                "High": [c * 1.01 for c in close_vals],
                "Low": [c * 0.99 for c in close_vals],
                "Volume": [1000000] * len(close_vals),
            }
        )

        spy_vals = [spy_sma_50] * 50 + [spy_close]
        spy_df = pd.DataFrame(
            {
                "Close": spy_vals,
                "Open": spy_vals,
                "High": [c * 1.01 for c in spy_vals],
                "Low": [c * 0.99 for c in spy_vals],
                "Volume": [5000000] * len(spy_vals),
            }
        )

        ts_seq = np.zeros((1, 60, 27))
        peer_seq = None
        tabular_row = np.zeros((1, 27))
        current_price = curr_close
        updated_config = {"data": {"max_seq_length": 128, "time_steps": 60}}
        market_regime = "BULL"
        req_conf = 60.0
        vol_ratio = 1.0
        tech_snapshot = {"ATR": 2.5, "RSI": 55.0, "ADX": 25.0}

        return (
            ts_seq,
            peer_seq,
            tabular_row,
            current_price,
            updated_config,
            market_regime,
            req_conf,
            vol_ratio,
            tech_snapshot,
            ticker_df,
            spy_df,
        )

    def test_asymmetric_veto_blocks_xgb_sell_when_secondary_buys(self):
        """
        Verify that when XGB_AGENT issues a SELL (P >= 0.60),
        but a secondary model (LGBM_AGENT or DQN_AGENT) signals BUY with P >= 0.65,
        the Bidirectional Asymmetric Veto triggers is_vetoed=True and forces dominant_direction="HOLD".
        """
        base_probs = {
            "XGB_AGENT": np.array([0.75, 0.15, 0.10]),   # SELL (idx 0)
            "LGBM_AGENT": np.array([0.10, 0.15, 0.75]),  # Strong BUY (idx 2 >= 0.65)
            "DQN_AGENT": np.array([0.20, 0.40, 0.40]),
        }
        res = self.consensus_engine.compute_asymmetric_veto(
            base_probs,
            primary_key="XGB_AGENT",
            primary_threshold=0.60,
            veto_threshold=0.65,
            veto_short=True,
        )
        self.assertTrue(res["is_vetoed"])
        self.assertEqual(res["dominant_direction"], "HOLD")
        self.assertEqual(res["dominant_idx"], 1)
        self.assertEqual(res["vetoed_by"], "LGBM_AGENT")
        self.assertIn("Vetoed by LGBM_AGENT", res["veto_reason"])
        self.assertIn("Counter-trend bullish conviction", res["veto_reason"])

    def test_asymmetric_veto_blocks_xgb_buy_when_secondary_sells(self):
        """
        Verify that when XGB_AGENT issues a BUY (P >= 0.60),
        but a secondary model (DQN_AGENT) signals SELL with P >= 0.65,
        the veto triggers is_vetoed=True and forces dominant_direction="HOLD".
        """
        base_probs = {
            "XGB_AGENT": np.array([0.10, 0.15, 0.75]),   # BUY (idx 2)
            "LGBM_AGENT": np.array([0.20, 0.40, 0.40]),
            "DQN_AGENT": np.array([0.70, 0.15, 0.15]),   # Strong SELL (idx 0 >= 0.65)
        }
        res = self.consensus_engine.compute_asymmetric_veto(
            base_probs,
            primary_key="XGB_AGENT",
            primary_threshold=0.60,
            veto_threshold=0.65,
            veto_short=True,
        )
        self.assertTrue(res["is_vetoed"])
        self.assertEqual(res["dominant_direction"], "HOLD")
        self.assertEqual(res["dominant_idx"], 1)
        self.assertEqual(res["vetoed_by"], "DQN_AGENT")
        self.assertIn("Vetoed by DQN_AGENT", res["veto_reason"])
        self.assertIn("Bearish conviction", res["veto_reason"])

    @patch("src.execution.inference_service.fetch_live_data")
    @patch("src.execution.inference_service.fetch_live_news")
    async def test_live_inference_veto_suppression(self, mock_news, mock_data):
        """
        Integration test verifying that a live inference call suppresses a SELL signal
        to HOLD when a secondary model issues a high-conviction BUY veto.
        """
        mock_data.return_value = self._generate_mock_market_data()
        mock_news.return_value = (None, None, "Market update")

        # Configure XGB to SELL, LightGBM to strong BUY
        self.mock_xgb.predict_proba.return_value = np.array([[0.75, 0.15, 0.10]])
        self.mock_lgbm.predict_proba.return_value = np.array([[0.10, 0.15, 0.75]])
        self.mock_dqn.predict_proba.return_value = np.array([0.20, 0.40, 0.40])

        config = {"data": {"max_seq_length": 128, "time_steps": 60}}
        metadata = {"ticker": "AAPL", "currency": "USD", "market": "US"}

        response = await self.service.get_prediction("AAPL", config, metadata)

        # The final signal must be HOLD due to the secondary veto
        self.assertEqual(response["signal"], "HOLD")
        self.assertIn("Vetoed", response["signal_note"])
        self.assertEqual(response["models"]["XGB_AGENT"]["signal"], "SELL")
        self.assertEqual(response["models"]["LGBM_AGENT"]["signal"], "BUY")

    @patch("src.execution.inference_service.fetch_live_data")
    @patch("src.execution.inference_service.fetch_live_news")
    async def test_macro_regime_filter_suppresses_buy_when_underlying_below_sma200(
        self, mock_news, mock_data
    ):
        """
        Verify that when Close < SMA200 on the underlying ticker,
        any BUY signal is suppressed to HOLD with a clear explanatory note.
        """
        # Current Close: 120, SMA200: 150 -> Below SMA200
        mock_data.return_value = self._generate_mock_market_data(
            curr_close=120.0, sma_200=150.0, spy_close=450.0, spy_sma_50=420.0
        )
        mock_news.return_value = (None, None, "Market update")

        # Models agree on BUY
        self.mock_xgb.predict_proba.return_value = np.array([[0.05, 0.15, 0.80]])
        self.mock_lgbm.predict_proba.return_value = np.array([[0.05, 0.15, 0.80]])
        self.mock_dqn.predict_proba.return_value = np.array([0.05, 0.15, 0.80])

        config = {"data": {"max_seq_length": 128, "time_steps": 60}}
        metadata = {"ticker": "AAPL", "currency": "USD", "market": "US"}

        response = await self.service.get_prediction("AAPL", config, metadata)

        self.assertIn("macro_regime_filter", response)
        self.assertFalse(response["macro_regime_filter"]["long_allowed"])
        self.assertEqual(response["macro_regime_filter"]["underlying_close"], 120.0)
        self.assertEqual(response["macro_regime_filter"]["underlying_sma_200"], 149.85)
        self.assertLess(
            response["macro_regime_filter"]["underlying_close"],
            response["macro_regime_filter"]["underlying_sma_200"],
        )
        self.assertEqual(response["signal"], "HOLD")
        self.assertIn("Suppressed by Macro Regime Filter", response["signal_note"])
        self.assertIn("SMA200", response["signal_note"])

    @patch("src.execution.inference_service.fetch_live_data")
    @patch("src.execution.inference_service.fetch_live_news")
    async def test_macro_regime_filter_suppresses_buy_when_spy_below_sma50(
        self, mock_news, mock_data
    ):
        """
        Verify that when SPY Close < SPY SMA50,
        any BUY signal is suppressed to HOLD even if the underlying is above SMA200.
        """
        # Underlying healthy (160 > 140), but SPY unhealthy (390 < 420)
        mock_data.return_value = self._generate_mock_market_data(
            curr_close=160.0, sma_200=140.0, spy_close=390.0, spy_sma_50=420.0
        )
        mock_news.return_value = (None, None, "Market update")

        # Models agree on BUY
        self.mock_xgb.predict_proba.return_value = np.array([[0.05, 0.15, 0.80]])
        self.mock_lgbm.predict_proba.return_value = np.array([[0.05, 0.15, 0.80]])
        self.mock_dqn.predict_proba.return_value = np.array([0.05, 0.15, 0.80])

        config = {"data": {"max_seq_length": 128, "time_steps": 60}}
        metadata = {"ticker": "AAPL", "currency": "USD", "market": "US"}

        response = await self.service.get_prediction("AAPL", config, metadata)

        self.assertIn("macro_regime_filter", response)
        self.assertFalse(response["macro_regime_filter"]["long_allowed"])
        self.assertEqual(response["macro_regime_filter"]["spy_close"], 390.0)
        self.assertEqual(response["macro_regime_filter"]["spy_sma_50"], 419.4)
        self.assertLess(
            response["macro_regime_filter"]["spy_close"],
            response["macro_regime_filter"]["spy_sma_50"],
        )
        self.assertEqual(response["signal"], "HOLD")
        self.assertIn("Suppressed by Macro Regime Filter", response["signal_note"])
        self.assertIn("SPY", response["signal_note"])

    @patch("src.execution.inference_service.fetch_live_data")
    @patch("src.execution.inference_service.fetch_live_news")
    async def test_macro_regime_filter_allows_buy_when_both_regimes_healthy(
        self, mock_news, mock_data
    ):
        """
        Verify that when Close >= SMA200 and SPY Close >= SPY SMA50,
        long_allowed is True in macro_regime_filter.
        """
        mock_data.return_value = self._generate_mock_market_data(
            curr_close=160.0, sma_200=140.0, spy_close=450.0, spy_sma_50=420.0
        )
        mock_news.return_value = (None, None, "Market update")

        self.mock_xgb.predict_proba.return_value = np.array([[0.05, 0.15, 0.80]])
        self.mock_lgbm.predict_proba.return_value = np.array([[0.05, 0.15, 0.80]])
        self.mock_dqn.predict_proba.return_value = np.array([0.05, 0.15, 0.80])

        config = {"data": {"max_seq_length": 128, "time_steps": 60}}
        metadata = {"ticker": "AAPL", "currency": "USD", "market": "US"}

        response = await self.service.get_prediction("AAPL", config, metadata)

        self.assertIn("macro_regime_filter", response)
        self.assertTrue(response["macro_regime_filter"]["long_allowed"])
        self.assertGreaterEqual(
            response["macro_regime_filter"]["underlying_close"],
            response["macro_regime_filter"]["underlying_sma_200"],
        )
        self.assertGreaterEqual(
            response["macro_regime_filter"]["spy_close"],
            response["macro_regime_filter"]["spy_sma_50"],
        )

    @patch("src.execution.inference_service.fetch_live_data")
    @patch("src.execution.inference_service.fetch_live_news")
    async def test_dl_fusion_quarantined_skips_neural_inference(
        self, mock_news, mock_data
    ):
        """
        Verify that when DL_FUSION has QUARANTINED role,
        inference service completely bypasses mm.lstm_model.predict and assigns neutral output.
        """
        mock_data.return_value = self._generate_mock_market_data()
        mock_news.return_value = (None, None, "Market update")

        # Provide a dummy lstm_model to verify .predict is NEVER invoked
        dummy_lstm = MagicMock()
        self.mock_mm.lstm_model = dummy_lstm

        self.mock_xgb.predict_proba.return_value = np.array([[0.10, 0.20, 0.70]])
        self.mock_lgbm.predict_proba.return_value = np.array([[0.10, 0.20, 0.70]])
        self.mock_dqn.predict_proba.return_value = np.array([0.10, 0.20, 0.70])

        config = {"data": {"max_seq_length": 128, "time_steps": 60}}
        metadata = {"ticker": "AAPL", "currency": "USD", "market": "US"}

        # Ensure MODEL_REGISTRY has DL_FUSION as QUARANTINED
        self.assertEqual(MODEL_REGISTRY["DL_FUSION"]["role"], ModelRole.QUARANTINED)

        response = await self.service.get_prediction("AAPL", config, metadata)

        # Neural predict must not have been called
        dummy_lstm.predict.assert_not_called()

        # In models breakdown, DL_FUSION must be reported as QUARANTINED
        self.assertEqual(response["models"]["DL_FUSION"]["role"], "QUARANTINED")
        self.assertEqual(response["models"]["DL_FUSION"]["status"], "QUARANTINED")
        self.assertEqual(response["models"]["DL_FUSION"]["signal"], "HOLD")


if __name__ == "__main__":
    unittest.main()
