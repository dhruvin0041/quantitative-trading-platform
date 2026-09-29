import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.data_ingestion.nlp_processor import NewsTokenizer
from src.data_ingestion.technical_indicators import clean_multiindex_columns
from src.execution.asset_intelligence import (
    AdaptiveWeightingEngine,
    AssetProfileEngine,
    MultiTimeframeEngine,
)
from src.execution.confidence_engine import ConfidenceBreakdownEngine
from src.execution.consensus_engine import WeightedConsensusEngine
from src.execution.execution_authority import ExecutionAuthorityEngine
from src.execution.forecast_engine import ForecastCalibrationEngine
from src.execution.governance_engine import SignalGovernanceAnalytics
from src.execution.live_inference import (
    FEATURE_COLUMNS,
    add_upgraded_features,
    compute_shap_explanation,
    fetch_live_data,
    fetch_live_news,
    get_meta_prediction,
)
from src.execution.risk_manager import (
    InstitutionalRiskArbitrator,
    calculate_beta,
    detect_stampede_risk,
    get_position_sizing,
)
from src.execution.signal_intelligence import (
    ConfidenceCalibrationEngine,
    ExpectedValueEngine,
    RegimeEngineV2,
    SignalQualityEngine,
)
from src.execution.signal_ledger import SignalLedger
from src.execution.timing_engine import PredictiveTimingEngine
from src.execution.trade_engine import TradeConstructionEngine
from src.models.regime.calibration import ModelCalibrator
from src.schemas import (
    ExpectedValueMetrics,
    SignalQuality,
)

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        model_manager,
        gemini_analyzer,
        physical_edge,
        dependency_graph,
        orchestrator,
        smart_router,
        report_gen,
        paper_engine,
        perf_analyzer,
        signal_journal=None,
        signal_ledger=None,
        use_veto: bool = False,
    ):
        self.use_veto = use_veto
        self.mm = model_manager
        self.gemini = gemini_analyzer
        self.physical = physical_edge
        self.graph = dependency_graph
        self.orchestrator = orchestrator
        self.router = smart_router
        self.report_gen = report_gen
        self.signal_ledger = signal_ledger if signal_ledger is not None else SignalLedger()
        self.consensus_engine = WeightedConsensusEngine()
        self.forecast_engine = ForecastCalibrationEngine()
        self.trade_engine = TradeConstructionEngine()
        self.timing_engine = PredictiveTimingEngine()
        self.confidence_engine = ConfidenceBreakdownEngine()
        self.execution_authority = ExecutionAuthorityEngine()
        self.governance_analytics = SignalGovernanceAnalytics()
        self.risk_arbitrator = InstitutionalRiskArbitrator()
        self.paper_engine = paper_engine
        self.perf_analyzer = perf_analyzer
        self.journal = signal_journal

        # Isotonic probability calibrator (fitted during training)
        try:
            self.model_calibrator = ModelCalibrator.load("artifacts/model_calibrator.joblib")
            logger.info("Loaded isotonic model calibrator")
        except Exception as e:
            logger.warning("No isotonic calibrator found (%s). Raw probs will be used.", e)
            self.model_calibrator = None

        # V2.0 Engines
        self.regime_v2 = RegimeEngineV2()
        self.calibration_engine = ConfidenceCalibrationEngine()
        self.ev_engine = ExpectedValueEngine()
        self.quality_engine = SignalQualityEngine()
        self.asset_engine = AssetProfileEngine()
        self.weight_engine = AdaptiveWeightingEngine()
        self.mtf_engine = MultiTimeframeEngine()

    def _replay_causal_ml_signals(
        self,
        ticker: str,
        ticker_df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """
        Point-in-Time Sequential Causal Replay.
        Evaluates the XGBoost consensus model sequentially over historical bars.
        At each bar t, features use strictly trailing data <= t.
        Fills are modeled at Open[t+1] +/- 5bps slippage (never High/Low).
        Confirmed signals are committed to the append-only SignalLedger.
        """
        try:
            df = clean_multiindex_columns(ticker_df.copy())
            if df.empty or len(df) < 30:
                return []

            spy = clean_multiindex_columns(spy_df.copy()) if spy_df is not None else None
            df_feat = add_upgraded_features(df.copy(), spy, None)
            df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()].copy()
            df_filtered = df_feat.reindex(columns=FEATURE_COLUMNS).dropna()

            if df_filtered.empty or len(df_filtered) < 15:
                return []

            # Pre-fitted production scaler
            scaler = None
            for s_path in [
                "artifacts/latest_scaler.joblib",
                "backend/artifacts/latest_scaler.joblib",
            ]:
                p = Path(s_path)
                if p.exists():
                    try:
                        scaler = joblib.load(p)
                        break
                    except Exception:
                        pass
            if scaler is None:
                scaler = getattr(self.mm, "scaler", None)

            if scaler is None:
                logger.warning("No scaler available for causal ML replay.")
                return []

            scaled_rows = scaler.transform(df_filtered.values)

            # Pre-trained XGBoost Alpha Model
            model = getattr(self.mm, "xgb_model", None)
            if model is None:
                for m_path in [
                    "artifacts/xgb_ensemble.json",
                    "backend/artifacts/xgb_ensemble.json",
                ]:
                    mp = Path(m_path)
                    if mp.exists():
                        try:
                            import xgboost as xgb

                            model = xgb.XGBClassifier()
                            model.load_model(str(mp))
                            break
                        except Exception:
                            pass

            if model is None:
                logger.warning("No XGBoost model available for causal ML replay.")
                return []

            all_probs = model.predict_proba(scaled_rows)
            valid_indices = df_filtered.index
            if len(all_probs) == 1 and len(valid_indices) > 1:
                all_probs = np.tile(all_probs, (len(valid_indices), 1))

            # SMA200 and SPY SMA50 for causal macro trend filter
            close_s = df["Close"].reindex(valid_indices).ffill()
            sma200_s = close_s.rolling(window=200, min_periods=20).mean()
            if spy is not None and "Close" in spy.columns:
                spy_close_s = spy["Close"].reindex(valid_indices).ffill()
                spy_sma50_s = spy_close_s.rolling(window=50, min_periods=10).mean()
            else:
                spy_close_s = close_s
                spy_sma50_s = close_s

            current_pos = "FLAT"
            last_trade_idx = -10
            min_cooldown_bars = 5

            for i in range(len(valid_indices)):
                v_idx = valid_indices[i]
                bar_time = (
                    v_idx.strftime("%Y-%m-%d")
                    if hasattr(v_idx, "strftime")
                    else str(v_idx)
                )
                prob_vec = all_probs[i]
                if self.model_calibrator is not None:
                    prob_vec = self.model_calibrator.calibrate("XGB", prob_vec)

                p_sell, p_hold, p_buy = (
                    float(prob_vec[0]),
                    float(prob_vec[1]),
                    float(prob_vec[2]),
                )

                # Conviction threshold >= 0.60
                if p_buy >= 0.60:
                    raw_signal = "BUY"
                    conf = p_buy
                elif p_sell >= 0.60:
                    raw_signal = "SELL"
                    conf = p_sell
                else:
                    raw_signal = "HOLD"
                    conf = p_hold

                # Macro filter at bar t
                cur_c = float(close_s.iloc[i])
                cur_sma200 = (
                    float(sma200_s.iloc[i]) if pd.notna(sma200_s.iloc[i]) else cur_c
                )
                cur_spy = float(spy_close_s.iloc[i])
                cur_spy_sma50 = (
                    float(spy_sma50_s.iloc[i])
                    if pd.notna(spy_sma50_s.iloc[i])
                    else cur_spy
                )

                long_ok = (cur_c >= cur_sma200) and (cur_spy >= cur_spy_sma50)
                short_ok = (cur_c < cur_sma200) or (cur_spy < cur_spy_sma50)

                if raw_signal == "BUY" and not long_ok:
                    filtered_signal = "HOLD"
                elif raw_signal == "SELL" and not short_ok:
                    filtered_signal = "HOLD"
                else:
                    filtered_signal = raw_signal

                # State machine alternation: BUY -> SELL -> BUY
                # Target execution is Open[t+1]
                locs = df.index.get_indexer([valid_indices[i]])
                orig_idx = int(locs[0]) if len(locs) > 0 and locs[0] >= 0 else i
                if (
                    filtered_signal == "BUY"
                    and current_pos != "LONG"
                    and (i - last_trade_idx >= min_cooldown_bars)
                ):
                    current_pos = "LONG"
                    last_trade_idx = i
                    if orig_idx + 1 < len(df):
                        next_idx = df.index[orig_idx + 1]
                        exec_target = (
                            next_idx.strftime("%Y-%m-%d")
                            if hasattr(next_idx, "strftime")
                            else str(next_idx)
                        )
                        exec_price = (
                            float(df["Open"].iloc[orig_idx + 1]) * 1.0005
                        )  # 5 bps slippage
                    else:
                        exec_target = "NEXT_SESSION_OPEN"
                        exec_price = float(df["Close"].iloc[orig_idx]) * 1.0005

                    self.signal_ledger.record_signal(
                        symbol=ticker,
                        bar_timestamp=bar_time,
                        signal="BUY",
                        confidence=conf,
                        execution_target_bar=exec_target,
                        execution_price=round(exec_price, 2),
                        model_version="Institutional_Mesh_V2.1",
                        raw_features_hash=SignalLedger.compute_features_hash(
                            scaled_rows[i]
                        ),
                        metadata={
                            "source": "causal_replay",
                            "rule": "Open[t+1] + 5bps",
                        },
                    )

                elif (
                    filtered_signal == "SELL"
                    and current_pos == "LONG"
                    and (i - last_trade_idx >= min_cooldown_bars)
                ):
                    current_pos = "FLAT"
                    last_trade_idx = i
                    if orig_idx + 1 < len(df):
                        next_idx = df.index[orig_idx + 1]
                        exec_target = (
                            next_idx.strftime("%Y-%m-%d")
                            if hasattr(next_idx, "strftime")
                            else str(next_idx)
                        )
                        exec_price = (
                            float(df["Open"].iloc[orig_idx + 1]) * 0.9995
                        )  # 5 bps slippage
                    else:
                        exec_target = "NEXT_SESSION_OPEN"
                        exec_price = float(df["Close"].iloc[orig_idx]) * 0.9995

                    self.signal_ledger.record_signal(
                        symbol=ticker,
                        bar_timestamp=bar_time,
                        signal="SELL",
                        confidence=conf,
                        execution_target_bar=exec_target,
                        execution_price=round(exec_price, 2),
                        model_version="Institutional_Mesh_V2.1",
                        raw_features_hash=SignalLedger.compute_features_hash(
                            scaled_rows[i]
                        ),
                        metadata={
                            "source": "causal_replay",
                            "rule": "Open[t+1] - 5bps",
                        },
                    )

            return self.signal_ledger.get_signals(
                symbol=ticker, actions_filter=["BUY", "SELL"]
            )

        except Exception as e:
            logger.error("Error during causal ML replay for %s: %s", ticker, e)
            return []

    def get_causal_chart_markers(
        self,
        ticker: str,
        ticker_df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves verified causal markers from the immutable SignalLedger.
        If the ledger does not have existing signals for this ticker,
        it executes a point-in-time sequential replay using strictly trailing data (t <= bar),
        commits the signals to the ledger, and returns them.
        """
        existing = self.signal_ledger.get_signals(
            symbol=ticker, actions_filter=["BUY", "SELL"]
        )
        if len(existing) >= 2:
            return existing

        return self._replay_causal_ml_signals(ticker, ticker_df, spy_df)

    async def get_prediction(self, ticker, config, metadata):
        import uuid

        signal_id = f"SIG_{datetime.now().strftime('%Y%m%d%H%M%S')}_{ticker}_{str(uuid.uuid4())[:8]}"

        # 1. Fetch live data
        try:
            (
                ts_sequence,
                peer_sequence,
                tabular_row,
                current_price,
                updated_config,
                market_regime,
                req_conf,
                vol_ratio,
                tech_snapshot,
                ticker_df_risk,
                spy_df_risk,
            ) = await asyncio.to_thread(fetch_live_data, ticker, config)
        except Exception as e:
            logger.error(f"Error fetching live data for {ticker}: {str(e)}")
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail=f"Market data unavailable or insufficient for {ticker}: {str(e)}")

        # 2. Pre-Inference Intelligence
        regime_detailed = self.regime_v2.detect_regime_v2(ticker_df_risk, spy_df_risk)
        asset_class = self.asset_engine.get_asset_class(ticker)
        self.asset_engine.enrich_context(ticker, ticker_df_risk)
        model_weights_raw = self.weight_engine.calculate_weights(
            regime_detailed, asset_class
        )

        # 3. Model Predictions
        from src.execution.asset_intelligence import MODEL_REGISTRY, ModelRole

        is_dl_quarantined = (
            MODEL_REGISTRY.get("DL_FUSION", {}).get("role") == ModelRole.QUARANTINED
            or MODEL_REGISTRY.get("DL_FUSION", {}).get("status") == "QUARANTINED"
            or self.mm.lstm_model is None
        )

        if not is_dl_quarantined and self.mm.lstm_model is not None:
            dl_outputs = self.mm.lstm_model.predict(
                x=[
                    ts_sequence,
                    ts_sequence,
                    ts_sequence,
                    ts_sequence,
                    ts_sequence,
                    peer_sequence if peer_sequence is not None else ts_sequence,
                ],
                verbose=0,
            )
            dl_preds_raw = dl_outputs[2][0]
        else:
            # DL_FUSION is permanently QUARANTINED: bypass tensor execution and assign neutral output
            dl_preds_raw = np.array([0.0, 1.0, 0.0])

        xgb_preds_raw = self.mm.xgb_model.predict_proba(tabular_row)[0]
        lgbm_preds_raw = (
            self.mm.lgbm_model.predict_proba(tabular_row)[0]
            if self.mm.lgbm_model
            else np.array([0.33, 0.33, 0.33])
        )

        # Apply isotonic calibration (fitted on validation data during training)
        if self.model_calibrator is not None:
            if not is_dl_quarantined:
                dl_preds_raw = self.model_calibrator.calibrate("DL_FUSION", dl_preds_raw)
            xgb_preds_raw = self.model_calibrator.calibrate("XGB", xgb_preds_raw)
            lgbm_preds_raw = self.model_calibrator.calibrate("LGBM", lgbm_preds_raw)
            logger.debug("Applied isotonic calibration to model predictions")

        # DQN Probabilistic Prediction (Soft Temperature-Scaled Calibration)
        dqn_state = np.hstack(
            (tabular_row, dl_preds_raw.reshape(1, -1), xgb_preds_raw.reshape(1, -1))
        )
        if hasattr(self.mm.dqn_agent, "predict_proba"):
            dqn_p = self.mm.dqn_agent.predict_proba(dqn_state[0], temperature=1.5)
        else:
            dqn_action = self.mm.dqn_agent.act(dqn_state[0])
            acc = self.mm.accuracies.get("dqn_accuracy", 0.50)
            dqn_p = np.full(3, (1.0 - acc) / 2.0)
            dqn_p[dqn_action] = acc

        # Enforce conviction threshold: if max(P_sell, P_hold, P_buy) < 0.60, model outputs HOLD ([0, 1, 0])
        if np.max(dl_preds_raw) < 0.60:
            dl_preds_raw = np.array([0.0, 1.0, 0.0])
        if np.max(xgb_preds_raw) < 0.60:
            xgb_preds_raw = np.array([0.0, 1.0, 0.0])
        if np.max(lgbm_preds_raw) < 0.60:
            lgbm_preds_raw = np.array([0.0, 1.0, 0.0])
        if np.max(dqn_p) < 0.60:
            dqn_p = np.array([0.0, 1.0, 0.0])

        # 4. Consensus & Decision Architecture
        # Flagship Default: Pure XGBoost as primary alpha generator
        # Secondary Option: Asymmetric Veto Consensus (when use_veto=True)
        base_probs = {
            "LSTM": dl_preds_raw,
            "XGBoost": xgb_preds_raw,
            "LightGBM": lgbm_preds_raw,
            "DQN": dqn_p,
        }

        if not getattr(self, "use_veto", False):
            # Production Flagship Default: Pure XGBoost Primary Alpha Driver (Secondary Veto Disabled)
            agreement_data = self.consensus_engine.compute_asymmetric_veto(
                base_probs,
                primary_key="XGB_AGENT",
                primary_threshold=0.60,
                veto_threshold=1.01,  # Veto hurdle > 1.0 ensures secondary models cannot veto XGBoost
                veto_short=False,
            )
        else:
            # Secondary Asymmetric Veto Consensus Architecture Enabled
            agreement_data = self.consensus_engine.compute_asymmetric_veto(
                base_probs,
                primary_key="XGB_AGENT",
                primary_threshold=0.60,
                veto_threshold=0.65,
                veto_short=True,
            )

        final_prob_raw = agreement_data["agreement_score"] / 100.0
        cal_results = self.calibration_engine.calibrate(
            final_prob_raw * 100, ticker, asset_class
        )
        calibrated_prob = cal_results["calibrated_prob"]
        cal_results["metrics"]

        # 5. Timing & Meta-Selection
        timing_data = self.timing_engine.calculate_timing_features(ticker_df_risk)
        regime_id_map = {"BEAR": 0, "NEUTRAL": 1, "BULL": 2}
        regime_id = regime_id_map.get(market_regime, 1)
        vol_id = (
            2
            if tech_snapshot["ATR"] / current_price > 0.04
            else (0 if tech_snapshot["ATR"] / current_price < 0.01 else 1)
        )

        final_probs_meta, uncertainty = get_meta_prediction(
            base_probs,
            regime_id,
            vol_id,
            vol_ratio,
            tech_snapshot["RSI"],
            tech_snapshot["ADX"],
        )

        # 6. Risk & EV Logic
        beta = calculate_beta(ticker_df_risk["Close"], spy_df_risk["Close"])
        stampede = detect_stampede_risk(vol_ratio, final_prob_raw)
        risk_profile = get_position_sizing(final_prob_raw, self.paper_engine.history)

        ev_metrics = self.ev_engine.calculate_ev(
            win_prob=calibrated_prob / 100,
            avg_gain=0.08 if "TREND" in regime_detailed else 0.03,
            avg_loss=0.04,
        )

        consensus_risk_input = {
            "beta": float(beta),
            "uncertainty_score": uncertainty,
            "stampede_risk": stampede,
            "suggested_allocation": risk_profile["suggested_allocation"],
            "hedge_ratio_spy": f"{beta:.2f}",
        }

        consensus_result = self.orchestrator.run_consensus(
            agreement_data, consensus_risk_input, market_regime=regime_detailed
        )
        final_signal_idx = consensus_result["final_action_idx"]
        signals_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        pre_signal = signals_map[final_signal_idx]

        # Macro Regime Filter: 200-day SMA on underlying & 50-day SMA on SPY
        try:
            close_s = ticker_df_risk["Close"].dropna()
            spy_close_s = spy_df_risk["Close"].dropna()
            curr_close = float(close_s.iloc[-1])
            curr_spy_close = float(spy_close_s.iloc[-1])
            sma_200 = float(close_s.rolling(window=200, min_periods=20).mean().iloc[-1])
            spy_sma_50 = float(spy_close_s.rolling(window=50, min_periods=10).mean().iloc[-1])
            long_allowed = bool((curr_close >= sma_200) and (curr_spy_close >= spy_sma_50))
            short_allowed = bool((curr_close < sma_200) or (curr_spy_close < spy_sma_50))
        except Exception as e:
            logger.warning(f"Error computing macro regime filter: {e}")
            long_allowed = True
            short_allowed = True
            sma_200, spy_sma_50 = current_price, 1.0
            curr_close, curr_spy_close = current_price, 1.0

        macro_filter_data = {
            "underlying_close": round(curr_close, 2),
            "underlying_sma_200": round(sma_200, 2),
            "spy_close": round(curr_spy_close, 2),
            "spy_sma_50": round(spy_sma_50, 2),
            "long_allowed": long_allowed,
            "short_allowed": short_allowed,
        }

        # 7. Quality & Confidence Decomposition
        confidence_data = self.confidence_engine.decompose_confidence(
            regime=regime_detailed,
            volatility_ratio=vol_ratio,
            agreement_score=agreement_data["agreement_score"],
            ev_pct=ev_metrics["ev_pct"],
            timing_score=timing_data["timing_score"],
            asset_class=asset_class,
            direction=pre_signal,
        )

        quality_metrics = self.quality_engine.calculate_score(
            consensus_agreement=float(agreement_data["agreement_score"]),
            calibrated_confidence=calibrated_prob,
            ev_metrics=ev_metrics,
            regime_v2=regime_detailed,
            risk_veto=consensus_result["consensus_status"] == "VETOED",
        )

        # Signal Suppression logic
        final_signal = pre_signal
        signal_note = None
        if consensus_result["consensus_status"] == "VETOED":
            final_signal = "VETOED"
            signal_note = consensus_result.get("veto_reason", "Vetoed by Governance")
        elif agreement_data.get("is_vetoed"):
            final_signal = "HOLD"
            signal_note = agreement_data.get("veto_reason", "Vetoed by secondary model risk gate")
        elif pre_signal == "BUY" and not long_allowed:
            final_signal = "HOLD"
            signal_note = (
                f"Suppressed by Macro Regime Filter: Close ({curr_close:.2f} < SMA200 {sma_200:.2f}) "
                f"or SPY ({curr_spy_close:.2f} < SMA50 {spy_sma_50:.2f})"
            )
        elif pre_signal == "SELL" and not short_allowed:
            final_signal = "HOLD"
            signal_note = (
                f"Suppressed by Macro Regime Filter: Counter-trend SHORT forbidden in confirmed bull regime "
                f"(Close {curr_close:.2f} >= SMA200 {sma_200:.2f} and SPY {curr_spy_close:.2f} >= SMA50 {spy_sma_50:.2f})"
            )
        elif quality_metrics["grade"] == "NO_TRADE":
            final_signal = "HOLD"
            signal_note = f"Suppressed: Low Signal Quality ({quality_metrics['score']})"
        elif ev_metrics["ev_pct"] <= 0:
            final_signal = "HOLD"
            signal_note = "Suppressed: Negative Expected Value"

        # 8. Forecast & Trade Construction
        tft_preds = self.mm.tft_model.predict(ts_sequence, verbose=0)[0]
        recent_vol = float(
            tech_snapshot.get("ATR", current_price * 0.02) / current_price
        )
        forecast_data = self.forecast_engine.calibrate_forecast(
            raw_forecasts=tft_preds,
            current_price=current_price,
            atr=float(tech_snapshot.get("ATR", current_price * 0.02)),
            volatility=recent_vol,
            asset_class=asset_class,
            regime=regime_detailed,
            volatility_state="HIGH"
            if vol_id == 2
            else ("LOW" if vol_id == 0 else "MEDIUM"),
        )

        trade_construction = self.trade_engine.construct_trade(
            current_price=current_price,
            atr=float(tech_snapshot.get("ATR", current_price * 0.02)),
            direction=final_signal if final_signal in ["BUY", "SELL"] else "HOLD",
            regime=regime_detailed,
            asset_class=asset_class,
            volatility=recent_vol,
        )

        # Phase 2: Final Execution Authority
        auth_data = self.execution_authority.determine_execution_state(
            {
                "quality": quality_metrics,
                "expected_value": ev_metrics,
                "explainable_confidence": confidence_data["explainable_confidence"],
                "agreement": agreement_data["agreement_score"],
                "signal": final_signal,
                "signal_note": signal_note,
                "uncertainty_score": uncertainty * 100,
                "calibration": cal_results,
            }
        )

        # News & Sentiment
        tokenizer = NewsTokenizer(max_length=updated_config["data"]["max_seq_length"])
        _, _, news_text = fetch_live_news(ticker, tokenizer, updated_config)
        sentiment_score, qual_reason = await asyncio.to_thread(
            self.gemini.analyze_fundamental_alpha, news_text, ticker
        )

        # Performance & Metrics
        signal_df = self.journal.get_all_signals() if self.journal else None
        self.perf_analyzer.analyze(
            self.paper_engine.portfolio_snapshots,
            self.paper_engine.history,
            self.paper_engine.initial_capital,
            signal_data=signal_df,
        )

        # 9. Final Response Assembly (Phase 3 Semantic Separation)
        response_data = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "signal": final_signal,
            "confidence_score": round(calibrated_prob, 1),
            "uncertainty_score": round(uncertainty * 100, 1),
            "signal_note": signal_note,
            # Phase 3: Semantic Separation
            "structural_regime": regime_detailed,
            "signal_bias": forecast_data["forecast_bias"],
            "execution_state": auth_data["execution_state"],
            "execution_reasoning": auth_data["decision_reasoning"],
            # Institutional Logic
            "signal_reasoning": f"Consensus directional agreement ({agreement_data['dominant_direction']}) crossed confidence threshold.",
            "timing_reason": f"Momentum acc {timing_data.get('momentum_acceleration', 0.0):.2f}, Volatility expansion {timing_data.get('volatility_expansion', 0.0):.2f}",
            "forecast_interpretation": forecast_data["forecast_interpretation"],
            "forecast_explanation": forecast_data["interpretation_explanation"],
            "consensus_intelligence": agreement_data["consensus_interpretation"],
            "agreement_data": agreement_data,
            "market_regime": market_regime,
            "volatility_state": "HIGH"
            if vol_id == 2
            else ("LOW" if vol_id == 0 else "MEDIUM"),
            "volume_ratio": round(vol_ratio, 2),
            "model_weights": model_weights_raw,
            "models": {
                "DL_FUSION": {
                    "role": "QUARANTINED",
                    "status": "QUARANTINED",
                    "signal": signals_map[int(np.argmax(dl_preds_raw))],
                    "probability": float(np.max(dl_preds_raw)),
                },
                "XGB_AGENT": {
                    "role": "PRIMARY_ALPHA_DRIVER",
                    "status": "ACTIVE",
                    "signal": signals_map[int(np.argmax(xgb_preds_raw))],
                    "probability": float(np.max(xgb_preds_raw)),
                },
                "LGBM_AGENT": {
                    "role": "SECONDARY_VETO",
                    "status": "ACTIVE",
                    "signal": signals_map[int(np.argmax(lgbm_preds_raw))],
                    "probability": float(np.max(lgbm_preds_raw)),
                },
                "DQN_AGENT": {
                    "role": "SECONDARY_VETO",
                    "status": "ACTIVE",
                    "signal": signals_map[int(np.argmax(dqn_p))],
                    "probability": float(np.max(dqn_p)),
                },
            },
            "projections": {
                "floor": round(forecast_data["p10_price"], 2),
                "p50": round(forecast_data["p50_price"], 2),
                "ceiling": round(forecast_data["p90_price"], 2),
                "confidence": round(forecast_data["forecast_confidence"], 1),
                "reliability": forecast_data["forecast_reliability"],
                "drift": forecast_data.get("forecast_drift", 0.0),
                "expected_move": forecast_data.get("expected_move_10d", 0.0),
            },
            "trade_parameters": trade_construction,
            "quality": SignalQuality(**quality_metrics),
            "expected_value": ExpectedValueMetrics(**ev_metrics),
            "confidence_breakdown": confidence_data["confidence_breakdown"],
            "explainable_confidence": confidence_data["explainable_confidence"],
            "asset_class": asset_class,
            "metadata": metadata,
            "governance": self.governance_analytics.analyze_throughput(signal_df)
            if signal_df is not None
            else {},
            "technical_snapshot": tech_snapshot,
            "macro_regime_filter": macro_filter_data,
            "qualitative_alpha": qual_reason,
            "sentiment_score": sentiment_score,
            "xai": compute_shap_explanation(self.mm.xgb_model, tabular_row, signal_idx=final_signal_idx),
            "risk": {
                "var_95": float(risk_profile.get("var_95", beta * 1.5)),
                "cvar": float(risk_profile.get("cvar", beta * 2.0)),
                "beta": float(beta),
                "kelly_fraction": float(risk_profile.get("raw_fraction", 0.0)),
                "target_size": float(risk_profile.get("raw_fraction", 0.0)) * 100,
                "max_drawdown": float(risk_profile.get("max_drawdown", 0.0)),
                "institutional_risk_index": 0.0,
                "risk_regime": "STABLE",
                "win_probability": float(calibrated_prob / 100),
                "expected_value": float(ev_metrics["ev_pct"]),
                "risk_reward_ratio": float(risk_profile.get("rr_ratio", 0.0)),
                "peak_equity": float(self.paper_engine.get_portfolio_summary({}).get("peak_equity", self.paper_engine.initial_capital)),
                "peak_date": "",
                "trough_equity": float(self.paper_engine.get_portfolio_summary({}).get("trough_equity", self.paper_engine.initial_capital)),
                "trough_date": ""
            }
        }

        # Check forming vs confirmed status
        is_bar_forming = bool(tech_snapshot.get("is_bar_forming", False))
        bar_state = str(tech_snapshot.get("bar_state", "CONFIRMED"))
        if not ticker_df_risk.empty:
            last_idx = ticker_df_risk.index[-1]
            bar_date = (
                last_idx.strftime("%Y-%m-%d")
                if hasattr(last_idx, "strftime")
                else str(last_idx)
            )
        else:
            bar_date = datetime.now().strftime("%Y-%m-%d")

        # Causal execution pricing & latency modeling (t -> t+1 Open)
        execution_target = "NEXT_SESSION_OPEN"
        if final_signal == "BUY":
            exec_price = round(current_price * 1.0005, 2)  # 5 bps slippage
        elif final_signal == "SELL":
            exec_price = round(current_price * 0.9995, 2)  # 5 bps slippage
        else:
            exec_price = round(current_price, 2)

        response_data["is_bar_forming"] = is_bar_forming
        response_data["bar_state"] = bar_state
        response_data["provisional"] = is_bar_forming
        response_data["execution_target_bar"] = execution_target
        response_data["execution_price"] = exec_price

        # Zero-Repainting Mandate: Commit confirmed signals to immutable SQLite ledger
        if not is_bar_forming and final_signal in ["BUY", "SELL", "HOLD"]:
            features_hash = SignalLedger.compute_features_hash(tabular_row)
            self.signal_ledger.record_signal(
                symbol=ticker,
                bar_timestamp=bar_date,
                signal=final_signal,
                confidence=float(calibrated_prob / 100.0),
                execution_target_bar=execution_target,
                execution_price=exec_price,
                model_version="Institutional_Mesh_V2.1",
                raw_features_hash=features_hash,
                metadata={
                    "quality_score": quality_metrics["score"],
                    "ev_pct": ev_metrics["ev_pct"],
                    "regime": regime_detailed,
                    "signal_note": signal_note,
                    "agreement_score": agreement_data["agreement_score"],
                },
            )

        # Retrieve verified causal markers (sourced from SignalLedger / Causal ML Consensus)
        causal_markers = self.get_causal_chart_markers(
            ticker, ticker_df_risk, spy_df=spy_df_risk
        )

        ai_report_stub = {
            "Models": {"Meta_Model_Status": "Institutional Mesh V2.1"},
            "Risk": {"Quality": quality_metrics["score"]},
        }

        reporting_data = self.report_gen.package_chart_data(
            ticker,
            ticker_df_risk,
            ai_report_stub,
            causal_markers,
        )
        response_data.update(reporting_data)
        response_data["signal_id"] = signal_id

        # Paper Trade Execution
        if auth_data["execution_state"] in ["EXECUTE LONG", "EXECUTE SHORT"]:
            trade = self.paper_engine.execute_trade(
                ticker,
                final_signal,
                current_price,
                risk_profile["raw_fraction"],
                market_regime,
                currency=metadata["currency"],
                market=metadata["market"],
                signal_id=signal_id,
            )
            if trade:
                response_data["paper_trade"] = trade

        response_data["portfolio"] = self.paper_engine.get_portfolio_summary(
            {ticker: current_price}
        )

        # Log to Journal
        if self.journal:
            self.journal.log_signal(
                {
                    "signal_id": signal_id,
                    "timestamp": datetime.now().isoformat(),
                    "asset": ticker,
                    "signal_type": final_signal,
                    "entry_price": current_price,
                    "confidence": calibrated_prob,
                    "agreement": agreement_data["agreement_score"],
                    "quality_score": quality_metrics["score"],
                    "ev_pct": ev_metrics["ev_pct"],
                    "asset_class": asset_class,
                    "execution_state": auth_data["execution_state"],
                }
            )

        return response_data
