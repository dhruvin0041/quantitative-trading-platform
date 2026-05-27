import asyncio
import logging
import numpy as np
from datetime import datetime

from src.execution.consensus_engine import WeightedConsensusEngine
from src.execution.forecast_engine import ForecastCalibrationEngine
from src.execution.trade_engine import TradeConstructionEngine
from src.execution.timing_engine import PredictiveTimingEngine
from src.execution.confidence_engine import ConfidenceBreakdownEngine

from src.execution.live_inference import (
    fetch_live_data,
    fetch_live_news,
    compute_shap_explanation,
    get_meta_prediction,
)
from src.execution.risk_manager import (
    get_position_sizing,
    calculate_beta,
    detect_stampede_risk,
)
from src.data_ingestion.nlp_processor import NewsTokenizer
from src.execution.signal_intelligence import (
    RegimeEngineV2,
    ConfidenceCalibrationEngine,
    ExpectedValueEngine,
    SignalQualityEngine,
)
from src.execution.asset_intelligence import (
    AssetProfileEngine,
    AdaptiveWeightingEngine,
    MultiTimeframeEngine,
)
from src.schemas import (
    RiskMetrics,
    SignalQuality,
    ModelWeight,
    ExpectedValueMetrics,
    CalibrationMetrics,
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
    ):
        self.mm = model_manager
        self.gemini = gemini_analyzer
        self.physical = physical_edge
        self.graph = dependency_graph
        self.orchestrator = orchestrator
        self.router = smart_router
        self.report_gen = report_gen
        self.consensus_engine = WeightedConsensusEngine()
        self.forecast_engine = ForecastCalibrationEngine()
        self.trade_engine = TradeConstructionEngine()
        self.timing_engine = PredictiveTimingEngine()
        self.confidence_engine = ConfidenceBreakdownEngine()
        self.paper_engine = paper_engine
        self.perf_analyzer = perf_analyzer
        self.journal = signal_journal

        # V2.0 Engines
        self.regime_v2 = RegimeEngineV2()
        self.calibration_engine = ConfidenceCalibrationEngine()
        self.ev_engine = ExpectedValueEngine()
        self.quality_engine = SignalQualityEngine()
        self.asset_engine = AssetProfileEngine()
        self.weight_engine = AdaptiveWeightingEngine()
        self.mtf_engine = MultiTimeframeEngine()

    async def get_prediction(self, ticker, config, metadata):
        import uuid
        import json

        signal_id = f"SIG_{datetime.now().strftime('%Y%m%d%H%M%S')}_{ticker}_{str(uuid.uuid4())[:8]}"

        # 1. Fetch live data
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

        # 2. V2.0 Pre-Inference Intelligence
        regime_detailed = self.regime_v2.detect_regime_v2(ticker_df_risk, spy_df_risk)
        asset_class = self.asset_engine.get_asset_class(ticker)
        asset_context = self.asset_engine.enrich_context(ticker, ticker_df_risk)
        model_weights_raw = self.weight_engine.calculate_weights(
            regime_detailed, asset_class
        )

        # 3. Model Predictions (Using Dynamically weighted probabilities)
        dl_outputs = self.mm.lstm_model.predict(
            x=[
                ts_sequence,
                ts_sequence,
                ts_sequence,
                ts_sequence,
                ts_sequence,
                peer_sequence,
            ],
            verbose=0,
        )
        dl_preds_raw = dl_outputs[2][0]

        xgb_preds_raw = self.mm.xgb_model.predict_proba(tabular_row)[0]

        lgbm_preds_raw = np.array([0.33, 0.33, 0.33])
        if self.mm.lgbm_model:
            lgbm_preds_raw = self.mm.lgbm_model.predict_proba(tabular_row)[0]

        # DQN Prediction
        dqn_state = np.hstack(
            (tabular_row, dl_preds_raw.reshape(1, -1), xgb_preds_raw.reshape(1, -1))
        )
        dqn_action = self.mm.dqn_agent.act(dqn_state[0])
        dqn_p = np.array([0.0, 1.0, 0.0])
        if dqn_action == 0:
            dqn_p = np.array([1.0, 0.0, 0.0])
        elif dqn_action == 2:
            dqn_p = np.array([0.0, 0.0, 1.0])

        # 4. Meta-Ensemble & Calibration
        base_probs = {
            "LSTM": dl_preds_raw,
            "XGBoost": xgb_preds_raw,
            "LightGBM": lgbm_preds_raw,
            "DQN": dqn_p,
        }

        # Phase 1: Weighted Consensus Engine
        extracted_weights = {k: v.get("weight", 0.25) for k, v in model_weights_raw.items()}
        agreement_data = self.consensus_engine.compute_agreement(base_probs, extracted_weights)
        
        final_prob_raw = agreement_data["agreement_score"] / 100.0
        cal_results = self.calibration_engine.calibrate(
            final_prob_raw * 100, ticker, asset_class
        )
        calibrated_prob = cal_results["calibrated_prob"]
        calibration_metrics = cal_results["metrics"]

        # 5. Signal Selection & Predictive Timing
        timing_data = self.timing_engine.calculate_timing_features(ticker_df_risk)
        
        regime_id_map = {"BEAR": 0, "NEUTRAL": 1, "BULL": 2}
        regime_id = regime_id_map.get(market_regime, 1)
        vol_id = 1
        if tech_snapshot["ATR"] / current_price > 0.04:
            vol_id = 2
        elif tech_snapshot["ATR"] / current_price < 0.01:
            vol_id = 0

        final_probs_meta, uncertainty = get_meta_prediction(
            base_probs,
            regime_id,
            vol_id,
            vol_ratio,
            tech_snapshot["RSI"],
            tech_snapshot["ADX"],
        )

        # 6. Agentic Orchestration & EV
        beta = calculate_beta(ticker_df_risk["Close"], spy_df_risk["Close"])
        stampede = detect_stampede_risk(vol_ratio, final_prob_raw)
        risk_profile = get_position_sizing(final_prob_raw, self.paper_engine.history)

        # EV Engine
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

        # Run multi-agent consensus with agreement data
        consensus_result = self.orchestrator.run_consensus(
            agreement_data, consensus_risk_input
        )
        final_signal_idx = consensus_result["final_action_idx"]
        signals_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        pre_signal = signals_map[final_signal_idx]

        # Phase 5: Confidence Decomposition
        confidence_data = self.confidence_engine.decompose_confidence(
            regime=regime_detailed,
            volatility_ratio=vol_ratio,
            agreement_score=agreement_data["agreement_score"],
            ev_pct=ev_metrics["ev_pct"],
            timing_score=timing_data["timing_score"],
            asset_class=asset_class
        )

        # Multi-Layer Quality Score
        quality_metrics = self.quality_engine.calculate_score(
            consensus_agreement=float(agreement_data["agreement_score"]),
            calibrated_confidence=calibrated_prob,
            ev_metrics=ev_metrics,
            regime_v2=regime_detailed,
            risk_veto=consensus_result["consensus_status"] == "VETOED",
        )

        # Final Signal Logic
        final_signal = pre_signal
        signal_note = None

        if consensus_result["consensus_status"] == "VETOED":
            final_signal = "VETOED"
            signal_note = consensus_result["agent_responses"]["risk"]["veto_reason"]
        elif quality_metrics["grade"] == "NO_TRADE":
            final_signal = "HOLD"
            signal_note = f"Suppressed: Low Signal Quality ({quality_metrics['score']})"
        elif ev_metrics["ev_pct"] <= 0:
            final_signal = "HOLD"
            signal_note = "Suppressed: Negative Expected Value"

        # MTF check
        mtf_consensus = self.mtf_engine.get_mtf_consensus(ticker, final_signal)

        # SHAP Explainability
        shap_xai = await asyncio.to_thread(
            compute_shap_explanation, self.mm.lgbm_model, tabular_row, final_signal_idx
        )

        # Phase 2: Forecast Calibration Engine
        tft_preds = self.mm.tft_model.predict(ts_sequence, verbose=0)[0]
        recent_vol = float(tech_snapshot.get("ATR", current_price * 0.02) / current_price)
        
        forecast_data = self.forecast_engine.calibrate_forecast(
            raw_forecasts=tft_preds,
            current_price=current_price,
            atr=float(tech_snapshot.get("ATR", current_price * 0.02)),
            volatility=recent_vol,
            asset_class=asset_class,
            regime=regime_detailed
        )
        
        # Phase 3: Trade Construction Engine
        trade_construction = self.trade_engine.construct_trade(
            current_price=current_price,
            atr=float(tech_snapshot.get("ATR", current_price * 0.02)),
            direction=final_signal if final_signal in ["BUY", "SELL"] else "HOLD",
            regime=regime_detailed,
            asset_class=asset_class,
            volatility=recent_vol
        )

        # News & Sentiment
        tokenizer = NewsTokenizer(max_length=updated_config["data"]["max_seq_length"])
        _, _, news_text = fetch_live_news(ticker, tokenizer, updated_config)
        sentiment_score, qual_reason = await asyncio.to_thread(
            self.gemini.analyze_fundamental_alpha, news_text, ticker
        )

        # 7. Risk Metrics (Transparency)
        signal_data = self.journal.get_all_signals() if self.journal else None
        perf_summary = self.perf_analyzer.analyze(
            self.paper_engine.portfolio_snapshots,
            self.paper_engine.history,
            self.paper_engine.initial_capital,
            signal_data=signal_data,
        )
        perf_data = perf_summary.get("summary", {})

        # Dynamic Risk Metrics from Paper Engine
        returns_history = [t.get("pnl", 0) / self.paper_engine.initial_capital for t in self.paper_engine.history]
        var_95 = self.paper_engine.calculate_var(returns_history)
        cvar = self.paper_engine.calculate_expected_shortfall(returns_history)

        risk_metrics_obj = RiskMetrics(
            var_95=float(var_95),
            cvar=float(cvar),
            beta=float(beta),
            kelly_fraction=min(0.05, float(risk_profile["raw_fraction"])),
            target_size=float(
                self.paper_engine.capital * float(risk_profile["raw_fraction"])
            ),
            max_drawdown=float(perf_data.get("max_drawdown", 0.0)),
            win_probability=float(calibrated_prob / 100),
            expected_value=ev_metrics["ev_pct"],
        )

        def map_model_output(probs):
            idx = np.argmax(probs)
            signals = ["SELL", "HOLD", "BUY"]
            return {"signal": signals[idx], "probability": round(float(probs[idx]), 3)}

        # 8. Final Response Assembly
        response_data = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "signal": final_signal,
            "confidence_score": round(calibrated_prob, 1),
            "uncertainty_score": round(uncertainty * 100, 1),
            "signal_note": signal_note,
            
            # Phase 10: Institutional Explainability
            "signal_reasoning": f"Consensus directional agreement ({agreement_data['dominant_direction']}) crossed confidence threshold with favorable EV.",
            "veto_reason": signal_note if final_signal in ["VETOED", "HOLD"] else "None",
            "timing_reason": f"Momentum acc {timing_data['momentum_acceleration']:.2f}, Volatility expansion {timing_data['volatility_expansion']:.2f}",
            "forecast_reason": f"Bounded by {asset_class} volatility constraints. Recent Vol: {recent_vol:.2f}",
            "rr_reason": trade_construction.get("reject_reason", "Dynamic construction via ATR multiples aligned with regime and volatility state."),
            
            "market_regime": market_regime,
            "market_regime_v2": regime_detailed,
            "volatility_state": "HIGH" if vol_id == 2 else "LOW" if vol_id == 0 else "MEDIUM",
            "volume_ratio": round(vol_ratio, 2),
            "is_point_forecast": False,
            "models": {
                "DL_FUSION": map_model_output(dl_preds_raw),
                "XGB_AGENT": map_model_output(xgb_preds_raw),
                "LGBM_AGENT": map_model_output(lgbm_preds_raw),
                "DQN_AGENT": map_model_output(dqn_p),
            },
            "model_weights": {
                k: ModelWeight(
                    weight=v["weight"],
                    reason=v["reason"],
                    recent_accuracy=v["recent_accuracy"],
                )
                for k, v in model_weights_raw.items()
            },
            "projections": {
                "p10": round(forecast_data["p10_price"], 2),
                "p50": round(forecast_data["p50_price"], 2),
                "p90": round(forecast_data["p90_price"], 2),
                "confidence": round(forecast_data["forecast_confidence"], 1),
                "reliability": forecast_data["forecast_reliability"]
            },
            "trade_parameters": trade_construction,
            "technical_snapshot": tech_snapshot,
            "qualitative_alpha": qual_reason,
            "xai": shap_xai,
            "sentiment_score": float(sentiment_score),
            "risk": risk_metrics_obj,
            "quality": SignalQuality(
                score=quality_metrics["score"],
                grade=quality_metrics["grade"],
                explanation=quality_metrics["explanation"],
                layers_passed=["CONSENSUS", "REGIME"] if quality_metrics["score"] > 50 else [],
                layers_failed=["EV"] if ev_metrics["ev_pct"] <= 0 else [],
            ),
            "calibration": CalibrationMetrics(**calibration_metrics),
            "expected_value": ExpectedValueMetrics(**ev_metrics),
            "multi_timeframe_consensus": mtf_consensus,
            "timing_intelligence": timing_data,
            "confidence_breakdown": confidence_data["confidence_breakdown"],
            "explainable_confidence": confidence_data["explainable_confidence"],
            "asset_class": asset_class,
            "asset_context": asset_context,
            "metadata": metadata,
        }

        # Historical Markers & Packaging
        historical_markers, df_full = self.report_gen.generate_historical_markers(
            ticker, ticker_df_risk
        )
        ai_report_stub = {
            "Models": {"Meta_Model_Status": "Live Consensus V2.1 Active"},
            "Risk": {"V2_Quality_Score": quality_metrics["score"]},
        }

        system_signals = None
        if self.journal:
            system_signals = self.journal.get_all_signals()
            system_signals = system_signals[system_signals["asset"] == ticker].head(30)

        reporting_data = self.report_gen.package_chart_data(
            ticker,
            df_full,
            ai_report_stub,
            historical_markers,
            system_signals=system_signals,
        )
        response_data.update(reporting_data)
        response_data["signal_id"] = signal_id

        # Paper Trade Execution
        if final_signal in ["BUY", "SELL"]:
            trade = self.paper_engine.execute_trade(
                ticker,
                final_signal,
                current_price,
                risk_metrics_obj.kelly_fraction,
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
                    "market": metadata["market"],
                    "exchange": metadata.get("exchange", "UNKNOWN"),
                    "signal_type": final_signal,
                    "entry_price": current_price,
                    "position_size": risk_metrics_obj.kelly_fraction,
                    "confidence": calibrated_prob,
                    "uncertainty": uncertainty * 100,
                    "agreement": agreement_data["agreement_score"],
                    "market_regime": market_regime,
                    "market_regime_v2": regime_detailed,
                    "volatility_regime": "HIGH" if vol_id == 2 else ("LOW" if vol_id == 0 else "MEDIUM"),
                    "model_consensus": json.dumps(response_data["models"]),
                    "quality_score": quality_metrics["score"],
                    "quality_grade": quality_metrics["grade"],
                    "ev_pct": ev_metrics["ev_pct"],
                    "expected_gain": ev_metrics["avg_gain_pct"],
                    "expected_loss": ev_metrics["avg_loss_pct"],
                    "asset_class": asset_class,
                }
            )

        return response_data
