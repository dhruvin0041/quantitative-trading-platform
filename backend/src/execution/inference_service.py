import asyncio
import logging
import numpy as np
from datetime import datetime

from src.execution.live_inference import (
    fetch_live_data,
    fetch_live_news,
    is_near_earnings,
    compute_shap_explanation,
    get_meta_prediction,
)
from src.execution.risk_manager import (
    get_position_sizing,
    calculate_beta,
    detect_stampede_risk,
)
from src.data_ingestion.nlp_processor import NewsTokenizer
from src.schemas import RiskMetrics

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
        self.paper_engine = paper_engine
        self.perf_analyzer = perf_analyzer
        self.journal = signal_journal

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

        # 2. Model Predictions
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

        # 3. Meta-Ensemble Consensus
        base_probs = {
            "LSTM": dl_preds_raw,
            "XGBoost": xgb_preds_raw,
            "LightGBM": lgbm_preds_raw,
            "DQN": dqn_p,
        }
        regime_id_map = {"BEAR": 0, "NEUTRAL": 1, "BULL": 2}
        regime_id = regime_id_map.get(market_regime, 1)
        vol_id = 1
        if tech_snapshot["ATR"] / current_price > 0.04:
            vol_id = 2
        elif tech_snapshot["ATR"] / current_price < 0.01:
            vol_id = 0

        final_probs, uncertainty = get_meta_prediction(
            base_probs,
            regime_id,
            vol_id,
            vol_ratio,
            tech_snapshot["RSI"],
            tech_snapshot["ADX"],
        )
        final_idx = np.argmax(final_probs)
        final_prob = float(final_probs[final_idx])

        # 4. Agentic Orchestration
        # Build risk metrics for consensus
        beta = calculate_beta(ticker_df_risk["Close"], spy_df_risk["Close"])
        stampede = detect_stampede_risk(vol_ratio, final_prob)
        
        # Sizing and Hedging
        risk_profile = get_position_sizing(final_prob, self.paper_engine.history)
        kelly_frac = min(0.05, float(risk_profile["raw_fraction"]))
        hedge_ratio = f"{beta:.2f}"

        consensus_risk_input = {
            "beta": float(beta),
            "uncertainty_score": uncertainty,
            "stampede_risk": stampede,
            "suggested_allocation": risk_profile["suggested_allocation"],
            "hedge_ratio_spy": hedge_ratio
        }

        consensus_result = self.orchestrator.run_consensus(
            final_probs, consensus_risk_input
        )
        final_signal_idx = consensus_result["final_action_idx"]
        signals_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        final_signal = signals_map[final_signal_idx]

        if consensus_result["consensus_status"] == "VETOED":
            final_signal = "VETOED"
            signal_note = consensus_result["agent_responses"]["risk"]["veto_reason"]
        else:
            signal_note = None
            if await asyncio.to_thread(is_near_earnings, ticker):
                final_signal = "HOLD"
                signal_note = "Suppressed: Earnings window"
            elif vol_ratio < 0.7:
                final_signal = "HOLD"
                signal_note = "Suppressed: Low volume (ratio: {:.2f})".format(vol_ratio)

        confidence_score = final_prob * 100

        # SHAP Explainability
        shap_xai = await asyncio.to_thread(
            compute_shap_explanation, self.mm.lgbm_model, tabular_row, final_signal_idx
        )

        # TFT Multi-Horizon Projections
        tft_preds = self.mm.tft_model.predict(ts_sequence, verbose=0)[0]
        constrained_rets = np.clip(tft_preds, -0.20, 0.20)
        is_point_forecast = np.all(np.isclose(constrained_rets, constrained_rets[0]))
        floor_ret, median_ret, ceiling_ret = (
            float(constrained_rets[0]),
            float(constrained_rets[2]),
            float(constrained_rets[4]),
        )
        forecast_low = current_price * (1 + floor_ret)
        forecast_median = current_price * (1 + median_ret)
        forecast_high = current_price * (1 + ceiling_ret)

        # Qualitative Alpha
        tokenizer = NewsTokenizer(max_length=updated_config["data"]["max_seq_length"])
        _, _, news_text = fetch_live_news(ticker, tokenizer, updated_config)
        sentiment_score, qual_reason = await asyncio.to_thread(
            self.gemini.analyze_fundamental_alpha, news_text, ticker
        )

        # 5. Risk & Performance
        returns_history = [
            t.get("pnl", 0) / (t.get("cost_base", 1e-9))
            for t in self.paper_engine.history
            if "pnl" in t
        ]

        if not returns_history:
            var_95, cvar = 0.0, 0.0
        elif len(returns_history) < 20:
            hist_daily_rets = ticker_df_risk["Close"].pct_change().dropna().values
            var_95 = (
                float(np.percentile(hist_daily_rets, 5))
                if len(hist_daily_rets) > 0
                else 0.0
            )
            cvar = (
                float(np.mean(hist_daily_rets[hist_daily_rets <= var_95]))
                if len(hist_daily_rets) > 0
                else 0.0
            )
        else:
            var_95 = float(self.paper_engine.calculate_var(returns_history))
            cvar = float(
                self.paper_engine.calculate_expected_shortfall(returns_history)
            )

        risk_profile = get_position_sizing(final_prob, self.paper_engine.history)
        kelly_frac = min(0.05, float(risk_profile["raw_fraction"]))

        perf_summary = self.perf_analyzer.analyze(
            self.paper_engine.portfolio_snapshots,
            self.paper_engine.history,
            self.paper_engine.initial_capital,
        )
        max_dd = perf_summary.get("summary", {}).get("max_drawdown", 0.0)

        risk_metrics = RiskMetrics(
            var_95=var_95,
            cvar=cvar,
            beta=float(beta),
            kelly_fraction=kelly_frac,
            target_size=float(self.paper_engine.capital * kelly_frac),
            max_drawdown=float(max_dd),
        )

        def map_model_output(probs):
            idx = np.argmax(probs)
            signals = ["SELL", "HOLD", "BUY"]
            return {"signal": signals[idx], "probability": round(float(probs[idx]), 3)}

        signals_only = [
            m["signal"]
            for m in [
                map_model_output(dl_preds_raw),
                map_model_output(xgb_preds_raw),
                map_model_output(lgbm_preds_raw),
            ]
        ]
        bullish_count = signals_only.count("BUY")
        bearish_count = signals_only.count("SELL")
        neutral_count = signals_only.count("HOLD")
        majority_count = max(bullish_count, bearish_count, neutral_count)
        model_agreement = (majority_count / 3.0) * 100

        response_data = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "signal": final_signal,
            "confidence_score": round(confidence_score, 1),
            "uncertainty_score": round(uncertainty * 100, 1),
            "signal_note": signal_note,
            "market_regime": market_regime,
            "volatility_state": "HIGH"
            if vol_id == 2
            else ("LOW" if vol_id == 0 else "MEDIUM"),
            "volume_ratio": round(vol_ratio, 2),
            "is_point_forecast": is_point_forecast,
            "model_agreement": round(model_agreement, 1),
            "bullish_models": bullish_count,
            "bearish_models": bearish_count,
            "neutral_models": neutral_count,
            "timestamp": datetime.now().isoformat(),
            "models": {
                "DL_FUSION": map_model_output(dl_preds_raw),
                "XGB_AGENT": map_model_output(xgb_preds_raw),
                "LGBM_AGENT": map_model_output(lgbm_preds_raw),
            },
            "projections": {
                "floor": round(float(forecast_low), 2),
                "median": round(float(forecast_median), 2),
                "ceiling": round(float(forecast_high), 2),
            },
            "technical_snapshot": tech_snapshot,
            "qualitative_alpha": qual_reason,
            "xai": shap_xai,
            "sentiment_score": float(sentiment_score),
            "risk": risk_metrics,
            "metadata": metadata,
        }

        historical_markers, df_full = self.report_gen.generate_historical_markers(
            ticker, ticker_df_risk
        )
        ai_report_stub = {
            "Models": {
                "Primary_Deep_Learning": {
                    "Suggested_Action": final_signal,
                    "Confidence": f"{confidence_score:.1f}%",
                },
                "Secondary_XGBoost": {
                    "Suggested_Action": "BUY" if xgb_preds_raw[2] > 0.5 else "SELL",
                    "Confidence": f"{xgb_preds_raw[2] * 100:.1f}%",
                },
            },
            "Risk_Management": {
                "Meta_Model_Status": "Live Consensus Active",
                "Dynamic_10_Day_Range": {"Low": forecast_low, "High": forecast_high},
            },
            "Context": {"Top_Headline_Processed": news_text},
        }
        reporting_data = self.report_gen.package_chart_data(
            ticker, df_full, ai_report_stub, historical_markers
        )
        response_data.update(reporting_data)

        response_data["signal_id"] = signal_id

        # Paper Trading with Currency Context
        if final_signal in ["BUY", "SELL"]:
            atr = tech_snapshot.get("ATR", current_price * 0.02)
            sl = (
                (current_price - 2 * atr)
                if final_signal == "BUY"
                else (current_price + 2 * atr)
            )
            tp = forecast_high if final_signal == "BUY" else forecast_low
            trade = self.paper_engine.execute_trade(
                ticker,
                final_signal,
                current_price,
                kelly_frac,
                market_regime,
                currency=metadata["currency"],
                market=metadata["market"],
                stop_loss=sl,
                take_profit=tp,
                signal_id=signal_id,
            )
            if trade:
                response_data["paper_trade"] = trade

        closed_trades = self.paper_engine.update_positions({ticker: current_price})
        response_data["portfolio"] = self.paper_engine.get_portfolio_summary(
            {ticker: current_price}
        )

        # Log to Signal Journal
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
                    "position_size": kelly_frac,
                    "confidence": confidence_score,
                    "uncertainty": uncertainty * 100,
                    "agreement": model_agreement,
                    "market_regime": market_regime,
                    "volatility_regime": "HIGH"
                    if vol_id == 2
                    else ("LOW" if vol_id == 0 else "MEDIUM"),
                    "model_consensus": json.dumps(
                        {
                            "DL_FUSION": map_model_output(dl_preds_raw),
                            "XGB_AGENT": map_model_output(xgb_preds_raw),
                            "LGBM_AGENT": map_model_output(lgbm_preds_raw),
                        }
                    ),
                }
            )

            # Update any closed trades in the journal
            for ct in closed_trades:
                if ct.get("signal_id"):
                    # compute holding time roughly in days based on trade open time
                    holding_time = 0
                    if ct.get("entry_time"):
                        entry_dt = datetime.fromisoformat(ct["entry_time"])
                        holding_time = max(1, (datetime.now() - entry_dt).days)
                    self.journal.update_signal_exit(
                        signal_id=ct["signal_id"],
                        exit_price=ct["price"],
                        realized_pnl=ct["pnl"],
                        holding_time=holding_time,
                    )

        return response_data
