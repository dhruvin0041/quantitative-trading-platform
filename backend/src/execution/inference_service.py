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
    check_bar_forming_status,
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
from src.utils.timezone_utils import format_new_york_display

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
        cal_path = None
        for cp in ["artifacts/model_calibrator.joblib", "backend/artifacts/model_calibrator.joblib"]:
            if Path(cp).exists():
                cal_path = cp
                break
        if cal_path:
            try:
                self.model_calibrator = ModelCalibrator.load(cal_path)
                logger.info("Loaded isotonic model calibrator from %s", cal_path)
            except Exception as e:
                logger.warning("Error loading calibrator (%s). Raw probs will be used.", e)
                self.model_calibrator = None
        else:
            logger.info("No isotonic calibrator found on disk. Raw probs will be used.")
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

            # Multi-Model Alpha Ensemble: XGBoost + LightGBM
            xgb_model = getattr(self.mm, "xgb_model", None)
            if xgb_model is None:
                for m_path in [
                    "artifacts/xgb_ensemble.json",
                    "backend/artifacts/xgb_ensemble.json",
                ]:
                    mp = Path(m_path)
                    if mp.exists():
                        try:
                            import xgboost as xgb

                            xgb_model = xgb.XGBClassifier()
                            xgb_model.load_model(str(mp))
                            break
                        except Exception:
                            pass

            if xgb_model is None:
                logger.warning("No XGBoost model available for causal ML replay.")
                return []

            xgb_probs = xgb_model.predict_proba(scaled_rows)

            # Secondary Alpha: LightGBM
            lgbm_model = getattr(self.mm, "lgbm_model", None)
            if lgbm_model is None:
                for l_path in [
                    "artifacts/lgbm_agent.joblib",
                    "backend/artifacts/lgbm_agent.joblib",
                ]:
                    lp = Path(l_path)
                    if lp.exists():
                        try:
                            lgbm_model = joblib.load(lp)
                            break
                        except Exception:
                            pass

            if lgbm_model is not None:
                try:
                    lgbm_probs = lgbm_model.predict_proba(scaled_rows)
                    all_probs = 0.55 * xgb_probs + 0.45 * lgbm_probs
                except Exception:
                    all_probs = xgb_probs
            else:
                all_probs = xgb_probs

            valid_indices = df_filtered.index
            if len(all_probs) == 1 and len(valid_indices) > 1:
                all_probs = np.tile(all_probs, (len(valid_indices), 1))

            # Zero-Repainting Mandate: Determine if the last candle in ticker_df is forming
            is_forming, _ = check_bar_forming_status(ticker_df)
            if is_forming and len(valid_indices) > 0:
                # Strictly exclude the unfinished candle from confirmed signal evaluation
                valid_indices = valid_indices[:-1]
                all_probs = all_probs[:-1]

            # Price series and indicators
            close_s = df["Close"].reindex(valid_indices).ffill()
            high_s = df["High"].reindex(valid_indices).ffill()
            low_s = df["Low"].reindex(valid_indices).ffill()
            open_s = df["Open"].reindex(valid_indices).ffill()
            sma200_s = close_s.rolling(window=200, min_periods=20).mean()
            if spy is not None and "Close" in spy.columns:
                spy_close_s = spy["Close"].reindex(valid_indices).ffill()
                spy_sma50_s = spy_close_s.rolling(window=50, min_periods=10).mean()
            else:
                spy_close_s = close_s
                spy_sma50_s = close_s

            # Point-in-time swing exhaustion indicators from feature frame
            rsi_s = (
                df_feat["RSI"].reindex(valid_indices).ffill()
                if "RSI" in df_feat.columns
                else pd.Series(50.0, index=valid_indices)
            )
            bb_pos_s = (
                df_feat["BB_Position"].reindex(valid_indices).ffill()
                if "BB_Position" in df_feat.columns
                else pd.Series(0.5, index=valid_indices)
            )

            current_pos = "FLAT"
            last_trade_idx = -10
            last_trade_price = 0.0

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

                p_sell, _p_hold, p_buy = (
                    float(prob_vec[0]),
                    float(prob_vec[1]),
                    float(prob_vec[2]),
                )

                cur_c = float(close_s.iloc[i])
                cur_rsi = float(rsi_s.iloc[i]) if pd.notna(rsi_s.iloc[i]) else 50.0
                cur_bb = float(bb_pos_s.iloc[i]) if pd.notna(bb_pos_s.iloc[i]) else 0.5
                cur_sma200 = (
                    float(sma200_s.iloc[i]) if pd.notna(sma200_s.iloc[i]) else cur_c
                )
                cur_spy = float(spy_close_s.iloc[i])
                cur_spy_sma50 = (
                    float(spy_sma50_s.iloc[i])
                    if pd.notna(spy_sma50_s.iloc[i])
                    else cur_spy
                )
                cur_l = float(low_s.iloc[i])
                cur_h = float(high_s.iloc[i])
                cur_o = float(open_s.iloc[i])

                # Mean-reversion swing exhaustion overlay
                is_dip_oversold = (cur_rsi < 30.0 or cur_bb < 0.05)
                is_peak_overbought = (cur_rsi > 75.0 or cur_bb > 0.95)

                # Local price action pivot detection (strictly causal, using t-2 and t-1)
                is_trough = False
                is_crest = False
                if i >= 2:
                    prev_l = float(low_s.iloc[i - 1])
                    prev2_l = float(low_s.iloc[i - 2])
                    prev_h = float(high_s.iloc[i - 1])
                    prev2_h = float(high_s.iloc[i - 2])

                    # More responsive trough detection: price rejection from recent lows
                    # Require candle to be green OR close in the top half of its range
                    is_bullish_candle = (cur_c > cur_o) or (cur_c > cur_l + (cur_h - cur_l) * 0.5)
                    is_trough = (
                        is_bullish_candle
                        and (cur_c > prev_l)
                        and (prev_l <= prev2_l or cur_l <= prev_l)
                    )
                    # More responsive crest detection: price rejection from recent highs
                    # Require candle to be red OR close in the bottom half of its range
                    is_bearish_candle = (cur_c < cur_o) or (cur_c < cur_l + (cur_h - cur_l) * 0.5)
                    is_crest = (
                        is_bearish_candle
                        and (cur_c < prev_h)
                        and (prev_h >= prev2_h or cur_h >= prev_h)
                    )


                eff_p_buy = (
                    p_buy
                    + (0.28 if is_dip_oversold else 0.0)
                    + (0.35 if is_trough else 0.0)
                )
                eff_p_sell = (
                    p_sell
                    + (0.28 if is_peak_overbought else 0.0)
                    + (0.35 if is_crest else 0.0)
                )

                if not is_crest:
                    eff_p_sell = 0.0
                if not is_trough:
                    eff_p_buy = 0.0

                # Conviction threshold >= 0.58
                if i < 30:
                    print(f"Bar {i} {bar_time}: is_trough={is_trough}, is_crest={is_crest}, p_buy={p_buy:.2f}, eff_p_buy={eff_p_buy:.2f}, eff_p_sell={eff_p_sell:.2f}")

                if eff_p_buy >= 0.58 and eff_p_buy > eff_p_sell:
                    raw_signal = "BUY"
                    conf = min(1.0, eff_p_buy)
                elif eff_p_sell >= 0.58 and eff_p_sell > eff_p_buy:
                    raw_signal = "SELL"
                    conf = min(1.0, eff_p_sell)
                else:
                    raw_signal = "HOLD"
                    conf = float(prob_vec[1])

                # Macro filter at bar t
                base_long_ok = (cur_c >= cur_sma200 * 0.85) and (cur_spy >= cur_spy_sma50 * 0.90)
                long_ok = True if (is_dip_oversold or is_trough) else base_long_ok
                short_ok = (cur_c < cur_sma200) or (cur_spy < cur_spy_sma50)

                # An existing LONG position can ALWAYS exit/take profit on a SELL signal.
                # Only opening a new naked SHORT when FLAT requires macro short_ok confirmation.
                if current_pos == "LONG":
                    sell_allowed = True
                else:
                    sell_allowed = short_ok

                if raw_signal == "BUY" and not long_ok:
                    filtered_signal = "HOLD"
                elif raw_signal == "SELL" and not sell_allowed:
                    filtered_signal = "HOLD"
                else:
                    filtered_signal = raw_signal

                # Price-Aware Adaptive Cooldown
                bars_since_trade = i - last_trade_idx
                cooldown_ok = False
                if bars_since_trade >= 3:
                    cooldown_ok = True
                elif bars_since_trade >= 1:
                    if (
                        current_pos != "LONG"
                        and last_trade_price > 0
                        and cur_c <= last_trade_price * 0.985
                    ):
                        cooldown_ok = True  # Bought dip >= 1.5% below prior trade price
                    elif (
                        current_pos == "LONG"
                        and last_trade_price > 0
                        and cur_c >= last_trade_price * 1.015
                    ):
                        cooldown_ok = True  # Exited rally >= 1.5% above prior entry price

                # State machine alternation: BUY -> SELL -> BUY
                # Target execution is Open[t+1]
                locs = df.index.get_indexer([valid_indices[i]])
                orig_idx = int(locs[0]) if len(locs) > 0 and locs[0] >= 0 else i
                if (
                    filtered_signal == "BUY"
                    and current_pos != "LONG"
                    and cooldown_ok
                ):
                    current_pos = "LONG"
                    last_trade_idx = i
                    last_trade_price = cur_c
                    if orig_idx + 1 < len(df):
                        next_idx = df.index[orig_idx + 1]
                        exec_target = (
                            next_idx.strftime("%Y-%m-%d")
                            if hasattr(next_idx, "strftime")
                            else str(next_idx)[:10]
                        )
                        exec_price = (
                            float(df["Open"].iloc[orig_idx + 1]) * 1.0005
                        )  # 5 bps slippage
                        exec_time_str = format_new_york_display(f"{exec_target} 09:30:00")
                    else:
                        exec_target = "NEXT_SESSION_OPEN"
                        exec_price = float(df["Close"].iloc[orig_idx]) * 1.0005
                        exec_time_str = "NEXT_SESSION_OPEN 09:30:00 ET"

                    self.signal_ledger.record_signal(
                        symbol=ticker,
                        bar_timestamp=bar_time,
                        signal="BUY",
                        confidence=conf,
                        execution_target_bar=exec_target,
                        execution_price=round(exec_price, 2),
                        model_version="Institutional_Mesh_V2.2_Swing",
                        raw_features_hash=SignalLedger.compute_features_hash(
                            scaled_rows[i]
                        ),
                        metadata={
                            "source": "causal_replay",
                            "rule": "Open[t+1] + 5bps",
                            "source_candle_timestamp": format_new_york_display(f"{bar_time} 16:00:00"),
                            "signal_generation_timestamp": format_new_york_display(f"{bar_time} 16:00:00"),
                            "execution_timestamp": exec_time_str,
                            "execution_price": round(exec_price, 2),
                            "signal_state": "CONFIRMED",
                        },
                    )

                elif (
                    filtered_signal == "SELL"
                    and current_pos == "LONG"
                    and cooldown_ok
                ):
                    current_pos = "FLAT"
                    last_trade_idx = i
                    last_trade_price = cur_c
                    if orig_idx + 1 < len(df):
                        next_idx = df.index[orig_idx + 1]
                        exec_target = (
                            next_idx.strftime("%Y-%m-%d")
                            if hasattr(next_idx, "strftime")
                            else str(next_idx)[:10]
                        )
                        exec_price = (
                            float(df["Open"].iloc[orig_idx + 1]) * 0.9995
                        )  # 5 bps slippage
                        exec_time_str = format_new_york_display(f"{exec_target} 09:30:00")
                    else:
                        exec_target = "NEXT_SESSION_OPEN"
                        exec_price = float(df["Close"].iloc[orig_idx]) * 0.9995
                        exec_time_str = "NEXT_SESSION_OPEN 09:30:00 ET"

                    self.signal_ledger.record_signal(
                        symbol=ticker,
                        bar_timestamp=bar_time,
                        signal="SELL",
                        confidence=conf,
                        execution_target_bar=exec_target,
                        execution_price=round(exec_price, 2),
                        model_version="Institutional_Mesh_V2.2_Swing",
                        raw_features_hash=SignalLedger.compute_features_hash(
                            scaled_rows[i]
                        ),
                        metadata={
                            "source": "causal_replay",
                            "rule": "Open[t+1] - 5bps",
                            "source_candle_timestamp": format_new_york_display(f"{bar_time} 16:00:00"),
                            "signal_generation_timestamp": format_new_york_display(f"{bar_time} 16:00:00"),
                            "execution_timestamp": exec_time_str,
                            "execution_price": round(exec_price, 2),
                            "signal_state": "CONFIRMED",
                        },
                    )

            # Record checkpoint for the latest evaluated bar if no BUY/SELL occurred on it
            if len(valid_indices) > 0:
                last_v_idx = valid_indices[-1]
                last_bar_time = (
                    last_v_idx.strftime("%Y-%m-%d")
                    if hasattr(last_v_idx, "strftime")
                    else str(last_v_idx)[:10]
                )
                self.signal_ledger.record_signal(
                    symbol=ticker,
                    bar_timestamp=last_bar_time,
                    signal="HOLD" if current_pos == "FLAT" else "HOLD_LONG",
                    confidence=0.5,
                    execution_target_bar="NEXT_SESSION_OPEN",
                    execution_price=round(float(df["Close"].loc[last_v_idx]), 2),
                    model_version="Institutional_Mesh_V2.2_Swing",
                    raw_features_hash="latest_bar_marker",
                    metadata={
                        "source": "causal_replay_checkpoint",
                        "source_candle_timestamp": format_new_york_display(f"{last_bar_time} 16:00:00"),
                        "signal_generation_timestamp": format_new_york_display(f"{last_bar_time} 16:00:00"),
                        "execution_timestamp": "NEXT_SESSION_OPEN 09:30:00 ET",
                        "signal_state": "CONFIRMED",
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
        force_recompute: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves verified causal markers from the immutable SignalLedger.
        If the ledger does not have existing signals for this ticker up to the latest completed bar,
        it executes a point-in-time sequential replay using strictly trailing completed data (t <= bar),
        commits the signals to the ledger, and returns them.
        """
        latest_completed_bar_date = None
        if ticker_df is not None and not ticker_df.empty:
            is_forming, _ = check_bar_forming_status(ticker_df)
            completed_slice = ticker_df.iloc[:-1] if is_forming and len(ticker_df) > 1 else ticker_df
            if not completed_slice.empty:
                last_idx = completed_slice.index[-1]
                latest_completed_bar_date = (
                    last_idx.strftime("%Y-%m-%d")
                    if hasattr(last_idx, "strftime")
                    else str(last_idx)[:10]
                )

        latest_ledger_bar = self.signal_ledger.get_latest_bar_timestamp(ticker)
        all_signals = self.signal_ledger.get_signals(
            symbol=ticker, actions_filter=["BUY", "SELL"]
        )

        if force_recompute or len(all_signals) < 2:
            self.signal_ledger.clear_historical_replay(ticker)
            return self._replay_causal_ml_signals(ticker, ticker_df, spy_df)

        if (
            latest_completed_bar_date is not None
            and latest_ledger_bar is not None
            and latest_ledger_bar >= latest_completed_bar_date
        ):
            all_signals = self.signal_ledger.get_signals(
                symbol=ticker, actions_filter=["BUY", "SELL"]
            )
            # Only return markers up to the completed bar; forming bar is NEVER included
            return [s for s in all_signals if s.get("bar_timestamp", "") <= latest_completed_bar_date]

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

        # Position awareness: Exiting an existing LONG position in paper trading is a risk-reducing action,
        # never a counter-trend naked short.
        is_long_exit = bool(
            self.paper_engine
            and isinstance(getattr(self.paper_engine, "positions", None), dict)
            and ticker in self.paper_engine.positions
            and isinstance(self.paper_engine.positions[ticker], dict)
            and self.paper_engine.positions[ticker].get("shares", 0) > 0
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
        elif pre_signal == "SELL" and not short_allowed and not is_long_exit:
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
                else str(last_idx)[:10]
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

        # Causal Signal States Mandate: CONFIRMED vs PROVISIONAL
        if is_bar_forming:
            signal_state = "PROVISIONAL"
            provisional_signal = final_signal if final_signal in ["BUY", "SELL"] else None
            confirmed_signal = "HOLD"
            provisional_marker = {
                "time": bar_date,
                "action": provisional_signal,
                "position": "belowBar" if provisional_signal == "BUY" else "aboveBar",
                "color": "#F59E0B",
                "shape": "arrowUp" if provisional_signal == "BUY" else "arrowDown",
                "text": f"PROVISIONAL {provisional_signal} (UNCONFIRMED)",
                "signal_state": "PROVISIONAL",
                "is_provisional": True,
                "source_candle_timestamp": f"{bar_date} (FORMING)",
                "execution_target": "PENDING_BAR_CLOSE",
                "execution_price": exec_price,
            } if provisional_signal else None
            sig_gen_time = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (INTRADAY)"
            source_candle_time = f"{bar_date} (FORMING)"
            exec_time = "PENDING_BAR_CLOSE"
        else:
            signal_state = "CONFIRMED"
            provisional_signal = None
            provisional_marker = None
            confirmed_signal = final_signal
            sig_gen_time = format_new_york_display(f"{bar_date} 16:00:00")
            source_candle_time = format_new_york_display(f"{bar_date} 16:00:00")
            exec_time = "NEXT_SESSION_OPEN 09:30:00 ET"

        response_data["signal"] = confirmed_signal
        response_data["signal_state"] = signal_state
        response_data["is_bar_forming"] = is_bar_forming
        response_data["bar_state"] = bar_state
        response_data["provisional"] = is_bar_forming
        response_data["provisional_signal"] = provisional_signal
        response_data["provisional_marker"] = provisional_marker
        response_data["source_candle_timestamp"] = source_candle_time
        response_data["signal_generation_timestamp"] = sig_gen_time
        response_data["execution_timestamp"] = exec_time
        response_data["execution_target_bar"] = execution_target
        response_data["execution_price"] = exec_price

        # Zero-Repainting Mandate: Commit confirmed signals to immutable SQLite ledger
        if not is_bar_forming and confirmed_signal in ["BUY", "SELL", "HOLD"]:
            features_hash = SignalLedger.compute_features_hash(tabular_row)
            self.signal_ledger.record_signal(
                symbol=ticker,
                bar_timestamp=bar_date,
                signal=confirmed_signal,
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
                    "source_candle_timestamp": source_candle_time,
                    "signal_generation_timestamp": sig_gen_time,
                    "execution_timestamp": exec_time,
                    "execution_price": exec_price,
                    "signal_state": "CONFIRMED",
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

        # Preserve the causal fields after package_chart_data update
        response_data["signal"] = confirmed_signal
        response_data["signal_state"] = signal_state
        response_data["is_bar_forming"] = is_bar_forming
        response_data["bar_state"] = bar_state
        response_data["provisional_signal"] = provisional_signal
        response_data["provisional_marker"] = provisional_marker
        response_data["source_candle_timestamp"] = source_candle_time
        response_data["signal_generation_timestamp"] = sig_gen_time
        response_data["execution_timestamp"] = exec_time
        response_data["execution_target_bar"] = execution_target
        response_data["execution_price"] = exec_price

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
