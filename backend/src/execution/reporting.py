import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from src.data_ingestion.technical_indicators import (
    add_advanced_features,
    clean_multiindex_columns,
)
from src.execution.asset_intelligence import AssetExpectancyFilter
from src.execution.live_inference import FEATURE_COLUMNS

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("ReportGenerator")


class ReportGenerator:
    """
    Handles generation of historical markers, chart data, and AI reports
    to decouple logic from the primary API routing.

    Implements a 3-layer quantitative strategy architecture:
    1. Trend Alignment & Macro Directional Filter (Moving average hierarchy, momentum slope, SPY >= SMA50)
    2. State-Machine Cooldown / Refractory Period (Eliminates signal clustering & over-allocation)
    3. Dynamic ATR-Ratcheting Trailing Exits (Beta-calibrated, eliminates premature peak-fading exits)
    """

    def __init__(
        self,
        kept_features_list: Optional[List[str]] = None,
        k: int = 3,
        cooldown_bars: int = 7,
        atr_period: int = 14,
        trail_mult: float = 2.5,
        xgb_model: Any = None,
        scaler: Any = None,
        expectancy_filter: Optional[AssetExpectancyFilter] = None,
    ):
        self.kept_features_list = kept_features_list
        self.kept_features = list(kept_features_list) if kept_features_list else None
        self.k = k
        self.cooldown_bars = cooldown_bars
        self.atr_period = atr_period
        self.trail_mult = trail_mult
        self.xgb_model = xgb_model
        self.scaler = scaler

        self.expectancy_filter = expectancy_filter or AssetExpectancyFilter(
            lookback_days=90,
            min_trades=4,
            trip_hurdle=1.15,
            recovery_hurdle=1.20,
        )

        self._load_production_artifacts()

    def _load_production_artifacts(self) -> None:
        """
        Dynamically loads the production model artifacts in-process:
        - Scaler: backend/artifacts/latest_scaler.joblib
        - Flagship Alpha: backend/artifacts/xgb_ensemble.json
        - Features: backend/artifacts/kept_features.json (or configs/kept_features.json)
        """
        artifacts_dir = BACKEND_DIR / "artifacts"
        configs_dir = BACKEND_DIR / "configs"

        # 1. Kept Features Definition
        if not self.kept_features:
            for kf_path in [
                artifacts_dir / "kept_features.json",
                configs_dir / "kept_features.json",
                Path("configs/kept_features.json"),
                Path("artifacts/kept_features.json"),
            ]:
                if kf_path.exists():
                    try:
                        with open(kf_path, "r") as f:
                            self.kept_features = json.load(f)
                            break
                    except Exception as e:
                        logger.warning("Could not load kept_features from %s: %s", kf_path, e)

        if not self.kept_features:
            try:
                self.kept_features = list(FEATURE_COLUMNS)
            except Exception:
                self.kept_features = []

        # 2. Production Feature Scaler
        if self.scaler is None and joblib is not None:
            for s_path in [
                artifacts_dir / "latest_scaler.joblib",
                Path("artifacts/latest_scaler.joblib"),
                Path("backend/artifacts/latest_scaler.joblib"),
            ]:
                if s_path.exists():
                    try:
                        self.scaler = joblib.load(str(s_path))
                        break
                    except Exception as e:
                        logger.warning("Could not load scaler from %s: %s", s_path, e)

        # 3. Flagship Alpha: XGBoost Ensemble
        if self.xgb_model is None and xgb is not None:
            for x_path in [
                artifacts_dir / "xgb_ensemble.json",
                Path("artifacts/xgb_ensemble.json"),
                Path("backend/artifacts/xgb_ensemble.json"),
            ]:
                if x_path.exists():
                    try:
                        model = xgb.XGBClassifier()
                        model.load_model(str(x_path))
                        self.xgb_model = model
                        break
                    except Exception as e:
                        logger.warning("Could not load XGBoost model from %s: %s", x_path, e)

    def generate_post_hoc_zigzag_overlay(
        self,
        ticker: str,
        df_raw: pd.DataFrame,
        k: Optional[int] = None,
        cooldown_bars: Optional[int] = None,
        atr_period: Optional[int] = None,
        trail_mult: Optional[float] = None,
        spy_df: Optional[pd.DataFrame] = None,
        vix_df: Optional[pd.DataFrame] = None,
        mock_probabilities: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Post-Hoc Retrospective ZigZag Overlay (EX-POST ANALYSIS ONLY).

        WARNING - NON-CAUSAL LOGIC (INTENTIONAL LOOK-AHEAD BIAS):
        This function identifies swing highs and swing lows by scanning future bars
        (`highs[i + r_offset]`, `lows[i + r_offset]`). It creates retrospective peak/trough
        markers that were impossible to know at time t, and repaints dynamically at the right
        edge as new highs or lows develop.

        CRITICAL ARCHITECTURAL POLICY:
        - MUST NEVER be used for live signal generation or automated order execution.
        - MUST NEVER drive primary real-time candlestick chart markers.
        - Strictly reserved for retrospective post-trade visual analysis overlays.
        """
        k = k if k is not None else self.k
        atr_period = atr_period if atr_period is not None else self.atr_period

        df_full = clean_multiindex_columns(df_raw.copy())
        n = len(df_full)
        min_required_bars = max(k + 1, atr_period + 1, 20)

        if df_full.empty or n < min_required_bars:
            df_full["trailing_stop"] = np.nan
            return [], df_full

        closes = df_full["Close"].values
        highs = df_full["High"].values
        lows = df_full["Low"].values
        dates = df_full.index.strftime("%Y-%m-%d").tolist()

        # Dynamic ATR for volatility-adjusted minimum return threshold
        high_s = df_full["High"]
        low_s = df_full["Low"]
        close_s = df_full["Close"]
        tr1 = high_s - low_s
        tr2 = (high_s - close_s.shift(1)).abs()
        tr3 = (low_s - close_s.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_vals = tr.rolling(atr_period).mean().bfill().values

        left_bars = 5
        right_bars = 5
        min_swing_bars = 5
        min_pct_base = 0.045

        # 1. Detect candidate Swing Highs (Peaks) and Swing Lows (Dips)
        pivots: List[Tuple[int, str, float]] = []

        for i in range(left_bars, n - right_bars):
            # Peak test: higher than left and right neighbors
            is_ph = True
            for l_offset in range(1, left_bars + 1):
                if highs[i] < highs[i - l_offset]:
                    is_ph = False
                    break
            if is_ph:
                for r_offset in range(1, right_bars + 1):
                    if highs[i] < highs[i + r_offset]:
                        is_ph = False
                        break

            # Dip test: lower than left and right neighbors
            is_pl = True
            for l_offset in range(1, left_bars + 1):
                if lows[i] > lows[i - l_offset]:
                    is_pl = False
                    break
            if is_pl:
                for r_offset in range(1, right_bars + 1):
                    if lows[i] > lows[i + r_offset]:
                        is_pl = False
                        break

            if is_ph and not is_pl:
                pivots.append((i, "SELL", float(highs[i])))
            elif is_pl and not is_ph:
                pivots.append((i, "BUY", float(lows[i])))

        # 2. Strict Alternation State Machine (BUY -> SELL -> BUY -> SELL)
        signals: List[Tuple[int, str, float]] = []
        current_action = None

        for idx, action, price in pivots:
            cur_price = closes[idx]
            min_return_pct = max(min_pct_base, float((2.2 * atr_vals[idx]) / max(cur_price, 1e-4)))

            if current_action is None:
                if action == "BUY":
                    current_action = "BUY"
                    signals.append((idx, action, price))
            elif action == current_action:
                # Same action in a row: pick the best price
                # For BUY: lower dip is better
                # For SELL: higher peak is better
                prev_idx, prev_action, prev_price = signals[-1]
                if action == "BUY" and price < prev_price:
                    signals[-1] = (idx, action, price)
                elif action == "SELL" and price > prev_price:
                    signals[-1] = (idx, action, price)
            else:
                # Opposite action: check minimum swing length and price displacement
                prev_idx, prev_action, prev_price = signals[-1]
                if (idx - prev_idx) < min_swing_bars:
                    continue

                if action == "SELL":
                    # Must have risen from previous BUY
                    if price > prev_price * (1.0 + min_return_pct) or (idx - prev_idx) >= 7:
                        signals.append((idx, action, price))
                        current_action = "SELL"
                elif action == "BUY":
                    # Must have fallen from previous SELL
                    if price < prev_price * (1.0 - min_return_pct) or (idx - prev_idx) >= 7:
                        signals.append((idx, action, price))
                        current_action = "BUY"

        # 3. Right-Edge Handling (Unconfirmed recent bars at the right edge)
        if current_action == "BUY" and signals:
            prev_idx, prev_action, prev_price = signals[-1]
            min_return_pct = max(min_pct_base, float((2.2 * atr_vals[-1]) / max(closes[-1], 1e-4)))
            recent_window = list(range(max(prev_idx + min_swing_bars, n - right_bars), n))
            if recent_window:
                best_high_idx = max(recent_window, key=lambda k: highs[k])
                if highs[best_high_idx] > prev_price * (1.0 + min_return_pct):
                    signals.append((best_high_idx, "SELL", float(highs[best_high_idx])))
        elif current_action == "SELL" and signals:
            prev_idx, prev_action, prev_price = signals[-1]
            min_return_pct = max(min_pct_base, float((2.2 * atr_vals[-1]) / max(closes[-1], 1e-4)))
            recent_window = list(range(max(prev_idx + min_swing_bars, n - right_bars), n))
            if recent_window:
                best_low_idx = min(recent_window, key=lambda k: lows[k])
                if lows[best_low_idx] < prev_price * (1.0 - min_return_pct):
                    signals.append((best_low_idx, "BUY", float(lows[best_low_idx])))

        # 4. Construct Markers
        markers: List[Dict[str, Any]] = []
        for idx, action, price in signals:
            markers.append(
                {
                    "time": dates[idx],
                    "action": action,
                    "label": "Post-Hoc Pivot (Non-Causal)",
                    "text": "",
                    "probability": 100,
                    "price": round(float(price), 2),
                }
            )

        df_full["trailing_stop"] = np.nan
        return markers, df_full

    def generate_historical_markers(
        self,
        ticker: str,
        df_raw: pd.DataFrame,
        *args,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        DEPRECATED: Retrospective swing pivot marker generator.
        Redirects to `generate_post_hoc_zigzag_overlay`.
        Notice: Contains look-ahead bias (highs[i + r_offset]). For live/causal trading,
        use `SignalLedger` or causal point-in-time ML signals.
        """
        logger.warning(
            "[DEPRECATION WARNING] `generate_historical_markers` was called. "
            "This logic contains retrospective look-ahead bias and is deprecated for live execution. "
            "Redirecting to `generate_post_hoc_zigzag_overlay`."
        )
        return self.generate_post_hoc_zigzag_overlay(ticker, df_raw, *args, **kwargs)


    def package_chart_data(
        self, ticker, df_full, ai_report_dict, historical_markers, system_signals=None
    ):
        """
        Formats data for the Next.js institutional dashboard.
        Returns continuous un-gated historical_markers without filtering out
        consecutive signals.
        """
        df_full = clean_multiindex_columns(df_full)
        df_features = add_advanced_features(df_full.copy())
        df_features = clean_multiindex_columns(df_features)

        # Data Alignment
        df_full["ribbon_upper"] = df_features["Ribbon_Fast"]
        df_full["ribbon_lower"] = df_features["Ribbon_Slow"]
        df_full["bb_upper"] = df_features["BB_120_Upper"]
        df_full["bb_lower"] = df_features["BB_120_Lower"]
        if "trailing_stop" not in df_full.columns:
            df_full["trailing_stop"] = np.nan

        df_chart = df_full.reset_index()
        date_col = "Date" if "Date" in df_chart.columns else "index"

        # Filter for 2026 onwards for UI clarity if 2026 data exists
        df_2026 = df_chart[df_chart[date_col] >= pd.Timestamp("2026-01-01")]
        if not df_2026.empty:
            df_chart = df_2026

        df_chart["time"] = df_chart[date_col].dt.strftime("%Y-%m-%d")
        df_chart = df_chart.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        candles = df_chart[["time", "open", "high", "low", "close", "volume"]].to_dict(
            orient="records"
        )

        # Clouds: include all timestamps that have at least one indicator to prevent early termination
        df_cloud_json = df_chart.copy()

        # Replace 0.0 with np.nan for chart indicators (caused by ML fillna)
        for col in ["ribbon_upper", "ribbon_lower", "bb_upper", "bb_lower", "trailing_stop"]:
            if col in df_cloud_json.columns:
                df_cloud_json[col] = df_cloud_json[col].replace(0.0, np.nan)

        # Only drop if ALL essential indicators are missing
        df_cloud_json = df_cloud_json.dropna(
            subset=["ribbon_upper", "ribbon_lower", "bb_upper", "bb_lower"], how="all"
        )

        cloud_cols = ["time", "ribbon_upper", "ribbon_lower", "bb_upper", "bb_lower"]
        if "trailing_stop" in df_cloud_json.columns:
            cloud_cols.append("trailing_stop")

        clouds = df_cloud_json[cloud_cols].replace({np.nan: None}).to_dict(orient="records")

        # Single Source of Truth: historical_markers from the simulation loop ONLY.
        raw_markers = list(historical_markers) if historical_markers else []

        # Filter out markers that fall before our chart window, strictly BUY and SELL
        min_date = (
            str(df_chart["time"].min())
            if not df_chart.empty and pd.notna(df_chart["time"].min())
            else ""
        )
        window_markers = [
            m
            for m in raw_markers
            if m.get("time")
            and (not min_date or str(m["time"]) >= min_date)
            and m.get("action") in ["BUY", "SELL"]
        ]
        window_markers.sort(key=lambda m: m["time"])

        return {
            "candles": candles,
            "clouds": clouds,
            "ai_report": ai_report_dict,
            "historical_markers": window_markers,
            "markers": window_markers,
        }
