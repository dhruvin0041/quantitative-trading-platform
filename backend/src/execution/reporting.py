import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

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

    def generate_historical_markers(
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
        Detects continuous raw momentum trade execution markers governed purely by localized
        price action relative to the moving average ribbon and volatility envelope:
        - BUY: Close > slow_ma, Low <= fast_ma, Close > Open
        - SELL: (Close < fast_ma OR High >= upper_band), Close < Open
        """
        k = k if k is not None else self.k
        atr_period = atr_period if atr_period is not None else self.atr_period

        df_full = clean_multiindex_columns(df_raw.copy())
        n = len(df_full)
        min_required_bars = max(k + 1, atr_period + 1, 24)

        if df_full.empty or n < min_required_bars:
            df_full["trailing_stop"] = np.nan
            return [], df_full

        # =========================================================================
        # INDICATOR DEFINITIONS (MA Ribbon & Upper Volatility Band)
        # =========================================================================
        df_advanced = add_advanced_features(df_full.copy())
        df_advanced = clean_multiindex_columns(df_advanced)

        if "Ribbon_Fast" in df_advanced.columns:
            fast_ma = df_advanced["Ribbon_Fast"].reindex(df_full.index).ffill().bfill().values
        else:
            fast_ma = ta.trend.EMAIndicator(close=df_full["Close"], window=12).ema_indicator().values

        if "Ribbon_Slow" in df_advanced.columns:
            slow_ma = df_advanced["Ribbon_Slow"].reindex(df_full.index).ffill().bfill().values
        else:
            slow_ma = ta.trend.EMAIndicator(close=df_full["Close"], window=24).ema_indicator().values

        # Upper Volatility Band (BB_120_Upper or BB_Upper)
        if "BB_120_Upper" in df_advanced.columns and not df_advanced["BB_120_Upper"].dropna().empty:
            upper_band = df_advanced["BB_120_Upper"].reindex(df_full.index).ffill().bfill().values
        elif "BB_Upper" in df_advanced.columns and not df_advanced["BB_Upper"].dropna().empty:
            upper_band = df_advanced["BB_Upper"].reindex(df_full.index).ffill().bfill().values
        else:
            bb = ta.volatility.BollingerBands(close=df_full["Close"], window=min(20, n), window_dev=2)
            upper_band = bb.bollinger_hband().ffill().bfill().values

        opens = df_full["Open"].values
        closes = df_full["Close"].values
        highs = df_full["High"].values
        lows = df_full["Low"].values
        dates = df_full.index.strftime("%Y-%m-%d").tolist()

        # =========================================================================
        # EDGE-TRIGGERED MOMENTUM SIGNALS WITH DEBOUNCE (ANTI-SPAM)
        # =========================================================================
        valid_ma = ~np.isnan(fast_ma) & ~np.isnan(slow_ma)

        last_buy_bar = -999
        last_sell_bar = -999

        markers: List[Dict[str, Any]] = []

        for t in range(20, n):
            if not valid_ma[t]:
                continue

            is_green = closes[t] > opens[t]
            is_red = closes[t] < opens[t]

            upper_band_val_t1 = (
                float(upper_band[t - 1])
                if not np.isnan(upper_band[t - 1])
                else float("inf")
            )
            upper_band_val_t = (
                float(upper_band[t])
                if not np.isnan(upper_band[t])
                else float("inf")
            )

            # 1. Edge-Triggered BUY (Dip & Bounce Pivot)
            # Tests support/ribbon
            zone_test_buy = (lows[t - 1] <= fast_ma[t - 1]) or (lows[t] <= fast_ma[t])
            bullish_regime = closes[t] > slow_ma[t]
            # 2-bar bullish reversal
            reversal_buy = is_green and (closes[t] > highs[t - 1])
            # Must be a dip, NOT already overextended
            buy_triggered = (
                bullish_regime
                and zone_test_buy
                and reversal_buy
            )

            # 2. Edge-Triggered SELL (Peak & Exhaustion Pivot)
            # Overextension: tested upper band OR extended above fast MA AND strictly above slow MA
            overextended = (
                (highs[t - 1] >= upper_band_val_t1 * 0.995)
                or (highs[t] >= upper_band_val_t * 0.995)
                or (
                    highs[t - 1] > fast_ma[t - 1] * 1.025
                    and highs[t - 1] > slow_ma[t - 1] * 1.01
                )
            )
            above_slow = highs[t - 1] > slow_ma[t - 1]
            reversal_sell = is_red and (closes[t] < lows[t - 1])

            sell_triggered = overextended and above_slow and reversal_sell

            if buy_triggered and (t - last_buy_bar) >= 3:
                markers.append(
                    {
                        "time": dates[t],
                        "action": "BUY",
                        "label": "",
                        "text": "",
                        "probability": 100,
                        "price": round(float(closes[t]), 2),
                    }
                )
                last_buy_bar = t
            elif sell_triggered and (t - last_sell_bar) >= 3:
                markers.append(
                    {
                        "time": dates[t],
                        "action": "SELL",
                        "label": "",
                        "text": "",
                        "probability": 100,
                        "price": round(float(closes[t]), 2),
                    }
                )
                last_sell_bar = t

        df_full["trailing_stop"] = np.nan
        return markers, df_full

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
