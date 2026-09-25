import numpy as np
import pandas as pd
import ta

from src.data_ingestion.technical_indicators import (
    add_advanced_features,
    clean_multiindex_columns,
)


class ReportGenerator:
    """
    Handles generation of historical markers, chart data, and AI reports
    to decouple logic from the primary API routing.

    Implements a 3-layer quantitative strategy architecture:
    1. Trend Alignment & Macro Directional Filter (Moving average hierarchy & momentum slope)
    2. State-Machine Cooldown / Refractory Period (Eliminates signal clustering & over-allocation)
    3. Dynamic ATR-Ratcheting Trailing Exits (Eliminates premature peak-fading exits)
    """

    def __init__(
        self,
        kept_features_list,
        k: int = 3,
        cooldown_bars: int = 7,
        atr_period: int = 14,
        trail_mult: float = 2.5,
    ):
        self.kept_features_list = kept_features_list
        self.k = k
        self.cooldown_bars = cooldown_bars
        self.atr_period = atr_period
        self.trail_mult = trail_mult

    def generate_historical_markers(
        self,
        ticker,
        df_raw,
        k: int | None = None,
        cooldown_bars: int | None = None,
        atr_period: int | None = None,
        trail_mult: float | None = None,
    ):
        """
        Detects causal, non-repainting historical trade execution markers
        and dynamic ATR trailing stop overlays governed by three algorithmic layers:
        1. Trend Alignment & Macro Directional Filter
        2. State-Machine Cooldown (Refractory Period)
        3. Dynamic ATR-Ratcheting Trailing Exits
        """
        k = k if k is not None else self.k
        cooldown_bars = cooldown_bars if cooldown_bars is not None else self.cooldown_bars
        atr_period = atr_period if atr_period is not None else self.atr_period
        trail_mult = trail_mult if trail_mult is not None else self.trail_mult

        df_full = clean_multiindex_columns(df_raw.copy())
        n = len(df_full)
        min_required_bars = max(k + 1, atr_period + 1, 24)

        if df_full.empty or n < min_required_bars:
            df_full["trailing_stop"] = np.nan
            return [], df_full

        df_features = add_advanced_features(df_full.copy())
        df_features = clean_multiindex_columns(df_features)

        # =========================================================================
        # 1. INDICATOR DEFINITIONS
        # =========================================================================
        # Align features to df_full index to prevent shape mismatch if macro indicators drop edge rows
        if "Ribbon_Fast" in df_features.columns:
            fast_ma = (
                df_features["Ribbon_Fast"].reindex(df_full.index).ffill().bfill().values
            )
        else:
            fast_ma = (
                ta.trend.EMAIndicator(close=df_full["Close"], window=12)
                .ema_indicator()
                .values
            )

        if "Ribbon_Slow" in df_features.columns:
            slow_ma = (
                df_features["Ribbon_Slow"].reindex(df_full.index).ffill().bfill().values
            )
        else:
            slow_ma = (
                ta.trend.EMAIndicator(close=df_full["Close"], window=24)
                .ema_indicator()
                .values
            )

        # Volatility: ATR(14)
        if atr_period == 14 and "ATR" in df_features.columns:
            atr = df_features["ATR"].reindex(df_full.index).ffill().bfill().values
        else:
            atr_ins = ta.volatility.AverageTrueRange(
                high=df_full["High"],
                low=df_full["Low"],
                close=df_full["Close"],
                window=atr_period,
            )
            atr = atr_ins.average_true_range().ffill().bfill().values

        closes = df_full["Close"].values
        highs = df_full["High"].values
        lows = df_full["Low"].values
        dates = df_full.index.strftime("%Y-%m-%d").tolist()

        trailing_stop_series = np.full(n, np.nan)
        markers = []

        # =========================================================================
        # 2. STATE-MACHINE VARIABLES (Persistent across bars, strictly causal)
        # =========================================================================
        in_long_position = False
        current_long_stop = np.nan
        in_short_position = False
        current_short_stop = np.nan

        last_long_bar = -9999
        last_short_bar = -9999
        last_exit_bar = -9999

        start_idx = min_required_bars

        for t in range(start_idx, n):
            # Guard against any NaN values in calculation window
            if (
                np.isnan(fast_ma[t])
                or np.isnan(slow_ma[t])
                or np.isnan(fast_ma[t - k])
                or np.isnan(slow_ma[t - k])
                or np.isnan(atr[t])
            ):
                continue

            # Loop-scoped flag: Prevents same-bar exit and re-entry
            exit_triggered_this_bar = False

            # =====================================================================
            # LAYER 3: DYNAMIC ATR-RATCHETING TRAILING EXITS (Position Monitoring)
            # =====================================================================
            if in_long_position:
                # Exit Execution: Triggered if current bar Low breaches the active Stop
                if lows[t] <= current_long_stop:
                    markers.append(
                        {
                            "time": dates[t],
                            "action": "EXIT",
                            "label": f"Stop Exit @ {current_long_stop:.2f}",
                            "probability": 100,
                        }
                    )
                    in_long_position = False
                    current_long_stop = np.nan
                    last_exit_bar = t
                    exit_triggered_this_bar = True
                else:
                    # Ratcheting Mechanism: Stop_t = max(Stop_{t-1}, High_t - trail_mult * ATR)
                    # Constraint: Stop line can only move upward; never downward while long
                    candidate_stop = highs[t] - (trail_mult * atr[t])
                    current_long_stop = max(current_long_stop, candidate_stop)
                    trailing_stop_series[t] = current_long_stop

            elif in_short_position:
                # Exit Execution: Triggered if current bar High breaches the active Short Stop
                if highs[t] >= current_short_stop:
                    markers.append(
                        {
                            "time": dates[t],
                            "action": "EXIT",
                            "label": f"Short Stop Exit @ {current_short_stop:.2f}",
                            "probability": 100,
                        }
                    )
                    in_short_position = False
                    current_short_stop = np.nan
                    last_exit_bar = t
                    exit_triggered_this_bar = True
                else:
                    # Ratcheting Mechanism: Short Stop_t = min(Stop_{t-1}, Low_t + trail_mult * ATR)
                    # Constraint: Stop line can only move downward; never upward while short
                    candidate_short_stop = lows[t] + (trail_mult * atr[t])
                    current_short_stop = min(current_short_stop, candidate_short_stop)
                    trailing_stop_series[t] = current_short_stop

            # =====================================================================
            # LAYER 1: TREND ALIGNMENT & MACRO DIRECTIONAL FILTER
            # =====================================================================
            # Slope Calculation over lookback window k: slope(MA) = MA_t - MA_{t-k}
            slope_fast = fast_ma[t] - fast_ma[t - k]
            slope_slow = slow_ma[t] - slow_ma[t - k]

            # Long Entry Filter:
            # 1. fast_ma > slow_ma
            # 2. slope(fast_ma) > 0 and slope(slow_ma) > 0
            # 3. Close > slow_ma
            long_trend_aligned = (
                (fast_ma[t] > slow_ma[t])
                and (slope_fast > 0)
                and (slope_slow > 0)
                and (closes[t] > slow_ma[t])
            )

            # Short Entry Filter:
            # - Hard Prohibition: Completely suppress all SELL/Short signals while Close > slow_ma
            # - Valid only if Close < slow_ma AND fast_ma < slow_ma AND slope(slow_ma) < 0
            short_prohibited = closes[t] > slow_ma[t]
            short_trend_aligned = (
                (not short_prohibited)
                and (closes[t] < slow_ma[t])
                and (fast_ma[t] < slow_ma[t])
                and (slope_slow < 0)
            )

            # Technical Entry Triggers (Confirmed on bar close):
            # Bullish: Golden cross OR pullback bounce off the moving average ribbon OR confirmed pivot low
            raw_buy_trigger = (
                (fast_ma[t] > slow_ma[t] and fast_ma[t - 1] <= slow_ma[t - 1])
                or (lows[t - 1] <= fast_ma[t - 1] and closes[t] > fast_ma[t] and closes[t] > closes[t - 1])
                or (lows[t - 1] <= lows[t - 2] and closes[t] > highs[t - 1])
            )

            # Bearish: Death cross OR pullback rejection at ribbon OR confirmed pivot high
            raw_sell_trigger = (
                (fast_ma[t] < slow_ma[t] and fast_ma[t - 1] >= slow_ma[t - 1])
                or (highs[t - 1] >= fast_ma[t - 1] and closes[t] < fast_ma[t] and closes[t] < closes[t - 1])
                or (highs[t - 1] >= highs[t - 2] and closes[t] < lows[t - 1])
            )

            # =====================================================================
            # LAYER 2: STATE-MACHINE COOLDOWN & ORDER EXECUTION
            # =====================================================================
            # Refractory Period Check: Must be at least cooldown_bars since the last stop exit
            post_exit_ok = (t - last_exit_bar) >= cooldown_bars

            # Process Long Entry
            if (
                raw_buy_trigger
                and long_trend_aligned
                and not in_long_position
                and not exit_triggered_this_bar
                and post_exit_ok
            ):
                # State Gating: Suppress any new BUY if (current_bar - last_long_bar) < cooldown_bars
                if (t - last_long_bar) >= cooldown_bars:
                    # If in short position, reverse exit first
                    if in_short_position:
                        markers.append(
                            {
                                "time": dates[t],
                                "action": "EXIT",
                                "label": "Reverse Exit",
                                "probability": 100,
                            }
                        )
                        in_short_position = False
                        current_short_stop = np.nan
                        last_exit_bar = t

                    in_long_position = True
                    entry_price = closes[t]
                    last_long_bar = t
                    last_short_bar = -9999

                    # Initial Stop on Entry: Stop_0 = Entry Price - (trail_mult * ATR_14)
                    current_long_stop = entry_price - (trail_mult * atr[t])
                    trailing_stop_series[t] = current_long_stop

                    markers.append(
                        {
                            "time": dates[t],
                            "action": "BUY",
                            "label": f"BUY (Entry: {entry_price:.2f})",
                            "probability": 100,
                        }
                    )

            # Process Short Entry / Reverse Exit
            elif (
                raw_sell_trigger
                and short_trend_aligned
                and not short_prohibited
                and not in_short_position
                and not exit_triggered_this_bar
                and post_exit_ok
            ):
                # State Gating: Suppress any new SELL if (current_bar - last_short_bar) < cooldown_bars
                if (t - last_short_bar) >= cooldown_bars:
                    # If long position is active, reverse exit long first
                    if in_long_position:
                        markers.append(
                            {
                                "time": dates[t],
                                "action": "EXIT",
                                "label": "Reverse Exit",
                                "probability": 100,
                            }
                        )
                        in_long_position = False
                        current_long_stop = np.nan
                        last_exit_bar = t

                    in_short_position = True
                    entry_price = closes[t]
                    last_short_bar = t
                    last_long_bar = -9999

                    # Initial Stop on Short Entry: Stop_0 = Entry Price + (trail_mult * ATR_14)
                    current_short_stop = entry_price + (trail_mult * atr[t])
                    trailing_stop_series[t] = current_short_stop

                    markers.append(
                        {
                            "time": dates[t],
                            "action": "SELL",
                            "label": f"SELL (Close: {closes[t]:.2f})",
                            "probability": 100,
                        }
                    )

        # Attach trailing stop to df_full for chart visualization
        df_full["trailing_stop"] = trailing_stop_series
        return markers, df_full

    def package_chart_data(
        self, ticker, df_full, ai_report_dict, historical_markers, system_signals=None
    ):
        """
        Formats data for the Next.js institutional dashboard.
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

        # Merge System Signals (from Journal) with Historical Pivots
        final_markers = historical_markers.copy()
        if system_signals is not None and not system_signals.empty:
            for _, sig in system_signals.iterrows():
                # Convert timestamp to date string
                sig_time = pd.to_datetime(sig["timestamp"]).strftime("%Y-%m-%d")
                # Avoid duplicates with historical markers on same date
                if not any(m["time"] == sig_time for m in final_markers):
                    final_markers.append(
                        {
                            "time": sig_time,
                            "action": sig["signal_type"],
                            "label": f"Hydra {sig['signal_type']}",
                            "probability": sig["confidence"],
                        }
                    )

        # Filter out markers that fall before our 150-day chart window
        min_date = df_chart["time"].min()
        final_markers = [m for m in final_markers if m["time"] >= min_date]

        return {
            "candles": candles,
            "clouds": clouds,
            "ai_report": ai_report_dict,
            "historical_markers": final_markers,
        }

