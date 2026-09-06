import numpy as np
import pandas as pd

from src.data_ingestion.technical_indicators import (
    add_advanced_features,
    clean_multiindex_columns,
)


class ReportGenerator:
    """
    Handles generation of historical markers, chart data, and AI reports
    to decouple logic from the primary API routing.
    """

    def __init__(self, kept_features_list):
        self.kept_features_list = kept_features_list

    def generate_historical_markers(self, ticker, df_raw):
        """
        Detects swing highs and lows to provide context.
        """
        df_full = clean_multiindex_columns(df_raw.copy())

        # Detect Pivots
        prices = df_full["Close"].values
        highs = df_full["High"].values
        lows = df_full["Low"].values
        dates = df_full.index.strftime("%Y-%m-%d").tolist()

        window = 3
        markers = []

        for i in range(window, len(prices) - window):
            action = "SKIP"
            if lows[i] == np.min(lows[i - window : i + window + 1]):
                action = "BUY"
            elif highs[i] == np.max(highs[i - window : i + window + 1]):
                action = "SELL"

            if action != "SKIP":
                markers.append(
                    {
                        "time": dates[i],
                        "action": action,
                        "label": action,
                        "probability": 100,
                    }
                )
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
        for col in ["ribbon_upper", "ribbon_lower", "bb_upper", "bb_lower"]:
            df_cloud_json[col] = df_cloud_json[col].replace(0.0, np.nan)

        # Only drop if ALL essential indicators are missing
        df_cloud_json = df_cloud_json.dropna(
            subset=["ribbon_upper", "ribbon_lower", "bb_upper", "bb_lower"], how="all"
        )

        clouds = df_cloud_json[
            ["time", "ribbon_upper", "ribbon_lower", "bb_upper", "bb_lower"]
        ].replace({np.nan: None}).to_dict(orient="records")

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
