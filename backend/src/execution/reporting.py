import numpy as np
from datetime import datetime, timedelta
from src.data_ingestion.market_data import fetch_historical_data
from src.data_ingestion.technical_indicators import add_advanced_features

class ReportGenerator:
    """
    Handles generation of historical markers, chart data, and AI reports
    to decouple logic from the primary API routing.
    """
    def __init__(self, kept_features_list):
        self.kept_features_list = kept_features_list

    def generate_historical_markers(self, ticker, df_raw):
        """
        Detects swing highs and lows over a 5-year period to provide context.
        """
        # NEW: Fetch 5 years of data for historical context
        start_date = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
        df_full = fetch_historical_data(
            ticker, start_date=start_date, end_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Detect Pivots
        prices = df_full["Close"].values
        highs = df_full["High"].values
        lows = df_full["Low"].values
        dates = df_full.index.strftime('%Y-%m-%d').tolist()
        
        window = 10
        markers = []
        
        for i in range(window, len(prices) - window):
            action = "SKIP"
            if lows[i] == np.min(lows[i - window : i + window + 1]):
                action = "BUY"
            elif highs[i] == np.max(highs[i - window : i + window + 1]):
                action = "SELL"
                
            if action != "SKIP":
                markers.append({
                    "time": dates[i],
                    "action": action,
                    "label": action,
                    "probability": 100
                })
        return markers, df_full

    def package_chart_data(self, ticker, df_full, xai_log, historical_markers):
        """
        Formats data for the Next.js institutional dashboard.
        """
        df_features = add_advanced_features(df_full.copy())
        
        # Data Alignment
        df_full['ribbon_upper'] = df_features['Ribbon_Fast']
        df_full['ribbon_lower'] = df_features['Ribbon_Slow']
        df_full['bb_upper'] = df_features['BB_120_Upper']
        df_full['bb_lower'] = df_features['BB_120_Lower']
        
        df_chart = df_full.reset_index()
        date_col = "Date" if "Date" in df_chart.columns else "index"
        df_chart["time"] = df_chart[date_col].dt.strftime("%Y-%m-%d")
        df_chart = df_chart.rename(
            columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
        )
        
        candles = df_chart[["time", "open", "high", "low", "close"]].to_dict(orient="records")
        
        df_cloud_json = df_chart.dropna(subset=["ribbon_upper", "ribbon_lower", "bb_upper", "bb_lower"])
        clouds = df_cloud_json[["time", "ribbon_upper", "ribbon_lower", "bb_upper", "bb_lower"]].to_dict(orient="records")
        
        return {
            "candles": candles,
            "clouds": clouds,
            "ai_report": xai_log,
            "historical_markers": historical_markers
        }
