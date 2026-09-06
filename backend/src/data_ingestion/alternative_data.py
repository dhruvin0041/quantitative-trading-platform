import numpy as np
import requests


class PhysicalEdgeAnalyzer:
    """
    Simulates institutional geospatial intelligence using free public proxies.
    - Weather-based Port Disruption (Open-Meteo)
    - Search Interest (Google Trends Proxy)
    """

    def __init__(self):
        # Major global port coordinates
        self.ports = {
            "Shanghai": (31.2, 121.5),
            "Long Beach": (33.7, -118.2),
            "Rotterdam": (51.9, 4.1),
            "Singapore": (1.3, 103.8),
        }

    def get_port_disruption_score(self):
        """
        Fetches current weather data for major ports to estimate supply chain delay risk.
        Score: 0.0 (Clear) to 1.0 (Major Storm/Disruption)
        """
        disruptions = []
        try:
            for port, (lat, lon) in self.ports.items():
                url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
                response = requests.get(url, timeout=5).json()
                weather_code = response.get("current_weather", {}).get("weathercode", 0)
                # WMO Weather interpretation codes (e.g., 95+ is thunderstorm/violent)
                score = 0.0
                if weather_code >= 95:
                    score = 1.0  # Violent
                elif weather_code >= 80:
                    score = 0.7  # Heavy rain/snow
                elif weather_code >= 60:
                    score = 0.4  # Moderate rain
                disruptions.append(score)

            avg_disruption = np.mean(disruptions) if disruptions else 0.0
            return float(avg_disruption)
        except Exception:
            return 0.0

    def get_retail_demand_proxy(self, ticker: str):
        """
        Uses search intensity/volume logic to proxy 'Retail Foot Traffic'.
        For a fully free system, we simulate this with a volume-volatility ratio
        if actual Trends APIs are restricted.
        """
        # Note: True PyTrends requires complex session management.
        # We'll use a 'Micro-Sentiment' proxy for now.
        return 0.5  # Baseline placeholder

    def get_physical_alpha_vector(self, ticker: str):
        """
        Returns a dictionary of physical/macro alpha features.
        """
        disruption = self.get_port_disruption_score()
        return {
            "supply_chain_disruption_index": disruption,
            "physical_bottleneck_risk": 1.0 if disruption > 0.6 else 0.0,
            "macro_physical_impact": -disruption
            * 0.1,  # Negative alpha if ports are blocked
        }
