class SectorMapper:
    """
    Maps tickers to GICS sectors.
    Used for tracking exposure and performance attribution.
    """

    def __init__(self):
        # Sample mapping - in production, this would be fetched from an API/Database
        self.mapping = {
            "AAPL": "Information Technology",
            "MSFT": "Information Technology",
            "NVDA": "Information Technology",
            "AMZN": "Consumer Discretionary",
            "TSLA": "Consumer Discretionary",
            "META": "Communication Services",
            "GOOGL": "Communication Services",
            "BRK-B": "Financials",
            "JPM": "Financials",
            "V": "Financials",
            "UNH": "Health Care",
            "LLY": "Health Care",
            "XOM": "Energy",
            "CVX": "Energy",
        }

    def get_sector(self, ticker):
        return self.mapping.get(ticker.upper(), "Unknown")
