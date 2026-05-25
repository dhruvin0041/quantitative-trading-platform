class AnalystRevisionTracker:
    """
    Tracks institutional analyst upgrades, downgrades, and price target changes.
    """

    def get_analyst_sentiment(self, ticker: str) -> dict:
        """
        Mocks analyst consensus data.
        """
        return {
            "consensus_rating": "BUY",
            "upgrades_30d": 4,
            "downgrades_30d": 1,
            "price_target_upside": 0.15,
            "analyst_signal": "BULLISH",
        }
