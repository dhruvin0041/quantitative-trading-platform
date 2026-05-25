
class InsiderTradeTracker:
    """
    Tracks Form 4 filings from SEC EDGAR to capture management conviction.
    """
    def get_insider_sentiment(self, ticker: str) -> dict:
        """
        Mocks SEC Form 4 extraction.
        """
        return {
            "net_insider_buying": True,
            "buy_volume_30d": 1500000,
            "sell_volume_30d": 200000,
            "insider_signal": "BULLISH"
        }
