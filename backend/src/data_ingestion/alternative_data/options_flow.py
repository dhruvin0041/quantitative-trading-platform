import pandas as pd

class OptionsFlowAnalyzer:
    """
    Analyzes Options Flow to detect institutional positioning vs speculative retail flow.
    Separates "Real" flow (hedging/institutional conviction) from "Lottery" flow (deep OTM retail).
    """
    def __init__(self):
        pass

    def analyze_flow(self, ticker: str, options_data: pd.DataFrame) -> dict:
        """
        Input: DataFrame of recent options transactions.
        Extracts Put/Call Ratio, Lottery %, and Anomaly flags.
        """
        if options_data.empty:
            return {"put_call_ratio": 1.0, "lottery_pct": 0.0, "signal": "NEUTRAL"}
            
        # Mock logic representing the algorithm
        calls = options_data[options_data['type'] == 'CALL']
        puts = options_data[options_data['type'] == 'PUT']
        
        lottery_calls = calls[(calls['premium'] < 0.10) & (calls['delta'] < 0.05)]
        real_calls = calls[~calls.index.isin(lottery_calls.index)]
        
        pc_ratio = len(puts) / len(real_calls) if len(real_calls) > 0 else 1.0
        lottery_pct = len(lottery_calls) / len(calls) if len(calls) > 0 else 0.0
        
        signal = "BULLISH" if pc_ratio < 0.7 else "BEARISH" if pc_ratio > 1.3 else "NEUTRAL"
        if lottery_pct > 0.6:
            signal = "BEARISH"  # Extreme retail euphoria is a contrarian indicator
            
        return {
            "adj_put_call_ratio": round(pc_ratio, 2),
            "lottery_pct": round(lottery_pct, 2),
            "options_signal": signal
        }
