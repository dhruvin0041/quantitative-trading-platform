# src/execution/signal_generator.py
import json
from datetime import datetime


class SignalFormatter:
    def __init__(self):
        self.signal_map = {0: "Sell", 1: "Hold", 2: "Buy"}

    def format_output(self, ticker, current_price, raw_preds, sentiment_meta):
        """
        Parses raw model outputs into the specified JSON schema.
        """
        dir_prob = float(raw_preds[0][0][0])
        range_preds = raw_preds[1][0]
        signal_probs = raw_preds[2][0]

        direction = "Up" if dir_prob > 0.5 else "Down"
        dir_confidence = dir_prob if direction == "Up" else (1 - dir_prob)

        signal_idx = int(signal_probs.argmax())

        output = {
            "stock_predictions": [
                {
                    "ticker": ticker,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "predicted_direction": direction,
                    "predicted_price_range": {
                        "min": round(float(current_price + range_preds[0]), 2),
                        "max": round(float(current_price + range_preds[1]), 2),
                    },
                    "trading_signal": self.signal_map[signal_idx],
                    "confidence_score": {
                        "direction": round(dir_confidence, 2),
                        "signal": round(float(signal_probs[signal_idx]), 2),
                    },
                    "news_sentiment_impact": {
                        "sentiment": sentiment_meta.get("label", "Neutral"),
                        "influence_score": round(sentiment_meta.get("score", 0.0), 2),
                    },
                }
            ]
        }
        return json.dumps(output, indent=2)
