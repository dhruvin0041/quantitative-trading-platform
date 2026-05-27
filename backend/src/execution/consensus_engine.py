import numpy as np
from typing import Dict


class WeightedConsensusEngine:
    """
    Computes directional agreement using weighted pressures rather than binary unanimity.
    Outputs: agreement_score, bullish_weight, bearish_weight, neutral_weight, dominant_direction
    """

    def __init__(self):
        # Indices: 0=SELL, 1=HOLD, 2=BUY
        self.idx_to_dir = {0: "SELL", 1: "HOLD", 2: "BUY"}

    def compute_agreement(
        self, base_probs: Dict[str, np.ndarray], model_weights: Dict[str, float]
    ) -> Dict:
        pressures = {0: 0.0, 1: 0.0, 2: 0.0}  # SELL, HOLD, BUY
        total_weight = 0.0

        for model_name, p in base_probs.items():
            w = model_weights.get(model_name, 0.25)
            # Find the primary direction for this model
            signal_idx = int(np.argmax(p))
            confidence = float(p[signal_idx])

            # The pressure added is the weight * confidence in that direction
            pressures[signal_idx] += w * confidence
            total_weight += w

        # Normalize
        if total_weight > 0:
            for k in pressures:
                pressures[k] /= total_weight

        # Dominant direction
        dominant_idx = max(pressures, key=pressures.get)
        dominant_direction = self.idx_to_dir[dominant_idx]

        # Agreement score is the pressure of the dominant direction * 100
        # representing directional consensus intensity
        agreement_score = pressures[dominant_idx] * 100.0

        return {
            "agreement_score": agreement_score,
            "bearish_weight": pressures[0],
            "neutral_weight": pressures[1],
            "bullish_weight": pressures[2],
            "dominant_direction": dominant_direction,
            "dominant_idx": dominant_idx,
        }
