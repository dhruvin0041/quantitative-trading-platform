import numpy as np
from typing import Dict, Any


class ConsensusIntelligenceEngine:
    """
    Phase 5: Consensus Intelligence Engine.
    Analyzes ensemble coherence, entropy, and disagreement severity.
    """

    def analyze_consensus(
        self, base_probs: Dict[str, np.ndarray], agreement_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep analysis of model ensemble relationships.
        """
        # 1. Directional Entropy
        # Calculate how spread out the model votes are
        all_votes = []
        for p in base_probs.values():
            all_votes.append(int(np.argmax(p)))

        unique_votes = set(all_votes)
        entropy_score = len(unique_votes) / 3.0  # 1=low entropy, 3=max fragmentation

        # 2. Ensemble Coherence
        coherence = "Fragmented"
        disagreement_severity = "HIGH"

        if len(unique_votes) == 1:
            coherence = "High Conviction"
            disagreement_severity = "NONE"
        elif len(unique_votes) == 2:
            coherence = "Moderate Consensus"
            disagreement_severity = "LOW"

        # 3. Institutional Interpretation
        dominant_dir = agreement_data.get("dominant_direction", "HOLD")
        interpretation = f"{coherence} {dominant_dir}"

        # Special conflict cases
        buy_votes = all_votes.count(2)
        sell_votes = all_votes.count(0)

        if buy_votes > 0 and sell_votes > 0:
            interpretation = "Regime Conflict"
            disagreement_severity = "EXTREME"
        elif buy_votes == 0 and sell_votes == 0:
            interpretation = "Neutral Consolidation"
            disagreement_severity = "NONE"

        return {
            "consensus_interpretation": interpretation,
            "ensemble_coherence": coherence,
            "disagreement_severity": disagreement_severity,
            "directional_entropy": round(entropy_score, 2),
            "vote_distribution": {
                "BUY": buy_votes,
                "SELL": sell_votes,
                "HOLD": all_votes.count(1),
            },
        }


class WeightedConsensusEngine:
    """
    Computes directional agreement using weighted pressures rather than binary unanimity.
    Outputs: agreement_score, bullish_weight, bearish_weight, neutral_weight, dominant_direction
    """

    def __init__(self):
        # Indices: 0=SELL, 1=HOLD, 2=BUY
        self.idx_to_dir = {0: "SELL", 1: "HOLD", 2: "BUY"}
        self.intelligence = ConsensusIntelligenceEngine()

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

        agreement_res = {
            "agreement_score": agreement_score,
            "bearish_weight": pressures[0],
            "neutral_weight": pressures[1],
            "bullish_weight": pressures[2],
            "dominant_direction": dominant_direction,
            "dominant_idx": dominant_idx,
        }

        # Apply Phase 5 Intelligence
        intelligence = self.intelligence.analyze_consensus(base_probs, agreement_res)
        agreement_res.update(intelligence)

        return agreement_res
