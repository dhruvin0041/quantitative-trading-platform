from typing import Any, Dict

import numpy as np


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

    ALIAS_MAP = {
        "LSTM": "DL_FUSION",
        "DL_FUSION": "DL_FUSION",
        "XGBOOST": "XGB_AGENT",
        "XGB": "XGB_AGENT",
        "XGB_AGENT": "XGB_AGENT",
        "LIGHTGBM": "LGBM_AGENT",
        "LGBM": "LGBM_AGENT",
        "LGBM_AGENT": "LGBM_AGENT",
        "DQN": "DQN_AGENT",
        "DQN_AGENT": "DQN_AGENT",
    }

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
            canonical_key = self.ALIAS_MAP.get(model_name.upper(), model_name)
            w = model_weights.get(canonical_key, model_weights.get(model_name, 0.25))
            p_arr = np.array(p, dtype=float)

            # Soften discrete one-hot vectors (e.g., DQN action) so discrete agents do not overwhelm calibrated models
            if np.max(p_arr) >= 0.99 and np.count_nonzero(p_arr) == 1:
                action_idx = int(np.argmax(p_arr))
                p_soft = np.full(3, 0.20)
                p_soft[action_idx] = 0.60
                p_arr = p_soft

            for k in range(3):
                pressures[k] += w * float(p_arr[k])
            total_weight += w

        # Normalize across classes
        if total_weight > 0:
            for k in pressures:
                pressures[k] /= total_weight

        # Dominant direction from full soft consensus distribution
        dominant_idx = max(pressures, key=pressures.get)
        dominant_direction = self.idx_to_dir[dominant_idx]

        # Agreement score is the probability of the dominant direction * 100
        agreement_score = pressures[dominant_idx] * 100.0

        # Normalized Model Reliability Metadata
        model_intelligence = {}
        for model_name, p in base_probs.items():
            canonical_key = self.ALIAS_MAP.get(model_name.upper(), model_name)
            w = model_weights.get(canonical_key, model_weights.get(model_name, 0.25))
            reliability = 0.85 if ("XGB" in canonical_key or "DL" in canonical_key) else 0.75
            model_intelligence[model_name] = {
                "weight": float(w / total_weight) if total_weight > 0 else 0.25,
                "recent_reliability": reliability,
                "confidence": float(np.max(p)),
                "is_dominant": int(np.argmax(p)) == dominant_idx,
            }

        agreement_res = {
            "agreement_score": agreement_score,
            "bearish_weight": pressures[0],
            "neutral_weight": pressures[1],
            "bullish_weight": pressures[2],
            "dominant_direction": dominant_direction,
            "dominant_idx": dominant_idx,
            "model_intelligence": model_intelligence,
        }

        # Apply Phase 5 Intelligence
        intelligence = self.intelligence.analyze_consensus(base_probs, agreement_res)
        agreement_res.update(intelligence)

        return agreement_res
