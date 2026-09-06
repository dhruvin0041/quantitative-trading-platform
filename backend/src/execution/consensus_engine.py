import json
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

    def _get_empirical_accuracy(self, model_key: str) -> float:
        """Anchor smoothing probability strictly to empirical out-of-sample validation accuracy."""
        try:
            with open("configs/model_accuracies.json", "r") as f:
                accs = json.load(f)
            if "DQN" in model_key.upper():
                return float(accs.get("dqn_accuracy", 0.50))
            elif "XGB" in model_key.upper():
                return float(accs.get("xgb_accuracy", 0.55))
            elif "LGBM" in model_key.upper():
                return float(accs.get("lgbm_accuracy", 0.53))
            elif "DL" in model_key.upper():
                return float(accs.get("dl_accuracy", 0.50))
        except Exception:
            pass
        return 0.50

    def compute_agreement(
        self, base_probs: Dict[str, np.ndarray], model_weights: Dict[str, float]
    ) -> Dict:
        pressures = {0: 0.0, 1: 0.0, 2: 0.0}  # SELL, HOLD, BUY
        total_weight = 0.0

        for model_name, p in base_probs.items():
            canonical_key = self.ALIAS_MAP.get(model_name.upper(), model_name)
            w = model_weights.get(canonical_key, model_weights.get(model_name, 0.25))
            p_arr = np.array(p, dtype=float)

            # Soften discrete one-hot vectors by anchoring primary probability alpha
            # strictly to empirical out-of-sample directional accuracy (e.g., 50% for DQN)
            if np.max(p_arr) >= 0.99 and np.count_nonzero(p_arr) == 1:
                action_idx = int(np.argmax(p_arr))
                alpha = self._get_empirical_accuracy(canonical_key)
                p_soft = np.full(3, (1.0 - alpha) / 2.0)
                p_soft[action_idx] = alpha
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

    def compute_asymmetric_veto(
        self,
        base_probs: Dict[str, np.ndarray],
        primary_key: str = "XGB_AGENT",
        primary_threshold: float = 0.60,
        veto_threshold: float = 0.65,
        veto_short: bool = False,
    ) -> Dict[str, Any]:
        """
        Lead Driver + Asymmetric Veto Architecture.

        - Primary Driver: Designated alpha generator (default: XGB_AGENT) is the sole trade generator
          when conviction >= primary_threshold (default: 0.60).
        - Asymmetric Veto Filter: Secondary models (LightGBM, DQN) do NOT add positive directional weight;
          they are only consulted to veto high-risk entries:
          * If primary signals BUY, but any secondary model assigns P(SELL) >= veto_threshold (0.65),
            the entry is vetoed and downgraded to HOLD.
          * If veto_short is True and primary signals SELL, but secondary assigns P(BUY) >= veto_threshold,
            the short entry is vetoed and downgraded to HOLD.
          * Otherwise, the primary signal passes through unmuted.
        - DL_FUSION is quarantined and excluded from veto authority.
        """
        canon_probs: Dict[str, np.ndarray] = {}
        for k, v in base_probs.items():
            canon_k = self.ALIAS_MAP.get(k.upper(), k)
            p_arr = np.array(v, dtype=float)
            if np.max(p_arr) >= 0.99 and np.count_nonzero(p_arr) == 1:
                action_idx = int(np.argmax(p_arr))
                alpha = self._get_empirical_accuracy(canon_k)
                p_soft = np.full(3, (1.0 - alpha) / 2.0)
                p_soft[action_idx] = alpha
                p_arr = p_soft
            canon_probs[canon_k] = p_arr

        primary_canonical = self.ALIAS_MAP.get(primary_key.upper(), primary_key)
        p_primary = canon_probs.get(primary_canonical)
        if p_primary is None:
            p_primary = np.array([0.0, 1.0, 0.0])

        primary_conviction = float(np.max(p_primary))
        primary_idx = int(np.argmax(p_primary))
        primary_dir = self.idx_to_dir[primary_idx]

        is_vetoed = False
        vetoed_by = None
        veto_reason = None

        if primary_conviction < primary_threshold or primary_idx == 1:
            dominant_idx = 1
            dominant_direction = "HOLD"
            agreement_score = primary_conviction * 100.0
        else:
            dominant_idx = primary_idx
            dominant_direction = primary_dir
            agreement_score = primary_conviction * 100.0

            # Consult active secondary models for asymmetric veto
            # DL_FUSION is strictly quarantined and excluded from veto checks
            veto_candidates = ["LGBM_AGENT", "DQN_AGENT"]
            for sec_key in veto_candidates:
                if sec_key in canon_probs:
                    p_sec = canon_probs[sec_key]
                    if primary_idx == 2:  # BUY signal
                        if p_sec[0] >= veto_threshold:  # Opposing SELL conviction
                            is_vetoed = True
                            vetoed_by = sec_key
                            veto_reason = (
                                f"Vetoed by {sec_key}: Bearish conviction "
                                f"({p_sec[0]:.2f} >= {veto_threshold:.2f})"
                            )
                            dominant_idx = 1
                            dominant_direction = "HOLD"
                            break
                    elif primary_idx == 0 and veto_short:  # SELL signal
                        if p_sec[2] >= veto_threshold:  # Opposing BUY conviction
                            is_vetoed = True
                            vetoed_by = sec_key
                            veto_reason = (
                                f"Vetoed by {sec_key}: Counter-trend bullish conviction "
                                f"({p_sec[2]:.2f} >= {veto_threshold:.2f})"
                            )
                            dominant_idx = 1
                            dominant_direction = "HOLD"
                            break

        model_intelligence = {}
        for m_name, p in canon_probs.items():
            if m_name == primary_canonical:
                role = "PRIMARY_ALPHA_DRIVER"
            elif m_name == "DL_FUSION":
                role = "QUARANTINED"
            else:
                role = "SECONDARY_VETO"
            model_intelligence[m_name] = {
                "role": role,
                "confidence": float(np.max(p)),
                "direction": self.idx_to_dir[int(np.argmax(p))],
                "is_dominant": int(np.argmax(p)) == dominant_idx,
            }

        pressures = {
            0: float(p_primary[0]),
            1: float(p_primary[1]),
            2: float(p_primary[2]),
        }

        coherence = "Lead Driver"
        disagreement_severity = "HIGH" if is_vetoed else "LOW"
        interpretation = f"Lead Driver: {primary_canonical} ({primary_dir})"
        if is_vetoed:
            interpretation += f" | [VETOED by {vetoed_by}]"

        return {
            "agreement_score": agreement_score,
            "bearish_weight": pressures[0],
            "neutral_weight": pressures[1],
            "bullish_weight": pressures[2],
            "dominant_direction": dominant_direction,
            "dominant_idx": dominant_idx,
            "primary_driver": primary_canonical,
            "primary_conviction": primary_conviction,
            "is_vetoed": is_vetoed,
            "vetoed_by": vetoed_by,
            "veto_reason": veto_reason,
            "model_intelligence": model_intelligence,
            "consensus_interpretation": interpretation,
            "ensemble_coherence": coherence,
            "disagreement_severity": disagreement_severity,
            "directional_entropy": 0.0,
            "vote_distribution": {
                "BUY": 1 if dominant_direction == "BUY" else 0,
                "SELL": 1 if dominant_direction == "SELL" else 0,
                "HOLD": 1 if dominant_direction == "HOLD" else 0,
            },
        }
