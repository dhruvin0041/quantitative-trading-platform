import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class SignalGovernanceAnalytics:
    """
    Phase 6: Signal Governance Analytics.
    Tracks veto rates, approval rates, and prevents signal starvation/paralysis.
    """

    def analyze_throughput(self, signal_journal: pd.DataFrame) -> Dict[str, Any]:
        if signal_journal.empty:
            return {
                "veto_rate": 0.0,
                "approval_rate": 0.0,
                "signal_density": 0.0,
                "governance_status": "STABLE",
            }

        total = len(signal_journal)
        vetoes = len(signal_journal[signal_journal["signal_type"] == "VETOED"])
        approvals = total - vetoes

        veto_rate = (vetoes / total) * 100
        approval_rate = (approvals / total) * 100

        # Determine governance status
        status = "STABLE"
        if veto_rate > 85:
            status = "PARALYSIS_RISK"
        elif veto_rate < 10:
            status = "CROWDING_RISK"

        # Signal Starvation (check if recent window has any trades)
        # assuming journal is sorted by timestamp
        recent_window = signal_journal.tail(20)
        recent_trades = len(
            recent_window[recent_window["signal_type"].isin(["BUY", "SELL"])]
        )

        starvation = False
        if recent_trades == 0 and total > 20:
            starvation = True
            status = "SIGNAL_STARVATION"

        return {
            "veto_rate": round(veto_rate, 1),
            "approval_rate": round(approval_rate, 1),
            "total_signals": total,
            "governance_status": status,
            "signal_starvation": starvation,
            "throughput_coherence": "HIGH"
            if not starvation and 20 <= veto_rate <= 60
            else "LOW",
        }
