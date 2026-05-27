import pandas as pd
from typing import List, Dict, Any


class ForensicPortfolioResetEngine:
    """
    Phase 7: Forensic Portfolio Reset Engine.
    Segments pre-repair (contaminated) vs post-repair (validated) telemetry.
    Ensures legacy contamination never pollutes institutional analytics.
    """

    def __init__(self, validated_start_date: str = "2026-05-27"):
        self.validated_start_date = pd.Timestamp(validated_start_date)

    def segment_telemetry(
        self, snapshots: List[Dict], history: List[Dict]
    ) -> Dict[str, Any]:
        """
        Filters snapshots and trade history to only include validated data eras.
        """
        df_snapshots = pd.DataFrame(snapshots)
        if not df_snapshots.empty:
            df_snapshots["time"] = pd.to_datetime(df_snapshots["time"])
            post_repair_snapshots = df_snapshots[
                df_snapshots["time"] >= self.validated_start_date
            ].to_dict("records")
            pre_repair_snapshots = df_snapshots[
                df_snapshots["time"] < self.validated_start_date
            ].to_dict("records")
        else:
            post_repair_snapshots = []
            pre_repair_snapshots = []

        df_history = pd.DataFrame(history)
        if not df_history.empty:
            # Assuming history has entry_time
            df_history["entry_time"] = pd.to_datetime(df_history["entry_time"])
            post_repair_history = df_history[
                df_history["entry_time"] >= self.validated_start_date
            ].to_dict("records")
            pre_repair_history = df_history[
                df_history["entry_time"] < self.validated_start_date
            ].to_dict("records")
        else:
            post_repair_history = []
            pre_repair_history = []

        return {
            "validated_snapshots": post_repair_snapshots,
            "validated_history": post_repair_history,
            "contaminated_snapshots_count": len(pre_repair_snapshots),
            "contaminated_history_count": len(pre_repair_history),
            "validated_start_date": str(self.validated_start_date.date()),
            "trusted_era_active": True,
        }
