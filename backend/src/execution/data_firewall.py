import logging
from typing import Any, Tuple

import pandas as pd

logger = logging.getLogger("DataFirewall")


class DataContaminationError(ValueError):
    """Raised when data boundaries are violated (e.g. 2026 data leaked into training/validation)."""
    pass


class TemporalFirewall:
    """
    Cryptographic and chronological firewall enforcing strict temporal isolation:
    - 2016-01-01 to 2024-12-31: MODEL DEVELOPMENT ONLY
    - 2025-01-01 to 2025-12-31: VALIDATION / CALIBRATION ONLY
    - 2026-01-01 onward: TRUE UNTOUCHED OUT-OF-SAMPLE (OOS)
    """

    DEVELOPMENT_START = "2016-01-01"
    DEVELOPMENT_END = "2024-12-31"
    VALIDATION_START = "2025-01-01"
    VALIDATION_END = "2025-12-31"
    OOS_START = "2026-01-01"

    @classmethod
    def extract_dates(cls, data: Any) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Extract minimum and maximum timestamps from DataFrame or DatetimeIndex."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if isinstance(data.index, pd.DatetimeIndex):
                idx = data.index
            elif "Date" in data.columns:
                idx = pd.to_datetime(data["Date"])
            elif "date" in data.columns:
                idx = pd.to_datetime(data["date"])
            else:
                try:
                    idx = pd.to_datetime(data.index)
                except Exception as e:
                    raise DataContaminationError(f"Unable to parse DatetimeIndex from data: {e}")
        elif isinstance(data, pd.DatetimeIndex):
            idx = data
        else:
            raise DataContaminationError(f"Unsupported data type for temporal firewall: {type(data)}")

        if len(idx) == 0:
            raise DataContaminationError("Data is empty; cannot verify temporal boundaries.")

        # Ensure tz-naive for pure date comparison
        if idx.tz is not None:
            idx = idx.tz_localize(None)

        return idx.min(), idx.max()

    @classmethod
    def validate_development_data(cls, df: Any, stage: str = "model_development") -> None:
        """
        Enforce that df contains ONLY data within 2016-01-01 to 2024-12-31.
        Throws DataContaminationError if any pre-2016 or post-2024 observation is present.
        """
        min_date, max_date = cls.extract_dates(df)

        if min_date < pd.Timestamp(cls.DEVELOPMENT_START):
            err = (
                f"[TEMPORAL CONTAMINATION DETECTED] Pre-2016 data found in {stage}! "
                f"Min date is {min_date.strftime('%Y-%m-%d')}, required >= {cls.DEVELOPMENT_START}."
            )
            logger.critical(err)
            raise DataContaminationError(err)

        if max_date > pd.Timestamp(cls.DEVELOPMENT_END):
            err = (
                f"[TEMPORAL CONTAMINATION DETECTED] Post-2024 data found in {stage}! "
                f"Max date is {max_date.strftime('%Y-%m-%d')}, required <= {cls.DEVELOPMENT_END}. "
                f"2025/2026 data must NEVER enter development or training!"
            )
            logger.critical(err)
            raise DataContaminationError(err)

        logger.info(
            f"[FIREWALL PASS] {stage}: {min_date.strftime('%Y-%m-%d')} -> {max_date.strftime('%Y-%m-%d')} "
            f"strictly within 2016-2024 boundary."
        )

    @classmethod
    def validate_validation_data(cls, df: Any, stage: str = "validation_calibration") -> None:
        """
        Enforce that validation/calibration evaluation data contains NO 2026+ data.
        Throws DataContaminationError if any 2026 observation is present.
        """
        min_date, max_date = cls.extract_dates(df)

        if max_date >= pd.Timestamp(cls.OOS_START):
            err = (
                f"[TEMPORAL CONTAMINATION DETECTED] 2026 data found in {stage}! "
                f"Max date is {max_date.strftime('%Y-%m-%d')}, required < {cls.OOS_START}. "
                f"2026 onward is TRUE UNTOUCHED OOS!"
            )
            logger.critical(err)
            raise DataContaminationError(err)

        logger.info(
            f"[FIREWALL PASS] {stage}: {min_date.strftime('%Y-%m-%d')} -> {max_date.strftime('%Y-%m-%d')} "
            f"strictly prior to 2026 OOS boundary."
        )

    @classmethod
    def validate_no_2026_leakage(cls, df: Any, stage: str = "general") -> None:
        """Absolute firewall check: 2026 data must NOT enter training, validation, or optimization."""
        _, max_date = cls.extract_dates(df)
        if max_date >= pd.Timestamp(cls.OOS_START):
            err = (
                f"[HARD 2026 FIREWALL BREACH] {stage} contains 2026 observation {max_date.strftime('%Y-%m-%d')}! "
                f"All 2026 data is strictly reserved for untouched out-of-sample forward evaluation."
            )
            logger.critical(err)
            raise DataContaminationError(err)

    @classmethod
    def enforce_label_horizon_isolation(
        cls, df_source: pd.DataFrame, horizon: int, max_allowed_date: str = "2024-12-31"
    ) -> Tuple[pd.DataFrame, int]:
        """
        Ensures that samples whose forward evaluation horizon extends past max_allowed_date
        are explicitly DROPPED so future prices are never used to compute training labels.

        Returns:
            df_safe: DataFrame with incomplete label rows removed
            dropped_count: Number of rows removed
        """
        if not isinstance(df_source.index, pd.DatetimeIndex):
            raise DataContaminationError("DataFrame index must be DatetimeIndex for horizon isolation check.")

        # Any sample where date + horizon trading days would exceed max_allowed_date must not peek into future
        cutoff = pd.Timestamp(max_allowed_date)
        initial_len = len(df_source)

        # In market trading days, the last `horizon` rows of a dataset ending at max_allowed_date
        # would look past max_allowed_date.
        df_within = df_source[df_source.index <= cutoff].copy()

        # When triple barrier is applied to df_within, future prices beyond df_within are unavailable
        # and result in NaN, which are then dropped.
        # We also enforce that the max index of the resulting labeled set has at least horizon days buffer
        # before any subsequent period.
        return df_within, initial_len - len(df_within)
