import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class CorporateCalendarGate:
    """
    Corporate Calendar Gate for Pre-Earnings & Binary Event Shield.
    Protects quantitative swing trading capital by:
    1. Identifying scheduled quarterly earnings releases within a 48 to 72-hour window.
    2. Blocking new BUY entries right before binary 50/50 earnings gaps.
    3. Signal-triggering tightened Chandelier ratchet stops on active long positions.
    """

    _earnings_cache: Dict[str, List[pd.Timestamp]] = {}
    _cache_timestamps: Dict[str, datetime] = {}
    CACHE_TTL_SECONDS: int = 86400  # Cache for 24 hours per ticker

    @classmethod
    def get_earnings_dates(cls, ticker: str, force_refresh: bool = False) -> List[pd.Timestamp]:
        """
        Fetches and caches all historical and future earnings dates for a ticker using yfinance.
        Dates are normalized to UTC-naive midnight timestamps for fast date arithmetic.
        """
        now = datetime.now()
        ticker = ticker.upper()

        if (
            not force_refresh
            and ticker in cls._earnings_cache
            and ticker in cls._cache_timestamps
            and (now - cls._cache_timestamps[ticker]).total_seconds() < cls.CACHE_TTL_SECONDS
        ):
            return cls._earnings_cache[ticker]

        try:
            stock = yf.Ticker(ticker)
            ed = stock.earnings_dates
            if ed is not None and not ed.empty:
                # Convert timestamps to tz-naive UTC midnight
                clean_dates = []
                for idx in ed.index:
                    ts = pd.Timestamp(idx)
                    if ts.tz is not None:
                        ts = ts.tz_convert("UTC").tz_localize(None)
                    clean_dates.append(ts.normalize())

                # Sort ascending
                clean_dates = sorted(list(set(clean_dates)))
                cls._earnings_cache[ticker] = clean_dates
                cls._cache_timestamps[ticker] = now
                logger.info(f"[CorporateCalendarGate] Loaded {len(clean_dates)} earnings dates for {ticker}")
                return clean_dates
        except Exception as e:
            logger.warning(f"[CorporateCalendarGate] Could not fetch earnings dates for {ticker}: {e}")

        # Fallback to existing cache if available, else empty
        return cls._earnings_cache.get(ticker, [])

    @classmethod
    def check_earnings_proximity(
        cls,
        ticker: str,
        current_date: Any,
        blackout_days_before: int = 3,
        blackout_days_after: int = 1,
    ) -> Dict[str, Any]:
        """
        Checks whether current_date is within the pre-earnings or post-earnings blackout window.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL')
            current_date: Bar date (str 'YYYY-MM-DD', pd.Timestamp, datetime, or date)
            blackout_days_before: Number of calendar/trading days before earnings to block entries (default 3 days / 72h)
            blackout_days_after: Number of days after earnings to wait for initial reaction bar (default 1 day / 24h)

        Returns:
            Dict containing blackout status, proximity metrics, and action directives.
        """
        earnings_dates = cls.get_earnings_dates(ticker)

        # Normalize current_date to pd.Timestamp midnight
        if isinstance(current_date, str):
            curr_ts = pd.Timestamp(current_date[:10]).normalize()
        elif hasattr(current_date, "normalize"):
            curr_ts = pd.Timestamp(current_date).normalize()
            if curr_ts.tz is not None:
                curr_ts = curr_ts.tz_convert("UTC").tz_localize(None).normalize()
        else:
            curr_ts = pd.Timestamp(current_date).normalize()

        if not earnings_dates:
            return {
                "is_blackout": False,
                "days_to_earnings": None,
                "nearest_earnings_date": None,
                "earnings_shield_active": False,
                "action": "NORMAL",
                "message": "No corporate earnings calendar data available.",
            }

        # Find the upcoming earnings date (first date >= curr_ts)
        upcoming_dates = [ed for ed in earnings_dates if ed >= curr_ts]
        past_dates = [ed for ed in earnings_dates if ed < curr_ts]

        next_earnings = upcoming_dates[0] if upcoming_dates else None
        last_earnings = past_dates[-1] if past_dates else None

        days_to_next = (next_earnings - curr_ts).days if next_earnings is not None else 999
        days_since_last = (curr_ts - last_earnings).days if last_earnings is not None else 999

        # Pre-earnings blackout: 0 <= days_to_next <= blackout_days_before
        is_pre_earnings = 0 <= days_to_next <= blackout_days_before
        # Post-earnings blackout: 0 <= days_since_last <= blackout_days_after
        is_post_earnings = 0 <= days_since_last <= blackout_days_after

        is_blackout = is_pre_earnings or is_post_earnings

        if is_pre_earnings:
            action = "PRE_EARNINGS_BLACKOUT"
            target_date_str = next_earnings.strftime("%Y-%m-%d")
            msg = (
                f"PRE-EARNINGS SHIELD ACTIVE: Quarterly earnings scheduled for {target_date_str} "
                f"({days_to_next}d away). New BUY entries suppressed to eliminate overnight gap risk; "
                f"active long ratchet stops tightened."
            )
            nearest_date = target_date_str
            days_reported = days_to_next
        elif is_post_earnings:
            action = "POST_EARNINGS_DIGESTION"
            target_date_str = last_earnings.strftime("%Y-%m-%d")
            msg = (
                f"POST-EARNINGS REACTION WINDOW: Earnings reported on {target_date_str} "
                f"({days_since_last}d ago). Awaiting post-event price stabilization."
            )
            nearest_date = target_date_str
            days_reported = -days_since_last
        else:
            action = "NORMAL"
            nearest_date = next_earnings.strftime("%Y-%m-%d") if next_earnings else None
            days_reported = days_to_next if next_earnings else None
            msg = f"Corporate calendar clear. Next earnings on {nearest_date} ({days_to_next}d away)."

        return {
            "is_blackout": is_blackout,
            "days_to_earnings": days_reported,
            "nearest_earnings_date": nearest_date,
            "earnings_shield_active": is_blackout,
            "action": action,
            "message": msg,
        }
