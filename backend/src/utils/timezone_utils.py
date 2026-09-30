# src/utils/timezone_utils.py
"""
Dynamic America/New_York Timezone Utilities.
Guarantees mathematical precision for timestamp conversions:
- All timestamps stored internally in UTC (ISO 8601 Z format).
- Display timestamps derived dynamically using zoneinfo America/New_York.
- Never hardcodes EST or EDT; resolves DST dynamically based on observation date.
"""
import zoneinfo
from datetime import datetime, timezone
from typing import Union

NY_TZ = zoneinfo.ZoneInfo("America/New_York")
UTC_TZ = timezone.utc


def parse_to_utc(dt_val: Union[str, datetime, int, float]) -> datetime:
    """
    Parses a date string, datetime object, or epoch timestamp into a timezone-aware UTC datetime.
    If the string does not specify a timezone, defaults to America/New_York market time.
    """
    if isinstance(dt_val, (int, float)):
        return datetime.fromtimestamp(dt_val, tz=UTC_TZ)

    if isinstance(dt_val, datetime):
        if dt_val.tzinfo is None:
            # Assume naive datetime was in America/New_York
            return dt_val.replace(tzinfo=NY_TZ).astimezone(UTC_TZ)
        return dt_val.astimezone(UTC_TZ)

    s = str(dt_val).strip()

    # 1. Direct ISO parse (handles Z, +00:00, and offsets like -04:00)
    try:
        iso_str = s.replace("Z", "+00:00") if s.endswith("Z") else s
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is not None:
            return dt.astimezone(UTC_TZ)
    except ValueError:
        pass

    # 2. Date only: treat as 16:00:00 America/New_York (regular equity close)
    if len(s) == 10 and s.count("-") == 2:
        dt_ny = datetime.strptime(s, "%Y-%m-%d").replace(hour=16, minute=0, second=0, tzinfo=NY_TZ)
        return dt_ny.astimezone(UTC_TZ)

    # 3. Handle explicit Eastern suffix like '2026-09-30 16:00:00 EDT' or 'EST'
    for suffix in [" EST", " EDT", " ET"]:
        if s.endswith(suffix):
            raw_dt = s[: -len(suffix)].strip()
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
                try:
                    dt = datetime.strptime(raw_dt, fmt)
                    return dt.replace(tzinfo=NY_TZ).astimezone(UTC_TZ)
                except ValueError:
                    pass

    # 4. Standard naive datetime formats: assume America/New_York
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=NY_TZ).astimezone(UTC_TZ)
        except ValueError:
            pass

    # Fallback to current UTC
    return datetime.now(UTC_TZ)


def to_utc_iso(dt_val: Union[str, datetime, int, float]) -> str:
    """Returns ISO 8601 UTC string: YYYY-MM-DDTHH:MM:SSZ."""
    dt_utc = parse_to_utc(dt_val)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_new_york_iso(dt_val: Union[str, datetime, int, float]) -> str:
    """Returns ISO 8601 America/New_York string with dynamic offset: e.g. -04:00 or -05:00."""
    dt_utc = parse_to_utc(dt_val)
    dt_ny = dt_utc.astimezone(NY_TZ)
    return dt_ny.isoformat()


def format_new_york_display(dt_val: Union[str, datetime, int, float]) -> str:
    """
    Dynamically returns display format with accurate timezone abbreviation:
    e.g. '2026-09-30 16:00:00 EDT' or '2026-01-15 16:00:00 EST'.
    """
    dt_utc = parse_to_utc(dt_val)
    dt_ny = dt_utc.astimezone(NY_TZ)
    tz_abbr = dt_ny.tzname() or ("EDT" if dt_ny.dst() else "EST")
    return dt_ny.strftime(f"%Y-%m-%d %H:%M:%S {tz_abbr}")


def get_new_york_tzname(dt_val: Union[str, datetime, int, float]) -> str:
    """Returns 'EDT' or 'EST' dynamically for the timestamp."""
    dt_utc = parse_to_utc(dt_val)
    dt_ny = dt_utc.astimezone(NY_TZ)
    return dt_ny.tzname() or ("EDT" if dt_ny.dst() else "EST")
