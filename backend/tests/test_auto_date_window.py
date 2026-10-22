"""
Unit tests for Automatic Dynamic Rolling Date Window Engine.
Validates automatic date advancement, configuration parsing, boundary constraints,
and firewall compatibility with dynamic calendar progression.
"""
from datetime import datetime

import pandas as pd
import pytest

from src.execution.data_firewall import DataContaminationError, TemporalFirewall
from src.utils.date_window import load_window_config, resolve_training_dates


def test_load_window_config():
    """Validates configuration schema and defaults."""
    cfg = load_window_config()
    assert "mode" in cfg
    assert "rolling_years" in cfg
    assert cfg["train_split_ratio"] == 0.75
    assert cfg["test_split_ratio"] == 0.25
    assert cfg["warmup_bars"] == 119


def test_resolve_training_dates_auto():
    """Validates automatic date resolution based on current calendar date."""
    start_date, end_date, is_auto = resolve_training_dates(start_date=None, end_date=None)
    assert is_auto is True

    today_str = datetime.now().strftime("%Y-%m-%d")
    assert end_date == today_str

    cfg = load_window_config()
    rolling_years = cfg.get("rolling_years", 8)
    expected_start = (pd.to_datetime(today_str) - pd.DateOffset(years=rolling_years)).strftime("%Y-%m-%d")
    assert start_date == expected_start


def test_resolve_training_dates_auto_string_inputs():
    """Validates resolution when explicit string 'auto' is passed."""
    start_date, end_date, is_auto = resolve_training_dates(start_date="auto", end_date="auto")
    assert is_auto is True
    today_str = datetime.now().strftime("%Y-%m-%d")
    assert end_date == today_str


def test_resolve_training_dates_explicit_override():
    """Validates that explicit dates override automatic calculation."""
    explicit_start = "2019-01-01"
    explicit_end = "2023-12-31"
    start_date, end_date, is_auto = resolve_training_dates(
        start_date=explicit_start, end_date=explicit_end
    )
    assert is_auto is False
    assert start_date == explicit_start
    assert end_date == explicit_end


def test_dynamic_firewall_allows_current_date():
    """Validates that TemporalFirewall accepts current date data when allow_dynamic=True."""
    # Data up to today
    dates = pd.date_range("2020-01-01", datetime.now().strftime("%Y-%m-%d"), freq="B")
    df = pd.DataFrame({"close": 100.0}, index=dates)

    # When allow_dynamic=True, passes without error
    TemporalFirewall.validate_development_data(df, "test_dynamic", allow_dynamic=True)
    TemporalFirewall.validate_no_2026_leakage(df, "test_dynamic", allow_dynamic=True)


def test_dynamic_firewall_rejects_future_dates():
    """Validates that TemporalFirewall rejects data beyond today even when allow_dynamic=True."""
    future_date = (pd.Timestamp.now() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    dates = pd.date_range("2020-01-01", future_date, freq="B")
    df = pd.DataFrame({"close": 100.0}, index=dates)

    with pytest.raises(DataContaminationError, match="FUTURE CONTAMINATION DETECTED"):
        TemporalFirewall.validate_development_data(df, "test_future", allow_dynamic=True)
