"""
Automatic Dynamic Rolling Window Engine.
Automatically rolls training and optimization dates forward based on calendar progression.
Provides an institutional 8-year trailing window with 75% train / 25% test chronological splits.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger("DateWindow")

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


def load_window_config() -> dict:
    """Loads configuration for dynamic training windows."""
    configs_dir = BACKEND_DIR / "configs"
    cfg_file = configs_dir / "training_window_config.json"
    default_cfg = {
        "mode": "auto_rolling",
        "rolling_years": 8,
        "train_split_ratio": 0.75,
        "test_split_ratio": 0.25,
        "warmup_bars": 119,
        "fixed_start": "2016-01-01",
        "fixed_end": "2024-12-31",
    }
    if cfg_file.exists():
        try:
            with open(cfg_file, "r") as f:
                data = json.load(f)
                default_cfg.update(data)
        except Exception as e:
            logger.warning(f"Could not parse {cfg_file}: {e}. Using defaults.")
    return default_cfg


def resolve_training_dates(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[str, str, bool]:
    """
    Resolves start and end dates automatically.

    If start_date or end_date is None or 'auto', it calculates dynamic dates:
    - In 'auto_rolling' mode:
        end_date = Current date (YYYY-MM-DD)
        start_date = end_date minus rolling_years (e.g. 8 years trailing)
    - In 'fixed' mode:
        uses fixed_start ('2016-01-01') and fixed_end ('2024-12-31')

    If explicit dates are passed, they are strictly respected.

    Returns:
        Tuple[start_date (str), end_date (str), is_auto (bool)]
    """
    cfg = load_window_config()
    mode = cfg.get("mode", "auto_rolling").lower()
    rolling_years = int(cfg.get("rolling_years", 8))

    is_auto = False
    now = datetime.now()

    # Check if end_date needs auto-resolution
    is_end_auto = not end_date or str(end_date).strip().lower() in ("auto", "none", "latest", "today", "")
    # Check if start_date needs auto-resolution
    is_start_auto = not start_date or str(start_date).strip().lower() in ("auto", "none", "")

    if is_end_auto and is_start_auto and mode == "fixed":
        return cfg.get("fixed_start", "2016-01-01"), cfg.get("fixed_end", "2024-12-31"), False

    if is_end_auto:
        end_date = now.strftime("%Y-%m-%d")
        is_auto = True
    else:
        end_date = str(end_date).strip()

    if is_start_auto:
        end_dt = pd.to_datetime(end_date)
        start_dt = end_dt - pd.DateOffset(years=rolling_years)
        start_date = start_dt.strftime("%Y-%m-%d")
        is_auto = True
    else:
        start_date = str(start_date).strip()

    if is_auto:
        logger.info(
            f"[AUTO ROLLING WINDOW] Calendar progression active: {start_date} -> {end_date} (Trailing {rolling_years} Years)"
        )

    return start_date, end_date, is_auto
