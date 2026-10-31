"""
Institutional Ops Script: Re-seeds SignalLedger with Adaptive Mesh signals for all universe assets.
Ensures 100% parity between historical chart markers and the institutional backtest.
"""
import json
import logging
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from unittest.mock import MagicMock

import pandas as pd
import yfinance as yf

from src.execution.inference_service import InferenceService
from src.execution.live_inference import load_config
from src.execution.signal_ledger import SignalLedger
from src.models.model_loader import ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def reseed_all():
    config = load_config()
    kf_path = backend_dir / "configs" / "kept_features.json"
    with open(kf_path) as f:
        kept_features = json.load(f)

    config["data"]["num_features"] = len(kept_features)
    mm = ModelManager(config, kept_features)
    mm.load_all_models()

    ledger = SignalLedger()
    service = InferenceService(
        mm,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        signal_ledger=ledger,
    )

    # Fetch 2y SPY data
    spy_df = None
    try:
        spy_dl = yf.download("SPY", period="2y", progress=False)
        if not spy_dl.empty:
            if hasattr(spy_dl.columns, "droplevel"):
                try:
                    spy_dl.columns = spy_dl.columns.droplevel(1)
                except Exception:
                    pass
            spy_df = spy_dl
    except Exception as e:
        logger.warning(f"Could not download fresh SPY: {e}")

    tickers = [
        # US Stable Champions & Growth
        "AAPL", "JNJ", "V", "BRK-B", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD",
        "COST", "WMT", "JPM", "LLY",
        # Indian Stable Champions & Leaders (NSE)
        "LT.NS", "TITAN.NS", "SUNPHARMA.NS", "RELIANCE.NS", "TCS.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS"
    ]
    logger.info("=" * 80)
    logger.info(f"RE-SEEDING SIGNAL LEDGER FOR ALL {len(tickers)} UNIVERSE ASSETS")
    logger.info("=" * 80)

    vix_df = None
    try:
        vix_dl = yf.download("^VIX", period="2y", progress=False)
        if not vix_dl.empty:
            if hasattr(vix_dl.columns, "droplevel"):
                try:
                    vix_dl.columns = vix_dl.columns.droplevel(1)
                except Exception:
                    pass
            vix_df = vix_dl
    except Exception as e:
        logger.warning(f"Could not download fresh VIX: {e}")

    results = {}
    for sym in tickers:
        pq_path = backend_dir / "data" / "cache" / f"{sym}_2y_features.parquet"
        if not pq_path.exists():
            logger.info(f"Feature cache missing for {sym}. Generating via fetch_and_prepare_data...")
            try:
                from scripts.evaluation.backtest import fetch_and_prepare_data
                df = fetch_and_prepare_data(
                    ticker=sym,
                    spy_df=spy_df,
                    vix_df=vix_df,
                    period="2y",
                    cache_dir=backend_dir / "data" / "cache",
                )
            except Exception as exc:
                logger.error(f"Failed to fetch and cache features for {sym}: {exc}")
                continue
        else:
            df = pd.read_parquet(pq_path)

        markers = service.get_causal_chart_markers(sym, df, spy_df=spy_df, force_recompute=True)
        audit = service.last_replay_audit or {}
        strat_mode = markers[0].get("metadata", {}).get("strategy_mode", "UNKNOWN") if markers else "EMPTY"
        results[sym] = {
            "staged": audit.get("staged_count", 0),
            "markers": len(markers),
            "success": audit.get("persistence_success", False),
            "mode": strat_mode,
        }
        logger.info(
            f"[{sym:<14}] Staged: {results[sym]['staged']:<3} | Committed Markers: {results[sym]['markers']:<3} | "
            f"Mode: {results[sym]['mode']:<28} | Success: {results[sym]['success']}"
        )

    logger.info("=" * 80)
    logger.info("RE-SEEDING COMPLETE")
    logger.info("=" * 80)
    return results


if __name__ == "__main__":
    reseed_all()
