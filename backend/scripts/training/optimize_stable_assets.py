"""
Grid Optimization for Newly Added Stable Assets:
BRK-B, V, JNJ, SUNPHARMA.NS, TITAN.NS.
Identifies the best institutional swing strategy configuration for each asset,
maximizes profit/alpha with clean trade execution (no whipsaws),
and updates backend/configs/optimized_champion_configs.json.
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from scripts.evaluation.backtest import fetch_and_prepare_data
from scripts.training.optimize_universe_grid import backtest_ticker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STABLE_TARGETS = ["BRK-B", "V", "JNJ", "SUNPHARMA.NS", "TITAN.NS"]


def main():
    logger.info("=" * 80)
    logger.info("OPTIMIZING CHAMPION STRATEGY CONFIGS FOR NEW STABLE ASSETS")
    logger.info("=" * 80)

    # Load SPY and VIX for macro regime
    spy_df = yf.download("SPY", period="2y", progress=False)
    if hasattr(spy_df.columns, "droplevel"):
        try:
            spy_df.columns = spy_df.columns.droplevel(1)
        except Exception:
            pass

    vix_df = yf.download("^VIX", period="2y", progress=False)
    if hasattr(vix_df.columns, "droplevel"):
        try:
            vix_df.columns = vix_df.columns.droplevel(1)
        except Exception:
            pass

    # Load existing configs
    cfg_path = backend_dir / "configs" / "optimized_champion_configs.json"
    with open(cfg_path, "r") as f:
        champion_configs = json.load(f)

    strat_types = [
        "CORE_TACTICAL_ACCUMULATION",
        "VOLATILITY_BREAKOUT_SWING",
        "ASYMMETRIC_MEAN_REVERSION"
    ]

    for sym in STABLE_TARGETS:
        logger.info(f"Optimizing parameters for {sym}...")
        pq_path = backend_dir / "data" / "cache" / f"{sym}_2y_features.parquet"
        if not pq_path.exists():
            df = fetch_and_prepare_data(
                ticker=sym,
                spy_df=spy_df,
                vix_df=vix_df,
                period="2y",
                cache_dir=backend_dir / "data" / "cache",
            )
        else:
            df = pd.read_parquet(pq_path)

        is_indian = sym.endswith(".NS")

        best_res = None
        best_params = None
        best_st = None

        for st in strat_types:
            for batr in [3.0, 3.5, 4.0, 4.5]:
                for drsi in [40.0, 44.0, 48.0]:
                    for ie in [True, False]:
                        for cd in [1, 2, 3]:
                            for ef, es in [(5, 13), (6, 15), (8, 21), (9, 26)]:
                                p = {
                                    "init_entry": ie,
                                    "base_atr": batr,
                                    "ratchet_15": batr * 0.8,
                                    "ratchet_25": batr * 0.6,
                                    "climax_rsi": 82.0,
                                    "climax_gain": 0.18,
                                    "dip_rsi": drsi,
                                    "cooldown": cd,
                                    "ema_fast": ef,
                                    "ema_slow": es,
                                    "macro_bear_exit": True,
                                    "bb_window": 20,
                                    "bb_mult": 2.1,
                                    "bb_adaptive": True,
                                    "ema_adaptive": True
                                }
                                res = backtest_ticker(df, st, p, is_indian=is_indian, spy_df=spy_df)
                                # We want positive alpha, decent win rate, and clean non-zero trades
                                if res["trades"] < 2:
                                    continue

                                # Composite score: Alpha + Win rate bonus - Trade count penalty if too many
                                score = res["alpha"] + (res["win_rate"] * 0.2) - (max(0, res["trades"] - 12) * 1.5)

                                if best_res is None or score > best_res["score"]:
                                    res["score"] = score
                                    best_res = res
                                    best_params = p
                                    best_st = st

        if best_res is None:
            # Fallback to standard Core Tactical Accumulation
            logger.warning(f"No custom parameter beat baseline for {sym}, using robust fallback.")
            best_st = "CORE_TACTICAL_ACCUMULATION"
            best_params = {
                "init_entry": True,
                "base_atr": 3.8,
                "ratchet_15": 3.04,
                "ratchet_25": 2.28,
                "climax_rsi": 82.0,
                "climax_gain": 0.18,
                "dip_rsi": 44.0,
                "cooldown": 2,
                "ema_fast": 6,
                "ema_slow": 15,
                "macro_bear_exit": True,
                "bb_window": 20,
                "bb_mult": 2.1,
                "bb_adaptive": True,
                "ema_adaptive": True
            }
            best_res = backtest_ticker(df, best_st, best_params, is_indian=is_indian, spy_df=spy_df)

        champion_configs[sym] = {
            "strategy_type": best_st,
            "strategy_name": best_st.replace("_", " ").title(),
            "params": best_params,
            "performance": best_res
        }
        logger.info(
            f"[{sym:<14}] Strategy: {best_st:<28} | Return: {best_res['strat_ret']:>+6.2f}% vs B&H {best_res['bh_ret']:>+6.2f}% | "
            f"Alpha: {best_res['alpha']:>+6.2f}% | Win Rate: {best_res['win_rate']:>5.1f}% | Trades: {best_res['trades']}"
        )

    # Save updated champion configs
    with open(cfg_path, "w") as f:
        json.dump(champion_configs, f, indent=2)

    logger.info(f"Successfully updated champion configs in {cfg_path}")


if __name__ == "__main__":
    main()

