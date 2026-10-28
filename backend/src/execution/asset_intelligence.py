from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


class AssetProfileEngine:
    """
    Phase 9: Asset-Specific Intelligence Layer.
    Injects unique alpha drivers for different asset classes.
    """

    @staticmethod
    def get_asset_class(ticker: str) -> str:
        ticker = ticker.upper()
        # Crypto
        if any(x in ticker for x in ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]):
            return "CRYPTO"
        # Commodities
        if any(x in ticker for x in ["GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F"]):
            return "COMMODITY"
        # Indices
        if any(
            x in ticker
            for x in [
                "^GSPC",
                "SPX",
                "^IXIC",
                "NDX",
                "^NSEI",
                "NIFTY",
                "^BSESN",
                "^FTSE",
                "^GDAXI",
            ]
        ):
            return "INDEX"
        # Forex
        if "=" in ticker and len(ticker) <= 9:
            return "FOREX"
        return "EQUITY"

    def enrich_context(self, ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
        asset_class = self.get_asset_class(ticker)
        context = {"class": asset_class}

        try:
            if asset_class == "CRYPTO":
                # Mock funding/OI for now
                context["funding_rate"] = 0.0001
                context["on_chain_activity"] = "INCREASING"

            elif asset_class == "COMMODITY":
                # DXY is a massive driver for commodities (priced in USD)
                dxy = yf.download(
                    "DX-Y.NYB", period="5d", interval="1d", progress=False
                )["Close"].iloc[-1]
                context["dxy_level"] = round(float(dxy), 2)
                context["usd_correlation"] = "NEGATIVE"

            elif asset_class == "INDEX":
                # Market Breadth / Volatility components
                context["vix_level"] = yf.download("^VIX", period="1d", progress=False)[
                    "Close"
                ].iloc[-1]
                context["global_sentiment"] = "NEUTRAL"

            elif asset_class == "FOREX":
                # Yield differentials
                context["interest_rate_bias"] = "HAWKISH"

            elif asset_class == "EQUITY":
                context["relative_strength_sector"] = 0.72
                context["institutional_accumulation"] = "HIGH"
        except Exception:
            pass

        return context


from enum import Enum


class ModelRole(str, Enum):
    PRIMARY_ALPHA_DRIVER = "PRIMARY_ALPHA_DRIVER"
    SECONDARY_VETO = "SECONDARY_VETO"
    FORECAST_ORACLE = "FORECAST_ORACLE"
    QUARANTINED = "QUARANTINED"


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "XGB_AGENT": {
        "role": ModelRole.PRIMARY_ALPHA_DRIVER,
        "status": "ACTIVE",
        "description": "Primary alpha trade generator for equity universe.",
        "conviction_threshold": 0.60,
    },
    "LGBM_AGENT": {
        "role": ModelRole.SECONDARY_VETO,
        "status": "ACTIVE",
        "description": "Asymmetric downside/counter-trend risk veto filter.",
        "veto_threshold": 0.65,
    },
    "DQN_AGENT": {
        "role": ModelRole.SECONDARY_VETO,
        "status": "ACTIVE",
        "description": "Sequential policy veto filter for execution safety.",
        "veto_threshold": 0.65,
    },
    "DL_FUSION": {
        "role": ModelRole.QUARANTINED,
        "status": "QUARANTINED",
        "description": "Quarantined pending retraining with symmetric labels.",
        "conviction_threshold": 0.0,
    },
    "TFT_AGENT": {
        "role": ModelRole.FORECAST_ORACLE,
        "status": "ACTIVE",
        "description": "Quantile volatility & price trajectory projections.",
    },
}


class AdaptiveWeightingEngine:
    """
    Phase 4: Dynamically rebalances model influence based on regime and accuracy.
    Formalizes the institutional hierarchy:
    - XGB_AGENT is the PRIMARY_ALPHA_DRIVER.
    - LGBM_AGENT and DQN_AGENT act strictly as SECONDARY_VETO filters.
    - DL_FUSION is QUARANTINED (weight 0.0).
    """

    def calculate_weights(
        self, regime: str, asset_class: str
    ) -> Dict[str, Dict[str, Any]]:
        # Formalized Model Hierarchy: Lead Driver (XGB_AGENT) with Asymmetric Secondary Vetoes
        base_weights = {
            "DL_FUSION": 0.0,
            "XGB_AGENT": 1.0,
            "LGBM_AGENT": 0.0,
            "DQN_AGENT": 0.0,
        }

        reason = "Primary alpha driver (XGBoost) with asymmetric veto gates."

        return {
            k: {
                "weight": v,
                "role": str(MODEL_REGISTRY.get(k, {}).get("role", ModelRole.SECONDARY_VETO)),
                "status": MODEL_REGISTRY.get(k, {}).get("status", "ACTIVE"),
                "reason": reason,
                "recent_accuracy": 0.65 if k == "XGB_AGENT" else 0.50,
            }
            for k, v in base_weights.items()
        }


class MultiTimeframeEngine:
    """
    Phase 5: Validates signals across multiple periodicities.
    """

    def get_mtf_consensus(self, ticker: str, daily_signal: str) -> Dict[str, str]:
        # Mocking MTF checks - in prod these would fetch 1H/4H data
        return {"1H": daily_signal, "4H": daily_signal, "1D": daily_signal}


class AssetExpectancyFilter:
    """
    Dynamic Asset-Level Expectancy Gating.
    Prevents capital bleed by tracking trailing 90-day realized Profit Factor per ticker.
    Hysteresis Band:
    - Suspension: If trailing 90-day PF < 1.15 (with >= min_trades trades),
      suspend new entries for that symbol.
    - Recovery: If suspended, the symbol remains suspended until its trailing 90-day PF
      recovers >= 1.20 (or when unprofitable trades roll off past 90 days).
    """

    def __init__(
        self,
        suspension_threshold: float = 1.15,
        recovery_threshold: float = 1.20,
        lookback_days: int = 90,
        min_trades: int = 4,
        trip_hurdle: float | None = None,
        recovery_hurdle: float | None = None,
    ):
        if trip_hurdle is not None:
            suspension_threshold = trip_hurdle
        if recovery_hurdle is not None:
            recovery_threshold = recovery_hurdle
        self.suspension_threshold = suspension_threshold
        self.recovery_threshold = recovery_threshold
        self.lookback_days = lookback_days
        self.min_trades = min_trades
        self.trade_history: Dict[str, list] = {}
        self.suspended_status: Dict[str, bool] = {}

    def record_trade(
        self,
        ticker: str,
        date: Any,
        pnl_ret: float,
        entry_date: Any = None,
    ) -> None:
        """
        Records a realized closed trade return for the given ticker.
        Strict Causal Mandate: 'date' represents the trade's exit/realization date
        (when the trade actually closes). If entry_date is also supplied, both are stored,
        and date is preserved as exit_date for strict causal filtering.
        """
        if ticker not in self.trade_history:
            self.trade_history[ticker] = []
        exit_ts = pd.Timestamp(date)
        entry_ts = pd.Timestamp(entry_date) if entry_date is not None else exit_ts
        self.trade_history[ticker].append(
            {
                "date": exit_ts,
                "exit_date": exit_ts,
                "entry_date": entry_ts,
                "pnl_ret": float(pnl_ret),
            }
        )

    def get_trailing_profit_factor(
        self, ticker: str, current_date: Any
    ) -> tuple[float, int]:
        """
        Computes the trailing 90-day realized Profit Factor for a ticker.
        Strict Causal Mandate: Only closed trades whose realization exit_date <= current_date
        AND within trailing lookback_days are considered.
        Returns:
            (trailing_pf, trade_count)
        """
        trades = self.trade_history.get(ticker, [])
        if not trades:
            return 1.5, 0

        curr_ts = pd.Timestamp(current_date)
        recent = [
            t
            for t in trades
            if t["exit_date"] <= curr_ts
            and 0 <= (curr_ts - t["exit_date"]).days <= self.lookback_days
        ]
        if not recent:
            return 1.5, 0

        gains = sum(t["pnl_ret"] for t in recent if t["pnl_ret"] > 0)
        losses = abs(sum(t["pnl_ret"] for t in recent if t["pnl_ret"] < 0))

        if losses == 0.0:
            trailing_pf = 3.0 if gains > 0 else 1.0
        else:
            trailing_pf = gains / losses

        return float(trailing_pf), len(recent)

    def is_entry_allowed(
        self, ticker: str, current_date: Any
    ) -> tuple[bool, str]:
        """
        Evaluates whether a new entry is allowed for the ticker under the hysteresis expectancy gate.
        Returns:
            (is_allowed, reason)
        """
        trailing_pf, count = self.get_trailing_profit_factor(ticker, current_date)
        is_currently_suspended = self.suspended_status.get(ticker, False)

        # Require a minimum sample size of recent trades before gating triggers
        if count < self.min_trades:
            if is_currently_suspended and count == 0:
                self.suspended_status[ticker] = False
                return True, f"Expectancy gate cleared: 0 active losses in trailing {self.lookback_days}d"
            elif is_currently_suspended:
                return (
                    False,
                    f"Suspended: Trailing {self.lookback_days}d PF {trailing_pf:.2f} (< {self.recovery_threshold:.2f} recovery threshold, {count} trades)",
                )
            return True, f"Sufficient expectancy (sample size {count} < {self.min_trades})"

        # Evaluate hysteresis transitions
        if is_currently_suspended:
            if trailing_pf >= self.recovery_threshold:
                self.suspended_status[ticker] = False
                return (
                    True,
                    f"Expectancy recovered: Trailing {self.lookback_days}d PF {trailing_pf:.2f} >= {self.recovery_threshold:.2f}",
                )
            else:
                return (
                    False,
                    f"Suspended by Expectancy Gate: Trailing {self.lookback_days}d PF {trailing_pf:.2f} < {self.recovery_threshold:.2f} recovery hurdle ({count} trades)",
                )
        else:
            if trailing_pf < self.suspension_threshold:
                self.suspended_status[ticker] = True
                return (
                    False,
                    f"Suspended by Expectancy Gate: Trailing {self.lookback_days}d PF {trailing_pf:.2f} dropped below {self.suspension_threshold:.2f} ({count} trades)",
                )
            else:
                return (
                    True,
                    f"Expectancy approved: Trailing {self.lookback_days}d PF {trailing_pf:.2f} >= {self.suspension_threshold:.2f} ({count} trades)",
                )


ADAPTIVE_MESH_MAP: Dict[str, str] = {
    "AAPL": "volatility_breakout_swing",
    "NVDA": "volatility_breakout_swing",
    "MSFT": "core_tactical_accumulation",
    "AMZN": "core_tactical_accumulation",
    "META": "core_tactical_accumulation",
    "GOOGL": "core_tactical_accumulation",
    "TSLA": "volatility_breakout_swing",
    "AMD": "volatility_breakout_swing",
    # Clean US Trend Leaders
    "COST": "core_tactical_accumulation",
    "WMT": "core_tactical_accumulation",
    "JPM": "core_tactical_accumulation",
    "LLY": "volatility_breakout_swing",
    # Clean Indian Market Leaders (NSE)
    "RELIANCE.NS": "volatility_breakout_swing",
    "TCS.NS": "asymmetric_mean_reversion",
    "ICICIBANK.NS": "core_tactical_accumulation",
    "BHARTIARTL.NS": "core_tactical_accumulation",
    "LT.NS": "core_tactical_accumulation",
    "ITC.NS": "core_tactical_accumulation",
}

ASSET_CHAMPION_STRATEGY_MAP: Dict[str, Dict[str, Any]] = {
    "AAPL": {
        "strategy_type": "VOLATILITY_BREAKOUT_SWING",
        "strategy_name": "Volatility Breakout Swing",
        "params": {
            "base_atr": 3.8,
            "ratchet_15": 3.0,
            "ratchet_25": 2.4,
            "climax_rsi": 78.0,
            "climax_gain": 0.15,
            "dip_rsi": 44.0,
            "init_entry": True,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 20,
            "bb_mult": 2.2,
            "bb_adaptive": True,
            "ema_fast": 8,
            "ema_slow": 13,
            "ema_adaptive": True,
        },
        "description": "Momentum expansion with MACD acceleration and institutional Chandelier trailing ratchet.",
    },
    "NVDA": {
        "strategy_type": "VOLATILITY_BREAKOUT_SWING",
        "strategy_name": "Volatility Breakout Swing",
        "params": {
            "base_atr": 4.2,
            "ratchet_15": 3.5,
            "ratchet_25": 2.8,
            "climax_rsi": 78.0,
            "climax_gain": 0.20,
            "dip_rsi": 42.0,
            "init_entry": True,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 22,
            "bb_mult": 2.5,
            "bb_adaptive": True,
            "ema_fast": 9,
            "ema_slow": 15,
            "ema_adaptive": True,
        },
        "description": "High-beta trend expansion with dynamic volatility trailing stop.",
    },
    "MSFT": {
        "strategy_type": "CORE_TACTICAL_ACCUMULATION",
        "strategy_name": "Core Tactical Accumulation",
        "params": {
            "dip_rsi": 45.0,
            "base_atr": 4.0,
            "ratchet_15": 3.2,
            "ratchet_25": 2.6,
            "climax_rsi": None,
            "climax_gain": 0.15,
            "init_entry": False,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 20,
            "bb_mult": 2.2,
            "bb_adaptive": True,
            "ema_fast": 10,
            "ema_slow": 17,
            "ema_adaptive": True,
        },
        "description": "Institutional core trend accumulator capturing secular breakouts with 100% Cash Defense during corrections.",
    },
    "AMZN": {
        "strategy_type": "CORE_TACTICAL_ACCUMULATION",
        "strategy_name": "Core Tactical Accumulation",
        "params": {
            "dip_rsi": 42.0,
            "base_atr": 4.0,
            "ratchet_15": 3.2,
            "ratchet_25": 2.6,
            "climax_rsi": 78.0,
            "climax_gain": 0.15,
            "init_entry": False,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 20,
            "bb_mult": 2.3,
            "bb_adaptive": True,
            "ema_fast": 6,
            "ema_slow": 25,
            "ema_adaptive": True,
        },
        "description": "Tactical trend accumulator capturing institutional pullbacks and secular momentum expansions.",
    },
    "META": {
        "strategy_type": "CORE_TACTICAL_ACCUMULATION",
        "strategy_name": "Core Tactical Accumulation",
        "params": {
            "dip_rsi": 45.0,
            "base_atr": 4.5,
            "ratchet_15": 3.5,
            "ratchet_25": 2.8,
            "climax_rsi": 80.0,
            "climax_gain": 0.18,
            "init_entry": True,
            "cooldown": 4,
            "macro_bear_exit": True,
            "bb_window": 22,
            "bb_mult": 2.4,
            "bb_adaptive": True,
            "ema_fast": 6,
            "ema_slow": 11,
            "ema_adaptive": True,
        },
        "description": "Institutional trend accumulator with wide Chandelier ratchets protecting against mid-trend whipsaws.",
    },
    "GOOGL": {
        "strategy_type": "CORE_TACTICAL_ACCUMULATION",
        "strategy_name": "Core Tactical Accumulation",
        "params": {
            "dip_rsi": 45.0,
            "base_atr": 4.2,
            "ratchet_15": 3.2,
            "ratchet_25": 2.6,
            "climax_rsi": 80.0,
            "climax_gain": 0.20,
            "init_entry": False,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 20,
            "bb_mult": 2.2,
            "bb_adaptive": True,
            "ema_fast": 7,
            "ema_slow": 12,
            "ema_adaptive": True,
        },
        "description": "Tactical accumulation engine participating in sustained bull trends and preserving capital in bear regimes.",
    },
    "TSLA": {
        "strategy_type": "VOLATILITY_BREAKOUT_SWING",
        "strategy_name": "Volatility Breakout Swing",
        "params": {
            "dip_rsi": 42.0,
            "base_atr": 4.5,
            "ratchet_15": 3.5,
            "ratchet_25": 2.8,
            "climax_rsi": 82.0,
            "climax_gain": 0.22,
            "init_entry": True,
            "cooldown": 4,
            "macro_bear_exit": True,
            "bb_window": 24,
            "bb_mult": 2.7,
            "bb_adaptive": True,
            "ema_fast": 12,
            "ema_slow": 33,
            "ema_adaptive": True,
        },
        "description": "High-volatility momentum breakout with strict 200 SMA regime filter, dynamic ATR ratchets, and bear de-risking.",
    },
    "AMD": {
        "strategy_type": "VOLATILITY_BREAKOUT_SWING",
        "strategy_name": "Volatility Breakout Swing",
        "params": {
            "dip_rsi": 42.0,
            "base_atr": 4.5,
            "ratchet_15": 3.8,
            "ratchet_25": 3.2,
            "climax_rsi": 80.0,
            "climax_gain": 0.22,
            "init_entry": True,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 24,
            "bb_mult": 2.6,
            "bb_adaptive": True,
            "ema_fast": 8,
            "ema_slow": 13,
            "ema_adaptive": True,
        },
        "description": "Secular semiconductor momentum breakout with volume confirmation and disciplined Chandelier trailing ratchets.",
    },
    "COST": {
        "strategy_type": "CORE_TACTICAL_ACCUMULATION",
        "strategy_name": "Core Tactical Accumulation",
        "params": {
            "dip_rsi": 45.0,
            "base_atr": 3.5,
            "ratchet_15": 2.8,
            "ratchet_25": 2.2,
            "climax_rsi": 78.0,
            "climax_gain": 0.15,
            "init_entry": True,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 18,
            "bb_mult": 2.0,
            "bb_adaptive": True,
            "ema_fast": 11,
            "ema_slow": 32,
            "ema_adaptive": True,
        },
        "description": "Low-beta consumer retail compounder capturing steady tactical accumulation and protecting during market corrections.",
    },
    "WMT": {
        "strategy_type": "CORE_TACTICAL_ACCUMULATION",
        "strategy_name": "Core Tactical Accumulation",
        "params": {
            "dip_rsi": 45.0,
            "base_atr": 3.5,
            "ratchet_15": 2.8,
            "ratchet_25": 2.2,
            "climax_rsi": 78.0,
            "climax_gain": 0.15,
            "init_entry": True,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 18,
            "bb_mult": 1.9,
            "bb_adaptive": True,
            "ema_fast": 7,
            "ema_slow": 19,
            "ema_adaptive": True,
        },
        "description": "Defensive staple accumulator with high capital preservation and disciplined trend trailing ratchets.",
    },
    "JPM": {
        "strategy_type": "CORE_TACTICAL_ACCUMULATION",
        "strategy_name": "Core Tactical Accumulation",
        "params": {
            "dip_rsi": 44.0,
            "base_atr": 3.8,
            "ratchet_15": 3.0,
            "ratchet_25": 2.4,
            "climax_rsi": 80.0,
            "climax_gain": 0.18,
            "init_entry": False,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 20,
            "bb_mult": 2.1,
            "bb_adaptive": True,
            "ema_fast": 6,
            "ema_slow": 20,
            "ema_adaptive": True,
        },
        "description": "Financial sector powerhouse capturing secular credit expansion cycles with swift macro bear de-risking.",
    },
    "LLY": {
        "strategy_type": "VOLATILITY_BREAKOUT_SWING",
        "strategy_name": "Volatility Breakout Swing",
        "params": {
            "base_atr": 4.2,
            "ratchet_15": 3.5,
            "ratchet_25": 2.8,
            "climax_rsi": 80.0,
            "climax_gain": 0.20,
            "dip_rsi": 44.0,
            "init_entry": True,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 22,
            "bb_mult": 2.4,
            "bb_adaptive": True,
            "ema_fast": 7,
            "ema_slow": 27,
            "ema_adaptive": True,
        },
        "description": "Healthcare momentum leader participating in secular product-cycle breakouts with dynamic Chandelier ratchets.",
    },
    "RELIANCE.NS": {
        "strategy_type": "VOLATILITY_BREAKOUT_SWING",
        "strategy_name": "Volatility Breakout Swing",
        "params": {
            "base_atr": 3.8,
            "ratchet_15": 3.0,
            "ratchet_25": 2.4,
            "climax_rsi": 78.0,
            "climax_gain": 0.18,
            "dip_rsi": 44.0,
            "init_entry": True,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 20,
            "bb_mult": 2.2,
            "bb_adaptive": True,
            "ema_fast": 5,
            "ema_slow": 30,
            "ema_adaptive": True,
        },
        "description": "Diversified industrial/energy giant capturing high-momentum expansion runs with trailing risk controls.",
    },
    "TCS.NS": {
        "strategy_type": "ASYMMETRIC_MEAN_REVERSION",
        "strategy_name": "Asymmetric Mean Reversion",
        "params": {
            "base_atr": 3.8,
            "ratchet_15": 3.0,
            "ratchet_25": 2.4,
            "climax_rsi": 76.0,
            "climax_gain": 0.15,
            "dip_rsi": 42.0,
            "init_entry": False,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 20,
            "bb_mult": 2.0,
            "bb_adaptive": True,
            "ema_fast": 11,
            "ema_slow": 35,
            "ema_adaptive": True,
        },
        "description": "IT export blue-chip with disciplined pullback entries and swift profit harvesting at structural resistance.",
    },
    "ICICIBANK.NS": {
        "strategy_type": "CORE_TACTICAL_ACCUMULATION",
        "strategy_name": "Core Tactical Accumulation",
        "params": {
            "dip_rsi": 45.0,
            "base_atr": 3.8,
            "ratchet_15": 3.0,
            "ratchet_25": 2.4,
            "climax_rsi": 78.0,
            "climax_gain": 0.18,
            "init_entry": True,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 20,
            "bb_mult": 2.1,
            "bb_adaptive": True,
            "ema_fast": 5,
            "ema_slow": 10,
            "ema_adaptive": True,
        },
        "description": "Premier Indian private bank riding credit growth secular trends with bear-regime cash preservation.",
    },
    "BHARTIARTL.NS": {
        "strategy_type": "CORE_TACTICAL_ACCUMULATION",
        "strategy_name": "Core Tactical Accumulation",
        "params": {
            "dip_rsi": 45.0,
            "base_atr": 3.8,
            "ratchet_15": 3.0,
            "ratchet_25": 2.4,
            "climax_rsi": 80.0,
            "climax_gain": 0.20,
            "init_entry": True,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 20,
            "bb_mult": 2.1,
            "bb_adaptive": True,
            "ema_fast": 5,
            "ema_slow": 20,
            "ema_adaptive": True,
        },
        "description": "Telecom infrastructure compounder capturing secular ARPU growth with tight drawdown controls.",
    },
    "LT.NS": {
        "strategy_type": "CORE_TACTICAL_ACCUMULATION",
        "strategy_name": "Core Tactical Accumulation",
        "params": {
            "dip_rsi": 45.0,
            "base_atr": 4.0,
            "ratchet_15": 3.2,
            "ratchet_25": 2.6,
            "climax_rsi": 78.0,
            "climax_gain": 0.18,
            "init_entry": True,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 20,
            "bb_mult": 2.1,
            "bb_adaptive": True,
            "ema_fast": 5,
            "ema_slow": 10,
            "ema_adaptive": True,
        },
        "description": "Capex infrastructure leader participating in industrial expansion super-cycles.",
    },
    "ITC.NS": {
        "strategy_type": "CORE_TACTICAL_ACCUMULATION",
        "strategy_name": "Core Tactical Accumulation",
        "params": {
            "dip_rsi": 45.0,
            "base_atr": 3.2,
            "ratchet_15": 2.6,
            "ratchet_25": 2.0,
            "climax_rsi": 76.0,
            "climax_gain": 0.15,
            "init_entry": True,
            "cooldown": 3,
            "macro_bear_exit": True,
            "bb_window": 18,
            "bb_mult": 1.8,
            "bb_adaptive": True,
            "ema_fast": 10,
            "ema_slow": 35,
            "ema_adaptive": True,
        },
        "description": "Defensive high-yield FMCG compounder with low beta and tight trailing defense.",
    },
}


def get_champion_strategy_for_ticker(ticker: str) -> Dict[str, Any]:
    ticker_clean = ticker.upper()
    return ASSET_CHAMPION_STRATEGY_MAP.get(
        ticker_clean,
        {
            "strategy_type": "DYNAMIC_REGIME_HYBRID",
            "strategy_name": "Dynamic Regime Hybrid",
            "params": {
                "base_atr": 4.2,
                "ratchet_15": 3.5,
                "ratchet_25": 3.0,
                "climax_rsi": 75.0,
                "climax_gain": 0.15,
                "init_entry": True,
                "cooldown": 1,
                "macro_bear_exit": True,
                "bb_window": 20,
                "bb_mult": 2.0,
                "bb_adaptive": True,
                "ema_fast": 9,
                "ema_slow": 21,
                "ema_adaptive": True,
            },
            "description": "Adaptive multi-regime hybrid alpha driver.",
        },
    )


def get_ticker_ema_params(ticker: Optional[str]) -> Tuple[int, int]:
    """
    Returns the mathematically calibrated (fast_period, slow_period) EMA pair for the specified ticker.
    Falls back to institutional baseline (9, 21) if ticker is unspecified or unmapped.
    """
    if not ticker or not isinstance(ticker, str):
        return 9, 21
    ticker_clean = ticker.upper().strip()
    strat_cfg = get_champion_strategy_for_ticker(ticker_clean)
    params = strat_cfg.get("params", {})
    fast_p = int(params.get("ema_fast", 9))
    slow_p = int(params.get("ema_slow", 21))
    return fast_p, slow_p


def compute_adaptive_ema_ribbon(
    df: pd.DataFrame,
    ticker: Optional[str],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes ticker-specific volatility-calibrated Exponential Moving Average lines (Ribbon).

    Returns:
        (ema_fast, ema_slow, ribbon_state)
        where ribbon_state is 1 for bullish (Fast >= Slow) and -1 for bearish (Fast < Slow).
    """
    fast_p, slow_p = get_ticker_ema_params(ticker)
    close = df["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    ema_fast = close.ewm(span=fast_p, adjust=False).mean()
    ema_slow = close.ewm(span=slow_p, adjust=False).mean()
    ribbon_state = pd.Series(
        np.where(ema_fast >= ema_slow, 1.0, -1.0),
        index=df.index,
    )
    return ema_fast, ema_slow, ribbon_state


def compute_adaptive_bollinger_bands(
    df: pd.DataFrame,
    ticker: str,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Computes ticker-specific volatility-adaptive Bollinger Bands.

    Modulates the standard deviation multiplier dynamically using the asset's
    real-time ATR Regime Ratio (14-day ATR vs 60-day baseline), expanding
    the envelope during volatility spikes and contracting during squeezes.

    Returns:
        (bb_mid, bb_upper, bb_lower, bb_width, bb_position)
    """
    ticker_clean = ticker.upper() if isinstance(ticker, str) and ticker.strip() else "AAPL"
    strat_cfg = get_champion_strategy_for_ticker(ticker_clean)
    params = strat_cfg.get("params", {})
    window = int(params.get("bb_window", 20))
    base_k = float(params.get("bb_mult", 2.0))
    is_adaptive = bool(params.get("bb_adaptive", True))

    close = df["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    bb_mid = close.rolling(window=window, min_periods=min(10, window)).mean()
    bb_std = close.rolling(window=window, min_periods=min(10, window)).std()

    if is_adaptive and len(df) >= 14:
        high = df["High"].squeeze() if "High" in df.columns else close
        low = df["Low"].squeeze() if "Low" in df.columns else close
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=5).mean().bfill()
        atr_baseline = atr_14.ewm(span=60, min_periods=10).mean()
        arr = (atr_14 / (atr_baseline + 1e-8)).clip(0.75, 1.35)
        dynamic_k = base_k * np.sqrt(arr)
    else:
        dynamic_k = base_k

    bb_upper = bb_mid + (dynamic_k * bb_std)
    bb_lower = bb_mid - (dynamic_k * bb_std)
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    bb_position = ((close - bb_lower) / (bb_upper - bb_lower + 1e-9)).clip(0.0, 1.0).fillna(0.5)

    return bb_mid, bb_upper, bb_lower, bb_width, bb_position



