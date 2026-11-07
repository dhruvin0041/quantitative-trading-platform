import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CrossSectionalUniverseRanker:
    """
    Phase 6: Institutional Cross-Sectional Universe Ranker.
    Evaluates relative strength, volatility-adjusted momentum, and residual alpha
    across the entire multi-asset universe (23 institutional assets).

    Core Principles:
    - Alpha in institutional quants is cross-sectional (relative), not isolated.
    - Prevents buying laggards even if an individual asset's technicals flash green.
    - Prioritizes capital deployment into top-decile relative performers.
    """

    def __init__(self, benchmark_ticker: str = "SPY"):
        self.benchmark_ticker = benchmark_ticker

    def compute_cross_sectional_ranks(
        self,
        universe_prices: Dict[str, pd.Series],
        lookback_days: int = 20,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Computes cross-sectional metrics across a universe of price series.

        Parameters:
        -----------
        universe_prices : Dict[str, pd.Series]
            Mapping of ticker -> daily close price Series.
        lookback_days : int
            Lookback window for momentum and volatility (default 20 trading days).

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Mapping of ticker -> {
                'rank_percentile': float [0.0, 1.0],
                'decile': int [1, 10],
                'momentum_zscore': float,
                'vol_adjusted_momentum_zscore': float,
                'residual_alpha': float,
                'is_leader': bool,
                'is_laggard': bool,
            }
        """
        if not universe_prices or len(universe_prices) < 2:
            return {}

        # 1. Align and compute returns
        df_prices = pd.DataFrame(universe_prices).dropna(how="all").ffill().bfill()
        if len(df_prices) < lookback_days + 2:
            return {}

        # 20-day returns and annualized volatility
        pct_returns = df_prices.pct_change(lookback_days).iloc[-1]
        daily_returns = df_prices.pct_change().tail(lookback_days)
        vols = daily_returns.std() * np.sqrt(252)

        # Risk-adjusted returns (Sharpe-like ratio)
        sharpe_like = pct_returns / (vols.replace(0, np.nan) + 1e-6)

        # 2. Benchmark Residual Alpha (if benchmark present)
        residual_alpha = {}
        bm_ret = (
            pct_returns[self.benchmark_ticker]
            if self.benchmark_ticker in pct_returns and not np.isnan(pct_returns[self.benchmark_ticker])
            else pct_returns.median()
        )

        for ticker in df_prices.columns:
            ret = pct_returns[ticker]
            if np.isnan(ret):
                residual_alpha[ticker] = 0.0
            else:
                residual_alpha[ticker] = float(ret - bm_ret)

        # 3. Cross-Sectional Z-Scores
        mean_ret = pct_returns.mean()
        std_ret = pct_returns.std()
        if std_ret < 1e-7 or np.isnan(std_ret):
            mom_z = pd.Series(0.0, index=pct_returns.index)
        else:
            mom_z = (pct_returns - mean_ret) / std_ret

        mean_sharpe = sharpe_like.mean()
        std_sharpe = sharpe_like.std()
        if std_sharpe < 1e-7 or np.isnan(std_sharpe):
            vol_adj_z = pd.Series(0.0, index=sharpe_like.index)
        else:
            vol_adj_z = (sharpe_like - mean_sharpe) / std_sharpe

        # 4. Composite Score & Percentile Rank
        composite_score = (0.5 * mom_z.fillna(0.0)) + (0.5 * vol_adj_z.fillna(0.0))
        n_assets = len(composite_score)
        ranks = composite_score.rank(ascending=True, method="average")
        percentiles = (ranks - 1.0) / max(n_assets - 1.0, 1.0)

        results = {}
        for ticker in df_prices.columns:
            pct = float(percentiles.get(ticker, 0.5))
            decile = int(min(max(np.ceil(pct * 10), 1), 10))
            z_mom = float(mom_z.get(ticker, 0.0))
            z_vol = float(vol_adj_z.get(ticker, 0.0))
            alpha = float(residual_alpha.get(ticker, 0.0))

            results[ticker] = {
                "rank_percentile": round(pct, 4),
                "decile": decile,
                "momentum_zscore": round(z_mom, 3),
                "vol_adjusted_momentum_zscore": round(z_vol, 3),
                "residual_alpha_pct": round(alpha * 100.0, 2),
                "is_leader": bool(pct >= 0.60),
                "is_laggard": bool(pct < 0.35),
            }

        return results

    def gate_signal_with_cross_sectional_rank(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        rank_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Adjusts or gates model signals based on cross-sectional universe positioning.
        """
        if not rank_data:
            return {
                "gated_signal": signal,
                "adjusted_confidence": confidence,
                "cross_sectional_status": "NEUTRAL_STANDALONE",
                "execution_permitted": True,
                "reasoning": "Cross-sectional universe data not available; standalone execution permitted.",
            }

        pct = rank_data.get("rank_percentile", 0.5)
        decile = rank_data.get("decile", 5)
        is_laggard = rank_data.get("is_laggard", False)
        is_leader = rank_data.get("is_leader", False)

        gated_signal = signal
        adjusted_conf = confidence
        permitted = True
        status = "NEUTRAL"
        reasoning = f"Decile {decile} ({pct * 100:.1f}th percentile)."
        sig_clean = str(signal).upper().strip()

        if sig_clean == "BUY":
            if is_laggard:
                status = "LAGGARD_SUPPRESSION"
                # Penalize confidence by 25% for buying bottom-tier relative performers
                adjusted_conf = max(confidence * 0.75, 0.0)
                reasoning = (
                    f"Cross-sectional Laggard Warning: {ticker} ranks in bottom {pct * 100:.1f}% "
                    f"of universe. BUY confidence penalized to prevent catching falling knives."
                )
                if pct < 0.15:
                    gated_signal = "HOLD"
                    permitted = False
                    reasoning = (
                        f"Cross-sectional VETO: {ticker} ranks in bottom 15% (Decile {decile}). "
                        f"System forbids deploying long capital into extreme relative laggards."
                    )
            elif is_leader:
                status = "LEADER_CONFIRMED"
                # Boost confidence by 10% (capped at 98%)
                adjusted_conf = min(confidence * 1.10, 98.0)
                reasoning = (
                    f"Cross-sectional Leader: {ticker} ranks in top {pct * 100:.1f}% (Decile {decile}). "
                    f"Strong relative strength confirms institutional accumulation."
                )

        elif sig_clean == "SELL":
            if is_laggard:
                status = "BEARISH_LEADERSHIP"
                adjusted_conf = min(confidence * 1.10, 98.0)
                reasoning = (
                    f"Cross-sectional Weakness Confirmed: {ticker} ranks in bottom {pct * 100:.1f}%. "
                    f"High conviction short / exit."
                )
            elif is_leader:
                status = "COUNTER_TREND_WARNING"
                adjusted_conf = max(confidence * 0.80, 0.0)
                reasoning = (
                    f"Short Warning: {ticker} ranks in top {pct * 100:.1f}% relative strength. "
                    f"Shorting relative market leaders has low statistical expectancy."
                )

        return {
            "gated_signal": gated_signal,
            "adjusted_confidence": round(float(adjusted_conf), 2),
            "cross_sectional_status": status,
            "execution_permitted": permitted,
            "reasoning": reasoning,
            "rank_data": rank_data,
        }
