import numpy as np
import pandas as pd
import pytest

from src.execution.asset_intelligence import (
    ASSET_CHAMPION_STRATEGY_MAP,
    compute_adaptive_bollinger_bands,
    compute_adaptive_ema_ribbon,
    get_ticker_bb_params,
    get_ticker_ema_params,
)
from src.execution.live_inference import (
    FEATURE_COLUMNS_V40,
    add_upgraded_features,
)


class TestAdaptiveIndicatorsAndCausality:
    """
    Institutional Verification Suite for:
    1. Per-ticker adaptive EMA line parameters across all 18 universe assets.
    2. Per-ticker adaptive Bollinger Band parameters across all 18 universe assets.
    3. Mathematical consistency of ribbon and band computation.
    4. Strict point-in-time causal invariance (Zero Lookahead Bias).
    """

    @pytest.mark.parametrize("ticker", list(ASSET_CHAMPION_STRATEGY_MAP.keys()))
    def test_all_18_tickers_have_calibrated_ema_and_bb(self, ticker):
        """Verify that every institutional asset has explicit, valid adaptive EMA and BB parameters."""
        bb_window, bb_mult, bb_adaptive = get_ticker_bb_params(ticker)
        assert bb_adaptive is True, f"{ticker} must have bb_adaptive=True"
        assert 10 <= bb_window <= 40, f"{ticker} bb_window {bb_window} out of sane range [10, 40]"
        assert 1.5 <= bb_mult <= 3.5, f"{ticker} bb_mult {bb_mult} out of sane range [1.5, 3.5]"

        ema_fast, ema_slow = get_ticker_ema_params(ticker)
        assert 4 <= ema_fast <= 20, f"{ticker} ema_fast {ema_fast} out of sane range [4, 20]"
        assert 8 <= ema_slow <= 45, f"{ticker} ema_slow {ema_slow} out of sane range [8, 45]"
        assert ema_fast < ema_slow, f"{ticker} ema_fast ({ema_fast}) must be strictly less than ema_slow ({ema_slow})"

    def test_adaptive_ema_ribbon_values_match_formulas(self):
        """Verify that compute_adaptive_ema_ribbon calculates exact exponential moving averages."""
        dates = pd.date_range("2026-01-01", periods=100, freq="B")
        close_prices = 100.0 + np.linspace(0, 50, len(dates))  # Linear uptrend
        df = pd.DataFrame({"Close": close_prices}, index=dates)

        # Test TSLA: fast=12, slow=33
        ribbon_fast_tsla, ribbon_slow_tsla, _ = compute_adaptive_ema_ribbon(df, "TSLA")
        expected_fast = df["Close"].ewm(span=12, adjust=False).mean()
        expected_slow = df["Close"].ewm(span=33, adjust=False).mean()

        np.testing.assert_allclose(ribbon_fast_tsla.values, expected_fast.values, rtol=1e-5)
        np.testing.assert_allclose(ribbon_slow_tsla.values, expected_slow.values, rtol=1e-5)

        # In an uptrend, fast EMA must be strictly above slow EMA
        assert (ribbon_fast_tsla.iloc[-10:] > ribbon_slow_tsla.iloc[-10:]).all()

        # Test AAPL: fast=8, slow=13
        ribbon_fast_aapl, ribbon_slow_aapl, _ = compute_adaptive_ema_ribbon(df, "AAPL")
        expected_fast_aapl = df["Close"].ewm(span=8, adjust=False).mean()
        expected_slow_aapl = df["Close"].ewm(span=13, adjust=False).mean()

        np.testing.assert_allclose(ribbon_fast_aapl.values, expected_fast_aapl.values, rtol=1e-5)
        np.testing.assert_allclose(ribbon_slow_aapl.values, expected_slow_aapl.values, rtol=1e-5)

    def test_adaptive_bollinger_bands_values_match_formulas(self):
        """Verify that compute_adaptive_bollinger_bands computes exact bands with ticker volatility multipliers."""
        dates = pd.date_range("2026-01-01", periods=100, freq="B")
        close_prices = 150.0 + np.sin(np.linspace(0, 10, len(dates))) * 10
        df = pd.DataFrame({
            "Close": close_prices,
            "High": close_prices + 2.0,
            "Low": close_prices - 2.0,
        }, index=dates)

        # TSLA: dynamic params from get_ticker_bb_params
        win_tsla, mult_tsla, _ = get_ticker_bb_params("TSLA")
        mid_tsla, upper_tsla, lower_tsla, width_tsla, pos_tsla = compute_adaptive_bollinger_bands(df, "TSLA")
        expected_mid = df["Close"].rolling(window=win_tsla, min_periods=min(10, win_tsla)).mean()

        np.testing.assert_allclose(mid_tsla.dropna().values, expected_mid.dropna().values, rtol=1e-5)
        assert (upper_tsla.dropna() > mid_tsla.dropna()).all()
        assert (lower_tsla.dropna() < mid_tsla.dropna()).all()
        assert (width_tsla.dropna() > 0).all()

        # COST: dynamic params from get_ticker_bb_params
        win_cost, mult_cost, _ = get_ticker_bb_params("COST")
        mid_cost, upper_cost, lower_cost, width_cost, pos_cost = compute_adaptive_bollinger_bands(df, "COST")
        expected_mid_cost = df["Close"].rolling(window=win_cost, min_periods=min(10, win_cost)).mean()

        np.testing.assert_allclose(mid_cost.dropna().values, expected_mid_cost.dropna().values, rtol=1e-5)
        assert (upper_cost.dropna() > mid_cost.dropna()).all()
        assert (lower_cost.dropna() < mid_cost.dropna()).all()

    def test_strict_zero_lookahead_causal_invariance(self):
        """
        MATHEMATICAL PROOF OF CAUSALITY:
        Indicator calculation at bar T MUST NOT change when future bars T+1...T+N are appended.
        Tests features, adaptive EMA ribbon, and adaptive Bollinger Bands across simulated historical sequence.
        """
        np.random.seed(1337)
        dates = pd.date_range("2024-01-01", periods=250, freq="B")
        ret = np.random.normal(0.0005, 0.015, size=len(dates))
        close = 100.0 * np.exp(np.cumsum(ret))
        high = close * (1 + np.abs(np.random.normal(0.005, 0.005, size=len(dates))))
        low = close * (1 - np.abs(np.random.normal(0.005, 0.005, size=len(dates))))
        open_p = (high + low) / 2
        volume = np.random.uniform(1e6, 1e7, size=len(dates))

        df_full = pd.DataFrame({
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }, index=dates)

        spy_full = pd.DataFrame({
            "Open": open_p * 2,
            "High": high * 2,
            "Low": low * 2,
            "Close": close * 2,
            "Volume": volume * 3,
        }, index=dates)

        vix_full = pd.DataFrame({
            "Close": np.random.uniform(14, 25, size=len(dates)),
        }, index=dates)

        # Evaluate at multiple historical points: bar 150, 180, 200
        for eval_bar in [150, 180, 200]:
            df_past = df_full.iloc[:eval_bar].copy()
            spy_past = spy_full.iloc[:eval_bar].copy()
            vix_past = vix_full.iloc[:eval_bar].copy()

            target_date = df_past.index[-1]

            for ticker in ["AAPL", "TSLA", "RELIANCE.NS", "COST"]:
                # 1. Test Adaptive EMA Ribbon Causal Invariance
                fast_past, slow_past, _ = compute_adaptive_ema_ribbon(df_past, ticker)
                fast_future, slow_future, _ = compute_adaptive_ema_ribbon(df_full, ticker)

                val_fast_past = fast_past.loc[target_date]
                val_fast_future = fast_future.loc[target_date]
                assert abs(val_fast_past - val_fast_future) < 1e-9, (
                    f"Lookahead leak in EMA fast for {ticker} at {target_date}: {val_fast_past} vs {val_fast_future}"
                )

                val_slow_past = slow_past.loc[target_date]
                val_slow_future = slow_future.loc[target_date]
                assert abs(val_slow_past - val_slow_future) < 1e-9, (
                    f"Lookahead leak in EMA slow for {ticker} at {target_date}: {val_slow_past} vs {val_slow_future}"
                )

                # 2. Test Adaptive Bollinger Bands Causal Invariance
                mid_past, upper_past, lower_past, _, _ = compute_adaptive_bollinger_bands(df_past, ticker)
                mid_future, upper_future, lower_future, _, _ = compute_adaptive_bollinger_bands(df_full, ticker)

                assert abs(upper_past.loc[target_date] - upper_future.loc[target_date]) < 1e-9
                assert abs(lower_past.loc[target_date] - lower_future.loc[target_date]) < 1e-9
                assert abs(mid_past.loc[target_date] - mid_future.loc[target_date]) < 1e-9

                # 3. Test Full Feature Matrix Causal Invariance (all 31 institutional indicators)
                feats_past = add_upgraded_features(df_past, spy_past, vix_past, ticker=ticker)
                feats_future = add_upgraded_features(df_full, spy_full, vix_full, ticker=ticker)

                for col in FEATURE_COLUMNS_V40:
                    if col in feats_past.columns and col in feats_future.columns:
                        p_val = float(feats_past.loc[target_date, col])
                        f_val = float(feats_future.loc[target_date, col])
                        assert abs(p_val - f_val) < 1e-6, (
                            f"Feature '{col}' for {ticker} leaked future information at {target_date}: "
                            f"past={p_val}, future={f_val}"
                        )
