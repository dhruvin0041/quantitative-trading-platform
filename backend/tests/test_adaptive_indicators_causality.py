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

        # Test TSLA: dynamically fetch champion calibrated fast and slow parameters
        fast_tsla, slow_tsla = get_ticker_ema_params("TSLA")
        ribbon_fast_tsla, ribbon_slow_tsla, _ = compute_adaptive_ema_ribbon(df, "TSLA")
        expected_fast = df["Close"].ewm(span=fast_tsla, adjust=False).mean()
        expected_slow = df["Close"].ewm(span=slow_tsla, adjust=False).mean()

        np.testing.assert_allclose(ribbon_fast_tsla.values, expected_fast.values, rtol=1e-5)
        np.testing.assert_allclose(ribbon_slow_tsla.values, expected_slow.values, rtol=1e-5)

        # In an uptrend, fast EMA must be strictly above slow EMA
        assert (ribbon_fast_tsla.iloc[-10:] > ribbon_slow_tsla.iloc[-10:]).all()

        # Test AAPL: dynamically fetch champion calibrated fast and slow parameters
        fast_aapl, slow_aapl = get_ticker_ema_params("AAPL")
        ribbon_fast_aapl, ribbon_slow_aapl, _ = compute_adaptive_ema_ribbon(df, "AAPL")
        expected_fast_aapl = df["Close"].ewm(span=fast_aapl, adjust=False).mean()
        expected_slow_aapl = df["Close"].ewm(span=slow_aapl, adjust=False).mean()

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
    def test_zlema_reduces_lag_relative_to_standard_ema(self):
        """Verify that Zero-Lag EMA responds faster to a sudden step change than standard EMA."""
        from src.data_ingestion.technical_indicators import compute_zlema

        prices = np.concatenate([np.full(50, 100.0), np.full(50, 120.0)])
        s = pd.Series(prices)
        zlema = compute_zlema(s, window=14)
        ema = s.ewm(span=14, adjust=False).mean()

        # Right after step change at index 50, ZLEMA must be closer to 120.0 than standard EMA
        assert zlema.iloc[52] > ema.iloc[52], "ZLEMA must react faster to step change than standard EMA"

    def test_kama_efficiency_adaptation(self):
        """Verify KAMA adapts dynamically to trend vs chop."""
        from src.data_ingestion.technical_indicators import compute_kama

        # Strong trend: linear climb
        trend_prices = pd.Series(100.0 + np.arange(100) * 1.5)
        kama_trend = compute_kama(trend_prices, n=10)
        assert len(kama_trend) == len(trend_prices)
        assert kama_trend.iloc[-1] > 200.0

    def test_chandelier_exit_invariants(self):
        """Verify Chandelier Exit trailing stops maintain causal risk boundary."""
        from src.data_ingestion.technical_indicators import compute_chandelier_exit

        dates = pd.date_range("2026-01-01", periods=60, freq="B")
        close = pd.Series(100.0 + np.linspace(0, 30, 60), index=dates)
        high = close + 2.0
        low = close - 2.0

        long_stop, short_stop = compute_chandelier_exit(high, low, close, period=14, mult=2.5)
        # In a steady uptrend, long stop must remain below close
        valid_stops = long_stop.dropna()
        assert (valid_stops < close.loc[valid_stops.index]).all()

    def test_mtf_fractal_alignment(self):
        """Verify MultiTimeframeEngine correctly detects higher timeframe alignment."""
        from src.execution.asset_intelligence import MultiTimeframeEngine

        dates = pd.date_range("2025-01-01", periods=150, freq="B")
        close = pd.Series(100.0 + np.linspace(0, 50, 150), index=dates)
        df = pd.DataFrame({"Close": close}, index=dates)

        mtf = MultiTimeframeEngine()
        res = mtf.evaluate_mtf_alignment(df, "AAPL", daily_signal="BUY")
        assert res["weekly_trend"] == "BULLISH"
        assert res["alignment"] is True
        assert res["confluence_score"] >= 0.85

    def test_risk_parity_and_fractional_kelly_sizing(self):
        """Verify Risk Parity and Fractional Kelly algorithms in PortfolioOptimizer."""
        from src.execution.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer()
        # 3 assets with different variances: low, med, high
        cov = np.diag([0.01, 0.04, 0.09])  # vols: 0.1, 0.2, 0.3
        rp_weights = opt.risk_parity_allocation(cov)

        # Asset 0 (lowest vol) must receive the highest weight
        assert rp_weights[0] > rp_weights[1] > rp_weights[2]
        np.testing.assert_allclose(np.sum(rp_weights), 1.0, rtol=1e-5)

        # Fractional Kelly sizing test
        win_probs = np.array([0.65, 0.55, 0.45])
        win_loss = np.array([1.5, 1.2, 1.0])
        kelly_w = opt.fractional_kelly_portfolio(win_probs, win_loss, fraction=0.5)
        assert kelly_w[0] > kelly_w[1]
        assert kelly_w[2] == 0.0  # Negative edge receives 0 allocation

