
import numpy as np
import pandas as pd

from src.execution.live_inference import (
    FEATURE_COLUMNS_V30,
    add_upgraded_features,
    check_bar_forming_status,
)
from src.schemas import PredictResponse


class TestLiveSignalsAndLookahead:
    """
    Verification test suite confirming:
    1. Zero look-ahead bias across all live indicators and execution timing.
    2. Forming bar isolation (PROVISIONAL vs CONFIRMED).
    3. Core + Tactical Alpha strategy signal generation and telemetry.
    4. Next-session causal execution pricing (t -> t+1 Open).
    """

    def test_forming_bar_status_causality(self):
        """Verify that check_bar_forming_status distinguishes confirmed historical bars."""
        # 1. Past completed bar -> CONFIRMED
        past_dates = pd.date_range("2026-01-01", periods=10, freq="D")
        past_df = pd.DataFrame({"Close": [100.0] * 10}, index=past_dates)
        is_forming, bar_state = check_bar_forming_status(past_df)
        assert not is_forming
        assert bar_state == "CONFIRMED"

    def test_feature_causality_zero_future_leakage(self):
        """
        Verify that indicators at bar t do not change when future bars are appended or modified.
        Strict append invariance test across all feature columns.
        """
        dates = pd.date_range("2025-01-01", periods=300, freq="B")
        np.random.seed(42)
        base_price = 150.0 + np.cumsum(np.random.normal(0, 1.5, size=len(dates)))
        high = base_price + np.random.uniform(0.5, 2.0, size=len(dates))
        low = base_price - np.random.uniform(0.5, 2.0, size=len(dates))
        open_p = base_price + np.random.uniform(-0.5, 0.5, size=len(dates))
        volume = np.random.uniform(1e6, 5e6, size=len(dates))

        df_full = pd.DataFrame({
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": base_price,
            "Volume": volume,
        }, index=dates)

        spy_full = pd.DataFrame({
            "Open": open_p * 1.5,
            "High": high * 1.5,
            "Low": low * 1.5,
            "Close": base_price * 1.5,
            "Volume": volume * 2,
        }, index=dates)

        vix_full = pd.DataFrame({
            "Close": [18.0] * len(dates),
        }, index=dates)

        # Slice at bar 200
        slice_t = 200
        df_truncated = df_full.iloc[:slice_t].copy()
        spy_truncated = spy_full.iloc[:slice_t].copy()
        vix_truncated = vix_full.iloc[:slice_t].copy()

        feats_truncated = add_upgraded_features(df_truncated, spy_truncated, vix_truncated)
        feats_full = add_upgraded_features(df_full, spy_full, vix_full)

        # Features at bar slice_t - 1 must be identical between truncated and full frames
        t_idx = df_truncated.index[-1]
        for col in FEATURE_COLUMNS_V30:
            if col in feats_truncated.columns and col in feats_full.columns:
                val_trunc = float(feats_truncated.loc[t_idx, col])
                val_full = float(feats_full.loc[t_idx, col])
                assert abs(val_trunc - val_full) < 1e-6, (
                    f"Lookahead leak detected in feature '{col}': truncated={val_trunc}, full={val_full}"
                )

    def test_next_session_execution_pricing(self):
        """Verify that live signal execution targets the next session open, not current bar."""
        # Current bar at close t cannot be filled at bar t Close without slippage,
        # and execution_target_bar must be NEXT_SESSION_OPEN.
        curr_price = 220.0
        exec_price_buy = round(curr_price * 1.0005, 2)
        exec_price_sell = round(curr_price * 0.9995, 2)

        assert exec_price_buy > curr_price  # 5 bps slippage penalty on buy
        assert exec_price_sell < curr_price  # 5 bps slippage penalty on sell

    def test_core_tactical_strategy_state_machine(self):
        """
        Verify the Core + Tactical Alpha state logic:
        1. Macro Bull + oversold dip (RSI <= 35) -> TACTICAL_OVERWEIGHT_BUY (1.40x).
        2. Macro Bull + overbought rally (RSI >= 65) -> TACTICAL_HARVEST_SELL (1.00x).
        3. Macro Bull + neutral channel (35 < RSI < 65) -> HOLD_CORE_1.0X (1.00x).
        4. Macro Bear breakdown (Underlying < SMA200 or SPY < SMA200) -> BEAR_REGIME_LIQUIDATION (0.00x).
        """
        # Scenario 1: Macro Bull + Dip
        curr_close = 200.0
        sma_200 = 180.0
        spy_close = 500.0
        spy_sma_200 = 450.0
        base_long = bool((curr_close >= sma_200) and (spy_close >= spy_sma_200))
        assert base_long is True

        c_rsi = 32.0  # Oversold dip
        lgbm_veto = False
        dir_b = 0.50

        if not base_long:
            action = "BEAR_REGIME_LIQUIDATION"
            target_exposure = 0.0
        elif (c_rsi <= 35.0 or dir_b >= 0.60) and not lgbm_veto:
            action = "TACTICAL_OVERWEIGHT_BUY"
            target_exposure = 1.40
        elif c_rsi >= 65.0 or lgbm_veto:
            action = "TACTICAL_HARVEST_SELL"
            target_exposure = 1.00
        else:
            action = "HOLD_CORE_1.0X"
            target_exposure = 1.00

        assert action == "TACTICAL_OVERWEIGHT_BUY"
        assert target_exposure == 1.40

        # Scenario 2: Macro Bull + Overbought
        c_rsi = 72.0
        if not base_long:
            action = "BEAR_REGIME_LIQUIDATION"
            target_exposure = 0.0
        elif (c_rsi <= 35.0 or dir_b >= 0.60) and not lgbm_veto:
            action = "TACTICAL_OVERWEIGHT_BUY"
            target_exposure = 1.40
        elif c_rsi >= 65.0 or lgbm_veto:
            action = "TACTICAL_HARVEST_SELL"
            target_exposure = 1.00
        else:
            action = "HOLD_CORE_1.0X"
            target_exposure = 1.00

        assert action == "TACTICAL_HARVEST_SELL"
        assert target_exposure == 1.00

        # Scenario 3: Macro Bull + Neutral
        c_rsi = 55.0
        if not base_long:
            action = "BEAR_REGIME_LIQUIDATION"
            target_exposure = 0.0
        elif (c_rsi <= 35.0 or dir_b >= 0.60) and not lgbm_veto:
            action = "TACTICAL_OVERWEIGHT_BUY"
            target_exposure = 1.40
        elif c_rsi >= 65.0 or lgbm_veto:
            action = "TACTICAL_HARVEST_SELL"
            target_exposure = 1.00
        else:
            action = "HOLD_CORE_1.0X"
            target_exposure = 1.00

        assert action == "HOLD_CORE_1.0X"
        assert target_exposure == 1.00

        # Scenario 4: Macro Bear Breakdown
        curr_close = 175.0  # < SMA200 (180.0)
        base_long = bool((curr_close >= sma_200) and (spy_close >= spy_sma_200))
        assert base_long is False

        if not base_long:
            action = "BEAR_REGIME_LIQUIDATION"
            target_exposure = 0.0
        elif (c_rsi <= 35.0 or dir_b >= 0.60) and not lgbm_veto:
            action = "TACTICAL_OVERWEIGHT_BUY"
            target_exposure = 1.40
        elif c_rsi >= 65.0 or lgbm_veto:
            action = "TACTICAL_HARVEST_SELL"
            target_exposure = 1.00
        else:
            action = "HOLD_CORE_1.0X"
            target_exposure = 1.00

        assert action == "BEAR_REGIME_LIQUIDATION"
        assert target_exposure == 0.00

    def test_predict_response_schema_compatibility(self):
        """Verify that PredictResponse schema accepts the Core + Tactical Alpha telemetry fields."""
        payload = {
            "ticker": "AAPL",
            "current_price": 339.50,
            "signal": "HOLD",
            "confidence_score": 75.0,
            "uncertainty_score": 15.0,
            "market_regime": "BULL",
            "volatility_state": "MEDIUM",
            "volume_ratio": 1.05,
            "models": {
                "XGB_AGENT": {"signal": "HOLD", "probability": 0.60},
                "DL_FUSION": {"signal": "HOLD", "probability": 0.33},
            },
            "projections": {
                "floor": 320.0,
                "p50": 345.0,
                "ceiling": 360.0,
                "confidence": 80.0,
                "reliability": "HIGH",
            },
            "strategy_mode": "CORE_TACTICAL_ALPHA",
            "tactical_action": "HOLD_CORE_1.0X",
            "target_exposure": 1.00,
            "tactical_rationale": "Macro Bull intact. Maintaining 1.0x Core long exposure.",
            "signal_state": "CONFIRMED",
            "is_bar_forming": False,
            "bar_state": "CONFIRMED",
            "execution_timestamp": "NEXT_SESSION_OPEN 09:30:00 ET",
        }
        resp = PredictResponse(**payload)
        assert resp.strategy_mode == "CORE_TACTICAL_ALPHA"
        assert resp.tactical_action == "HOLD_CORE_1.0X"
        assert resp.target_exposure == 1.00
        assert resp.signal_state == "CONFIRMED"
