
with open('src/execution/inference_service.py', 'r') as f:
    content = f.read()

old_tft = '''        # TFT Projections
        tft_preds = self.mm.tft_model.predict(ts_sequence, verbose=0)[0]
        constrained_rets = np.clip(tft_preds, -0.20, 0.20)
        is_point_forecast = np.all(np.isclose(constrained_rets, constrained_rets[0]))'''

new_tft = '''        # TFT Projections via ForecastCalibrationEngine
        tft_preds = self.mm.tft_model.predict(ts_sequence, verbose=0)[0]
        recent_vol = float(tech_snapshot.get(\
ATR\\, current_price * 0.02) / current_price)

        forecast_data = self.forecast_engine.calibrate_forecast(
            raw_forecasts=tft_preds,
            current_price=current_price,
            atr=float(tech_snapshot.get(\\ATR\\, current_price * 0.02)),
            volatility=recent_vol,
            asset_class=asset_class,
            regime=regime_detailed
        )

        constrained_rets = [forecast_data[\\p10_return\\], 0, forecast_data[\\p50_return\\], 0, forecast_data[\\p90_return\\]]
        is_point_forecast = False'''

content = content.replace(old_tft, new_tft)

with open('src/execution/inference_service.py', 'w') as f:
    f.write(content)


