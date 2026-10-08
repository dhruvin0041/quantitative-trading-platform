import pandas as pd
import yfinance as yf
from src.execution.live_inference import ModelManager

mm = ModelManager()
ticker_df = yf.download('AAPL', start='2026-06-01', end='2026-08-30')
df_feat = mm.engineer_features(ticker_df)

for i in range(3, len(df_feat)):
    cur_c = df_feat['close'].iloc[i]
    cur_rsi = df_feat['rsi_14'].iloc[i]
    cur_bb = df_feat['bb_pos'].iloc[i]
    cur_sma200 = df_feat['sma_200'].iloc[i] if 'sma_200' in df_feat.columns else cur_c
    
    cur_l = df_feat['low'].iloc[i]
    prev_l = df_feat['low'].iloc[i-1]
    prev2_l = df_feat['low'].iloc[i-2]
    
    is_trough = (cur_c > prev_l) and (prev_l <= prev2_l or cur_l <= prev_l) and (cur_rsi < 45.0 or cur_bb < 0.20)
    is_dip_oversold = (cur_rsi < 40.0) or (cur_bb < 0.15)
    
    date = df_feat.index[i].strftime('%Y-%m-%d')
    if '2026-07' in date or ('2026-06' in date and i > len(df_feat)-40):
        print(f"{date} - C:{cur_c:.2f}, RSI:{cur_rsi:.2f}, BB:{cur_bb:.2f}, L:{cur_l:.2f}, prev_L:{prev_l:.2f}, p2_L:{prev2_l:.2f}, dip:{is_dip_oversold}, tr:{is_trough}")
