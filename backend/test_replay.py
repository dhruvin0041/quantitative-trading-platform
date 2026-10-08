import pandas as pd
import yfinance as yf
from src.execution.inference_service import InferenceService
from src.models.model_loader import ModelManager

class Dummy: pass

mm = ModelManager()
srv = InferenceService(mm, Dummy(), Dummy(), Dummy(), Dummy(), Dummy(), Dummy(), Dummy(), Dummy())

ticker_df = yf.download('AAPL', start='2026-06-01', end='2026-10-08')
spy_df = yf.download('SPY', start='2026-06-01', end='2026-10-08')

res = srv._replay_causal_ml_signals('AAPL', ticker_df, spy_df)
print(f'Got {len(res)} signals')
for r in res:
    if '2026-07' in r['bar_timestamp']:
        print(r['bar_timestamp'], r['signal'], r['confidence'], r['metadata'])
