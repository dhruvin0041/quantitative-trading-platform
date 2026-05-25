import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from src.execution.live_inference import add_upgraded_features, FEATURE_COLUMNS
from src.data_ingestion.market_data import fetch_historical_data, apply_dynamic_triple_barrier

def run_falsification_audit(tickers=["AAPL", "MSFT", "NVDA"], randomize=False, sanitize=True):
    print(f"--- STARTING ADVERSARIAL AUDIT (Actual Returns) (Sanitize: {sanitize}, Randomize: {randomize}) ---")
    
    full_start = "2021-01-01"
    full_end = "2026-05-23"
    
    spy_full = yf.download('SPY', start=full_start, end=full_end, progress=False)
    vix_full = yf.download('^VIX', start=full_start, end=full_end, progress=False)
    if isinstance(spy_full.columns, pd.MultiIndex): spy_full.columns = spy_full.columns.droplevel(1)
    if isinstance(vix_full.columns, pd.MultiIndex): vix_full.columns = vix_full.columns.droplevel(1)

    all_data = {}
    for t in tickers:
        df = fetch_historical_data(t, start_date=full_start, end_date=full_end)
        df = add_upgraded_features(df, spy_full, vix_full)
        # Calculate actual forward returns for PF (not proxy)
        df['forward_5d_ret'] = df['Close'].shift(-5) / df['Close'] - 1
        df['target_signal'] = apply_dynamic_triple_barrier(df.copy())['target_signal']
        all_data[t] = df.dropna()

    sim_start = pd.Timestamp("2024-01-01")
    sim_end = pd.Timestamp("2026-05-01")
    current_date = sim_start
    results = []
    
    while current_date < sim_end:
        next_date = current_date + timedelta(weeks=1)
        train_start = current_date - timedelta(days=365*2)
        
        X_train_list = []
        y_train_list = []
        
        for t in tickers:
            df = all_data[t]
            mask = (df.index >= train_start) & (df.index < current_date)
            train_chunk = df.loc[mask].copy()
            if sanitize: train_chunk = train_chunk.iloc[:-10]
            
            if len(train_chunk) > 100:
                X_vals = train_chunk[FEATURE_COLUMNS].values
                y_vals = train_chunk['target_signal'].values
                if randomize: np.random.shuffle(y_vals)
                X_train_list.append(X_vals)
                y_train_list.append(y_vals)
        
        if not X_train_list:
            current_date = next_date
            continue
            
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, objective='multi:softprob', num_class=3, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        for t in tickers:
            df = all_data[t]
            test_mask = (df.index >= current_date) & (df.index < next_date)
            test_chunk = df.loc[test_mask]
            if test_chunk.empty: continue
            
            X_test_scaled = scaler.transform(test_chunk[FEATURE_COLUMNS].values)
            y_test = test_chunk['target_signal'].values
            fwd_ret = test_chunk['forward_5d_ret'].values
            
            probs = model.predict_proba(X_test_scaled)
            preds = np.argmax(probs, axis=1)
            
            for i in range(len(test_chunk)):
                if preds[i] != 1: # Directional signals
                    actual_return = fwd_ret[i]
                    # If signal is SELL (0), return is negative of price move
                    trade_return = actual_return if preds[i] == 2 else -actual_return
                    results.append({
                        'is_correct': (preds[i] == 2 and actual_return > 0) or (preds[i] == 0 and actual_return < 0),
                        'ret': trade_return
                    })
        
        current_date = next_date

    res_df = pd.DataFrame(results)
    if res_df.empty: return {"wr": 0, "pf": 0}
    
    wr = res_df['is_correct'].mean() * 100
    profits = res_df[res_df['ret'] > 0]['ret'].sum()
    losses = abs(res_df[res_df['ret'] < 0]['ret'].sum())
    pf = profits / losses if losses > 0 else 0
    return {"wr": wr, "pf": pf, "count": len(res_df)}

if __name__ == "__main__":
    print("=== FINAL INTEGRITY AUDIT (ACTUAL PRICE MOVES) ===")
    
    # 1. Clean Simulation
    result_clean = run_falsification_audit(sanitize=True, randomize=False)
    print(f"CLEAN SIMULATION: Win Rate {result_clean['wr']:.1f}%, PF {result_clean['pf']:.2f}")
    
    # 2. Randomization Test
    result_rand = run_falsification_audit(sanitize=True, randomize=True)
    print(f"RANDOMIZED TEST: Win Rate {result_rand['wr']:.1f}%, PF {result_rand['pf']:.2f}")
