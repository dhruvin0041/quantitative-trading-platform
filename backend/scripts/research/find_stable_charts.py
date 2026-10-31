"""
Research Script: Quantitative Stability Screening and Alpha Performance.
Finds the most stable charts (low volatility, smooth trend like AAPL)
both within the existing universe and outside candidates (US and Indian markets).
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

US_CANDIDATES = [
    # Existing
    "AAPL", "MSFT", "GOOGL", "META", "COST", "WMT", "JPM", "LLY", "NVDA", "AMZN", "TSLA", "AMD",
    # Outside candidates
    "BRK-B", "V", "MA", "JNJ", "PG", "HD"
]

INDIA_CANDIDATES = [
    # Existing
    "LT.NS", "RELIANCE.NS", "TCS.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS",
    # Outside candidates
    "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", "SUNPHARMA.NS", "TITAN.NS", "KOTAKBANK.NS"
]


def calculate_stability_metrics(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 100:
        return {}

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # 1. Daily log returns & Annualized Volatility
    log_ret = np.log(close / close.shift(1)).dropna()
    ann_vol = float(log_ret.std() * np.sqrt(252) * 100.0)

    # 2. Normalized ATR (14-day)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    norm_atr = float((atr14 / close).dropna().mean() * 100.0)

    # 3. Maximum Drawdown
    cummax = close.cummax()
    drawdown = (close - cummax) / cummax
    max_dd = float(drawdown.min() * 100.0)

    # 4. Trend Smoothness (R^2 of log price vs time)
    x = np.arange(len(close))
    y = np.log(close.values)
    corr = np.corrcoef(x, y)[0, 1]
    r_squared = float(corr ** 2) if not np.isnan(corr) else 0.0

    # Composite Stability Score (Lower Vol, Lower ATR, Lower DD, Higher R^2)
    # Higher score = more stable trend like AAPL
    stability_score = round((r_squared * 50.0) - (ann_vol * 0.8) - (norm_atr * 5.0) + (max_dd * 0.3 + 30.0), 2)

    return {
        "ann_vol_pct": round(ann_vol, 2),
        "norm_atr_pct": round(norm_atr, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "trend_r2": round(r_squared, 3),
        "stability_score": stability_score
    }


def main():
    logger.info("=" * 80)
    logger.info("SCREENING US & INDIAN CANDIDATES FOR CHART STABILITY (AAPL-LIKE)")
    logger.info("=" * 80)

    all_tickers = US_CANDIDATES + INDIA_CANDIDATES
    stability_results = []

    # Download data for all tickers
    for ticker in all_tickers:
        try:
            dl = yf.download(ticker, period="3y", progress=False)
            if dl.empty:
                logger.warning(f"No data for {ticker}")
                continue
            if hasattr(dl.columns, "droplevel"):
                try:
                    dl.columns = dl.columns.droplevel(1)
                except Exception:
                    pass

            metrics = calculate_stability_metrics(dl)
            if not metrics:
                continue

            is_outside = ticker not in [
                "AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD", "COST", "WMT", "JPM", "LLY",
                "RELIANCE.NS", "TCS.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "LT.NS", "ITC.NS"
            ]
            market = "INDIA" if ticker.endswith(".NS") else "US"

            stability_results.append({
                "ticker": ticker,
                "market": market,
                "is_outside": is_outside,
                **metrics
            })
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")

    df_res = pd.DataFrame(stability_results)

    print("\n" + "=" * 90)
    print("US MARKETS: STABILITY RANKING (LOWER VOLATILITY, CLEAN STEADY TREND)")
    print("=" * 90)
    us_df = df_res[df_res["market"] == "US"].sort_values(by="stability_score", ascending=False)
    print(us_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("INDIAN MARKETS (NSE): STABILITY RANKING (LOWER VOLATILITY, CLEAN STEADY TREND)")
    print("=" * 90)
    ind_df = df_res[df_res["market"] == "INDIA"].sort_values(by="stability_score", ascending=False)
    print(ind_df.to_string(index=False))

    # Save screening results to scratch
    out_path = backend_dir / "reports" / "stability_screening_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stability_results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()

