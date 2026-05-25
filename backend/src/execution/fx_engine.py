# src/execution/fx_engine.py
import yfinance as yf
import pandas as pd
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FXEngine:
    def __init__(self):
        self.rates: Dict[str, float] = {"USD": 1.0}
        self.last_updated: Optional[datetime] = None
        self.pairs_map = {
            "INR": "USDINR=X",
            "EUR": "EURUSD=X",
            "GBP": "GBPUSD=X",
            "JPY": "USDJPY=X",
            "CAD": "USDCAD=X",
            "AUD": "AUDUSD=X",
        }
        # Invert logic: yfinance gives USDINR=X as 1 USD = X INR
        # For EURUSD=X it gives 1 EUR = X USD
        self.is_base_usd = {
            "INR": True,
            "EUR": False,
            "GBP": False,
            "JPY": True,
            "CAD": True,
            "AUD": False,
        }

    async def update_rates(self):
        """Fetch latest FX rates from Yahoo Finance."""
        try:
            tickers = list(self.pairs_map.values())
            data = await asyncio.to_thread(
                yf.download, tickers, period="1d", interval="1m", progress=False
            )

            if data.empty:
                logger.warning(
                    "FX Engine: No data returned from yfinance. Using cached rates."
                )
                return

            if isinstance(data.columns, pd.MultiIndex):
                close_data = data["Close"]
            else:
                close_data = data[["Close"]]

            new_rates = {"USD": 1.0}
            for currency, ticker in self.pairs_map.items():
                try:
                    if ticker in close_data.columns:
                        series = close_data[ticker].dropna()
                        if not series.empty:
                            raw_rate = float(series.iloc[-1])
                        else:
                            raw_rate = self.rates.get(currency, 1.0)

                        if self.is_base_usd[currency]:
                            # 1 USD = raw_rate Currency
                            # We want normalized value in USD: Val_USD = Val_Curr / raw_rate
                            new_rates[currency] = raw_rate
                        else:
                            # 1 Currency = raw_rate USD
                            # We want normalized value in USD: Val_USD = Val_Curr * raw_rate
                            # So rate stored is raw_rate
                            new_rates[currency] = raw_rate
                    else:
                        new_rates[currency] = self.rates.get(currency, 1.0)
                except Exception as e:
                    logger.error(f"FX Engine: Error parsing {currency}: {e}")
                    new_rates[currency] = self.rates.get(currency, 1.0)

            self.rates = new_rates
            self.last_updated = datetime.now()
            logger.info(f"FX Engine: Rates updated at {self.last_updated.isoformat()}")

        except Exception as e:
            logger.error(f"FX Engine Critical Failure: {e}")

    def get_rate(self, currency: str) -> float:
        """Returns the rate for the given currency relative to USD."""
        return self.rates.get(currency.upper(), 1.0)

    def convert_to_base(
        self, amount: float, from_currency: str, base_currency: str = "USD"
    ) -> float:
        """Normalizes an amount into the base currency."""
        if from_currency == base_currency:
            return amount

        # First convert from_currency to USD
        usd_val = 0.0
        rate_from = self.get_rate(from_currency)

        if from_currency == "USD":
            usd_val = amount
        elif self.is_base_usd.get(from_currency, True):
            # 1 USD = rate_from Curr
            usd_val = amount / rate_from if rate_from != 0 else 0
        else:
            # 1 Curr = rate_from USD
            usd_val = amount * rate_from

        if base_currency == "USD":
            return usd_val

        # Then convert USD to base_currency
        rate_to = self.get_rate(base_currency)
        if self.is_base_usd.get(base_currency, True):
            # 1 USD = rate_to Curr
            return usd_val * rate_to
        else:
            # 1 Curr = rate_to USD
            return usd_val / rate_to if rate_to != 0 else 0

    def get_summary(self) -> Dict:
        return {
            "rates": self.rates,
            "last_updated": self.last_updated.isoformat()
            if self.last_updated
            else None,
            "status": "LIVE" if self.last_updated else "INITIALIZING",
        }
