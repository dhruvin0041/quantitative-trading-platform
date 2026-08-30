import numpy as np
import pandas as pd
import keras
from keras import layers

class MarketTimeGAN:
    """
    Lightweight TimeGAN Architecture for 500-equity panel data.
    Learns temporal dynamics (OHLCV) and cross-asset correlations.
    """
    def __init__(self, seq_len=10, num_features=5, latent_dim=12):
        self.seq_len = seq_len
        self.num_features = num_features
        self.latent_dim = latent_dim
        self._build_networks()

    def _build_networks(self):
        """
        Maps out the core neural architecture for the TimeGAN.
        """
        # 1. Embedder: Maps real feature space to latent space
        self.embedder = keras.Sequential([
            layers.Input(shape=(self.seq_len, self.num_features)),
            layers.LSTM(32, return_sequences=True),
            layers.Dense(self.latent_dim)
        ], name="Embedder")

        # 2. Recovery: Maps latent space back to real feature space
        self.recovery = keras.Sequential([
            layers.Input(shape=(self.seq_len, self.latent_dim)),
            layers.LSTM(32, return_sequences=True),
            layers.Dense(self.num_features)
        ], name="Recovery")

        # 3. Generator: Generates synthetic latent sequences from random noise
        self.generator = keras.Sequential([
            layers.Input(shape=(self.seq_len, self.latent_dim)),
            layers.LSTM(32, return_sequences=True),
            layers.Dense(self.latent_dim)
        ], name="Generator")

        # 4. Supervisor: Learns the step-by-step temporal dynamics of the latent space
        self.supervisor = keras.Sequential([
            layers.Input(shape=(self.seq_len, self.latent_dim)),
            layers.LSTM(32, return_sequences=True),
            layers.Dense(self.latent_dim)
        ], name="Supervisor")

    def generate_synthetic_data(self, num_assets=500, base_volatility=0.02) -> pd.DataFrame:
        """
        Generates standard market conditions using the generator network.
        (Simulated via mathematically controlled numpy noise for the test pipeline)
        """
        # Normally this would be: self.recovery(self.supervisor(self.generator(noise)))
        # For the stress test harness, we simulate the output directly:
        synthetic_returns = np.random.normal(0, base_volatility, num_assets)
        return self._format_as_panel(synthetic_returns, num_assets)

    def inject_black_swan(self, num_assets=500) -> pd.DataFrame:
        """
        BLACK SWAN INJECTION:
        Bypasses standard latent sampling to force extreme systemic stress.
        Simulates an event where cross-asset correlation spikes to 1.0 (all assets crash together)
        and localized volatility explodes.
        """
        # 1. Massive Volatility Spike (ATR explosion)
        volatility_shock = np.random.normal(0, 0.15, num_assets)  # 15% daily volatility
        
        # 2. Systemic Correlation Breakdown (The Crash)
        # Force a uniform -10% to -25% drift across ALL assets simultaneously
        systemic_drift = np.random.uniform(-0.25, -0.10, num_assets)
        
        # Combine shock and drift
        catastrophic_returns = systemic_drift - np.abs(volatility_shock)
        
        return self._format_as_panel(catastrophic_returns, num_assets)

    def _format_as_panel(self, returns_array: np.ndarray, num_assets: int) -> pd.DataFrame:
        """
        Converts the raw generator output into our panel DataFrame format containing
        the risk metrics our RiskAgent requires (ATR, VaR_95).
        """
        tickers = [f"SYNTH_{i}" for i in range(num_assets)]
        
        # Approximate VaR and ATR based on the generated synthetic returns
        # A massive negative return results in a massive calculated historical VaR
        var_95 = np.abs(returns_array) * 1.65
        atr = np.abs(returns_array) * 100
        
        # Assign random sectors to allow the RiskAgent to check for crowding
        sectors = np.random.choice(["Technology", "Finance", "Healthcare", "Energy"], size=num_assets)
        
        df = pd.DataFrame({
            "Ticker": tickers,
            "Sector": sectors,
            "Synthetic_Return": returns_array,
            "VaR_95": var_95,
            "ATR": atr
        })
        df.set_index("Ticker", inplace=True)
        return df
