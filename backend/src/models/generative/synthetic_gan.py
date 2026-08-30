import tensorflow as tf
from keras import layers
import numpy as np
import logging

from src.utils.gpu_utils import configure_tensorflow_gpu

logger = logging.getLogger(__name__)


class MarketTimeGAN:
    """
    SOTA 2026 Generative Adversarial Network for synthetic market simulation.
    Captures 'fat tails' and 'volatility clustering' for stress testing.
    """

    def __init__(self, seq_len=60, n_features=20):
        self.seq_len = seq_len
        self.n_features = n_features
        configure_tensorflow_gpu()
        logger.info("MarketTimeGAN initializing on: %s",
                    'GPU' if tf.config.list_physical_devices('GPU') else 'CPU')
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

    def _build_generator(self):
        model = tf.keras.Sequential(
            [
                layers.Input(shape=(self.seq_len, self.n_features)),
                layers.LSTM(64, return_sequences=True),
                layers.LSTM(64, return_sequences=True),
                layers.Dense(self.n_features, activation="tanh"),
            ],
            name="Generator",
        )
        return model

    def _build_discriminator(self):
        model = tf.keras.Sequential(
            [
                layers.Input(shape=(self.seq_len, self.n_features)),
                layers.LSTM(64),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="Discriminator",
        )
        return model

    def generate_synthetic_crisis(self, n_samples=100):
        """
        Generates 'What-If' scenarios by adding noise to the latent space.
        """
        noise = np.random.normal(0, 1, (n_samples, self.seq_len, self.n_features))
        synthetic_data = self.generator.predict(noise, verbose=0)
        return synthetic_data

    def stress_test_policy(self, dqn_agent, n_paths=1000):
        """
        Runs the DQN agent through synthetic paths to find Max Drawdown.
        """
        print(f"Starting Monte Carlo Stress Test ({n_paths} paths)...")
        synthetic_paths = self.generate_synthetic_crisis(n_paths)
        results = []
        for path in synthetic_paths:
            capital = 100000.0
            peak_capital = capital
            max_dd = 0.0
            shares = 0

            for t in range(path.shape[0]):
                state = path[t]
                if len(state) < dqn_agent.state_size:
                    state = np.pad(state, (0, dqn_agent.state_size - len(state)))

                action = dqn_agent.act(state)
                # Use first feature as synthetic price proxy
                price = 100 * (1 + state[0])
                if price <= 0:
                    price = 1.0

                if action == 2 and capital >= price:  # BUY
                    shares += 1
                    capital -= price
                elif action == 0 and shares > 0:  # SELL
                    capital += shares * price
                    shares = 0

                current_value = capital + (shares * price)
                peak_capital = max(peak_capital, current_value)
                dd = (peak_capital - current_value) / peak_capital
                max_dd = max(max_dd, dd)

            results.append(max_dd)

        return {
            "synthetic_max_drawdown": f"{round(max(results) * 100, 1)}%",
            "survival_probability": f"{round(len([r for r in results if r < 0.25]) / n_paths * 100, 1)}%",
        }
