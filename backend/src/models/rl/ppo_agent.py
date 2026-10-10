import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.utils.gpu_utils import configure_gpu_optimizations, get_device

logger = logging.getLogger(__name__)


class ActorCriticNetwork(nn.Module):
    """
    Unified Actor-Critic Network for PPO.

    Produces:
    1. Discrete action distribution over [0: Sell/Short, 1: Hold/Flat, 2: Buy/Long].
    2. Continuous dynamic position sizing factor in [-1.0, 1.0].
    3. Baseline scalar state-value estimate V(s) for Generalized Advantage Estimation.
    """

    def __init__(self, state_size: int, action_size: int = 3, hidden_dim: int = 128):
        super(ActorCriticNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Shared representation backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
        )

        # Actor head: Discrete Action Logits
        self.actor_discrete = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.Tanh(),
            nn.Linear(32, action_size),
        )

        # Actor head: Continuous Position Sizing [-1.0, +1.0]
        self.actor_sizing = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        # Critic head: Value Function V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared_backbone(state)
        logits = self.actor_discrete(features)
        sizing = self.actor_sizing(features)
        value = self.critic(features)
        return logits, sizing, value


class PPOMemory:
    """Rollout buffer for PPO trajectory storage."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.position_sizes: List[float] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.position_sizes.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for End-to-End Execution and Dynamic Sizing.

    Roles:
    - Direct policy optimization over trading states (technical indicators + portfolio state).
    - Outputs discrete trade directives (Buy, Sell, Hold) alongside continuous sizing [-1.0, 1.0].
    - Governed by clipped surrogate objective (L_CLIP) and Generalized Advantage Estimation (GAE).
    """

    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        config: Optional[Dict] = None,
    ):
        if config is None:
            config = {}

        self.state_size = state_size
        self.action_size = action_size

        # PPO Hyperparameters
        self.lr = config.get("lr", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.k_epochs = config.get("k_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.c1 = config.get("value_coef", 0.5)
        self.c2 = config.get("entropy_coef", 0.01)

        self.device = get_device()
        configure_gpu_optimizations()

        # Networks
        self.network = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        self.memory = PPOMemory()
        logger.info("PPO Agent initialized on %s (state_size=%d)", self.device, state_size)

    def remember(
        self,
        state: np.ndarray,
        action: int,
        position_size: float,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.memory.states.append(np.asarray(state, dtype=np.float32))
        self.memory.actions.append(int(action))
        self.memory.position_sizes.append(float(position_size))
        self.memory.log_probs.append(float(log_prob))
        self.memory.rewards.append(float(reward))
        self.memory.dones.append(bool(done))
        self.memory.values.append(float(value))

    def act(self, state: np.ndarray) -> Tuple[int, float, float, float]:
        """
        Samples an action and position size given the state.
        Returns: (action: int, position_size: float, log_prob: float, state_value: float)
        """
        self.network.eval()
        with torch.no_grad():
            s_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, sizing, value = self.network(s_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        self.network.train()
        return (
            int(action.item()),
            float(sizing.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        """Returns calibrated discrete action probabilities [P(Sell), P(Hold), P(Buy)]."""
        self.network.eval()
        with torch.no_grad():
            s_tensor = torch.FloatTensor(state).to(self.device)
            if s_tensor.ndim == 1:
                s_tensor = s_tensor.unsqueeze(0)
            logits, _, _ = self.network(s_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        self.network.train()
        return probs[0] if probs.shape[0] == 1 else probs

    def get_position_size(self, state: np.ndarray) -> float:
        """Returns continuous dynamic position size in [-1.0, 1.0]."""
        self.network.eval()
        with torch.no_grad():
            s_tensor = torch.FloatTensor(state).to(self.device)
            if s_tensor.ndim == 1:
                s_tensor = s_tensor.unsqueeze(0)
            _, sizing, _ = self.network(s_tensor)
            pos_size = float(sizing.item())
        self.network.train()
        return pos_size

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates Generalized Advantage Estimates (GAE) and discounted returns."""
        advantages = []
        gae = 0.0
        values_with_next = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.gamma * values_with_next[t + 1] * (1.0 - float(dones[t]))
                - values_with_next[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)

        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = advantages_tensor + torch.FloatTensor(values).to(self.device)

        # Standardize advantages for gradient stability
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )

        return advantages_tensor, returns_tensor

    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """
        Executes PPO policy update with clipped surrogate loss and value regularization.
        """
        if len(self.memory) == 0:
            return {"loss": 0.0}

        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)

        advantages, returns = self.compute_gae(
            self.memory.rewards,
            self.memory.values,
            self.memory.dones,
            next_value=next_value,
        )

        n_samples = len(states)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(self.k_epochs):
            indices = np.random.permutation(n_samples)
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx : start_idx + self.batch_size]

                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]

                logits, _, values = self.network(b_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # Ratio r_t(theta)
                ratios = torch.exp(new_log_probs - b_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratios * b_advantages
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * b_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic value loss
                value_loss = nn.functional.mse_loss(values.squeeze(-1), b_returns)

                # Composite PPO loss
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        self.memory.clear()

        return {
            "loss": float(total_loss),
            "policy_loss": float(total_policy_loss),
            "value_loss": float(total_value_loss),
        }

    @staticmethod
    def shape_reward(
        raw_profit: float,
        max_drawdown: float = 0.0,
        holding_period: int = 1,
        transaction_cost: float = 0.0005,
    ) -> float:
        """
        Institutional reward shaping:
        - Penalizes portfolio drawdown severely.
        - Penalizes high turnover friction (transaction costs).
        - Penalizes idle capital stagnation.
        """
        drawdown_penalty = max_drawdown * 12.0
        time_penalty = holding_period * 0.0005
        cost_penalty = transaction_cost * 2.0
        return float(raw_profit - drawdown_penalty - time_penalty - cost_penalty)

    def save(self, save_path: Union[str, Path] = "artifacts/ppo_agent.pth") -> None:
        """Serializes network parameters and training state."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "state_size": self.state_size,
                "action_size": self.action_size,
            },
            path,
        )
        logger.info("PPOAgent saved to %s", path)

    def load(self, load_path: Union[str, Path] = "artifacts/ppo_agent.pth") -> None:
        """Loads serialized network parameters."""
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("PPOAgent loaded successfully from %s", path)
