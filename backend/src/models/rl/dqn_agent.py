import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.gpu_utils import configure_gpu_optimizations, get_device

logger = logging.getLogger(__name__)


class DuelingDQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_size)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        prios = self.priorities[: len(self.buffer)]
        probs = prios**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio


class DQNAgent:
    def __init__(self, state_size, action_size=3, config=None):
        if config is None:
            config = {}

        self.state_size = state_size
        self.action_size = action_size

        # Configuration
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 64)
        self.tau = config.get("tau", 0.005)  # Soft update parameter

        self.memory = PrioritizedReplayBuffer(capacity=config.get("buffer_size", 10000))
        self.device = get_device()
        configure_gpu_optimizations()
        logger.info("DQN Agent initialized on %s", self.device)

        # Double DQN architecture
        self.policy_net = DuelingDQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingDQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')

        # Transaction configuration
        self.transaction_cost = config.get("transaction_cost", 0.001)  # 0.1%

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.policy_net(state_tensor)
        self.policy_net.train()

        return torch.argmax(act_values[0]).item()

    def predict_q_values(self, state):
        """Extract continuous Q-values Q(s, a) for all actions."""
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.policy_net(state_tensor)
        self.policy_net.train()
        return act_values[0].cpu().numpy()

    def predict_proba(self, state, temperature=1.5):
        """
        Convert continuous Q-values into a well-calibrated soft probability distribution
        via temperature-scaled softmax:
        P(a) = exp(Q(s, a) / tau) / sum_j exp(Q(s, j) / tau)
        """
        q_vals = self.predict_q_values(state)
        tau = max(float(temperature), 1e-4)
        scaled = q_vals / tau
        exp_vals = np.exp(scaled - np.max(scaled))
        return exp_vals / np.sum(exp_vals)

    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        minibatch, indices, weights = self.memory.sample(self.batch_size)
        weights = torch.FloatTensor(weights).to(self.device)

        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = (
            torch.LongTensor(np.array([t[1] for t in minibatch]))
            .unsqueeze(1)
            .to(self.device)
        )
        rewards = (
            torch.FloatTensor(np.array([t[2] for t in minibatch]))
            .unsqueeze(1)
            .to(self.device)
        )
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(
            self.device
        )
        dones = (
            torch.FloatTensor(np.array([t[4] for t in minibatch]))
            .unsqueeze(1)
            .to(self.device)
        )

        # DDQN: use policy net to select best action, target net to evaluate it
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                next_q_targets = self.target_net(next_states).gather(1, next_actions)
                target_q = rewards + (1 - dones) * self.gamma * next_q_targets

            # Current Q-values
            current_q = self.policy_net(states).gather(1, actions)

            # Huber loss with prioritized weights
            loss = (
                weights
                * nn.functional.smooth_l1_loss(current_q, target_q, reduction="none")
            ).mean()

        # Compute TD errors for priority update
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors + 1e-6)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Soft update of target network
        self._soft_update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _soft_update_target_network(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def shape_reward(self, raw_profit, max_drawdown, holding_period):
        """
        Reward shaping based on risk-adjusted metrics
        """
        # Penalize for transaction costs implicitly handled in raw_profit
        # Penalize for drawdown (risk)
        risk_penalty = max_drawdown * 10.0

        # Penalize for holding too long without profit (opportunity cost)
        time_penalty = holding_period * 0.001

        shaped_reward = raw_profit - risk_penalty - time_penalty
        return shaped_reward

    def save(self, name):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            name,
        )

    def load(self, name):
        checkpoint = torch.load(name, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
