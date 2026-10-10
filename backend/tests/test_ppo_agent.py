import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models.rl.ppo_agent import PPOAgent


class TestPPOAgent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        self.state_size = 16
        self.agent = PPOAgent(
            state_size=self.state_size,
            action_size=3,
            config={"lr": 1e-3, "batch_size": 16, "k_epochs": 2},
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_act_and_continuous_position_sizing(self):
        state = np.random.randn(self.state_size).astype(np.float32)
        action, sizing, log_prob, value = self.agent.act(state)

        # Discrete action must be in [0, 1, 2]
        self.assertIn(action, [0, 1, 2])
        # Continuous position size must be bounded in [-1.0, 1.0]
        self.assertGreaterEqual(sizing, -1.0)
        self.assertLessEqual(sizing, 1.0)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)

    def test_predict_proba(self):
        state = np.random.randn(self.state_size).astype(np.float32)
        probs = self.agent.predict_proba(state)

        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertTrue(np.all(probs >= 0.0))

    def test_get_position_size(self):
        state = np.random.randn(self.state_size).astype(np.float32)
        sizing = self.agent.get_position_size(state)

        self.assertGreaterEqual(sizing, -1.0)
        self.assertLessEqual(sizing, 1.0)

    def test_trajectory_rollout_and_ppo_update(self):
        # Collect rollout trajectory
        for step in range(32):
            state = np.random.randn(self.state_size).astype(np.float32)
            action, sizing, log_prob, value = self.agent.act(state)
            reward = PPOAgent.shape_reward(raw_profit=0.01, max_drawdown=0.002)
            done = step == 31
            self.agent.remember(state, action, sizing, log_prob, reward, done, value)

        self.assertEqual(len(self.agent.memory), 32)
        metrics = self.agent.update(next_value=0.0)

        self.assertIn("loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        # Memory buffer should be purged after update
        self.assertEqual(len(self.agent.memory), 0)

    def test_serialization(self):
        save_path = Path(self.temp_dir) / "ppo_test.pth"
        self.agent.save(save_path)
        self.assertTrue(save_path.exists())

        new_agent = PPOAgent(state_size=self.state_size, action_size=3)
        new_agent.load(save_path)

        state = np.random.randn(self.state_size).astype(np.float32)
        p1 = self.agent.predict_proba(state)
        p2 = new_agent.predict_proba(state)
        np.testing.assert_allclose(p1, p2, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
