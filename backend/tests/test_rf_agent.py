import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.classical.rf_agent import RandomForestAgent, train_rf_agent


class TestRandomForestAgent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        self.n_samples = 120
        self.n_features = 6
        self.feature_names = [f"feat_{i}" for i in range(self.n_features)]

        self.X = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=self.feature_names,
        )
        # 3 classes: 0=Sell, 1=Hold, 2=Buy
        self.y = np.random.choice([0, 1, 2], size=self.n_samples)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fit_and_max_depth_constraint(self):
        agent = RandomForestAgent(n_estimators=30, max_depth=4, min_samples_leaf=5)
        agent.fit(self.X, self.y)

        # Verify max depth constraint is strictly respected on all decision trees
        for tree in agent.model.estimators_:
            self.assertLessEqual(tree.get_depth(), 4)

    def test_predict_and_proba_contract(self):
        agent = RandomForestAgent(n_estimators=20, max_depth=3)
        agent.fit(self.X, self.y)

        preds = agent.predict(self.X)
        self.assertEqual(len(preds), self.n_samples)
        self.assertTrue(set(preds).issubset({0, 1, 2}))

        probs = agent.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.n_samples, 3))
        # Ensure probabilities sum to 1.0 across rows
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(self.n_samples), atol=1e-5)

    def test_feature_importance_and_selection(self):
        agent = RandomForestAgent(n_estimators=25, max_depth=4)
        agent.fit(self.X, self.y)

        importances = agent.get_feature_importances()
        self.assertEqual(len(importances), self.n_features)
        # Ensure values are non-negative and sum to 1.0
        self.assertAlmostEqual(sum(importances.values()), 1.0, places=4)

        top_3 = agent.select_top_features(top_k=3)
        self.assertEqual(len(top_3), 3)
        self.assertTrue(all(name in self.feature_names for name in top_3))

    def test_serialization(self):
        save_path = Path(self.temp_dir) / "rf_test.joblib"
        agent = train_rf_agent(
            self.X, self.y, save_path=str(save_path), max_depth=3, n_estimators=15
        )
        self.assertTrue(save_path.exists())

        loaded = RandomForestAgent.load(str(save_path))
        orig_preds = agent.predict(self.X)
        loaded_preds = loaded.predict(self.X)
        np.testing.assert_array_equal(orig_preds, loaded_preds)


if __name__ == "__main__":
    unittest.main()
