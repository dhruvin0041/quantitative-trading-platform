import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.regime.svm_regime_classifier import (
    MarketRegime,
    SVMRegimeClassifier,
)


class TestSVMRegimeClassifier(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        self.n_samples = 150

        # Construct stationary features
        self.df = pd.DataFrame(
            {
                "ADX": np.random.uniform(10.0, 40.0, size=self.n_samples),
                "ATR_Percentile": np.random.uniform(0.1, 0.95, size=self.n_samples),
                "BB_Width": np.random.uniform(0.01, 0.12, size=self.n_samples),
                "HMA_Slope": np.random.uniform(-0.01, 0.01, size=self.n_samples),
            }
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_synthesize_regime_labels(self):
        labels = SVMRegimeClassifier.synthesize_regime_labels(self.df)
        self.assertEqual(len(labels), self.n_samples)
        # Should encompass valid regime classes 0, 1, 2
        unique_labels = set(labels)
        self.assertTrue(unique_labels.issubset({0, 1, 2}))

    def test_fit_and_predict_contracts(self):
        labels = SVMRegimeClassifier.synthesize_regime_labels(self.df)
        clf = SVMRegimeClassifier(C=1.0)
        clf.fit(self.df, labels)

        self.assertTrue(clf.is_fitted)

        preds = clf.predict(self.df)
        self.assertEqual(len(preds), self.n_samples)
        self.assertTrue(set(preds).issubset({0, 1, 2}))

        probs = clf.predict_proba(self.df)
        self.assertEqual(probs.shape, (self.n_samples, 3))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(self.n_samples), atol=1e-5)

    def test_classify_regime_live_inference(self):
        labels = SVMRegimeClassifier.synthesize_regime_labels(self.df)
        clf = SVMRegimeClassifier(C=1.0)
        clf.fit(self.df, labels)

        # High volatility record
        sample_high_vol = {
            "ADX": 15.0,
            "ATR_Percentile": 0.95,
            "BB_Width": 0.15,
            "HMA_Slope": 0.0,
        }
        regime, conf = clf.classify_regime(sample_high_vol)
        self.assertIn(
            regime,
            [
                MarketRegime.HIGH_VOLATILITY_NOISE.value,
                MarketRegime.STRONG_TREND.value,
                MarketRegime.MEAN_REVERTING.value,
            ],
        )
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_serialization(self):
        labels = SVMRegimeClassifier.synthesize_regime_labels(self.df)
        clf = SVMRegimeClassifier(C=1.0)
        clf.fit(self.df, labels)

        save_path = Path(self.temp_dir) / "svm_regime.joblib"
        clf.save(save_path)
        self.assertTrue(save_path.exists())

        loaded = SVMRegimeClassifier.load(save_path)
        np.testing.assert_array_equal(clf.predict(self.df), loaded.predict(self.df))


if __name__ == "__main__":
    unittest.main()
