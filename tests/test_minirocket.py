"""Unit tests for MiniRocket Pipeline."""

import unittest
import numpy as np
from src.models.minirocket_pipeline import MiniRocketPipeline


class TestMiniRocketPipeline(unittest.TestCase):

    def setUp(self):
        # Use a small number of kernels for fast unit testing
        self.pipeline = MiniRocketPipeline(
            num_kernels=500,
            max_dilations_per_kernel=28,
            n_jobs=1,
            random_state=42,
        )

    def test_fit_transform_predict(self):
        # 16 samples, length 1280, 4 classes
        X = np.random.randn(16, 1, 1280).astype(np.float32)
        y = np.array([0, 1, 2, 3] * 4)

        self.pipeline.fit(X, y)
        self.assertTrue(self.pipeline.is_fitted)
        self.assertGreater(self.pipeline.feature_dim_, 0)

        preds = self.pipeline.predict(X)
        self.assertEqual(len(preds), 16)
        self.assertTrue(all(p in [0, 1, 2, 3] for p in preds))

        probs = self.pipeline.predict_proba(X)
        self.assertEqual(probs.shape, (16, 4))
        # Verify probabilities sum to 1
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(16), atol=1e-5)

    def test_parameter_count(self):
        X = np.random.randn(8, 1, 1280).astype(np.float32)
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        self.pipeline.fit(X, y)
        params = self.pipeline.get_parameter_count()
        # 4 classes * feature_dim + 4
        expected = (4 * self.pipeline.feature_dim_) + 4
        self.assertEqual(params, expected)


if __name__ == "__main__":
    unittest.main()
