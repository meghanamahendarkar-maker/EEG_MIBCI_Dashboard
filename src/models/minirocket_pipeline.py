"""MiniRocket Feature Transform + Linear Ridge Classifier Pipeline.
Based on Hwaidi & Ghanem (NeuroImage 2026).
"""

import time
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sktime.transformations.panel.rocket import MiniRocket

logger = logging.getLogger(__name__)


class MiniRocketPipeline:
    """End-to-end MiniRocket feature extraction and linear classification pipeline."""

    def __init__(
        self,
        num_kernels: int = 10000,
        max_dilations_per_kernel: int = 28, # Optimal 28 dilations found in paper
        alphas: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.alphas = alphas
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.minirocket = MiniRocket(
            num_kernels=self.num_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.scaler = StandardScaler(with_mean=False) # PPV is non-negative [0, 1]
        self.classifier = RidgeClassifierCV(alphas=self.alphas)

        self.is_fitted = False
        self.feature_dim_ = 0
        self.train_time_sec_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MiniRocketPipeline":
        """Fit MiniRocket random kernels and solve Ridge regression in closed form.

        Args:
            X: EEG data array of shape (N, 1, 1280) or (N, 1280)
            y: Labels array of shape (N,)
        """
        start_time = time.perf_counter()

        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)

        logger.info(f"Extracting MiniRocket features (kernels={self.num_kernels}, dilations<={self.max_dilations_per_kernel})...")
        X_feats = self.minirocket.fit_transform(X)
        if hasattr(X_feats, "values"):
            X_feats = X_feats.values

        self.feature_dim_ = X_feats.shape[1]
        logger.info(f"MiniRocket extracted {self.feature_dim_} features per sample.")

        X_scaled = self.scaler.fit_transform(X_feats)

        logger.info(f"Solving closed-form Ridge Regression for {len(np.unique(y))} classes...")
        self.classifier.fit(X_scaled, y)

        self.train_time_sec_ = time.perf_counter() - start_time
        self.is_fitted = True
        logger.info(f"MiniRocket pipeline trained in {self.train_time_sec_:.2f}s (best alpha={getattr(self.classifier, 'alpha_', None)}).")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract scaled MiniRocket features without classifying."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming.")

        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)

        X_feats = self.minirocket.transform(X)
        if hasattr(X_feats, "values"):
            X_feats = X_feats.values
        return self.scaler.transform(X_feats)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for given samples."""
        X_scaled = self.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate class posterior probabilities using softmax over Ridge decision values."""
        X_scaled = self.transform(X)
        decision = self.classifier.decision_function(X_scaled) # shape (N, n_classes)

        if decision.ndim == 1:
            # Binary case
            decision = np.vstack([-decision, decision]).T

        # Softmax over decision function
        exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        probs = exp_d / np.sum(exp_d, axis=1, keepdims=True)
        return probs

    def get_parameter_count(self) -> int:
        """Return the number of trainable parameters.

        For MiniRocket + Ridge, only the linear weights (feature_dim * n_classes + biases)
        are learned, matching the ~40k parameter estimation reported in Table 2 of the paper.
        """
        if not self.is_fitted:
            return 0
        n_classes = len(self.classifier.classes_)
        # weights: (n_classes, feature_dim), intercept: (n_classes,)
        return (n_classes * self.feature_dim_) + n_classes
