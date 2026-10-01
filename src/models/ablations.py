"""Ablation variants and comparative models.
Includes:
- Hybrid CNN-GRU model (ablation replacing LSTM with GRU)
- Alternative classifiers for MiniRocket (SVM, Logistic Regression)
- Full ROCKET transform comparison
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import Rocket
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HybridCNNGRU(nn.Module):
    """Ablation model replacing LSTM with Gated Recurrent Unit (GRU).

    Reported in paper to have slightly faster inference but slightly lower accuracy.
    """

    def __init__(
        self,
        input_channels: int = 1,
        sequence_length: int = 1280,
        gru_units: int = 100,
        lstm_timesteps: int = 62,
        num_classes: int = 4,
        conv1_filters: int = 16,
        conv2_filters: int = 32,
        kernel_size: int = 3,
        dropout_conv: float = 0.5,
        dropout_gru: float = 0.5,
        dropout_dense: float = 0.25,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, conv1_filters, kernel_size, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.dropout_conv = nn.Dropout(dropout_conv)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.adaptive_align = nn.AdaptiveAvgPool1d(lstm_timesteps)

        self.gru = nn.GRU(
            input_size=conv2_filters,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
        )
        self.dropout_gru = nn.Dropout(dropout_gru)
        self.dense1 = nn.Linear(gru_units, 100)
        self.relu_dense1 = nn.ReLU()
        self.dropout_dense1 = nn.Dropout(dropout_dense)
        self.dense2 = nn.Linear(100, 50)
        self.relu_dense2 = nn.ReLU()
        self.dropout_dense2 = nn.Dropout(dropout_dense)
        self.classifier = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.dropout_conv(out)
        out = self.pool(out)

        out = self.adaptive_align(out)
        out = out.permute(0, 2, 1)

        gru_out, hn = self.gru(out)
        last_hidden = hn[-1]

        out = self.dropout_gru(last_hidden)
        out = self.relu_dense1(self.dense1(out))
        out = self.dropout_dense1(out)
        out = self.relu_dense2(self.dense2(out))
        out = self.dropout_dense2(out)
        return self.classifier(out)

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_ablation_classifier(classifier_type: str = "ridge", random_state: int = 42):
    """Return classifier instance for classifier choice ablation experiments."""
    classifier_type = classifier_type.lower()
    if classifier_type == "svm_linear":
        return LinearSVC(C=1.0, random_state=random_state, max_iter=2000)
    elif classifier_type == "svm_rbf":
        return SVC(C=1.0, kernel="rbf", random_state=random_state)
    elif classifier_type == "logistic_regression":
        return LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


class RocketPipeline:
    """End-to-end ROCKET feature extraction and linear classification pipeline (Baseline).
    
    This implements the original ROCKET transform which uses random convolutional 
    kernels without the optimizations of MiniRocket.
    """

    def __init__(
        self,
        num_kernels: int = 10000,
        alphas: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.num_kernels = num_kernels
        self.alphas = alphas
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.rocket = Rocket(
            num_kernels=self.num_kernels,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.scaler = StandardScaler(with_mean=False)
        self.classifier = RidgeClassifierCV(alphas=self.alphas)

        self.is_fitted = False
        self.feature_dim_ = 0
        self.train_time_sec_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RocketPipeline":
        start_time = time.perf_counter()

        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)

        logger.info(f"Extracting ROCKET features (kernels={self.num_kernels})...")
        X_feats = self.rocket.fit_transform(X)
        if hasattr(X_feats, "values"):
            X_feats = X_feats.values

        self.feature_dim_ = X_feats.shape[1]
        X_scaled = self.scaler.fit_transform(X_feats)

        self.classifier.fit(X_scaled, y)
        self.train_time_sec_ = time.perf_counter() - start_time
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming.")
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
        X_feats = self.rocket.transform(X)
        if hasattr(X_feats, "values"):
            X_feats = X_feats.values
        return self.scaler.transform(X_feats)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.transform(X)
        decision = self.classifier.decision_function(X_scaled)
        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        return exp_d / np.sum(exp_d, axis=1, keepdims=True)

    def get_parameter_count(self) -> int:
        if not self.is_fitted:
            return 0
        n_classes = len(self.classifier.classes_)
        return (n_classes * self.feature_dim_) + n_classes
