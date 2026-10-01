"""Trainer orchestration for MiniRocket + Linear Classifier pipeline."""

import time
import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
try:
    from src.models.minirocket_pipeline import MiniRocketPipeline
except (ImportError, ValueError):
    from ..models.minirocket_pipeline import MiniRocketPipeline

logger = logging.getLogger(__name__)


class MiniRocketTrainer:
    """Orchestrates fitting, timing, and evaluating the MiniRocket + Ridge pipeline."""

    def __init__(
        self,
        num_kernels: int = 10000,
        max_dilations: int = 28,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.pipeline = MiniRocketPipeline(
            num_kernels=num_kernels,
            max_dilations_per_kernel=max_dilations,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.results: Dict[str, Any] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MiniRocketTrainer":
        """Fit MiniRocket features and Ridge classifier with timing."""
        start_time = time.perf_counter()
        self.pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start_time
        self.results["train_time_sec"] = elapsed
        self.results["train_params"] = self.pipeline.get_parameter_count()
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate predictions, accuracy, macro F1, and probabilities on test set."""
        start_inf = time.perf_counter()
        preds = self.pipeline.predict(X_test)
        inf_elapsed = time.perf_counter() - start_inf

        probs = self.pipeline.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        avg_latency_ms = (inf_elapsed / max(1, len(y_test))) * 1000.0

        eval_res = {
            "accuracy": acc,
            "f1_macro": f1,
            "predictions": preds,
            "probabilities": probs,
            "test_time_sec": inf_elapsed,
            "avg_latency_ms": avg_latency_ms,
        }
        self.results.update(eval_res)
        return eval_res
