"""Non-additive Information Fusion across Electrode Sources.
Implements the fuzzy Choquet integral and Shapley value attribution for multi-pair MI-BCI decoding
as outlined in Section 5.3 of Hwaidi & Ghanem (2026).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class ChoquetIntegralFusion:
    """Decision-level Choquet integral fusion for multi-electrode pair BCI decoding.

    Aggregates class probabilities from N information sources (electrode pairs)
    using a learned or specified fuzzy measure (capacity).
    """

    def __init__(self, num_sources: int = 5, lambda_param: float = 0.0):
        """Args:

        num_sources: Number of information sources (e.g. 5 symmetric electrode pairs).
        lambda_param: Sugeno lambda parameter for fuzzy measure combination.
                      lambda = 0 reduces to additive weighting.
                      lambda > 0 represents source redundancy.
                      lambda < 0 represents source synergy.
        """
        self.num_sources = num_sources
        self.lambda_param = lambda_param
        # Default symmetric base densities for 5 motor cortex pairs
        # Primary pair C3/C4 (index 1) has highest base density
        self.densities = np.array([0.18, 0.28, 0.18, 0.18, 0.18], dtype=np.float64)
        self.densities /= np.sum(self.densities)

    def set_densities(self, densities: np.ndarray):
        """Set custom base densities (importance) for each electrode pair."""
        assert len(densities) == self.num_sources
        self.densities = np.array(densities, dtype=np.float64)
        self.densities /= np.sum(self.densities)

    def compute_sugeno_lambda(self, densities: np.ndarray) -> float:
        """Numerically solve for Sugeno lambda parameter: 1 + lambda = prod(1 + lambda * g_i)."""
        if np.isclose(np.sum(densities), 1.0):
            return 0.0
        # In this implementation, we use lambda_param or additive normalized measure
        return self.lambda_param

    def evaluate_fuzzy_measure(self, subset_indices: List[int]) -> float:
        """Evaluate the fuzzy measure (capacity) of a coalition/subset of sources."""
        if len(subset_indices) == 0:
            return 0.0
        if len(subset_indices) == self.num_sources:
            return 1.0

        if np.isclose(self.lambda_param, 0.0):
            # Additive measure: sum of individual densities
            return float(np.sum(self.densities[subset_indices]))
        else:
            # Sugeno lambda-measure formulation
            val = 1.0
            for idx in subset_indices:
                val *= (1.0 + self.lambda_param * self.densities[idx])
            return float((val - 1.0) / self.lambda_param)

    def fuse_sample_scores(self, source_scores: np.ndarray) -> np.ndarray:
        """Compute the discrete Choquet integral for a single sample across classes.

        Args:
            source_scores: shape (num_sources, num_classes)
                           probability/score distribution for each source.

        Returns:
            fused_probabilities: shape (num_classes,)
        """
        num_classes = source_scores.shape[1]
        fused = np.zeros(num_classes)

        for c in range(num_classes):
            scores_c = source_scores[:, c]
            # Sort scores in ascending order
            sorted_idx = np.argsort(scores_c)
            sorted_scores = scores_c[sorted_idx]

            # Discrete Choquet integral:
            # sum_{i=1}^n (x_{(i)} - x_{(i-1)}) * mu(A_{(i)})
            integral = 0.0
            prev_val = 0.0
            for i in range(self.num_sources):
                subset = list(sorted_idx[i:])
                measure = self.evaluate_fuzzy_measure(subset)
                diff = sorted_scores[i] - prev_val
                integral += diff * measure
                prev_val = sorted_scores[i]

            fused[c] = integral

        # Normalize to valid probability distribution
        fused = np.maximum(fused, 0.0)
        sum_fused = np.sum(fused)
        if sum_fused > 0:
            fused /= sum_fused
        else:
            fused = np.ones(num_classes) / num_classes
        return fused

    def fuse_batch(self, batch_source_scores: np.ndarray) -> np.ndarray:
        """Compute Choquet integral fusion for a batch of samples.

        Args:
            batch_source_scores: shape (N_samples, num_sources, num_classes)

        Returns:
            fused_probs: shape (N_samples, num_classes)
        """
        N = batch_source_scores.shape[0]
        num_classes = batch_source_scores.shape[2]
        out = np.zeros((N, num_classes))
        for n in range(N):
            out[n] = self.fuse_sample_scores(batch_source_scores[n])
        return out

    def compute_shapley_values(self) -> np.ndarray:
        """Compute Shapley values measuring the average marginal contribution of each source."""
        # For additive/lambda measure, the Shapley value reflects the relative power of each source
        # across all 2^N coalitions
        shapley = np.zeros(self.num_sources)
        import itertools

        sources = list(range(self.num_sources))
        n = self.num_sources
        import math

        for i in sources:
            other_sources = [s for s in sources if s != i]
            phi_i = 0.0
            for k in range(n):
                for subset in itertools.combinations(other_sources, k):
                    coalition = list(subset)
                    coalition_with_i = coalition + [i]
                    marginal = self.evaluate_fuzzy_measure(coalition_with_i) - self.evaluate_fuzzy_measure(coalition)
                    weight = (math.factorial(k) * math.factorial(n - k - 1)) / math.factorial(n)
                    phi_i += weight * marginal
            shapley[i] = phi_i

        return shapley
