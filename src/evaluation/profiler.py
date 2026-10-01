"""Computational benchmarking profiler.
Measures:
- Trainable parameters
- CPU Inference latency per sample (ms) and per trial (ms, 9 samples) with batch size = 1
- Relative speedup ratios
- Peak RAM consumption
"""

import time
import tracemalloc
import logging
from typing import Dict, Any, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def profile_inference_latency(
    predict_fn: Callable[[np.ndarray], Any],
    sample_input: np.ndarray,
    warmup_runs: int = 30,
    timed_runs: int = 100,
    samples_per_trial: int = 9,
) -> Dict[str, float]:
    """Measure inference latency for a single sample (batch size = 1) on CPU."""
    # Warmup runs to ensure JIT/cache is primed
    for _ in range(warmup_runs):
        _ = predict_fn(sample_input)

    latencies = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = predict_fn(sample_input)
        latencies.append((time.perf_counter() - t0) * 1000.0) # ms

    avg_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    trial_ms = avg_ms * samples_per_trial

    return {
        "avg_latency_ms": avg_ms,
        "std_latency_ms": std_ms,
        "latency_per_trial_ms": trial_ms,
    }


def profile_torch_model(
    model: nn.Module,
    sample_shape: Tuple[int, ...] = (1, 1, 1280),
    warmup_runs: int = 30,
    timed_runs: int = 100,
    samples_per_trial: int = 9,
) -> Dict[str, Any]:
    """Profile PyTorch model inference latency on CPU with batch_size = 1."""
    model.eval()
    dummy_tensor = torch.randn(*sample_shape, dtype=torch.float32)

    def torch_predict(x_tensor):
        with torch.no_grad():
            return model(x_tensor)

    timing = profile_inference_latency(
        torch_predict, dummy_tensor, warmup_runs, timed_runs, samples_per_trial
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    timing["trainable_params"] = params
    return timing


def profile_minirocket_pipeline(
    pipeline,
    sample_shape: Tuple[int, ...] = (1, 1, 1280),
    warmup_runs: int = 30,
    timed_runs: int = 100,
    samples_per_trial: int = 9,
) -> Dict[str, Any]:
    """Profile MiniRocket + Ridge pipeline inference latency on CPU."""
    dummy_input = np.random.randn(*sample_shape).astype(np.float32)

    timing = profile_inference_latency(
        pipeline.predict, dummy_input, warmup_runs, timed_runs, samples_per_trial
    )
    timing["trainable_params"] = pipeline.get_parameter_count()
    return timing


def measure_peak_memory(fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Measure peak RAM consumption of a function call in megabytes."""
    tracemalloc.start()
    result = fn()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024.0 * 1024.0)
    return result, peak_mb
