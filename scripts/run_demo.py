"""Fast end-to-end demonstration script.
Tests synthetic data generation, MiniRocket pipeline, and CNN-LSTM model in ~10 seconds.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic import generate_synthetic_eeg_dataset
from src.data.dataset import split_train_val_test, create_dataloaders
from src.models.minirocket_pipeline import MiniRocketPipeline
from src.models.cnn_lstm import HybridCNNLSTM
from src.training.trainer_dl import DeepLearningTrainer
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.profiler import profile_minirocket_pipeline, profile_torch_model


def run_quick_demo():
    print("=" * 65)
    print("FAST DEMO: Motor Imagery EEG Classification")
    print("=" * 65)

    print("\n1. Generating synthetic 4-class EEG dataset (2 subjects, 4 trials/class)...")
    ds = generate_synthetic_eeg_dataset(num_subjects=2, trials_per_class=4, random_state=42)
    X, y = ds["X"], ds["y"]
    print(f"   Generated {X.shape[0]} samples with shape {X.shape}.")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_train_val_test(X, y, random_state=42)
    print(f"   Train samples: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")

    print("\n2. Training MiniRocket + Ridge Pipeline (num_kernels=500, max_dilations=28)...")
    mr = MiniRocketPipeline(num_kernels=500, max_dilations_per_kernel=28, random_state=42)
    t0 = time.perf_counter()
    mr.fit(X_tr, y_tr)
    mr_train_time = time.perf_counter() - t0
    mr_preds = mr.predict(X_te)
    mr_probs = mr.predict_proba(X_te)
    mr_metrics = compute_classification_metrics(y_te, mr_preds, mr_probs)
    mr_prof = profile_minirocket_pipeline(mr, warmup_runs=10, timed_runs=30)

    print(f"   MiniRocket Fit Time: {mr_train_time:.2f}s")
    print(f"   MiniRocket Test Accuracy: {mr_metrics['accuracy'] * 100:.2f}% | Macro F1: {mr_metrics['f1_macro']:.4f}")
    print(f"   MiniRocket CPU Latency: {mr_prof['avg_latency_ms']:.2f} ms/sample | Trial: {mr_prof['latency_per_trial_ms']:.2f} ms")
    print(f"   Trainable Parameters: {mr_prof['trainable_params']:,}")

    print("\n3. Training Hybrid CNN-LSTM Baseline (5 epochs for quick check)...")
    cnn_lstm = HybridCNNLSTM(num_classes=4)
    loaders = create_dataloaders((X_tr, y_tr), (X_va, y_va), (X_te, y_te), batch_size=32)
    trainer = DeepLearningTrainer(cnn_lstm, learning_rate=1e-4)
    t0 = time.perf_counter()
    trainer.fit(loaders["train"], loaders["val"], epochs=5, verbose=False)
    dl_train_time = time.perf_counter() - t0
    _, _, dl_preds, dl_probs = trainer.evaluate(loaders["test"])
    dl_metrics = compute_classification_metrics(y_te, dl_preds, dl_probs)
    dl_prof = profile_torch_model(cnn_lstm, warmup_runs=10, timed_runs=30)

    print(f"   CNN-LSTM Fit Time: {dl_train_time:.2f}s")
    print(f"   CNN-LSTM Test Accuracy: {dl_metrics['accuracy'] * 100:.2f}% | Macro F1: {dl_metrics['f1_macro']:.4f}")
    print(f"   CNN-LSTM CPU Latency: {dl_prof['avg_latency_ms']:.2f} ms/sample | Trial: {dl_prof['latency_per_trial_ms']:.2f} ms")
    print(f"   Trainable Parameters: {dl_prof['trainable_params']:,}")

    print("\n" + "=" * 65)
    print("DEMO VERIFIED SUCCESSFULLY!")
    print(f"Speedup of MiniRocket over CNN-LSTM: {dl_prof['avg_latency_ms'] / max(1e-5, mr_prof['avg_latency_ms']):.1f}x faster inference")
    print("=" * 65)


if __name__ == "__main__":
    run_quick_demo()
