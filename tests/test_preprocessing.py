"""Unit tests for preprocessing and data generators."""

import unittest
import numpy as np
from src.data.synthetic import generate_synthetic_eeg_trial, generate_synthetic_eeg_dataset
from src.data.preprocessor import EEGPreprocessor
from src.data.dataset import split_train_val_test, create_dataloaders


class TestEEGPreprocessing(unittest.TestCase):

    def test_synthetic_trial_generation(self):
        trial = generate_synthetic_eeg_trial(
            task_class=0,
            sampling_rate=128,
            duration_sec=4.0,
            num_electrode_pairs=5,
            random_state=42,
        )
        self.assertEqual(trial.shape, (5, 2, 512))
        self.assertTrue(np.all(np.isfinite(trial)))

    def test_synthetic_dataset_shape(self):
        # 2 subjects, 4 trials per class for fast unit testing
        ds = generate_synthetic_eeg_dataset(
            num_subjects=2,
            trials_per_class=4,
            sampling_rate=128,
            duration_sec=4.0,
            samples_per_trial=9,
            random_state=42,
        )
        # Expected total samples = 2 subjects * 4 classes * 4 trials * 9 samples = 288 samples
        expected_samples = 2 * 4 * 4 * 9
        self.assertEqual(ds["X"].shape, (expected_samples, 1, 1280))
        self.assertEqual(len(ds["y"]), expected_samples)
        self.assertEqual(len(np.unique(ds["y"])), 4)

    def test_preprocessor_resampling_and_filtering(self):
        preprocessor = EEGPreprocessor(raw_fs=160, target_fs=128, use_ica=False)
        # Create dummy signal: 10 trials, 4 channels, 640 timepoints (4s at 160Hz)
        raw_dummy = np.random.randn(10, 4, 640)
        resampled = preprocessor.resample_signal(raw_dummy, orig_fs=160, target_fs=128)
        # 640 * (128/160) = 512
        self.assertEqual(resampled.shape, (10, 4, 512))

        filtered = preprocessor.bandpass_filter(resampled, fs=128, lowcut=8.0, highcut=30.0)
        self.assertEqual(filtered.shape, (10, 4, 512))

    def test_split_and_dataloader_creation(self):
        X = np.random.randn(100, 1, 1280).astype(np.float32)
        y = np.random.choice([0, 1, 2, 3], size=100)

        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_train_val_test(
            X, y, split_ratio=(0.5, 0.2, 0.3), random_state=42
        )
        self.assertEqual(len(X_tr) + len(X_va) + len(X_te), 100)
        self.assertAlmostEqual(len(X_tr), 50, delta=2)
        self.assertAlmostEqual(len(X_va), 20, delta=2)
        self.assertEqual(len(X_te), 30)

        loaders = create_dataloaders(
            (X_tr, y_tr), (X_va, y_va), (X_te, y_te), batch_size=16
        )
        self.assertIn("train", loaders)
        self.assertIn("val", loaders)
        self.assertIn("test", loaders)

        batch_X, batch_y = next(iter(loaders["train"]))
        self.assertEqual(batch_X.shape[1:], (1, 1280))


if __name__ == "__main__":
    unittest.main()
