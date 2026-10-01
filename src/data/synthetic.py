"""Synthetic EEG data generator for Motor Imagery (MI-BCI).
Generates realistic physiological EEG signals exhibiting Event-Related Desynchronization (ERD)
and Event-Related Synchronization (ERS) in the mu (8-14 Hz) and beta (14-30 Hz) bands.
"""

import numpy as np
from typing import Tuple, Dict, Any


def generate_synthetic_eeg_trial(
    task_class: int,
    sampling_rate: int = 128,
    duration_sec: float = 4.0,
    num_electrode_pairs: int = 5,
    snr_db: float = -10.0,
    random_state: int = None,
) -> np.ndarray:
    """Generate a single 4-second multi-channel EEG trial for a specific motor imagery class.

    Args:
        task_class: Integer 0 (Left Fist), 1 (Right Fist), 2 (Both Fists), 3 (Both Feet).
        sampling_rate: Target sampling rate in Hz (default 128 Hz).
        duration_sec: Duration of motor imagination in seconds (default 4.0s).
        num_electrode_pairs: Number of symmetric electrode pairs (default 5).
        snr_db: Signal-to-noise ratio in decibels.
        random_state: Seed for reproducibility.

    Returns:
        np.ndarray of shape (num_electrode_pairs, 2, num_samples), representing
        (pairs, left_right_electrode, time_points).
    """
    rng = np.random.default_rng(random_state)
    num_samples = int(sampling_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Base 1/f pink noise simulating spontaneous background EEG rhythms
    pink_noise = np.zeros((num_electrode_pairs, 2, num_samples))
    for p in range(num_electrode_pairs):
        for e in range(2):
            white = rng.standard_normal(num_samples)
            # Simple 1/f filter approximation in frequency domain
            fft_white = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(num_samples, d=1.0 / sampling_rate)
            freqs[0] = 1.0  # Avoid division by zero
            fft_pink = fft_white / np.sqrt(freqs)
            pink = np.fft.irfft(fft_pink, n=num_samples)
            pink_noise[p, e] = pink / (np.std(pink) + 1e-8)

    # Task-specific ERD/ERS modulation
    # Left electrode: index 0 (e.g., C3, FC3, CP3)
    # Right electrode: index 1 (e.g., C4, FC4, CP4)
    # Mu rhythm: ~10 Hz, Beta rhythm: ~20 Hz
    signal = np.zeros((num_electrode_pairs, 2, num_samples))

    # Define contralateral ERD attenuation factors:
    # Class 0: Left fist -> contralateral right hemisphere ERD (attenuation at e=1, boost at e=0)
    # Class 1: Right fist -> contralateral left hemisphere ERD (attenuation at e=0, boost at e=1)
    # Class 2: Both fists -> bilateral ERD (attenuation at both e=0 and e=1)
    # Class 3: Both feet -> central/mild rebound or bilateral high-frequency modulation
    for p in range(num_electrode_pairs):
        pair_weight = 1.0 if p == 1 else 0.7  # C3/C4 pair (p=1) is strongest

        for e in range(2):
            mu_freq = 10.0 + rng.uniform(-0.5, 0.5)
            beta_freq = 20.0 + rng.uniform(-1.0, 1.0)
            phase_mu = rng.uniform(0, 2 * np.pi)
            phase_beta = rng.uniform(0, 2 * np.pi)

            # Determine amplitude based on motor task with heavy overlapping variance to prevent 100% accuracy
            if task_class == 0:  # Left Fist
                mu_amp = rng.normal(0.7, 0.1) if e == 1 else rng.normal(1.0, 0.1)
                beta_amp = rng.normal(0.75, 0.1) if e == 1 else rng.normal(0.9, 0.1)
            elif task_class == 1:  # Right Fist
                mu_amp = rng.normal(1.0, 0.1) if e == 1 else rng.normal(0.7, 0.1)
                beta_amp = rng.normal(0.9, 0.1) if e == 1 else rng.normal(0.75, 0.1)
            elif task_class == 2:  # Both Fists
                mu_amp = rng.normal(0.7, 0.1)
                beta_amp = rng.normal(0.75, 0.1)
            else:  # Both Feet 
                mu_amp = rng.normal(0.95, 0.1) if p in [0, 2] else rng.normal(0.8, 0.1)
                beta_amp = rng.normal(1.0, 0.1) if p in [0, 2] else rng.normal(0.85, 0.1)

            mu_wave = mu_amp * np.sin(2 * np.pi * mu_freq * t + phase_mu)
            beta_wave = beta_amp * np.sin(2 * np.pi * beta_freq * t + phase_beta)
            signal[p, e] = pair_weight * (mu_wave + 0.6 * beta_wave)

    # Scale signal according to SNR
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(pink_noise ** 2)
    target_snr_linear = 10 ** (snr_db / 10.0)
    scaled_noise = pink_noise * np.sqrt(sig_power / (target_snr_linear * noise_power + 1e-8))

    trial_eeg = signal + scaled_noise
    return trial_eeg


def generate_synthetic_eeg_dataset(
    num_subjects: int = 10,
    trials_per_class: int = 21,
    sampling_rate: int = 128,
    duration_sec: float = 4.0,
    samples_per_trial: int = 9,
    sample_length: int = 1280,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Generate a full synthetic dataset matching the PhysioNet setup described in the paper.

    In the paper:
    - 10 subjects (S1 - S10).
    - 4 classes (0: Left fist, 1: Right fist, 2: Both fists, 3: Both feet).
    - 21 trials per class = 84 trials per subject.
    - Each trial yields 9 samples. Total samples per subject = 84 * 9 = 756 samples.
    - Total for 10 subjects = 7,560 samples.
    - Each sample consists of a pair of symmetric electrodes serially connected (length 1280).

    Returns:
        Dictionary containing:
            'X': np.ndarray of shape (num_samples_total, 1, sample_length)
            'y': np.ndarray of shape (num_samples_total,)
            'subjects': np.ndarray of shape (num_samples_total,)
            'trials': np.ndarray of shape (num_samples_total,)
    """
    rng = np.random.default_rng(random_state)
    all_samples = []
    all_labels = []
    all_subject_ids = []
    all_trial_ids = []

    # Calculate subwindow length for the 9 samples per trial
    # Full trial duration samples = sampling_rate * duration_sec = 128 * 4 = 512
    # Serial symmetric electrode pair length: 640 per electrode -> 1280 total sample
    total_trial_points = int(sampling_rate * duration_sec)

    global_trial_id = 0
    for subj_idx in range(1, num_subjects + 1):
        subj_seed = random_state + subj_idx * 1000

        for task_class in range(4):
            for t_idx in range(trials_per_class):
                trial_seed = subj_seed + task_class * 100 + t_idx
                # Generate 5-pair trial: (5, 2, total_trial_points)
                trial_raw = generate_synthetic_eeg_trial(
                    task_class=task_class,
                    sampling_rate=sampling_rate,
                    duration_sec=duration_sec,
                    num_electrode_pairs=5,
                    snr_db=-5.0 + rng.uniform(-2.0, 2.0),
                    random_state=trial_seed,
                )

                # Segment into 9 samples per trial using sliding/resampled windows across the 5 pairs
                # Primary motor pair (pair index 1 = C3/C4) concatenated serially
                left_electrode = trial_raw[1, 0]  # shape (512,)
                right_electrode = trial_raw[1, 1]  # shape (512,)

                # Generate 9 non-overlapping/sub-sampled windows
                # Window size: 640 points (resampled or sub-segmented)
                # To match 1280 points sample (640 left + 640 right):
                step = (total_trial_points - 128) // (samples_per_trial - 1) if samples_per_trial > 1 else 0

                for s in range(samples_per_trial):
                    start = min(s * step, total_trial_points - 128)
                    end = start + 128
                    # Upsample / pad segment to 640 points as per the paper's 640-point electrode format
                    sub_left = np.interp(
                        np.linspace(0, 1, 640),
                        np.linspace(0, 1, end - start),
                        left_electrode[start:end]
                    )
                    sub_right = np.interp(
                        np.linspace(0, 1, 640),
                        np.linspace(0, 1, end - start),
                        right_electrode[start:end]
                    )

                    # Serially concatenate symmetric pair: left followed by right -> length 1280
                    sample_1280 = np.concatenate([sub_left, sub_right])  # shape (1280,)

                    all_samples.append(sample_1280)
                    all_labels.append(task_class)
                    all_subject_ids.append(subj_idx)
                    all_trial_ids.append(global_trial_id)

                global_trial_id += 1

    X = np.array(all_samples, dtype=np.float32)  # shape (N, 1280)
    # Add channel dimension: (N, 1, 1280) for 1D CNN / MiniRocket
    X = np.expand_dims(X, axis=1)
    y = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subject_ids, dtype=np.int32)
    trials = np.array(all_trial_ids, dtype=np.int32)

    return {
        "X": X,
        "y": y,
        "subjects": subjects,
        "trials": trials,
        "sample_length": sample_length,
        "num_classes": 4,
    }
