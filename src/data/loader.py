"""PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) Loader.
Utilizes MNE-Python to download, cache, and extract 4-class Motor Imagery recordings.
"""

import os
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    import mne
    from mne.datasets import eegbci
except ImportError:
    mne = None
    eegbci = None

logger = logging.getLogger(__name__)


class PhysioNetLoader:
    """Manages downloading and raw extraction of the PhysioNet EEGMMIDB dataset."""

    TASK1_RUNS = [4, 8, 12]  # Imagined Left fist (T1) vs Right fist (T2)
    TASK2_RUNS = [6, 10, 14] # Imagined Both fists (T1) vs Both feet (T2)

    # Class mappings:
    # Task 1: T1 -> Class 0 (Left Fist), T2 -> Class 1 (Right Fist)
    # Task 2: T1 -> Class 2 (Both Fists), T2 -> Class 3 (Both Feet)
    CLASS_NAMES = {
        0: "Left Fist",
        1: "Right Fist",
        2: "Both Fists",
        3: "Both Feet",
    }

    # Standard 64-channel 10-20 motor cortex channels
    DEFAULT_MOTOR_CHANNELS = [
        "FC3.", "FC4.",
        "C3..", "C4..",
        "CP3.", "CP4.",
        "FC1.", "FC2.",
        "CP1.", "CP2.",
        "Cz..",
    ]

    def __init__(
        self,
        data_dir: Optional[str] = None,
        raw_sampling_rate: int = 160,
    ):
        self.data_dir = data_dir
        self.raw_sampling_rate = raw_sampling_rate

    def load_subject_raw(
        self, subject_id: int
    ) -> Tuple[List["mne.io.Raw"], List["mne.io.Raw"]]:
        """Download and load raw EDF files for a given subject (S1 - S109).

        Args:
            subject_id: Integer subject ID (1 to 109).

        Returns:
            Tuple of (task1_raw_list, task2_raw_list)
        """
        if mne is None:
            raise ImportError("MNE is required to load PhysioNet data. Install via `pip install mne`.")

        logger.info(f"Fetching PhysioNet EEG data for Subject {subject_id}...")
        try:
            task1_paths = eegbci.load_data(
                subject_id, self.TASK1_RUNS, path=self.data_dir, verbose=False
            )
            task2_paths = eegbci.load_data(
                subject_id, self.TASK2_RUNS, path=self.data_dir, verbose=False
            )
        except Exception as e:
            logger.error(f"Error fetching data for Subject {subject_id}: {e}")
            raise

        task1_raws = [
            mne.io.read_raw_edf(p, preload=True, verbose=False) for p in task1_paths
        ]
        task2_raws = [
            mne.io.read_raw_edf(p, preload=True, verbose=False) for p in task2_paths
        ]

        # Standardize channel names (strip dots/spaces e.g., 'C3..' -> 'C3')
        for raw_list in (task1_raws, task2_raws):
            for r in raw_list:
                mne.datasets.eegbci.standardize(r)

        return task1_raws, task2_raws

    def extract_trials(
        self,
        raw: "mne.io.Raw",
        task_type: int,
        tmin: float = 0.0,
        tmax: float = 4.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 4-second motor imagery epochs and assign class labels (0, 1, 2, 3).

        Args:
            raw: MNE Raw object.
            task_type: 1 for Task 1 (L/R fist), 2 for Task 2 (both fists/feet).
            tmin: Start time relative to event (0.0s = cue onset).
            tmax: End time relative to event (4.0s = cue end).

        Returns:
            Tuple of (epochs_data, labels)
            epochs_data: shape (n_trials, n_channels, n_timepoints)
            labels: shape (n_trials,)
        """
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        # PhysioNet annotations: 'T1' -> class A, 'T2' -> class B, 'T0' -> rest

        t1_code = event_dict.get("T1")
        t2_code = event_dict.get("T2")

        target_events = []
        target_labels = []

        for ev in events:
            ev_id = ev[2]
            if ev_id == t1_code:
                label = 0 if task_type == 1 else 2
                target_events.append(ev)
                target_labels.append(label)
            elif ev_id == t2_code:
                label = 1 if task_type == 1 else 3
                target_events.append(ev)
                target_labels.append(label)

        if not target_events:
            return np.empty((0, len(raw.ch_names), int((tmax - tmin) * raw.info["sfreq"]))), np.empty((0,))

        target_events = np.array(target_events)
        target_labels = np.array(target_labels)

        epochs = mne.Epochs(
            raw,
            target_events,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )

        epochs_data = epochs.get_data()  # shape (n_trials, n_channels, n_times)
        return epochs_data, target_labels

    def load_subject_dataset(
        self, subject_id: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Download, parse, and collect all 4 MI classes for a subject.

        Returns:
            Tuple of (all_trials, all_labels, channel_names)
            all_trials: shape (n_total_trials, n_channels, n_times)
            all_labels: shape (n_total_trials,) with values in {0, 1, 2, 3}
            channel_names: list of channel name strings
        """
        task1_raws, task2_raws = self.load_subject_raw(subject_id)
        ch_names = task1_raws[0].ch_names

        trials_list = []
        labels_list = []

        # Process Task 1 runs (classes 0: Left Fist, 1: Right Fist)
        for raw in task1_raws:
            data, labels = self.extract_trials(raw, task_type=1)
            if len(labels) > 0:
                trials_list.append(data)
                labels_list.append(labels)

        # Process Task 2 runs (classes 2: Both Fists, 3: Both Feet)
        for raw in task2_raws:
            data, labels = self.extract_trials(raw, task_type=2)
            if len(labels) > 0:
                trials_list.append(data)
                labels_list.append(labels)

        all_trials = np.concatenate(trials_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        logger.info(
            f"Subject {subject_id} loaded: {all_trials.shape[0]} trials total, "
            f"{all_trials.shape[1]} channels, {all_trials.shape[2]} timepoints."
        )
        return all_trials, all_labels, ch_names
