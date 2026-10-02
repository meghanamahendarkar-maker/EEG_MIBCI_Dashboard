import re
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

@dataclass
class EEGTrial:
    subject_id: str
    run: int
    trial_index: int
    annotation: str
    class_id: int
    class_name: str
    onset: float
    duration: float
    sfreq: float
    channel_names: List[str]
    signal: np.ndarray  # shape (n_channels, n_times)

def parse_physionet_filename(filename: str):
    """
    Parses a filename like 'S001R10.edf' to extract subject, run, and experiment type.
    """
    match = re.search(r'(S\d+)(R\d+)\.edf', filename, re.IGNORECASE)
    if not match:
        return None, None, "unknown"
    
    subject_id = match.group(1).upper()
    run_str = match.group(2).upper()
    run_number = int(run_str.replace('R', ''))
    
    if run_number in [4, 8, 12]:
        experiment_type = "motor_imagery_left_right_fist"
    elif run_number in [6, 10, 14]:
        experiment_type = "motor_imagery_both_fists_both_feet"
    elif run_number in [3, 7, 11]:
        experiment_type = "motor_execution_left_right_fist"
    elif run_number in [5, 9, 13]:
        experiment_type = "motor_execution_both_fists_both_feet"
    elif run_number in [1, 2]:
        experiment_type = "baseline"
    else:
        experiment_type = "unknown"
        
    return subject_id, run_number, experiment_type
