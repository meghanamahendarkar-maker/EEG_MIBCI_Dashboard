import mne
import numpy as np
from typing import List
from .physionet_mapping import EEGTrial, parse_physionet_filename

class TrialExtractor:
    def __init__(self, tmin=0.0, tmax=4.0):
        self.tmin = tmin
        self.tmax = tmax
        
    def extract_trials_from_raw(self, raw: mne.io.Raw, filename: str) -> List[EEGTrial]:
        subject_id, run_number, exp_type = parse_physionet_filename(filename)
        
        if run_number not in [4, 6, 8, 10, 12, 14]:
            raise ValueError(f"Run {run_number} is not a supported motor imagery run.")
            
        mne.datasets.eegbci.standardize(raw)
        
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        sfreq = raw.info['sfreq']
        
        trials = []
        trial_idx = 1
        
        for ev in events:
            onset_samp = ev[0]
            ev_id = ev[2]
            
            # Find annotation string
            anno_str = None
            for k, v in event_dict.items():
                if v == ev_id:
                    anno_str = k
                    break
                    
            if anno_str == 'T0':
                continue # ignore rest
                
            if run_number in [4, 8, 12]:
                if anno_str == 'T1':
                    class_id = 0
                    class_name = "Left Fist Imagery"
                elif anno_str == 'T2':
                    class_id = 1
                    class_name = "Right Fist Imagery"
                else:
                    continue
            elif run_number in [6, 10, 14]:
                if anno_str == 'T1':
                    class_id = 2
                    class_name = "Both Fists Imagery"
                elif anno_str == 'T2':
                    class_id = 3
                    class_name = "Both Feet Imagery"
                else:
                    continue
            else:
                continue
                
            start_samp = onset_samp + int(self.tmin * sfreq)
            end_samp = onset_samp + int(self.tmax * sfreq)
            
            # Extract 4 second window
            data = raw.get_data(start=start_samp, stop=end_samp)
            
            # Pad if needed
            req_len = int((self.tmax - self.tmin) * sfreq)
            if data.shape[1] < req_len:
                pad_t = np.zeros((data.shape[0], req_len - data.shape[1]))
                data = np.hstack([data, pad_t])
            elif data.shape[1] > req_len:
                data = data[:, :req_len]
                
            trial = EEGTrial(
                subject_id=subject_id,
                run=run_number,
                trial_index=trial_idx,
                annotation=anno_str,
                class_id=class_id,
                class_name=class_name,
                onset=start_samp / sfreq,
                duration=self.tmax - self.tmin,
                sfreq=sfreq,
                channel_names=raw.ch_names,
                signal=data
            )
            trials.append(trial)
            trial_idx += 1
            
        return trials
