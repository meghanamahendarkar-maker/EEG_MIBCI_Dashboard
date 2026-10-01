import os
import sys
import torch
import joblib
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import PhysioNetLoader
from src.data.preprocessor import EEGPreprocessor
from src.data.dataset import split_train_val_test, create_dataloaders
from src.models.minirocket_pipeline import MiniRocketPipeline
from src.models.cnn_lstm import HybridCNNLSTM
from src.training.trainer_dl import DeepLearningTrainer

def main():
    print("Starting Training on 15 PhysioNet Subjects...")
    num_subjects = 15
    os.makedirs("checkpoints", exist_ok=True)
    
    loader = PhysioNetLoader(data_dir="data/physionet")
    preprocessor = EEGPreprocessor(raw_fs=160, target_fs=160, use_ica=False)
    
    all_X, all_y = [], []
    for s_idx in range(1, num_subjects + 1):
        try:
            print(f"Loading Subject {s_idx}...")
            trials_data, labels, ch_names = loader.load_subject_dataset(s_idx)
            X_s, y_s, _ = preprocessor.process_trials(trials_data, labels, ch_names, subject_id=s_idx)
            all_X.append(X_s)
            all_y.append(y_s)
        except Exception as e:
            print(f"Failed to load subject {s_idx}: {e}")
            
    if len(all_X) == 0:
        print("Failed to download or load any data.")
        return
        
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"Total dataset shape: {X.shape}")
    
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_train_val_test(X, y, split_ratio=(0.5, 0.2, 0.3), random_state=42)
    
    print("Training MiniRocket...")
    mr_pipe = MiniRocketPipeline(num_kernels=2000, random_state=42)
    mr_pipe.fit(X_tr, y_tr)
    joblib.dump(mr_pipe, "checkpoints/mr_pipe.pkl")
    print("MiniRocket saved to checkpoints/mr_pipe.pkl")
    
    print("Training CNN-LSTM...")
    cnn_lstm = HybridCNNLSTM(input_channels=1, sequence_length=X.shape[-1], lstm_units=64, num_classes=4)
    loaders = create_dataloaders((X_tr, y_tr), (X_va, y_va), (X_te, y_te), batch_size=32)
    
    trainer = DeepLearningTrainer(
        model=cnn_lstm,
        learning_rate=1e-4,
        early_stopping_patience=5
    )
    
    trainer.fit(loaders["train"], loaders["val"], epochs=15, verbose=True)
    
    torch.save(cnn_lstm.state_dict(), "checkpoints/cnn_lstm.pt")
    print("CNN-LSTM saved to checkpoints/cnn_lstm.pt")
    print("Done!")

if __name__ == "__main__":
    main()
