"""Deep Learning Trainer for Hybrid CNN-LSTM / CNN-GRU.
Implements exact optimizer and regularization settings from Hwaidi & Ghanem (2026):
- Adam optimizer with lr = 1e-5
- L2 weight regularization (weight_decay = 0.01)
- CrossEntropy loss
- Tracking epoch loss/accuracy curves
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DeepLearningTrainer:
    """Trains and evaluates PyTorch deep learning models (CNN-LSTM, CNN-GRU)."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-5,
        l2_weight_decay: float = 0.01,
        device: Optional[str] = None,
        early_stopping_patience: int = 15,
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = torch.device("xpu")
            else:
                try:
                    import torch_directml  # type: ignore
                    if torch_directml.is_available():
                        self.device = torch_directml.device()
                    else:
                        self.device = torch.device("cpu")
                except ImportError:
                    self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.l2_weight_decay = l2_weight_decay
        self.early_stopping_patience = early_stopping_patience

        # Loss function & Adam optimizer matching paper
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_weight_decay,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.train_time_sec_ = 0.0
        self.best_model_state_ = None

    def train_epoch(self, train_loader: DataLoader, use_mixup: bool = False, alpha: float = 0.2) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # --- ADVANCED PROTOTYPE UPGRADE: MixUp Data Augmentation ---
            if use_mixup and alpha > 0:
                lam = np.random.beta(alpha, alpha)
                batch_size = batch_X.size(0)
                index = torch.randperm(batch_size).to(self.device)
                
                mixed_X = lam * batch_X + (1 - lam) * batch_X[index, :]
                y_a, y_b = batch_y, batch_y[index]
                
                self.optimizer.zero_grad()
                logits = self.model(mixed_X)
                loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
            else:
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                
            loss.backward()
            
            # --- ADVANCED PROTOTYPE UPGRADE: Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            # For MixUp, accuracy is approximate based on dominant label
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        epoch_loss = total_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        return epoch_loss, epoch_acc

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)

                total_loss += loss.item() * len(batch_y)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        epoch_loss = total_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        preds_arr = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
        return epoch_loss, epoch_acc, preds_arr, probs_arr

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Run full training loop over specified number of epochs."""
        start_time = time.perf_counter()
        best_val_acc = -1.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_epoch(train_loader)
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)

            if val_loader is not None:
                va_loss, va_acc, _, _ = self.evaluate(val_loader)
                self.history["val_loss"].append(va_loss)
                self.history["val_acc"].append(va_acc)

                if va_acc > best_val_acc:
                    best_val_acc = va_acc
                    patience_counter = 0
                    self.best_model_state_ = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
                    logger.info(
                        f"Epoch [{epoch:03d}/{epochs:03d}] | "
                        f"Train Loss: {tr_loss:.4f} Acc: {tr_acc * 100:.2f}% | "
                        f"Val Loss: {va_loss:.4f} Acc: {va_acc * 100:.2f}%"
                    )

                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
            else:
                if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
                    logger.info(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {tr_loss:.4f} Acc: {tr_acc * 100:.2f}%")

        self.train_time_sec_ = time.perf_counter() - start_time

        # Restore best model state if available
        if self.best_model_state_ is not None:
            self.model.load_state_dict(self.best_model_state_)

        return self.history
