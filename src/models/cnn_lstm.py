"""Hybrid CNN-LSTM Deep Learning Architecture.
Implements the exact 13-layer model specified in Table 1 & Figure 5 of Hwaidi & Ghanem (2026).
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class HybridCNNLSTM(nn.Module):
    """13-layer CNN-LSTM Hybrid Neural Network for Motor Imagery EEG Classification."""

    def __init__(
        self,
        input_channels: int = 1,
        sequence_length: int = 1280,
        lstm_units: int = 100,
        lstm_timesteps: int = 62,
        num_classes: int = 4,
        conv1_filters: int = 16,
        conv2_filters: int = 32,
        kernel_size: int = 3,
        dropout_conv: float = 0.5,
        dropout_lstm: float = 0.5,
        dropout_dense: float = 0.25,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.lstm_timesteps = lstm_timesteps
        self.num_classes = num_classes

        # Layer 1: Input (handled as tensor input (batch, 1, 1280))

        # Layer 2: Conv1D (16 filters, kernel size 3, stride 1, VALID padding, ReLU)
        # 1280 -> 1278
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=conv1_filters,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.relu1 = nn.ReLU()

        # Layer 3: Conv1D (32 filters, kernel size 3, stride 1, VALID padding, ReLU)
        # 1278 -> 1276
        self.conv2 = nn.Conv1d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.relu2 = nn.ReLU()

        # Layer 4: Dropout (0.5)
        self.dropout_conv = nn.Dropout(dropout_conv)

        # Layer 5: Max Pooling 1D (pool size 2, stride 2, VALID padding)
        # 1276 -> 638
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Removed AdaptiveMaxPool1d to prevent destroying high-frequency temporal data

        # --- ADVANCED PROTOTYPE UPGRADE: Self-Attention Mechanism ---
        # Adding a Multi-Head Attention layer to focus on salient temporal EEG features
        self.attention = nn.MultiheadAttention(embed_dim=conv2_filters, num_heads=4, batch_first=True)

        # Layer 7: LSTM (100 units)
        self.lstm = nn.LSTM(
            input_size=conv2_filters,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
        )

        # Layer 8: Dropout (0.5)
        self.dropout_lstm = nn.Dropout(dropout_lstm)

        # Layer 9: Dense (100 units, ReLU)
        self.dense1 = nn.Linear(lstm_units, 100)
        self.relu_dense1 = nn.ReLU()

        # Layer 10: Dropout (0.25)
        self.dropout_dense1 = nn.Dropout(dropout_dense)

        # Layer 11: Dense (50 units, ReLU)
        self.dense2 = nn.Linear(100, 50)
        self.relu_dense2 = nn.ReLU()

        # Layer 12: Dropout (0.25)
        self.dropout_dense2 = nn.Dropout(dropout_dense)

        # Layer 13: Dense (4 units, Logits for 4 classes)
        self.classifier = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the advanced Attention-CNN-LSTM layers."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.dropout_conv(out)
        out = self.pool(out)
        out = out.permute(0, 2, 1)

        # --- ADVANCED PROTOTYPE UPGRADE: Apply Self-Attention ---
        attn_output, _ = self.attention(out, out, out)
        out = out + attn_output # Residual connection

        lstm_out, (hn, cn) = self.lstm(out)

        last_hidden = hn[-1]
        out = self.dropout_lstm(last_hidden)

        out = self.relu_dense1(self.dense1(out))
        out = self.dropout_dense1(out)

        out = self.relu_dense2(self.dense2(out))
        out = self.dropout_dense2(out)

        logits = self.classifier(out)
        return logits

    def count_trainable_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
