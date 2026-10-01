"""Unit tests for Hybrid CNN-LSTM Architecture."""

import unittest
import torch
from src.models.cnn_lstm import HybridCNNLSTM
from src.models.ablations import HybridCNNGRU


class TestCNNLSTMModel(unittest.TestCase):

    def test_cnn_lstm_forward_pass(self):
        model = HybridCNNLSTM(
            input_channels=1,
            sequence_length=1280,
            lstm_units=100,
            lstm_timesteps=62,
            num_classes=4,
        )
        dummy_input = torch.randn(8, 1, 1280)
        output = model(dummy_input)

        self.assertEqual(output.shape, (8, 4))
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_cnn_lstm_trainable_parameters(self):
        model = HybridCNNLSTM()
        params = model.count_trainable_parameters()
        self.assertGreater(params, 50000)
        print(f"HybridCNNLSTM trainable parameters: {params:,}")

    def test_cnn_gru_forward_pass(self):
        model = HybridCNNGRU(
            input_channels=1,
            sequence_length=1280,
            gru_units=100,
            lstm_timesteps=62,
            num_classes=4,
        )
        dummy_input = torch.randn(8, 1, 1280)
        output = model(dummy_input)

        self.assertEqual(output.shape, (8, 4))
        self.assertTrue(torch.all(torch.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
