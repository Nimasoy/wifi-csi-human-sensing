"""Unit tests for `CSISpatioTemporalNet` (`src/models/csi_net.py`).

Verifies:
1. Forward pass output shapes `(B, 5)` and `(B, 100, 1)` for both `K = 52`
   (ESP32-CSI-Tool default) and `K = 90` (Widar3.0 3x30 links).
2. Temporal self-attention weights `alpha` are non-negative and sum to `1.0`
   along the temporal sequence dimension (`dim=1`).
3. Non-zero backward gradients flow cleanly through all four stages:
   Stage 1 (2D CNN), Stage 2 (2-Layer BiLSTM + LayerNorm),
   Stage 3 (Temporal Self-Attention), and Stage 4 (Classification Head).
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.engine.trainer import set_deterministic_seed
from src.models.csi_net import CSISpatioTemporalNet


class TestCSISpatioTemporalNet(unittest.TestCase):
    """Test suite for `CSISpatioTemporalNet` architecture and gradient flow."""

    def setUp(self) -> None:
        """Seed RNGs and instantiate `CSISpatioTemporalNet`."""
        set_deterministic_seed(42)
        self.batch_size = 4
        self.window_size = 200
        self.num_classes = 5
        self.model = CSISpatioTemporalNet(
            in_channels=2,
            num_classes=self.num_classes,
            temporal_pooled_len=100,
            lstm_hidden_size=128,
            lstm_num_layers=2,
            lstm_dropout=0.3,
            attention_dim=128,
        )

    def test_forward_shapes_for_k52_and_k90(self) -> None:
        """Verify forward pass works seamlessly for both `K = 52` and `K = 90`."""
        self.model.eval()
        for num_k in (52, 90):
            x = torch.randn(self.batch_size, 2, num_k, self.window_size)
            with torch.no_grad():
                logits = self.model(x, return_attention=False)
                self.assertIsInstance(logits, torch.Tensor)
                self.assertEqual(
                    tuple(logits.shape),
                    (self.batch_size, self.num_classes),
                )

                logits_attn, alpha = self.model(x, return_attention=True)
                self.assertEqual(
                    tuple(logits_attn.shape),
                    (self.batch_size, self.num_classes),
                )
                self.assertEqual(
                    tuple(alpha.shape),
                    (self.batch_size, 100, 1),
                )

    def test_attention_weights_sum_to_one(self) -> None:
        """Verify temporal self-attention weights `alpha` sum to `1.0` along `dim=1`."""
        self.model.eval()
        x = torch.randn(self.batch_size, 2, 52, self.window_size)
        with torch.no_grad():
            _, alpha = self.model(x, return_attention=True)

        self.assertTrue(torch.all(alpha >= 0.0))
        alpha_sums = alpha.sum(dim=1)  # shape: (B, 1)
        torch.testing.assert_close(
            alpha_sums,
            torch.ones_like(alpha_sums),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_backward_gradients_across_all_stages(self) -> None:
        """Verify non-zero gradients across CNN, BiLSTM, Attention, and Classifier layers."""
        self.model.train()
        self.model.zero_grad()

        x = torch.randn(self.batch_size, 2, 52, self.window_size, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 4], dtype=torch.long)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        logits, alpha = self.model(x, return_attention=True)
        loss = criterion(logits, targets)
        loss.backward()

        self.assertFalse(torch.isnan(loss).item())
        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(tuple(alpha.shape), (self.batch_size, 100, 1))

        # Verify every trainable parameter across all 4 stages has a non-zero gradient
        stage_groups = {
            "spatial_encoder": self.model.spatial_encoder,
            "lstm": self.model.lstm,
            "lstm_norm": self.model.lstm_norm,
            "attn_proj": self.model.attn_proj,
            "attn_score": self.model.attn_score,
            "classifier": self.model.classifier,
        }
        for stage_name, module in stage_groups.items():
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(
                        param.grad,
                        msg=f"Missing gradient in {stage_name}.{param_name}",
                    )
                    grad_norm = float(param.grad.abs().sum().item())
                    self.assertGreater(
                        grad_norm,
                        0.0,
                        msg=f"Zero gradient norm in {stage_name}.{param_name}",
                    )


if __name__ == "__main__":
    unittest.main()
