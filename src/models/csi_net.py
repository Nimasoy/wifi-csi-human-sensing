"""Hybrid 2D-CNN + BiLSTM + Temporal Self-Attention network for Wi-Fi CSI sensing.

Architecture (`CSISpatioTemporalNet`):
    - Input: `(B, 2, K, 200)` where channel 0 is filtered amplitude `A_filt` and
      channel 1 is sanitized/filtered phase `phi_filt`, `K` is the number of OFDM
      subcarriers (`52` default or `90`), and `T = 200` packets (`2.0 s @ 100 Hz`).
    - Stage 1 (2D Spatial-Spectral Encoder):
      3-block strided 2D CNN over `(K, T)` followed by `AdaptiveAvgPool2d((1, 100))`
      to make the architecture agnostic to `K` (`K = 52` or `K = 90`), yielding `(B, 100, 128)`.
    - Stage 2 (Temporal Sequence Encoder):
      2-layer Bidirectional LSTM (`128 -> 256`) + `LayerNorm(256)` yielding `(B, 100, 256)`.
    - Stage 3 (Temporal Self-Attention Pooling):
      Additive tanh attention (`Linear(256, 128)` -> `tanh` -> `Linear(128, 1, bias=False)`)
      computing normalized temporal weights `alpha in R^{B x 100 x 1}` and context vector
      `z in R^{B x 256}`.
    - Stage 4 (Classification Head):
      Two-layer MLP with GELU and Dropout yielding class logits `(B, 5)`.
"""

from __future__ import annotations

import torch
from torch import nn


class CSISpatioTemporalNet(nn.Module):
    """Spatio-Temporal Deep Neural Network for Wi-Fi CSI Human Presence & Motion Sensing.

    Attributes:
        in_channels: Number of input CSI feature channels (default: 2 for [Amp, Phase]).
        num_classes: Number of target activity classes (default: 5).
        temporal_pooled_len: Sequence length after 2D spatial-spectral pooling (default: 100).
        spatial_encoder: Sequential 2D CNN feature extractor.
        adaptive_pool: Adaptive average pooling collapsing subcarrier dimension to `(1, 100)`.
        lstm: 2-layer Bidirectional LSTM temporal sequence encoder.
        lstm_norm: LayerNorm normalizing the 256-dim BiLSTM hidden states.
        attn_proj: Linear projection `(256 -> 128)` for temporal attention.
        attn_score: Linear scoring layer `(128 -> 1, bias=False)` for temporal attention.
        classifier: Classification MLP head `(256 -> 128 -> num_classes)`.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 5,
        temporal_pooled_len: int = 100,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        attention_dim: int = 128,
    ) -> None:
        """Initialize the `CSISpatioTemporalNet` architecture.

        Args:
            in_channels: Number of input channels (default: 2).
            num_classes: Number of output classes C (default: 5).
            temporal_pooled_len: Pooled temporal sequence length after Stage 1 (default: 100).
            lstm_hidden_size: Hidden size per direction in the BiLSTM (default: 128).
            lstm_num_layers: Number of stacked BiLSTM layers (default: 2).
            lstm_dropout: Inter-layer dropout probability for BiLSTM (default: 0.3).
            attention_dim: Hidden projection dimension for additive self-attention (default: 128).
        """
        super().__init__()
        self.in_channels: int = in_channels
        self.num_classes: int = num_classes
        self.temporal_pooled_len: int = temporal_pooled_len

        # Stage 1: 2D Spatial-Spectral Encoder
        self.spatial_encoder = nn.Sequential(
            # Block 1: (B, 2, K, 200) -> (B, 32, ceil(K/2), 200)
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=(5, 3),
                stride=(2, 1),
                padding=(2, 1),
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(p=0.15),
            # Block 2: (B, 32, ceil(K/2), 200) -> (B, 64, ceil(K/4), 100)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(p=0.15),
            # Block 3: (B, 64, ceil(K/4), 100) -> (B, 128, ceil(K/8), 100)
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, temporal_pooled_len))

        # Stage 2: Temporal Sequence Encoder (BiLSTM + LayerNorm)
        bilstm_out_dim = lstm_hidden_size * 2  # 256
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(bilstm_out_dim)

        # Stage 3: Temporal Self-Attention Pooling
        self.attn_proj = nn.Linear(bilstm_out_dim, attention_dim)
        self.attn_score = nn.Linear(attention_dim, 1, bias=False)

        # Stage 4: Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(bilstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass through the Spatio-Temporal CSI Network.

        Args:
            x: Input CSI tensor of shape `(B, 2, K, T)` where `K` is the number of
                subcarriers (e.g., 52 or 90) and `T = 200`.
            return_attention: If True, returns `(logits, alpha)` where `alpha` has
                shape `(B, 100, 1)` representing temporal self-attention weights.

        Returns:
            If `return_attention` is False:
                Logits tensor of shape `(B, 5)`.
            If `return_attention` is True:
                Tuple `(logits, alpha)` with shapes `((B, 5), (B, 100, 1))`.

        Raises:
            ValueError: If input tensor `x` does not have 4 dimensions `(B, C, K, T)`.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, 2, K, T), got shape {tuple(x.shape)}."
            )

        # Stage 1: 2D Spatial-Spectral Encoding
        feat_2d = self.spatial_encoder(x)              # (B, 128, K', 100)
        pooled = self.adaptive_pool(feat_2d)           # (B, 128, 1, 100)
        seq_in = pooled.squeeze(2).permute(0, 2, 1)    # (B, 100, 128)

        # Stage 2: Bidirectional LSTM Temporal Encoding
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(seq_in)                # (B, 100, 256)
        h_t = self.lstm_norm(lstm_out)                 # (B, 100, 256)

        # Stage 3: Temporal Self-Attention Pooling
        u_t = torch.tanh(self.attn_proj(h_t))          # (B, 100, 128)
        scores = self.attn_score(u_t)                  # (B, 100, 1)
        alpha = torch.softmax(scores, dim=1)           # (B, 100, 1)
        z = torch.sum(alpha * h_t, dim=1)              # (B, 256)

        # Stage 4: Classification Head
        logits = self.classifier(z)                    # (B, 5)

        if return_attention:
            return logits, alpha
        return logits
