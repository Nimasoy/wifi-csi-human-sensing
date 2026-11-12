"""Training-only RF/CSI data augmentations for normalized spatio-temporal tensors.

Implements `CSIAugmenter` operating on normalized CSI tensors `X in R^{2 x K x T}`
(or batched `(B, 2, K, T)`) during training only:
1. Additive Gaussian noise jitter with standard deviation `sigma ~ U(0.01, 0.05)`.
2. Contiguous subcarrier masking (zeroing a random block of up to 5 subcarriers).
3. Contiguous temporal masking (zeroing a random block of up to 15 time steps).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class CSIAugmenter:
    """Stochastic RF/CSI spectrogram & time-series augmenter for training only.

    Attributes:
        noise_std_range: Tuple `(min_sigma, max_sigma)` for additive Gaussian jitter
            (default: `(0.01, 0.05)`).
        max_subcarrier_mask: Maximum contiguous subcarriers to zero out (default: 5).
        max_temporal_mask: Maximum contiguous time steps to zero out (default: 15).
        p_noise: Probability of applying Gaussian noise jitter (default: 1.0).
        p_subcarrier_mask: Probability of applying contiguous subcarrier masking (default: 0.8).
        p_temporal_mask: Probability of applying contiguous temporal masking (default: 0.8).
        seed: Deterministic seed initializing the internal NumPy generator (default: 42).
    """

    def __init__(
        self,
        noise_std_range: tuple[float, float] = (0.01, 0.05),
        max_subcarrier_mask: int = 5,
        max_temporal_mask: int = 15,
        p_noise: float = 1.0,
        p_subcarrier_mask: float = 0.8,
        p_temporal_mask: float = 0.8,
        seed: int = 42,
    ) -> None:
        """Initialize the `CSIAugmenter`.

        Args:
            noise_std_range: Range `(sigma_min, sigma_max)` for uniform noise std sampling.
            max_subcarrier_mask: Maximum contiguous block size along subcarrier axis `K`.
            max_temporal_mask: Maximum contiguous block size along time axis `T`.
            p_noise: Probability of applying Gaussian noise jitter per call.
            p_subcarrier_mask: Probability of applying subcarrier masking per call.
            p_temporal_mask: Probability of applying temporal masking per call.
            seed: Random seed for reproducible augmentation sequences.

        Raises:
            ValueError: If `noise_std_range` is invalid or mask sizes are negative.
        """
        if (
            len(noise_std_range) != 2
            or noise_std_range[0] < 0.0
            or noise_std_range[0] > noise_std_range[1]
        ):
            raise ValueError(
                f"Invalid noise_std_range={noise_std_range}; require 0 <= min <= max."
            )
        if max_subcarrier_mask < 0 or max_temporal_mask < 0:
            raise ValueError("Mask lengths must be non-negative integers.")

        self.noise_std_range: tuple[float, float] = (
            float(noise_std_range[0]),
            float(noise_std_range[1]),
        )
        self.max_subcarrier_mask: int = int(max_subcarrier_mask)
        self.max_temporal_mask: int = int(max_temporal_mask)
        self.p_noise: float = float(p_noise)
        self.p_subcarrier_mask: float = float(p_subcarrier_mask)
        self.p_temporal_mask: float = float(p_temporal_mask)
        self.seed: int = int(seed)
        self._rng: np.random.Generator = np.random.default_rng(self.seed)

    def apply_gaussian_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add zero-mean Gaussian noise with `sigma ~ U(0.01, 0.05)` to tensor `x`.

        Args:
            x: Tensor of shape `(2, K, T)` or `(B, 2, K, T)`.

        Returns:
            Noise-jittered tensor of the same shape and dtype.
        """
        sigma = float(
            self._rng.uniform(self.noise_std_range[0], self.noise_std_range[1])
        )
        noise_np = self._rng.normal(loc=0.0, scale=sigma, size=tuple(x.shape))
        noise_tensor = torch.from_numpy(noise_np).to(device=x.device, dtype=x.dtype)
        return x + noise_tensor

    def apply_subcarrier_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out a contiguous block of `1..max_subcarrier_mask` subcarriers.

        Args:
            x: Tensor of shape `(2, K, T)` or `(B, 2, K, T)`.

        Returns:
            Subcarrier-masked tensor of the same shape.
        """
        num_k = x.shape[-2]
        max_block = min(self.max_subcarrier_mask, num_k)
        if max_block <= 0:
            return x

        block_len = int(self._rng.integers(1, max_block + 1))
        start_k = int(self._rng.integers(0, num_k - block_len + 1))
        out = x.clone()
        out[..., start_k : start_k + block_len, :] = 0.0
        return out

    def apply_temporal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out a contiguous block of `1..max_temporal_mask` time steps.

        Args:
            x: Tensor of shape `(2, K, T)` or `(B, 2, K, T)`.

        Returns:
            Time-masked tensor of the same shape.
        """
        num_t = x.shape[-1]
        max_block = min(self.max_temporal_mask, num_t)
        if max_block <= 0:
            return x

        block_len = int(self._rng.integers(1, max_block + 1))
        start_t = int(self._rng.integers(0, num_t - block_len + 1))
        out = x.clone()
        out[..., :, start_t : start_t + block_len] = 0.0
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply training-only RF/CSI augmentations to a normalized CSI tensor.

        Args:
            x: Normalized CSI tensor of shape `(2, K, T)` or `(B, 2, K, T)`.

        Returns:
            Augmented tensor of matching shape and dtype.

        Raises:
            ValueError: If `x.ndim` is not 3 or 4.
        """
        if x.ndim not in (3, 4):
            raise ValueError(
                f"CSIAugmenter expects a 3D (2, K, T) or 4D (B, 2, K, T) tensor, "
                f"got shape {tuple(x.shape)}."
            )

        out = x.clone()
        if self._rng.random() <= self.p_noise:
            out = self.apply_gaussian_jitter(out)
        if self._rng.random() <= self.p_subcarrier_mask:
            out = self.apply_subcarrier_mask(out)
        if self._rng.random() <= self.p_temporal_mask:
            out = self.apply_temporal_mask(out)
        return out


class AugmentedCSIDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset wrapper applying `CSIAugmenter` on the fly to a training subset.

    Attributes:
        base_dataset: Underlying dataset or `torch.utils.data.Subset`.
        augmenter: Optional `CSIAugmenter` applied when not `None`.
    """

    def __init__(
        self,
        base_dataset: Dataset[tuple[torch.Tensor, int]],
        augmenter: CSIAugmenter | None = None,
    ) -> None:
        """Initialize the `AugmentedCSIDataset` wrapper.

        Args:
            base_dataset: Base dataset returning `(tensor, label)`.
            augmenter: `CSIAugmenter` instance for training, or `None` for eval splits.
        """
        super().__init__()
        self.base_dataset: Dataset[tuple[torch.Tensor, int]] = base_dataset
        self.augmenter: CSIAugmenter | None = augmenter

    def __len__(self) -> int:
        """Return number of samples in the wrapped dataset."""
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Fetch `(x, y)` and apply `augmenter(x)` if configured."""
        x, y = self.base_dataset[index]
        if self.augmenter is not None:
            x = self.augmenter(x)
        return x, int(y)
