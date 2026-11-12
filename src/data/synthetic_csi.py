"""Physics-based synthetic Wi-Fi CSI dataset generator for human sensing.

Generates realistic complex Channel Frequency Response (CFR) windows
H in C^{K x T} over IEEE 802.11n 20 MHz OFDM subcarriers (K = 52 default or
K = 90 for Widar3.0) sampled at f_s = 100 Hz for W = 2.0 s (T = 200 packets).

Simulates five human presence and motion classes:
    0: Empty               - Static multipath reflections + low-variance AWGN.
    1: Stationary_Presence - Static multipath + periodic chest respiration (0.2-0.4 Hz).
    2: Walking             - Sustained multi-frequency Doppler oscillations (2.0-8.0 Hz).
    3: Sitting_Standing    - Smooth localized transient velocity envelope (1.0-4.0 Hz).
    4: Falling             - High-velocity sudden transient spike (6.0-14.0 Hz)
                             followed by abrupt post-impact stillness.

Hardware impairments modeled on every window:
    - Per-packet Carrier Frequency Offset (CFO) / SFO linear phase slope across
      symmetric logical subcarriers m_k and random per-packet phase offset.
    - Random impulsive amplitude and phase spike outliers.
    - Complex Additive White Gaussian Noise (AWGN).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

from src.dsp.preprocessor import SignalPreprocessor

CLASS_LABELS: dict[int, str] = {
    0: "Empty",
    1: "Stationary_Presence",
    2: "Walking",
    3: "Sitting_Standing",
    4: "Falling",
}


class SyntheticCSIDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch Dataset synthesizing physics-based complex Wi-Fi CSI windows.

    Attributes:
        num_samples: Total number of windows in the dataset.
        num_classes: Number of target activity classes (5).
        num_subcarriers: Number of OFDM subcarriers K (default: 52; supports 90).
        window_size: Number of time packets T per window (default: 200 for 2.0 s @ 100 Hz).
        stride: Sliding window stride in packets (default: 100 for 50% overlap).
        fs: Packet sampling frequency in Hz (default: 100.0 Hz).
        seed: Deterministic random seed (default: 42).
        preprocessor: Instance of `SignalPreprocessor` applied to raw complex CFR.
        labels: 1D NumPy int64 array of class labels in `{0, 1, 2, 3, 4}`.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_subcarriers: int = 52,
        window_size: int = 200,
        stride: int = 100,
        fs: float = 100.0,
        seed: int = 42,
        preprocessor: SignalPreprocessor | None = None,
        precompute: bool = False,
    ) -> None:
        """Initialize the physics-based SyntheticCSIDataset.

        Args:
            num_samples: Total number of CSI windows to generate across all 5 classes.
            num_subcarriers: Number of subcarriers K (even integer, e.g., 52 or 90).
            window_size: Number of packets T per window (default: 200).
            stride: Packet stride between consecutive windows (default: 100).
            fs: Sampling rate in Hz (default: 100.0).
            seed: Deterministic random seed (default: 42).
            preprocessor: Optional custom `SignalPreprocessor`. If None, a default
                `SignalPreprocessor(num_subcarriers=num_subcarriers, fs=fs)` is created.
            precompute: If True, precomputes and caches all preprocessed tensors in RAM
                upon initialization. If False, synthesizes and preprocesses deterministically
                on the fly per sample index.

        Raises:
            ValueError: If `num_samples <= 0`, `num_subcarriers` is not a positive even
                integer, or `window_size < 32`.
        """
        super().__init__()
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if num_subcarriers <= 0 or num_subcarriers % 2 != 0:
            raise ValueError(
                f"num_subcarriers must be a positive even integer, got {num_subcarriers}."
            )
        if window_size < 32:
            raise ValueError(f"window_size must be >= 32, got {window_size}.")

        self.num_samples: int = num_samples
        self.num_classes: int = len(CLASS_LABELS)
        self.num_subcarriers: int = num_subcarriers
        self.window_size: int = window_size
        self.stride: int = stride
        self.fs: float = float(fs)
        self.seed: int = seed

        self.preprocessor: SignalPreprocessor = (
            preprocessor
            if preprocessor is not None
            else SignalPreprocessor(num_subcarriers=num_subcarriers, fs=fs)
        )

        # Time vector t in [0, W) seconds of shape (1, T)
        self._time_vec: NDArray[np.float64] = (
            np.arange(self.window_size, dtype=np.float64)[np.newaxis, :] / self.fs
        )
        # Symmetric subcarrier index vector m_k of shape (K, 1)
        self._subcarrier_indices: NDArray[np.float64] = (
            self.preprocessor.subcarrier_indices[:, np.newaxis]
        )
        # Normalized subcarrier progression in [0, 1] of shape (K, 1) for multipath diversity
        self._norm_k: NDArray[np.float64] = np.linspace(
            0.0, 1.0, self.num_subcarriers, dtype=np.float64
        )[:, np.newaxis]

        # Deterministic balanced label assignment shuffled with fixed seed
        rng = np.random.default_rng(self.seed)
        base_labels = np.arange(self.num_samples, dtype=np.int64) % self.num_classes
        rng.shuffle(base_labels)
        self.labels: NDArray[np.int64] = base_labels

        # Per-sample deterministic child seeds so __getitem__(idx) is order-independent
        self._sample_seeds: NDArray[np.uint64] = rng.integers(
            low=0,
            high=2**62,
            size=self.num_samples,
            dtype=np.uint64,
        )

        self._cached_tensors: list[torch.Tensor] | None = None
        if precompute:
            self._cached_tensors = [
                self.preprocessor.process(self.generate_raw_csi(idx))
                for idx in range(self.num_samples)
            ]

    def __len__(self) -> int:
        """Return total number of CSI windows in the dataset."""
        return self.num_samples

    def _generate_static_multipath(
        self,
        rng: np.random.Generator,
        num_paths: int = 4,
    ) -> NDArray[np.complex128]:
        """Synthesize static background multipath CFR H_static(k, t) in C^{K x T}.

        Args:
            rng: Seeded NumPy random generator.
            num_paths: Number of static reflection paths.

        Returns:
            Complex128 static CFR array of shape (K, T).
        """
        h_static = np.zeros((self.num_subcarriers, 1), dtype=np.complex128)
        for _ in range(num_paths):
            amp = rng.uniform(0.35, 0.95)
            delay_slope = rng.uniform(-2.5 * np.pi, 2.5 * np.pi)
            init_phase = rng.uniform(-np.pi, np.pi)
            h_static += amp * np.exp(1j * (delay_slope * self._norm_k + init_phase))

        # Broadcast static response across all T packets with slow thermal drift
        slow_drift_freq = rng.uniform(0.01, 0.04)
        slow_drift = 0.01 * np.sin(2.0 * np.pi * slow_drift_freq * self._time_vec)
        return h_static * np.exp(1j * slow_drift)

    def _simulate_empty(
        self,
        rng: np.random.Generator,
    ) -> NDArray[np.complex128]:
        """Class 0 (Empty): Static multipath reflections + low-variance thermal floor.

         Additionally injects a micro-harmonic ambient vibration (0.12 - 0.18 Hz, very low
        amplitude) so the bandpass filter sees realistic baseline environmental noise.
        """
        h_static = self._generate_static_multipath(rng, num_paths=4)
        env_freq = rng.uniform(0.11, 0.17)
        env_phase = rng.uniform(-np.pi, np.pi)
        env_ripple = 0.008 * np.cos(2.0 * np.pi * env_freq * self._time_vec + env_phase)
        return h_static * (1.0 + env_ripple)

    def _simulate_stationary_presence(
        self,
        rng: np.random.Generator,
    ) -> NDArray[np.complex128]:
        """Class 1 (Stationary_Presence): Static paths + periodic chest respiration (0.2-0.4 Hz).

        Models sub-centimeter chest wall displacement modulating a dynamic scatterer path
        at breathing rate f_resp in [0.2, 0.4] Hz with subcarrier-selective sensitivity.
        """
        h_static = self._generate_static_multipath(rng, num_paths=4)
        resp_freq = rng.uniform(0.20, 0.40)
        resp_init_phase = rng.uniform(-np.pi, np.pi)
        resp_amp = rng.uniform(0.18, 0.35)

        # Subcarrier sensitivity profile (Fresnel zone variation across subcarriers)
        subcarrier_weights = 0.6 + 0.4 * np.sin(
            2.0 * np.pi * rng.uniform(1.0, 2.5) * self._norm_k + rng.uniform(0.0, np.pi)
        )
        chest_disp = np.sin(2.0 * np.pi * resp_freq * self._time_vec + resp_init_phase)
        # Second harmonic for realistic inhalation/exhalation asymmetry
        chest_disp += 0.20 * np.sin(
            4.0 * np.pi * resp_freq * self._time_vec + resp_init_phase + 0.4
        )

        dynamic_amp = resp_amp * subcarrier_weights * (1.0 + 0.35 * chest_disp)
        dynamic_phase = (
            rng.uniform(-np.pi, np.pi)
            + 1.2 * self._norm_k
            + rng.uniform(0.45, 0.85) * subcarrier_weights * chest_disp
        )
        h_dynamic = dynamic_amp * np.exp(1j * dynamic_phase)
        return h_static + h_dynamic

    def _simulate_walking(
        self,
        rng: np.random.Generator,
    ) -> NDArray[np.complex128]:
        """Class 2 (Walking): Sustained multi-frequency Doppler oscillations (2.0-8.0 Hz).

        Models torso and swinging limb scatterers producing continuous multi-tone Doppler
        shifts and amplitude fading across the entire 2.0 s window.
        """
        h_static = self._generate_static_multipath(rng, num_paths=3)
        h_dynamic = np.zeros(
            (self.num_subcarriers, self.window_size), dtype=np.complex128
        )

        # Torso primary Doppler component + 2 limb harmonic/independent components
        num_scatterers = 3
        freq_bands = [(2.0, 4.0), (3.8, 6.0), (5.5, 8.0)]
        for idx in range(num_scatterers):
            f_low, f_high = freq_bands[idx]
            f_doppler = rng.uniform(f_low, f_high)
            f_mod = rng.uniform(0.8, 1.8)  # Step cadence modulation
            scatter_amp = rng.uniform(0.22, 0.45) / (idx + 1.0) ** 0.4
            k_slope = rng.uniform(-3.0, 3.0)
            init_phi = rng.uniform(-np.pi, np.pi)

            amp_env = scatter_amp * (
                1.0
                + 0.45 * np.cos(2.0 * np.pi * f_doppler * self._time_vec + init_phi)
                + 0.25 * np.sin(2.0 * np.pi * f_mod * self._time_vec)
            )
            phase_evol = (
                2.0 * np.pi * f_doppler * self._time_vec
                + 0.6 * np.sin(2.0 * np.pi * f_mod * self._time_vec)
                + k_slope * self._norm_k
                + init_phi
            )
            h_dynamic += amp_env * np.exp(1j * phase_evol)

        return h_static + h_dynamic

    def _simulate_sitting_standing(
        self,
        rng: np.random.Generator,
    ) -> NDArray[np.complex128]:
        """Class 3 (Sitting_Standing): Smooth localized transient envelope (1.0-4.0 Hz).

        Models a sit-down or stand-up transition localized near the middle of the window
         via a Gaussian bell envelope modulating a 1.0-4.0 Hz vertical/horizontal torso path,
        plus subtle stationary breathing before and after the transition.
        """
        h_static = self._generate_static_multipath(rng, num_paths=4)
        duration_sec = self.window_size / self.fs
        center_t = rng.uniform(0.32 * duration_sec, 0.68 * duration_sec)
        width_sigma = rng.uniform(0.18, 0.30)

        # Smooth Gaussian transient envelope centered mid-window
        envelope = np.exp(-0.5 * ((self._time_vec - center_t) / width_sigma) ** 2)

        trans_freq = rng.uniform(1.0, 4.0)
        init_phase = rng.uniform(-np.pi, np.pi)
        subcarrier_profile = 0.7 + 0.3 * np.cos(
            2.0 * np.pi * self._norm_k + rng.uniform(-np.pi, np.pi)
        )

        # Transient amplitude and phase excursion
        trans_osc = np.sin(2.0 * np.pi * trans_freq * self._time_vec + init_phase)
        amp_mod = (
            rng.uniform(0.35, 0.65)
            * subcarrier_profile
            * envelope
            * (0.6 + 0.4 * trans_osc)
        )
        phase_mod = (
            rng.uniform(1.0, 1.8) * envelope * trans_osc
            + 1.5 * self._norm_k
            + init_phase
        )

        # Subtle background respiration outside the transition
        resp_freq = rng.uniform(0.22, 0.36)
        resp_wave = 0.06 * np.sin(2.0 * np.pi * resp_freq * self._time_vec)

        h_dynamic = (amp_mod + resp_wave) * np.exp(1j * phase_mod)
        return h_static + h_dynamic

    def _simulate_falling(
        self,
        rng: np.random.Generator,
    ) -> NDArray[np.complex128]:
        """Class 4 (Falling): High-velocity sudden spike (6.0-14.0 Hz) + abrupt stillness.

        Models rapid acceleration during a fall (chirped high-frequency Doppler burst in
        6.0-14.0 Hz) followed by a sharp floor-impact decay and post-fall stillness.
        """
        h_static = self._generate_static_multipath(rng, num_paths=3)
        duration_sec = self.window_size / self.fs
        fall_start = rng.uniform(0.20 * duration_sec, 0.45 * duration_sec)
        impact_time = fall_start + rng.uniform(0.28, 0.48)

        # Asymmetric envelope: rapid build-up until impact_time, then sharp exponential cutoff
        rise_sigma = rng.uniform(0.10, 0.16)
        decay_tau = rng.uniform(0.03, 0.06)
        pre_impact = np.exp(-0.5 * ((self._time_vec - impact_time) / rise_sigma) ** 2)
        post_impact = np.exp(-np.maximum(0.0, self._time_vec - impact_time) / decay_tau)
        fall_envelope = np.where(
            self._time_vec <= impact_time,
            pre_impact,
            post_impact,
        )

        # High-velocity Doppler chirp from f_start (6-9 Hz) to f_end (10-14 Hz)
        f_start = rng.uniform(6.0, 9.0)
        f_end = rng.uniform(10.0, 14.0)
        chirp_rate = (f_end - f_start) / max(impact_time - fall_start, 0.2)
        rel_t = np.clip(self._time_vec - fall_start, 0.0, duration_sec)
        inst_phase = 2.0 * np.pi * (f_start * rel_t + 0.5 * chirp_rate * (rel_t**2))

        subcarrier_gain = 0.65 + 0.35 * np.sin(
            2.0 * np.pi * 1.5 * self._norm_k + rng.uniform(0.0, np.pi)
        )
        burst_amp = (
            rng.uniform(0.60, 1.05)
            * subcarrier_gain
            * fall_envelope
            * (0.65 + 0.35 * np.cos(inst_phase))
        )
        burst_phase = (
            rng.uniform(1.4, 2.4) * fall_envelope * np.sin(inst_phase)
            + 2.0 * self._norm_k
        )

        h_dynamic = burst_amp * np.exp(1j * burst_phase)
        return h_static + h_dynamic

    def _inject_hardware_impairments(
        self,
        csi_clean: NDArray[np.complex128],
        rng: np.random.Generator,
        class_id: int,
    ) -> NDArray[np.complex128]:
        """Inject realistic Wi-Fi NIC hardware impairments into clean CFR H(k, t).

        Impairments injected:
            1. Random per-packet linear phase slope `a_t * m_k` (SFO/packet detection delay)
               and uniform per-packet constant phase offset `b_t in [-pi, pi]` (CFO/PLL).
            2. Impulsive spike outliers in amplitude and phase across random `(k, t)` entries.
            3. Complex circular Additive White Gaussian Noise (AWGN).

        Args:
            csi_clean: Clean complex128 CFR matrix of shape `(K, T)`.
            rng: Seeded NumPy random generator.
            class_id: Activity class index in `{0, 1, 2, 3, 4}`.

        Returns:
            Impaired complex128 CFR matrix of shape `(K, T)`.
        """
        # 1. Complex AWGN (lower variance for Empty/Stationary so subtle respiration is clear)
        if class_id in (0, 1):
            noise_std = rng.uniform(0.015, 0.035)
        else:
            noise_std = rng.uniform(0.025, 0.065)

        awgn = (
            rng.normal(0.0, noise_std, size=csi_clean.shape)
            + 1j * rng.normal(0.0, noise_std, size=csi_clean.shape)
        ) / np.sqrt(2.0)
        csi_noisy = csi_clean + awgn

        # 2. Impulsive spike outliers (1.2% of samples)
        outlier_prob = 0.012
        outlier_mask = rng.random(size=csi_clean.shape) < outlier_prob
        if np.any(outlier_mask):
            spike_scale = rng.uniform(2.5, 4.5, size=csi_clean.shape)
            spike_phase = rng.uniform(-np.pi, np.pi, size=csi_clean.shape)
            csi_noisy = np.where(
                outlier_mask,
                csi_noisy * spike_scale * np.exp(1j * spike_phase),
                csi_noisy,
            )

        # 3. Per-packet linear phase distortion exp(j * (a_t * m_k + b_t))
        cfo_slope_t = rng.normal(loc=0.0, scale=0.18, size=(1, self.window_size))
        cfo_offset_t = rng.uniform(-np.pi, np.pi, size=(1, self.window_size))
        phase_distortion = cfo_slope_t * self._subcarrier_indices + cfo_offset_t

        csi_impaired = csi_noisy * np.exp(1j * phase_distortion)
        return csi_impaired.astype(np.complex128)

    def generate_raw_csi(self, index: int) -> NDArray[np.complex128]:
        """Generate a single raw complex CFR window `H in C^{K x T}` for dataset `index`.

        Args:
            index: Sample index in `[0, num_samples)`.

        Returns:
            Complex128 NumPy array of shape `(K, T)`.

        Raises:
            IndexError: If `index` is out of bounds.
        """
        if index < 0 or index >= self.num_samples:
            raise IndexError(
                f"Sample index {index} out of range for dataset of size {self.num_samples}."
            )

        class_id = int(self.labels[index])
        sample_rng = np.random.default_rng(int(self._sample_seeds[index]))

        simulators: dict[int, Callable[[np.random.Generator], NDArray[np.complex128]]] = {
            0: self._simulate_empty,
            1: self._simulate_stationary_presence,
            2: self._simulate_walking,
            3: self._simulate_sitting_standing,
            4: self._simulate_falling,
        }
        csi_clean = simulators[class_id](sample_rng)
        csi_raw = self._inject_hardware_impairments(csi_clean, sample_rng, class_id)
        return csi_raw

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return `(preprocessed_tensor, label)` for sample `index`.

        Args:
            index: Sample index in `[0, num_samples)`.

        Returns:
            Tuple `(x, y)` where `x` is a preprocessed `torch.float32` tensor of
            shape `(2, K, T)` and `y` is the integer class label in `{0, 1, 2, 3, 4}`.
        """
        if self._cached_tensors is not None:
            return self._cached_tensors[index], int(self.labels[index])

        raw_csi = self.generate_raw_csi(index)
        preprocessed_tensor = self.preprocessor.process(raw_csi)
        label = int(self.labels[index])
        return preprocessed_tensor, label
